# Surrogate-Assisted Hybrid Evolutionary Algorithm with Local Estimation
# of Distribution (SHEALED)
# ------------------------------- Reference --------------------------------
# Liu Y, Wang H. Surrogate-Assisted Hybrid Evolutionary Algorithm with Local
# Estimation of Distribution for Expensive Mixed-Variable Optimization Problems
# in Applied Soft Computing.
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2022 HandingWangXD Group. Permission is granted to copy and
# use this code for research, noncommercial purposes, provided this
# copyright notice is retained and the origin of the code is cited. The
# code is provided "as is" and without any warranties, express or implied.
# ------------------------------- Developer --------------------------------
# This code is written by Yongcun Liu. Email: yongcunl@stu.xidian.edu.cn

import numpy as np
from scipy.stats import norm, mode

from EAs.DE import DE
from EAs.DEGA import DEGA
from Surrogate.RBFNmv import RBFNmv
from smt.surrogate_models import KRG

# Population class
class Pop:
    def __init__(self, X, F):
        self.X = X
        self.F = F    # evaluation function
        self.realObjV = None
        self.predObjV = None

    def pred_fit(self, sm):
        self.predObjV = sm.predict(self.X)

    # function evaluation
    def cal_fitness(self, X):
        return self.F(X)

# Algorithm class
class SHEALED(object):
    def __init__(self, maxFEs, popsize, dim, clb, cub, N_lst, prob, r, database=None):
        '''
        :param maxFEs: the maximum number of function evaluations
        :param popsize: the size of population
        :param dim: the dimension of decision variables
        :param clb: the lower bounds of continuous decision variables
        :param cub: the upper bounds of continuous decision variables
        :param N_lst: the number of values for discrete variables
        :param prob: an expensive mixed-variable optimization problem instance
        :param r: the dimension of continuous decision variables
        '''

        self.maxFEs = maxFEs
        self.popsize = popsize

        self.dim = dim
        self.cxmin = np.array(clb)
        self.cxmax = np.array(cub)
        self.N_lst = N_lst

        self.prob = prob
        self.r = r
        self.o = self.dim - self.r

        # surrogate models
        self.global_sm = None
        self.local_sm1 = None
        self.local_sm2 = None
        self.sm3 = None

        self.pop = None
        self.database = None
        self.init_size = 100  # the size of initial samples
        self.gen = None

        self.xbest = None
        self.ybest = None
        self.ybest_lst = []

        self.data = None
        self.melst = []

    # build or update the global surrogate model using all data
    def updateGSM(self):

        xtrain = self.database[0]
        ytrain = self.database[1]

        sm = RBFNmv(self.dim, self.N_lst, self.cxmin, self.cxmax)
        sm.fit(xtrain, ytrain)

        self.global_sm = sm

    def select_nearbest(self):
        select_inds = np.argsort(np.sum((self.xbest - self.database[0]) ** 2, axis=1))[:self.popsize]
        return select_inds

    # build or update the local surrogate model for Weighted EDA using the samples
    # near to the current best in decision space
    def updateLSM1(self):
        select_inds = self.select_nearbest()
        xtrain = self.database[0][select_inds]
        ytrain = self.database[1][select_inds]

        sm = RBFNmv(self.dim, self.N_lst, self.cxmin, self.cxmax)
        sm.fit(xtrain, ytrain)
        self.local_sm1 = sm

    # build or update the local surrogate model for Weighted EDA using the samples
    # closed to the current best in objective space
    def updateLSM2(self):
        xtrain = self.database[0][:min(self.gen, 5 * self.dim)]
        ytrain = self.database[1][:min(self.gen, 5 * self.dim)]

        sm = RBFNmv(self.dim, self.N_lst, self.cxmin, self.cxmax)
        sm.fit(xtrain, ytrain)
        self.local_sm2 = sm

    # calculate the diversity indicator of X in current population
    def calDI(self, X):
        # 求均值
        Xmean = np.zeros(self.dim)
        Xmean[:self.r] = np.mean(X[:, :self.r], axis = 0)
        for j in range(self.r, self.dim):
            Xmean[j] = mode(X[:,j])[0][0]

        d = np.hstack([X[:, :self.r] - Xmean[:self.r].reshape(1,-1), X[:, self.r:] != Xmean[self.r:].reshape(1,-1)])
        return np.sqrt((1/self.popsize)*np.sum(d**2))

    # Inilization
    def initPop(self):

        X = np.zeros((self.init_size, self.dim))
        area = self.cxmax - self.cxmin
        # continuous part
        for j in range(self.r):
            for i in range(self.init_size):
                X[i, j] = self.cxmin[j] + np.random.uniform(i / self.popsize * area[j],
                                                            (i + 1) / self.popsize * area[j])
            np.random.shuffle(X[:, j])
        # discrete part
        for j in range(self.r, self.dim):
            for i in range(self.init_size):
                X[i, j] = int(np.random.uniform(i / self.popsize * self.N_lst[j - self.r],
                                                (i + 1) / self.popsize * self.N_lst[j - self.r]))
            np.random.shuffle(X[:, j])

        inity = self.prob(X)
        self.gen = self.init_size
        inds = np.argsort(inity)
        self.database = [X[inds], inity[inds]]
        self.data = [X[inds], inity[inds]]

        popInds = inds[:self.popsize]
        self.pop = Pop(X[popInds], self.prob)
        self.pop.realObjV = inity[popInds]

        self.DIini = self.calDI(self.pop.X)
        self.xbest = self.database[0][0]
        self.ybest = self.database[1][0]

    # Check for duplicate samples
    def check(self, x):
        num = self.database[1].shape[0]
        for i in range(num):
            if (np.all(x[0] == self.database[0][i])):
                return False
        return True

    # update the database and current population
    def update_database(self, X, y):
        self.data[0] = np.r_[self.data[0], X]
        self.data[1] = np.append(self.data[1], y)

        size = len(self.database[1])
        for i in range(size):
            if (self.database[1][i] > y):
                self.database[0] = np.insert(self.database[0], i, X, axis=0)
                self.database[1] = np.insert(self.database[1], i, y)
                break

        self.pop.X = self.database[0][:self.popsize]
        self.pop.realObjV = self.database[1][:self.popsize]
        self.xbest = self.database[0][0]
        self.ybest = self.database[1][0]

    # Weighted EDA
    def EDAmv1(self, prom_inds):
        n_best = len(prom_inds)
        promX = self.pop.X[prom_inds]
        promy = self.pop.realObjV[prom_inds]

        # calculate the weights
        fits1 = np.exp((np.max(promy) - promy + (1e-10)) / (np.max(promy) - np.min(promy) + (1e-10)))
        ps1 = fits1 / np.sum(fits1)

        oXr1 = np.zeros((2 * self.popsize, self.r))

        # build Gaussian distribution for continuous variables
        mu_r = np.sum(promX[:, :self.r] * ps1.reshape(-1, 1), axis=0)
        sigma_r = np.sqrt(np.sum((promX[:, :self.r] - mu_r) ** 2, axis=0) / n_best)

        # sampling the continuous decision variables
        for i in range(2 * self.popsize):
            for j in range(self.r):
                oXr1[i, j] = np.clip(norm.rvs(mu_r[j], sigma_r[j]), self.cxmin[j],
                                     self.cxmax[j])

        # build the probability matrix for discrete variables
        prob_d = [np.zeros(self.N_lst[j]) for j in range(self.o)]
        for j in range(self.r, self.dim):
            for k in range(self.N_lst[j - self.r]):
                for i in range(n_best):
                    if (promX[i, j] == k):
                        prob_d[j - self.r][k] += fits1[i]

        # sampling the discrete decision variables
        oXc1 = np.empty((2 * self.popsize, self.o))
        for i in range(2 * self.popsize):
            for j in range(self.r, self.dim):
                oXc1[i, j - self.r] = np.random.choice(
                    np.arange(self.N_lst[j - self.r]), 1,
                    p=prob_d[j - self.r] / np.sum(prob_d[j - self.r]))

        oX1 = np.hstack([oXr1, oXc1])
        return oX1

    # Selected EDA
    def EDAmv2(self, prom_inds):
        n_best = len(prom_inds)
        promX = self.pop.X[prom_inds]
        promy = self.pop.realObjV[prom_inds]

        # Calculate the selection probability for all decision variables using the rank
        rank = np.ones(n_best)
        rank[np.argsort(promy)] = np.arange(n_best)
        ps2 = (n_best - rank)/(((1 + n_best) * n_best) / 2)

        oXr2 = np.zeros((2 * self.popsize, self.r))

        # sampling continuous
        for i in range(2 * self.popsize):
            for j in range(self.r):
                ind_sel = np.random.choice(range(n_best), 1, p=ps2)
                muj = promX[ind_sel, j]
                std = np.std(promX[:, j])

                oXr2[i, j] = np.clip(norm.rvs(muj, 0.1 * std), self.cxmin[j],
                                     self.cxmax[j])

        # for discrete variables
        prob_d = [np.zeros(self.N_lst[j]) for j in range(self.o)]

        for j in range(self.r, self.dim):
            for k in range(self.N_lst[j - self.r]):
                for i in range(n_best):
                    if (promX[i, j] == k):
                        prob_d[j - self.r][k] += ps2[i]

        oXc2 = np.empty((2 * self.popsize, self.o))
        for i in range(2 * self.popsize):
            for j in range(self.r, self.dim):
                oXc2[i, j - self.r] = np.random.choice(
                    np.arange(self.N_lst[j - self.r]), 1,
                    p=prob_d[j - self.r] / np.sum(prob_d[j - self.r]))

        oX2 = np.hstack([oXr2, oXc2])
        return oX2

    # Select samples with the same value as the current optimal discrete values for continuous refining
    def data_selection2(self):
        best_c = self.xbest[self.r:]
        inds = []
        for i in range(len(self.database[1])):
            if (np.all(self.database[0][i, self.r:] == best_c)):
                inds.append(i)

        X_r = self.database[0][inds, :self.r]
        y_r = self.database[1][inds]
        size = len(y_r)

        if (size > 5 * self.r):
            ssinds = np.argsort(y_r)
            effsamples = []
            effsamples.append(ssinds[0])
            for i in range(1, size):
                if ((y_r[ssinds[i]] - y_r[ssinds[i - 1]]) / y_r[ssinds[i - 1]] > 1e-3):
                    effsamples.append(ssinds[i])

            if len(effsamples) > 11 * self.r:
                X_r = X_r[effsamples[:11 * self.r]]
                y_r = y_r[effsamples[:11 * self.r]]
            else:
                X_r = X_r[effsamples]
                y_r = y_r[effsamples]

            size = len(y_r)

        return size, X_r, y_r

    # Continuous refining
    def SAR_local(self, X_r, y_r):
        self.sm3 = KRG(print_global=False)
        self.sm3.set_training_values(X_r, y_r)
        self.sm3.train()

        ga = DE(max_iter=30, func=self.sm3.predict_values, dim=self.r, lb=self.cxmin, ub=self.cxmax,
                initX=X_r)
        X_l = ga.run()

        return np.append(X_l, self.xbest[self.r:]).reshape(1, -1)

    def run(self):
        '''
        :return:
        self.xbest: the optimal solution
        self.ybest: the optimal objective value
        self.ybest_lst: the historical best solution set
        self.data: the database
        self.melst: the historical sampling method record:
        1：The global hybrid EA； 2：Local Weighted EDA； 3：Local Selected EDA；
        4：continuous refining； 5：continuous regining with fake data；
        '''
        if self.database is None:
            self.initPop()
        else:

            initX = self.database[0]
            inity = self.database[1]
            inds = np.argsort(inity)

            self.data = [initX[inds], inity[inds], self.database[2][inds]]
            self.database = [initX[inds], inity[inds]]

            self.pop = Pop(initX, self.prob)
            self.pop.realObjV = inity

            self.DIini = self.calDI(self.pop.X)

            self.xbest = self.database[0][0]
            self.ybest = self.database[1][0]
            self.gen = len(self.database[1])


        flag = "l1"
        while (self.gen < self.maxFEs):

            DI = self.calDI(self.pop.X)
            rDI = DI / self.DIini
            if rDI < 0.9 or self.gen == 100:
                # Global hybrid EA
                self.updateGSM()

                dega = DEGA(max_iter=30, func=self.global_sm.predict, dim=self.dim, lb=self.cxmin,
                            ub=self.cxmax, r=self.r, N_lst=self.N_lst, initX=self.pop.X)

                x1 = dega.run()[0].reshape(1, -1)
                if self.check(x1):
                    y1 = self.pop.cal_fitness(x1)

                    print("{}/{} gen x1: {}".format(self.gen, self.maxFEs, y1))

                    if y1[0] < self.ybest:
                        self.DIini = DI

                    self.update_database(x1, y1)
                    self.melst.append(1)
                    self.ybest_lst.append(self.ybest)
                    self.gen += 1

            # Select excellent solutions from current population
            n_best = int(0.45 * self.popsize)
            prom_inds = np.argsort(self.pop.realObjV)[:n_best]

            # Switch-based local Estimation of Distributions
            if flag == "l1":
                self.updateLSM1()

                # Weighted EDA
                cmX1 = self.EDAmv1(prom_inds)
                epop1 = Pop(cmX1, self.prob)
                epop1.pred_fit(self.local_sm1)
                x2 = cmX1[np.argmin(epop1.predObjV)].reshape(1, -1)

                if self.check(x2) and self.gen < 600:
                    y2 = self.pop.cal_fitness(x2)
                    print("{}/{} gen x2: {}".format(self.gen, self.maxFEs, y2))
                    if (y2[0] < self.ybest):
                        flag = "l1"
                    else:
                        flag = "l2"
                    self.update_database(x2, y2)
                    self.melst.append(2)
                    self.ybest_lst.append(self.ybest)
                    self.gen += 1

            else:
                self.updateLSM2()

                # Selected EDA
                cmX2 = self.EDAmv2(prom_inds)
                epop2 = Pop(cmX2, self.prob)
                epop2.pred_fit(self.local_sm2)
                x3 = cmX2[np.argmin(epop2.predObjV)].reshape(1, -1)

                if self.check(x3) and self.gen < 600:
                    y3 = self.pop.cal_fitness(x3)
                    print("{}/{} gen x3: {}".format(self.gen, self.maxFEs, y3))
                    if (y3[0] < self.ybest):
                        flag = "l2"
                    else:
                        flag = "l1"
                    self.update_database(x3, y3)
                    self.melst.append(3)
                    self.ybest_lst.append(self.ybest)
                    self.gen += 1

            # Local continuous search
            size, X_r, y_r = self.data_selection2()
            if (size >= 5 * self.r):
                x4 = self.SAR_local(X_r, y_r)
                if (self.check(x4) and self.gen < 600):
                    y4 = self.pop.cal_fitness(x4)
                    print("{}/600 gen x4: {}".format(self.gen, y4))
                    self.update_database(x4, y4)
                    self.melst.append(4)
                    self.ybest_lst.append(self.ybest)
                    self.gen += 1

            elif (self.gen > 550 and size >= self.r):

                sup_n = 5 * self.r - size
                sup_x = np.zeros((sup_n, self.r))
                sup_y = np.zeros(sup_n)
                for i in range(sup_n):
                    sel = i % size
                    for j in range(self.r):
                        sup_x[i, j] = np.clip(norm.rvs(X_r[sel][j], 0.01 * abs(X_r[sel][j])), self.cxmin[j],
                                              self.cxmax[j])
                    sup_y[i] = y_r[sel]

                Xr = np.vstack([X_r, sup_x])
                yr = np.hstack([y_r, sup_y])
                x4 = self.SAR_local(Xr, yr)

                if (self.check(x4) and self.gen < 600):
                    y4 = self.pop.cal_fitness(x4)
                    print("{}/600 gen x4_sup: {}".format(self.gen, y4))
                    self.update_database(x4, y4)
                    self.melst.append(5)
                    self.ybest_lst.append(self.ybest)
                    self.gen += 1

        return self.xbest, self.ybest, self.ybest_lst, self.data, self.melst


