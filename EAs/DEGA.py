import random
import numpy as np

class Pop:
    def __init__(self, X, func):
        self.X = X
        self.func = func
        self.ObjV = None

    def __add__(self, other):
        self.X = np.vstack([self.X, other.X])
        self.ObjV = np.hstack([self.ObjV, other.ObjV])
        return self

    def cal_fitness(self):  # 计算目标值
        self.ObjV = self.func(self.X).ravel()


class DEGA(object):
    def __init__(self, func, max_iter, dim, lb, ub, r, N_lst, initX=None):

        self.max_iter = max_iter
        if initX is None:
            self.popsize = 100
        else:
            self.popsize = initX.shape[0]
        self.dim = dim

        self.xmax = np.array(ub)  # 上界
        self.xmin = np.array(lb)  # 下界

        self.func = func
        self.N_lst = N_lst
        self.r = r  # 连续变量个数
        self.o = self.dim - self.r

        self.pop = None
        self.xbest = None
        self.ybest = None
        self.y_lst = []
        self.initX = initX

    def initPop(self):

        X = np.zeros((self.popsize, self.dim))
        area = self.xmax - self.xmin

        for j in range(self.r):
            for i in range(self.popsize):
                X[i, j] = self.xmin[j] + np.random.uniform(i / self.popsize * area[j],
                                                           (i + 1) / self.popsize * area[j])
            np.random.shuffle(X[:, j])

        for j in range(self.r, self.dim):
            for i in range(self.popsize):
                X[i, j] = i % self.N_lst[j - self.r]
            np.random.shuffle(X[:, j])

        self.pop = Pop(X, self.func)
        self.pop.cal_fitness()

    def DEGAoperator(self):
        muXr = np.empty((self.popsize, self.r))
        # DE/rand/1
        b = np.argmin(self.pop.ObjV)
        for i in range(self.popsize):
            r1 = r2 = r3 = 0
            while r1 == i or r2 == i or r3 == i or r2 == r1 or r3 == r1 or r3 == r2:
                r1 = random.randint(0, self.popsize - 1)
                r2 = random.randint(0, self.popsize - 1)
                r3 = random.randint(0, self.popsize - 1)

            mutationXr = 0.5 * self.pop.X[b, :self.r] + 0.5 * self.pop.X[r1, :self.r] + 0.5 * (
                    self.pop.X[r2, :self.r] - self.pop.X[r3, :self.r])

            for j in range(self.r):
                #  判断变异后的值是否满足边界条件，不满足需重新生成
                if self.xmin[j] <= mutationXr[j] <= self.xmax[j]:
                    muXr[i, j] = mutationXr[j]
                else:
                    rand_value = self.xmin[j] + random.random() * (self.xmax[j] - self.xmin[j])
                    muXr[i, j] = rand_value

        cmXr = np.empty((self.popsize, self.r))
        for i in range(self.popsize):
            for j in range(self.r):
                rj = random.randint(0, self.r - 1)
                rf = random.random()
                if rf <= 0.8 or rj == j:
                    cmXr[i, j] = muXr[i, j]
                else:
                    cmXr[i, j] = self.pop.X[i, j]

        cmXd = np.zeros((self.popsize, self.o))
        encodelengths = []
        for j in range(self.o):
            encodelengths.append(int(np.log2(self.N_lst[j] - 1) + 1))

        m, n = self.popsize, sum(encodelengths)
        encodeXc = np.empty((m, n), dtype=np.uint8)
        for i in range(m):
            chrome = []
            for j in range(self.r, self.dim):
                # ind = np.where(self.T[j - self.r] == self.pop.X[i, j])[0][0]
                b = [int(c) for c in bin(int(self.pop.X[i, j])).split('b')[1]]
                chrome += [0 for i in range(encodelengths[j - self.r] - len(b))] + b
            encodeXc[i, :] = np.array(chrome)
        # 变异
        mutationGeneIndex = random.sample(range(0, m * n), np.uint8(m * n * 0.3))  # Pm = 0.01
        for gene in mutationGeneIndex:
            chromosomeIndex = gene // n
            geneIndex = gene % n
            if encodeXc[chromosomeIndex, geneIndex] == 0:
                encodeXc[chromosomeIndex, geneIndex] = 1
            else:
                encodeXc[chromosomeIndex, geneIndex] = 0
        # 交叉
        # 确保进行交叉的染色体个数是偶数个
        numbers = np.uint8(self.popsize * 0.8)  # Pc = 0.8
        if numbers % 2 != 0:
            numbers += 1
        # 产生随机索引
        index = random.sample(range(self.popsize), numbers)
        encodeXc2 = encodeXc.copy()
        while len(index) > 0:
            a = index.pop()
            b = index.pop()
            # 随机产生一个交叉点
            crossoverPoint = random.sample(range(1, n - 1), 1)[0]
            # one-single-point crossover
            encodeXc2[a, 0:crossoverPoint] = encodeXc[a, 0:crossoverPoint]
            encodeXc2[a, crossoverPoint:] = encodeXc[b, crossoverPoint:]
            encodeXc2[b, 0:crossoverPoint] = encodeXc[b, 0:crossoverPoint]
            encodeXc2[b, crossoverPoint:] = encodeXc[a, crossoverPoint:]
        # 转化为整数
        for i in range(m):
            chrome = encodeXc2[i].ravel().tolist()  # 拆分
            L = []
            s = 0
            for j in range(self.o):
                L.append(chrome[s:s + encodelengths[j]])
                s = s + encodelengths[j]
            for j in range(self.o):
                intx = sum([L[j][t] * 2 ** (encodelengths[j] - t - 1) for t in range(encodelengths[j])])
                cmXd[i, j] = intx % self.N_lst[j]

        cmX1 = np.hstack([cmXr, cmXd])

        return cmX1

    def update_best(self):
        rank = np.argmin(self.pop.ObjV)
        self.xbest = self.pop.X[rank]
        self.ybest = self.pop.ObjV[rank]

    # 选择算子:从父代和子代中选
    def selection(self, pop, offs):
        mpop = pop + offs
        inds = np.argsort(mpop.ObjV)

        spop = Pop(mpop.X[inds[:self.popsize]], self.func)
        spop.ObjV = mpop.ObjV[inds[:self.popsize]]
        return spop

    def run(self):
        if self.initX is None:
            self.initPop()
        else:
            self.pop = Pop(self.initX, self.func)
            self.pop.cal_fitness()

        self.update_best()
        for i in range(self.max_iter):
            cmX = self.DEGAoperator()
            epop = Pop(cmX, self.func)
            epop.cal_fitness()

            self.pop = self.selection(self.pop, epop)
            self.y_lst.append(self.ybest)
            self.update_best()

        return self.xbest, self.ybest, self.y_lst


