# Type2 mixed-variable optimization problems: the search range [-100, 100]
# of j-th decision variables evenly divided into lj-1 parts,
# and each part randomly takes a value.

import numpy as np
from opfunu.cec_based.cec2013 import F172013, F212013, F242013

class MVOPT2(object):
    def __init__(self, prob_name, dim_c, dim_d, N_d):
        '''
        :param prob_name: problem name, including "ellipsoid", "rosenbrock",
        "ackley","griewank","rastrigin","sphere","weierstrass","Lunacek","CF1","CF4"

        :param dim_r: dimension of continuous decision variables, including 5, 15, 25 for "ellipsoid", "rosenbrock",
        "ackley","griewank","rastrigin"; 5, 15 for "sphere","weierstrass","Lunacek","CF1","CF4".
        :param dim_d: dimension of discrete decision variables, including 5, 15, 25 for "ellipsoid", "rosenbrock",
        "ackley","griewank","rastrigin"; 5, 15 for "sphere","weierstrass","Lunacek","CF1","CF4".
        :param N_d: the number of values for discrete variables
        '''

        self.r = dim_c
        self.o = dim_d
        self.dim = dim_c + dim_d

        self.N_lst = []
        self.bounds = [-100, 100]

        self.x_shift = np.loadtxt("Benchmarks/shift_data/data_" + prob_name + '.txt')[:self.dim].reshape(1, -1)

        def ellipsoid(X):
            if isinstance(X, list):
                X = np.array(X)[np.newaxis, :]
            # decode
            tmp = X.copy()
            for i in range(X.shape[0]):
                for j in range(self.r, self.dim):
                    tmp[i, j] = self.T[j - self.r, int(X[i, j])]
            z = tmp - self.x_shift
            f = np.sum(np.array([(i + 1) * z[:, i] ** 2 for i in range(self.dim)]), axis=0)
            return f

        def rosenbrock(X):
            if isinstance(X, list):
                X = np.array(X)[np.newaxis, :]

            tmp = X.copy()
            for i in range(X.shape[0]):
                for j in range(self.r, self.dim):
                    tmp[i, j] = self.T[j - self.r, int(X[i, j])]

            z = tmp - self.x_shift + 1
            f = np.zeros(X.shape[0])
            for k in range(self.dim - 1):
                f += (100 * (z[:, k] ** 2 - z[:, k + 1]) ** 2 + (z[:, k] - 1) ** 2)
            return f

        def ackley(X):
            if isinstance(X, list):
                X = np.array(X)[np.newaxis, :]
            # decode
            tmp = X.copy()
            for i in range(X.shape[0]):
                for j in range(self.r, self.dim):
                    tmp[i, j] = self.T[j - self.r, int(X[i, j])]

            z = 0.32 * (tmp - self.x_shift)
            f = 20 + np.e - 20 * np.exp(-0.2 * np.sqrt((1 / self.dim) * np.sum(z ** 2, axis=1))) - np.exp(
                (1 / self.dim) * np.sum(np.cos(2 * np.pi * z), axis=1))
            return f

        def griewank(X):
            if isinstance(X, list):
                X = np.array(X)[np.newaxis, :]
            # decode
            tmp = X.copy()
            for i in range(X.shape[0]):
                for j in range(self.r, self.dim):
                    tmp[i, j] = self.T[j - self.r, int(X[i, j])]

            if (X.ndim == 1):
                t = 1
            else:
                t = np.ones(X.shape[0])
            z = 6 * (tmp - self.x_shift)
            for i in range(self.dim):
                t *= np.cos(z[:, i] / np.sqrt(i + 1))
            f = 1 + (1 / 4000) * np.sum(z ** 2, axis=1) - t
            return f

        def rastrigin(X):
            if isinstance(X, list):
                X = np.array(X)[np.newaxis, :]
            # decode
            tmp = X.copy()
            for i in range(X.shape[0]):
                for j in range(self.r, self.dim):
                    tmp[i, j] = self.T[j - self.r, int(X[i, j])]

            z = 0.05 * (tmp - self.x_shift)
            f = np.sum(z ** 2 - 10 * np.cos(2 * np.pi * z) + 10, axis=1)
            return f

        # Shifted Sphere Function
        def Sphere(X):
            if isinstance(X, list):
                X = np.array(X)[np.newaxis, :]
            tmp = X.copy()
            # decode
            for i in range(X.shape[0]):
                for j in range(self.r, self.dim):
                    tmp[i, j] = self.T[j - self.r, int(X[i, j])]

            z = tmp - self.x_shift
            f = np.sum((z) ** 2, axis=1)
            return f

        # Shifted Weierstrass Function
        def Weierstrass(X):
            if isinstance(X, list):
                X = np.array(X)[np.newaxis, :]
            n = X.shape[0]
            # decode
            tmp = X.copy()
            for i in range(n):
                for j in range(self.r, self.dim):
                    tmp[i, j] = self.T[j - self.r, int(X[i, j])]

            z = (tmp - self.x_shift) * 0.5 / 100
            f = np.zeros(n)
            for k in range(self.dim):
                f += sum([(0.5 ** i) * np.cos(2 * np.pi * (3 ** i) * (z[:, k] + 0.5)) for i in range(20)])
            return f - self.dim * sum([(0.5 ** i) * np.cos(2 * np.pi * (3 ** i) * 0.5) for i in range(20)])


        # Lunacek Bi_Rastrigin Function (CEC2013 F17)
        def Lunacek(X):
            if isinstance(X, list):
                X = np.array(X)[np.newaxis, :]
            n = X.shape[0]
            tmp = X.copy()
            for i in range(X.shape[0]):
                for j in range(self.r, self.dim):
                    tmp[i, j] = self.T[j - self.r, int(X[i, j])]

            fm = F172013(self.dim)

            f = np.empty(n)
            for i in range(n):
                f[i] = fm.evaluate(tmp[i]) - 300
            return f

        # Composition Function 1 (cec2013 F21)
        def CF1(X):
            if isinstance(X, list):
                X = np.array(X)[np.newaxis, :]
            n = X.shape[0]
            tmp = X.copy()
            for i in range(X.shape[0]):
                for j in range(self.r, self.dim):
                    tmp[i, j] = self.T[j - self.r, int(X[i, j])]

            fm = F212013(self.dim)

            f = np.empty(n)
            for i in range(n):
                f[i] = fm.evaluate(tmp[i]) - 700
            return f

        # Composition Function 4 (CEC2013 F24)
        def CF4(X):
            if isinstance(X, list):
                X = np.array(X)[np.newaxis, :]
            n = X.shape[0]
            tmp = X.copy()
            for i in range(X.shape[0]):
                for j in range(self.r, self.dim):
                    tmp[i, j] = self.T[j - self.r, int(X[i, j])]

            fm = F242013(self.dim)
            f = np.empty(n)
            for i in range(n):
                f[i] = fm.evaluate(tmp[i]) - 1000
            return f

        dvdata = np.loadtxt("Benchmarks/DVdata/dvalue_" + prob_name + "_" + str(dim_d) + '_' + str(N_d) + ".txt")
        self.T = dvdata[:, :-1]
        self.indxs = dvdata[:, -1].astype(int)

        for i in range(self.o):
            self.N_lst.append(len(self.T[i]))

        if prob_name == "ellipsoid":
            self.F = ellipsoid

        elif prob_name == "rosenbrock":
            self.F = rosenbrock

        elif prob_name == "ackley":
            self.F = ackley

        elif prob_name == "griewank":
            self.F = griewank

        elif prob_name == "rastrigin":
            self.F = rastrigin

        elif prob_name == "sphere":
            self.F = Sphere

        elif prob_name == "weierstrass":
            self.F = Weierstrass

        elif prob_name == "Lunacek":
            self.F = Lunacek

        elif prob_name == "CF1":
            self.F = CF1

        elif prob_name == "CF4":
            self.F = CF4


