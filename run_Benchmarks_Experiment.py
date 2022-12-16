from Benchmarks.MVOP_type2 import MVOPT2
from Algorithm.SHEALED import SHEALED

import warnings
warnings.filterwarnings('ignore')


def run_benchmark(pname, n_c, n_d):
    # problem = MVOPT1(pname, n_c, n_d, 5)
    problem = MVOPT2(pname, n_c, n_d, 5)

    opt = SHEALED(maxFEs=600, popsize=100, dim=problem.dim, clb=[problem.bounds[0]] * problem.r,
                  cub=[problem.bounds[1]] * problem.r, N_lst=problem.N_lst, prob=problem.F, r=problem.r)

    x_best, y_best, y_lst, database, melst = opt.run()

    print("optimum on {}:{}".format(pname, y_best))


if __name__ == "__main__":

    type1_plst = [
        "F1",
        "F2",
        "F3",
        "F4",
        "F5"
    ]

    type2_lst = [

        "ellipsoid",
        "rosenbrock",
        "ackley",
        "griewank",
        "rastrigin"

        "sphere",
        "weierstrass",
        "Lunacek",
        "CF1",
        "CF4"
    ]

    K = 1
    for p in type2_lst:
        if p in ["sphere", "weierstrass", "Lunacek", "CF1", "CF4"]:
            d_lst = [5, 15]
        else:
            d_lst = [5, 15, 25]

        for d in d_lst:
            run_benchmark(p, d, d)
