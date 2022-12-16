import os, time
import pandas as pd

from Application.ParamOP import TPLeNet5, TPAlexNet
from Algorithm.SHEALED import SHEALED




def run_algoritms():
    k, pname = 0, "AlexNet-cifar10"

    problem = TPAlexNet(10)
    # problem = TPLeNet5(10)

    # load initial data (solution + accuracy + loss)
    #data = pd.read_csv("results/initial_data/LeNet5_cifar10_initial_data.csv", header=None)
    data = pd.read_csv("Application/initial_data/AlexNet_cifar10_initial_data.csv", header=None)

    database = [data.values[:,:-2], 1-data.values[:, -2], data.values[:, -1]]

    opt = SHEALED(maxFEs=300, popsize=100, dim=problem.dim, clb=problem.bounds[0],
                cub=problem.bounds[1], N_lst=problem.N_lst, prob=problem.F, r=problem.r, database=database)

    x_best, y_best, y_lst, database, melst = opt.run()

    print("optimum on {}:{}".format(pname, y_best))

if __name__ == "__main__":
    strart_t = time.time()
    run_algoritms()
    end_t = time.time()