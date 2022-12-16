import numpy as np

def generate_type1_dvalue(x_shift, N_lst):
    o = len(N_lst)
    r = len(x_shift) - o
    T = np.zeros((o, N_lst[0]))
    for j in range(o):
        T[j, :] = np.append(x_shift[j + r], np.random.normal(loc=x_shift[r + j], scale=10, size=N_lst[j] - 1))
    return T


def generate_type2_dvalue(pname, x_shift, bounds, dim, dim_d, N_d):
    area = bounds[1] - bounds[0]
    table = np.zeros((dim_d, N_d))
    z_best = x_shift[0, dim - dim_d:dim]
    table[:, 0] = z_best

    for i in range(dim_d):
        for j in range(1, N_d):
            table[i, j] = bounds[0] + np.random.uniform((j - 1) / (N_d - 1) * area, j / (N_d - 1) * area)
        np.random.shuffle(table[i])

    indxs = np.ones((dim_d, 1))
    for i in range(dim_d):
        indxs[i] = np.where(table[i] == z_best[i])[0]
    table = np.around(table, decimals=4)

    dvdata = np.hstack([table, indxs])

    np.savetxt("Benchmarks/DVdata/dvalue_" + pname + "_" + str(dim_d) + '_' + str(N_d) + ".txt", dvdata)

