import sys
import os
import math
from os.path import join, dirname, isfile, abspath
import numpy as np
from scipy.stats import t
import pandas as pd
sys.path.insert(1, join(dirname(dirname(abspath(__file__))), 'utils'))
from saving import save, clean_name


N_DIMS = 6
NOISE_STD = 0.1


def evaluate_hartmann6(X):
    try:
        X = np.array([X]).reshape(1, N_DIMS)
    except:
        raise ValueError(
            f'Did not provide the right format for X. Should be shape (1, {N_DIMS}) or ({N_DIMS}, ), got {X.shape}')

    def hartmann6_func(X):
        i = 0
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                        [0.05, 10, 17, 0.1, 8, 14],
                        [3, 3.5, 1.7, 10, 17, 8],
                        [17, 8, 0.05, 10, 0.1, 14]])
        P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                                [2329, 4135, 8307, 3736, 1004, 9991],
                                [2348, 1451, 3522, 2883, 3047, 6650],
                                [4047, 8828, 8732, 5743, 1091, 381]])

        outer = 0
        for ii in range(4):
            inner = 0
            for jj in range(6):
                xj = X[i, jj]
                Aij = A[ii, jj]
                Pij = P[ii, jj]
                inner = inner + Aij*(xj-Pij)**2

            new = alpha[ii] * np.exp(-inner)
            outer = outer + new

        return np.array([outer])

    res = hartmann6_func(X)
    noisy_res = res + np.random.normal(0, NOISE_STD)

    return res[0], noisy_res[0]


if __name__ == '__main__':

    if len(sys.argv) > N_DIMS + 3:
        raise IndexError(
            f'Too many arguments for the {N_DIMS}-dimensional benchmark')

    elif len(sys.argv) < N_DIMS + 3:
        raise IndexError(f'Too few arguments for the {N_DIMS}-dimensional benchmark')

    method = sys.argv[1]
    run_idx = sys.argv[2]
    if 'run_' not in run_idx:
        raise ValueError(
            'Run index does not have the proper signature. Should follow the structure run_*index*.')

    idx = int(run_idx.replace('run_', ''))
    X = [float(arg) for arg in sys.argv[3:]]

    res, noisy_res = evaluate_hartmann6(X)
    save(X, noisy_res, res, clean_name(__file__), method, run_idx)

    sys.stdout.write(str(noisy_res) + '\n')
