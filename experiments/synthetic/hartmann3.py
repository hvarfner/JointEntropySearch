import sys
import os
import math
from os.path import join, dirname, isfile, abspath
import numpy as np
from scipy.stats import t
import pandas as pd
sys.path.insert(1, join(dirname(dirname(abspath(__file__))), 'utils'))
from saving import save, clean_name


N_DIMS = 3
NOISE_STD = 0.1


def evaluate_hartmann3(X):
    try:
        X = np.array([X]).reshape(1, N_DIMS)
    except:
        raise ValueError(
            f'Did not provide the right format for X. Should be shape (1, {N_DIMS}) or ({N_DIMS}, ), got {X.shape}')

    def hartmann3_func(X):
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array([
            [3.0, 10, 30],
            [0.1, 10, 35],
            [3.0, 10, 30],
            [0.1, 10, 35]
        ])
        P = np.array([
            [3689, 1170, 2673],
            [4699, 4387, 7470],
            [1091, 8732, 5547],
            [381, 5743, 8828],
        ])

        inner_sum = np.sum(
            A * (X[:, np.newaxis, :] - 0.0001 * P) ** 2, axis=-1)
        H = -(np.sum(alpha * np.exp(-inner_sum), axis=-1))
        H_true = -(np.sum(alpha * np.exp(-inner_sum), axis=-1))

        return H_true

    res = hartmann3_func(X)
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

    res, noisy_res = evaluate_hartmann3(X)
    save(X, noisy_res, res, clean_name(__file__), method, run_idx)

    sys.stdout.write(str(noisy_res) + '\n')
