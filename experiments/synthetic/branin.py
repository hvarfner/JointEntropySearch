import sys
import os
import math
from os.path import join, dirname, isfile, abspath
import numpy as np
from scipy.stats import t
import pandas as pd
sys.path.insert(1, join(dirname(dirname(abspath(__file__))), 'utils'))
from saving import save, clean_name


N_DIMS = 2
NOISE_STD = 0.1


def evaluate_branin(X):
    try:
        X = np.array([X]).reshape(1, N_DIMS)
    except:
        raise ValueError(
            f'Did not provide the right format for X. Should be shape (1, {N_DIMS}) or ({N_DIMS}, ), got {X.shape}')

    def branin_func(X): 
        x1 = X[:, 0] * 15 - 5
        x2 = X[:, 1] * 15
        
        a = 1.0
        b = 5.1 / (4.0 * math.pi * math.pi)
        c = 5.0 / math.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8.0 * math.pi)

        y_value = a * (x2 - b * x1 * x1 + c * x1 - r) ** 2 + s * (1 - t) * math.cos(x1) + s

        return -y_value

    res = branin_func(X)
    noisy_res = res + np.random.normal(0, NOISE_STD)

    return res[0], noisy_res[0]


if __name__ == '__main__':

    if len(sys.argv) > N_DIMS + 3:
        raise IndexError(
            'Too many arguments for the two-dimensional benchmark')

    elif len(sys.argv) < N_DIMS + 3:
        raise IndexError('Too few arguments for the two-dimensional benchmark')

    method = sys.argv[1]
    run_idx = sys.argv[2]
    if 'run_' not in run_idx:
        raise ValueError(
            'Run index does not have the proper signature. Should follow the structure run_*index*.')

    idx = int(run_idx.replace('run_', ''))
    X = [float(arg) for arg in sys.argv[3:]]

    res, noisy_res = evaluate_branin(X)
    save(X, noisy_res, res, clean_name(__file__), method, run_idx)

    sys.stdout.write(str(noisy_res) + '\n')
