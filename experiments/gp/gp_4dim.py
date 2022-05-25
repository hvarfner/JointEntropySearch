import sys
import os
from os.path import join, dirname, isfile, abspath
import numpy as np
from scipy.stats import t
import pandas as pd
sys.path.insert(1, join(dirname(dirname(abspath(__file__))), 'utils'))
from saving import save, clean_name


N_DIMS = 4
GP_PATH = join(dirname(__file__), 'gp_{}dim/gp_{}dim_run{}.csv')
NOISE_STD = 0.1
SIGMA = 10
# LS = 0.15 (already included, really shouldn't be)


def evaluate_gp(gp_idx, X):
    data = pd.read_csv(GP_PATH.format(N_DIMS, N_DIMS, str(int(gp_idx) % 10)))
    try:
        X = np.array([X]).reshape(1, N_DIMS)
    except:
        raise ValueError(
            f'Did not provide the right format for X. Should be shape (1, {N_DIMS}) or ({N_DIMS}, ), got {X.shape}')
   
    w = data.iloc[:, 0:-2].to_numpy()
    b = data.iloc[:, -2].to_numpy().reshape(-1, 1)
    theta = data.iloc[:, -1].to_numpy().reshape(-1, 1)

    def gp_func(X, w, b, theta): return len(b) * np.mean(theta * np.sqrt(2 * SIGMA / len(b)) *
                                                         np.cos(np.dot(w, X.T) + b), axis=0)

    res = gp_func(X, w, b, theta)
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

    res, noisy_res = evaluate_gp(idx, X)
    save(X, noisy_res, res, clean_name(__file__), method, run_idx)

    sys.stdout.write(str(noisy_res) + '\n')
