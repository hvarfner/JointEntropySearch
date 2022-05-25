import sys
import os
from os.path import join, dirname, isfile, abspath
import numpy as np
import ConfigSpace
import warnings
import argparse

import pandas as pd
sys.path.insert(1, join(dirname(dirname(abspath(__file__))), 'utils'))
from saving import save, clean_name

from util import format_arguments, rescale_arguments
warnings.filterwarnings('ignore')
from ConfigSpace import Configuration

sys.path.append(join(dirname(abspath(__file__)),"libs/HPOBench"))
from hpobench.benchmarks.ml.nn_benchmark import NNBenchmark as Benchmark



N_DIMS = 4
EXP = 146818
def eval_function(args, idx, infer=1):
    if infer > 1:
        # if we want to do the inference regret for the task
        tot_res = 0
        each_res = []
        for i in range(infer):
            b = Benchmark(task_id=EXP, rng=i)
            config_space = b.get_configuration_space(seed=i)
            args_dict = rescale_arguments(args)
            config = Configuration(b.get_configuration_space(seed=i), args_dict)
            result_dict = b.objective_function(configuration=config, rng=i)
            tot_res += result_dict['function_value']
            each_res.append(result_dict['function_value'])
        res = 1 - tot_res / infer
        return each_res, res


    else:
        b = Benchmark(task_id=EXP, rng=idx)
        config_space = b.get_configuration_space(seed=idx)
        args_dict = rescale_arguments(args)
        config = Configuration(b.get_configuration_space(seed=idx), args_dict)
        result_dict = b.objective_function(configuration=config, rng=idx)
        res = 1 - result_dict['function_value']

    # returning with no noise
        return res, res


if __name__ == '__main__':
    
    method = sys.argv[1]
    run_idx = sys.argv[2]
    if 'run_' not in run_idx:
        raise ValueError(
            'Run index does not have the proper signature. Should follow the structure run_*index*.')
    idx = int(run_idx.replace('run_', ''))
    
        
    if len(sys.argv) == N_DIMS + 3:
        infer = 1
        X = [float(arg) for arg in sys.argv[3:]]



    elif len(sys.argv) == N_DIMS + 4:
        infer = int(sys.argv[3])
        X = [float(arg) for arg in sys.argv[4:]]

    else:
        raise IndexError('Incorrect nummber of arguments for the 4-dimensional benchmark')
    
    res, noisy_res = eval_function(X, idx, infer=infer)
    save(X, noisy_res, res, clean_name(__file__), method, run_idx)
    sys.stdout.write(str(noisy_res) + '\n')
