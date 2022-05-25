import sys
import os
import numpy as np
from os.path import join, dirname, isfile, abspath, exists
import pandas as pd
import copy
import warnings
import csv


def save(X, y, f, experiment_name, method, seed):
    file_name = f'{method.upper()}_{experiment_name}_{seed}.csv'
    result_path = join(join(join(dirname(dirname(abspath(__file__))), 'results'), experiment_name), method)
    dir_of_function = dirname(result_path)
    
    if not exists(dir_of_function):
        os.mkdir(dir_of_function)
    if not exists(result_path):
        os.mkdir(result_path)

    file_path = join(result_path, file_name)
    data = copy.copy(X)
    data.extend([y, f])
    if not exists(file_path):
        os.system(f'touch {file_path}')
        with open(file_path,'a') as fd:
            writer = csv.writer(fd)
            columns = [f'X{i}' for i in range(len(X))]
            columns.extend(['y', 'f'])
            writer.writerow(columns)
            writer.writerow(data)
    else:
        with open(file_path,'a') as fd:
            writer = csv.writer(fd)
            writer.writerow(data)
    
    
def clean_name(file):
    file = str(file)
    file_name = file.split('/')[-1].split('.py')[0]
    return file_name

