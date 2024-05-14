"""
Created on Thu Feb 15 2024
Comparing the performances of information criteria and cross-validation for model selection
@author: Ojaswi Jain
"""

import numpy as np
from info_crit import *
from algos import *
from utils import *
from make_plots import *
import argparse
import os

output1 = "reconstruction"
output2 = "rmse"
output3 = "support_set"
os.makedirs(output1, exist_ok=True)
os.makedirs(output2, exist_ok=True)
os.makedirs(output3, exist_ok=True)

np.random.seed(0)

# Generate data
p = 256
k = 10
list_n = range(50, 250, 20)
noise = False

def main(estimator, criterion):
    print(f'Estimator: {estimator}, Criterion: {criterion}')
    x = generate_sparse_signal(p, k)
    rmse = np.zeros((len(list_n), 1))
    support_sets = {}

    for i, n in enumerate(list_n):
        
        params = (n, p, k)
        x_reconstructed, s = simulate(x, params, estimator, criterion, noise)        
        plot_reconstruction(x, x_reconstructed, params, estimator, criterion)
        rmse[i] = rel_rmse(x, x_reconstructed)
        support_sets[params] = np.sort(s) 

    write_support_set(estimator, criterion, support_sets)
    plot_rmse(estimator, criterion, list_n, rmse)
    print(rmse)
        

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--estimator', type=str, default='OMP', help='Estimator to use (OMP or LASSO)')
    argparse.add_argument('--criterion', type=str, default='EBIC', help='Criterion to use (BIC, EBIC, EFIC, EBIC_robust)')

    args = argparse.parse_args()
    main(args.estimator, args.criterion)   