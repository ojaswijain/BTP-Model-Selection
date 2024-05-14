"""
Observing effects of increasing signal size on the performance of the MS criteria
Created on Mon Feb 20 2024
author: Ojaswi Jain
"""

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import numpy as np
from utils import *
import os
import matplotlib.pyplot as plt
from make_plots import *

np.random.seed(1)

output1 = "reconstruction"
output2 = "rmse"
output3 = "support_set"

os.makedirs(output1, exist_ok=True)
os.makedirs(output2, exist_ok=True)
os.makedirs(output3, exist_ok=True)

estimators = ['OMP', 'LASSO']
criteria = ['BIC', 'EBIC', 'EFIC', 'EBIC_robust']
algos = ['OMP', 'LASSO']

#p_list is everything from 64 to 1024 in geometric progression
p_list = [2**i for i in range(6, 11)]
ng = 0.4
kg = 0.1
noise = False

def main():
    rmse_omp = {cr: np.zeros(len(p_list)) for cr in criteria}
    rmse_lasso = {cr: np.zeros(len(p_list)) for cr in criteria}
    lambdas = {cr: np.zeros(len(p_list)) for cr in criteria}

    #Sensitivity and specificity
    sens_omp = {cr: np.zeros(len(p_list)) for cr in criteria}
    sens_lasso = {cr: np.zeros(len(p_list)) for cr in criteria}
    spec_omp = {cr: np.zeros(len(p_list)) for cr in criteria}
    spec_lasso = {cr: np.zeros(len(p_list)) for cr in criteria}

    rmse_cv = {al: np.zeros(len(p_list)) for al in algos}
    lambdas_cv = np.zeros(len(p_list))
    sens_cv = {al: np.zeros(len(p_list)) for al in algos}
    spec_cv = {al: np.zeros(len(p_list)) for al in algos}

    for i, p in enumerate(p_list):
        print(f'p = {p}')
        n = int(ng * p)
        k = int(kg * p)
        x = generate_sparse_signal(p, k)
        s_true = set(np.where(x)[0])
        params = (n, p, k)
        print("OMP")
        x_omp, s_omp = simulate(x, params, "OMP")      
        for cr in x_omp.keys():
            print(f'Criterion: {cr}')
            rmse_omp[cr][i] = rel_rmse(x, x_omp[cr])
            plot_reconstruction(x, x_omp[cr], params, 'OMP', cr)
            sens_omp[cr][i], spec_omp[cr][i] = sensitivity_specificity(s_true, s_omp[cr], p)
            
        print("LASSO")
        x_lasso, s_lasso, lambda_lasso = simulate(x, params, "LASSO")
        for cr in x_lasso.keys():
            print(f'Criterion: {cr}')
            rmse_lasso[cr][i] = rel_rmse(x, x_lasso[cr])
            lambdas[cr][i] = lambda_lasso[cr]
            plot_reconstruction(x, x_lasso[cr], params, 'LASSO', cr)
            sens_lasso[cr][i], spec_lasso[cr][i] = sensitivity_specificity(s_true, s_lasso[cr], p)
        
        print("CV_LASSO")
        x_cv_lasso, s_cv_lasso, lambda_cv = simulate(x, params, "CV_LASSO")
        rmse_cv['LASSO'][i] = rel_rmse(x, x_cv_lasso)
        lambdas_cv[i] = lambda_cv
        plot_reconstruction(x, x_cv_lasso, params, 'LASSO', 'CV')
        sens_cv['LASSO'][i], spec_cv['LASSO'][i] = sensitivity_specificity(set(np.where(x)[0]), s_cv_lasso, p)

        print("CV_OMP")
        x_cv_omp, s_cv_omp = simulate(x, params, "CV_OMP")
        rmse_cv['OMP'][i] = rel_rmse(x, x_cv_omp)
        plot_reconstruction(x, x_cv_omp, params, 'OMP', 'CV')
        sens_cv['OMP'][i], spec_cv['OMP'][i] = sensitivity_specificity(set(np.where(x)[0]), s_cv_omp, p)
        
    for cr in x_omp.keys():
        plot_rmse('OMP', cr, p_list, rmse_omp[cr], 'p')
        plot_rmse('LASSO', cr, p_list, rmse_lasso[cr], 'p')

    plot_rmse('LASSO', 'CV', p_list, rmse_cv['LASSO'], 'p')
    plot_rmse('OMP', 'CV', p_list, rmse_cv['OMP'], 'p')
    lambdas['CV'] = lambdas_cv

    plot_combined_rmse_p(p_list, rmse_omp, rmse_lasso, rmse_cv)
    plot_lambda_p(p_list, lambdas)
    plot_sensitivity_p(p_list, sens_omp, sens_lasso, sens_cv)
    plot_specificity_p(p_list, spec_omp, spec_lasso, spec_cv)

if __name__ == '__main__':

    for estimator in estimators:
        for criterion in criteria:
            os.makedirs(f'{output1}/{estimator}_{criterion}', exist_ok=True)
            os.makedirs(f'{output2}/{estimator}_{criterion}', exist_ok=True)
            os.makedirs(f'{output3}/{estimator}_{criterion}', exist_ok=True)

    for al in algos:
        os.makedirs(f'{output1}/{al}_CV', exist_ok=True)
        os.makedirs(f'{output2}/{al}_CV', exist_ok=True)
        os.makedirs(f'{output3}/{al}_CV', exist_ok=True)

    main()                        