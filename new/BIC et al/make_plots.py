"""
File to define plot macros, need not be edited again
author : Ojaswi Jain
Date : 11th March 2024
"""

from matplotlib import pyplot as plt

def plot_rmse(estimator, criterion, var, rmse, varname):
    plt.plot(rmse)
    plt.xlabel(varname)
    plt.xticks(range(len(var)), var)
    plt.ylabel('Relative RMSE')
    plt.title('Relative RMSE of ' + estimator + ' with ' + criterion)
    subfolder = estimator + '_' + criterion
    plt.savefig(f'rmse/' + subfolder + '/' + varname + '.png')
    plt.clf()

def plot_reconstruction(x, x_reconstructed, params, estimator, criterion):
    plt.subplot(2, 1, 1)
    plt.plot(x)
    plt.title('Original signal')
    plt.subplot(2, 1, 2)
    plt.plot(x_reconstructed)
    plt.title('Reconstructed signal')
    plt.suptitle(estimator + ' with ' + criterion + ' n=' + str(params[0]) + ', p=' + str(params[1]) + ', k=' + str(params[2]))
    subfolder = estimator + '_' + criterion
    file = f'n{params[0]}_p{params[1]}_k{params[2]}'
    plt.savefig('reconstruction/' + subfolder + '/' + file + '.png')
    plt.clf()

def plot_combined_rmse_p(p_list, rmse_omp, rmse_lasso, rmse_cv=None):
    for cr in rmse_omp.keys():
        plt.plot(rmse_omp[cr], label=f"OMP_{cr}")
        plt.plot(rmse_lasso[cr], label=f"LASSO_{cr}")

    if rmse_cv is not None:
        for keys in rmse_cv.keys():
            plt.plot(rmse_cv[keys], label=f"{keys}_CV")

    plt.xlabel('p')
    plt.xticks(range(len(p_list)), p_list)
    plt.ylabel('Relative RMSE')
    plt.title('Relative RMSE with different criteria and estimators, k = 0.15p, n = 0.4p')
    plt.legend()
    plt.savefig(f'rmse/combined_p.png')
    plt.clf()

def plot_lambda_p(p_list, lambdas):
    for cr in lambdas.keys():
        plt.plot(lambdas[cr], label=cr)
        
    plt.legend()
    plt.xlabel('p')
    plt.xticks(range(len(p_list)), p_list)
    plt.ylabel('Best lambda for LASSO')
    plt.title('Best lambda for LASSO with different criteria, k = 0.15p, n = 0.4p')
    plt.savefig(f'rmse/lasso_p.png')
    plt.clf()

def plot_sensitivity_p(p_list, sens_omp, sens_lasso, sens_cv = None):
    for cr in sens_omp.keys():
        plt.plot(sens_omp[cr], label=f"OMP_{cr}")
        plt.plot(sens_lasso[cr], label=f"LASSO_{cr}")

    if sens_cv is not None:
        for keys in sens_cv.keys():
            plt.plot(sens_cv[keys], label=f"{keys}_CV")

    plt.legend()
    plt.xlabel('p')
    plt.xticks(range(len(p_list)), p_list)
    plt.ylabel('Sensitivity')
    plt.title('Sensitivity with different criteria and estimators, k = 0.15p, n = 0.4p')
    plt.savefig(f'rmse/sensitivity_p.png')
    plt.clf()

def plot_specificity_p(p_list, spec_omp, spec_lasso, spec_cv = None):
    for cr in spec_omp.keys():
        plt.plot(spec_omp[cr], label=f"OMP_{cr}")
        plt.plot(spec_lasso[cr], label=f"LASSO_{cr}")

    if spec_cv is not None:
        for keys in spec_cv.keys():
            plt.plot(spec_cv[keys], label=f"{keys}_CV")

    plt.legend()
    plt.xlabel('p')
    plt.xticks(range(len(p_list)), p_list)
    plt.ylabel('Specificity')
    plt.title('Specificity with different criteria and estimators, k = 0.15p, n = 0.4p')
    plt.savefig(f'rmse/specificity_p.png')
    plt.clf()

def plot_combined_rmse_n(n_list, rmse_omp, rmse_lasso, rmse_cv=None):
    for cr in rmse_omp.keys():
        plt.plot(rmse_omp[cr], label=f"OMP_{cr}")
        plt.plot(rmse_lasso[cr], label=f"LASSO_{cr}")

    if rmse_cv is not None:
        for keys in rmse_cv.keys():
            plt.plot(rmse_cv[keys], label=f"{keys}_CV")

    plt.xlabel('n')
    plt.xticks(range(len(n_list)), n_list)
    plt.ylabel('Relative RMSE')
    plt.title('Relative RMSE with different criteria and estimators, k = 25, p = 256')
    plt.legend()
    plt.savefig(f'rmse/combined_n.png')
    plt.clf()

def plot_lambda_n(n_list, lambdas):
    for cr in lambdas.keys():
        plt.plot(lambdas[cr], label=cr)
        
    plt.legend()
    plt.xlabel('n')
    plt.xticks(range(len(n_list)), n_list)
    plt.ylabel('Best lambda for LASSO')
    plt.title('Best lambda for LASSO with different criteria, k = 25, p = 256')
    plt.savefig(f'rmse/lasso_n.png')
    plt.clf()

def plot_sensitivity_n(n_list, sens_omp, sens_lasso, sens_cv = None):
    for cr in sens_omp.keys():
        plt.plot(sens_omp[cr], label=f"OMP_{cr}")
        plt.plot(sens_lasso[cr], label=f"LASSO_{cr}")

    if sens_cv is not None:
        for keys in sens_cv.keys():
            plt.plot(sens_cv[keys], label=f"{keys}_CV")

    plt.legend()
    plt.xlabel('n')
    plt.xticks(range(len(n_list)), n_list)
    plt.ylabel('Sensitivity')
    plt.title('Sensitivity with different criteria and estimators, k = 25, p = 256')
    plt.savefig(f'rmse/sensitivity_n.png')
    plt.clf()

def plot_specificity_n(n_list, spec_omp, spec_lasso, spec_cv = None):
    for cr in spec_omp.keys():
        plt.plot(spec_omp[cr], label=f"OMP_{cr}")
        plt.plot(spec_lasso[cr], label=f"LASSO_{cr}")

    if spec_cv is not None:
        for keys in spec_cv.keys():
            plt.plot(spec_cv[keys], label=f"{keys}_CV")

    plt.legend()
    plt.xlabel('n')
    plt.xticks(range(len(n_list)), n_list)
    plt.ylabel('Specificity')
    plt.title('Specificity with different criteria and estimators, k = 25, p = 256')
    plt.savefig(f'rmse/specificity_n.png')
    plt.clf()

def plot_combined_rmse_k(k_list, rmse_omp, rmse_lasso, rmse_cv=None):
    for cr in rmse_omp.keys():
        plt.plot(rmse_omp[cr], label=f"OMP_{cr}")
        plt.plot(rmse_lasso[cr], label=f"LASSO_{cr}")

    if rmse_cv is not None:
        for keys in rmse_cv.keys():
            plt.plot(rmse_cv[keys], label=f"{keys}_CV")

    plt.xlabel('k')
    plt.xticks(range(len(k_list)), k_list)
    plt.ylabel('Relative RMSE')
    plt.title('Relative RMSE with different criteria and estimators, n = 100, p = 256')
    plt.legend()
    plt.savefig(f'rmse/combined_k.png')
    plt.clf()

def plot_lambda_k(k_list, lambdas):
    for cr in lambdas.keys():
        plt.plot(lambdas[cr], label=cr)
        
    plt.legend()
    plt.xlabel('k')
    plt.xticks(range(len(k_list)), k_list)
    plt.ylabel('Best lambda for LASSO')
    plt.title('Best lambda for LASSO with different criteria, n = 100, p = 256')
    plt.savefig(f'rmse/lasso_k.png')
    plt.clf()

def plot_sensitivity_k(k_list, sens_omp, sens_lasso, sens_cv = None):
    for cr in sens_omp.keys():
        plt.plot(sens_omp[cr], label=f"OMP_{cr}")
        plt.plot(sens_lasso[cr], label=f"LASSO_{cr}")

    if sens_cv is not None:
        for keys in sens_cv.keys():
            plt.plot(sens_cv[keys], label=f"{keys}_CV")

    plt.legend()
    plt.xlabel('k')
    plt.xticks(range(len(k_list)), k_list)
    plt.ylabel('Sensitivity')
    plt.title('Sensitivity with different criteria and estimators, n = 100, p = 256')
    plt.savefig(f'rmse/sensitivity_k.png')
    plt.clf()

def plot_specificity_k(k_list, spec_omp, spec_lasso, spec_cv = None):
    for cr in spec_omp.keys():
        plt.plot(spec_omp[cr], label=f"OMP_{cr}")
        plt.plot(spec_lasso[cr], label=f"LASSO_{cr}")

    if spec_cv is not None:
        for keys in spec_cv.keys():
            plt.plot(spec_cv[keys], label=f"{keys}_CV")

    plt.legend()
    plt.xlabel('k')
    plt.xticks(range(len(k_list)), k_list)
    plt.ylabel('Specificity')
    plt.title('Specificity with different criteria and estimators, n = 100, p = 256')
    plt.savefig(f'rmse/specificity_k.png')
    plt.clf()