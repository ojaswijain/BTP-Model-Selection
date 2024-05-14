"""
Diffusion MRI, implementation to find the best fit Lambdas.
Using NNLS, and various information criteria
Fix a set of lambdas, followed by a 3 dimensional grid search.
---------------------------
Main function to run the script
---------------------------
@author: Ojaswi Jain
Date: 21th March 2024
"""

import numpy as np
from info_crit import *
from funcs import *
from gen_data import *
from algos import *
from make_plots import *
from utils import *
from time import time

np.random.seed(1)

# Parameters
t_init = time()
V = generate_V()
g = generate_g()
lambdas_noisy = off_center_lambdas(lambdas)
N = n1*n2
true_w, true_s = generate_true_weights()
true_A = convert_to_linear_problem(g, b, m, N, V, lambdas_noisy)
S_g = true_A @ true_w
S_g = S_g * S_0
# S_g = add_noise_to_S_g(S_g)
criteria = ['BIC', 'EBIC', 'EFIC', 'EBIC_robust']
#print non-zero weights
true_idx = np.where(np.abs(true_w) > 1e-3)[0]
#sort indices of non-zero weights according to the true weights
ord = np.argsort(np.abs(true_w[true_idx]))
true_idx = true_idx[ord]
print('True non-zero weights at:', true_idx)
print('True weights:', true_w[true_idx])
print("True lambdas:", lambdas_noisy)

print("Initializations done")

t_init = time() - t_init
print('Time taken for initializations:', t_init, 'seconds')
t_ic = time()
print('-------------------------------------')
w_best, best_lambdas = find_best_lambdas(S_g, S_0, g, b, m, N, V, lambdas)
# w_best, best_lambdas = {crit: true_w for crit in criteria}, {crit: lambdas for crit in criteria}
t_ic = time() - t_ic
print('Time taken for information criteria:', t_ic, 'seconds')
print('-------------------------------------')
t_plot = time()
for crit in criteria:
    write_weights('noisy_off_center_weights_' + crit + '.txt', w_best[crit], best_lambdas[crit])
    idx = np.where(np.abs(w_best[crit]) > 1e-3)[0]
    ord = np.argsort(np.abs(w_best[crit][idx]))
    idx = idx[ord]
    print('Best fit lambdas using', crit, 'criterion:', best_lambdas[crit])
    print('Non-zero weights using', crit, 'criterion at:', idx)
    print('Weights using', crit, 'criterion:', w_best[crit][idx])
    rmse = rel_rmse(true_w, w_best[crit])
    print('Rel. RMSE:', rmse)
    s_best = np.where(np.abs(w_best[crit]) > 1e-3)[0]
    sens, spec = sensitivity_specificity(s_best, true_s, N)
    print('Sensitivity:', sens)
    print('Specificity:', spec)
    plot_with_l1(S_g/S_0, g, b, m, N, V, true_w, l1_range, best_lambdas[crit], crit)
    plot_with_l2(S_g/S_0, g, b, m, N, V, true_w, l2_range, best_lambdas[crit], crit)
    plot_with_l3(S_g/S_0, g, b, m, N, V, true_w, l3_range, best_lambdas[crit], crit)
    print('-------------------------------------')
t_plot = time() - t_plot
print('Time taken for plotting:', t_plot, 'seconds')
print('-------------------------------------')

exit()

t_cv = time()
# w_best, best_lambdas = CV_best_lambdas(S_g, S_0, g, b, m, N, V, lambdas)
w_best, best_lambdas = true_w, lambdas
idx = np.where(np.abs(w_best) > 1e-3)[0]
ord = np.argsort(np.abs(w_best[idx]))
idx = idx[ord]
print('Best fit lambdas using cross-validation:', best_lambdas)
print('Non-zero weights using cross-validation at:', idx)
print('Weights using cross-validation:', w_best[idx])
rmse = rel_rmse(true_w, w_best)
print('Rel. RMSE:', rmse)
s_best = np.where(np.abs(w_best) > 1e-3)[0]
sens, spec = sensitivity_specificity(s_best, true_s, N)
print('Sensitivity:', sens)
print('Specificity:', spec)
t_cv = time() - t_cv
print('Time taken for cross-validation:', t_cv, 'seconds')
print('-------------------------------------')
t_plot = time()
l1, l2, l3 = best_lambdas
plot_with_l1(S_g/S_0, g, b, m, N, V, true_w, l1_range, best_lambdas, 'CV')
plot_with_l2(S_g/S_0, g, b, m, N, V, true_w, l2_range, best_lambdas, 'CV')
plot_with_l3(S_g/S_0, g, b, m, N, V, true_w, l3_range, best_lambdas, 'CV')
t_plot = time() - t_plot
print('Time taken for plotting:', t_plot, 'seconds')
print('-------------------------------------')