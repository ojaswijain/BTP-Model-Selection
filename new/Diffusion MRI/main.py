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
criteria = ['BIC', 'EBIC', 'EFIC', 'EBIC_robust']
# criteria = ['BIC']

n_voxels = 1

def main(i, dict):
    folder_name = "voxel_" + str(i)
    print('Voxel:', i)
    print('-------------------------------------')

    os.makedirs(folder_name, exist_ok=True)
    os.chdir(folder_name)
    
    t_init = time()
    
    # Parameters
    V = generate_V()
    g = generate_g()
    V_off_center = generate_off_center_V()
    lambdas_noisy = off_center_lambdas(lambdas)
    true_w, true_s = generate_true_weights()
    true_A = convert_to_linear_problem(g, b, m, N, V_off_center, lambdas_noisy)
    S_g = true_A @ true_w
    S_g = S_g * S_0
    S_g = add_noise_to_S_g(S_g)

    #print non-zero weights
    true_idx = np.where(np.abs(true_w) > 1e-3)[0]
    ord = np.argsort(np.abs(true_w[true_idx]))
    true_idx = true_idx[ord]
    print('True non-zero weights at:', true_idx)
    print('True weights:', true_w[true_idx])
    print("True lambdas:", lambdas_noisy)

    print("Initializations done")

    t_init = time() - t_init
    print('Time taken for initializations:', t_init, 'seconds')
    print('-------------------------------------')

    t_ic = time()

    rmse_list = np.zeros(len(criteria)+1)
    sens_list = np.zeros(len(criteria)+1)
    spec_list = np.zeros(len(criteria)+1)
    angle_error_list = np.zeros((len(criteria)+1, 3))

    # toggle between the next two sets of lines to run the information criteria or read the weights from file

    # w_best = {crit: read_weights('perturbedV_and_shiftedLambdas' + crit + '.txt')[0] for crit in criteria}
    # best_lambdas = {crit: read_weights('perturbedV_and_shiftedLambdas' + crit + '.txt')[1] for crit in criteria}

    w_best, best_lambdas = find_best_lambdas(S_g, S_0, g, b, m, N, V, lambdas)

    t_ic = time() - t_ic
    print('Time taken for information criteria:', t_ic, 'seconds')
    print('-------------------------------------')

    t_plot = time()
    for crit in criteria:
        write_weights('perturbedV_and_shiftedLambdas' + crit + '.txt', w_best[crit], best_lambdas[crit])
        idx = np.where(np.abs(w_best[crit]) > 1e-3)[0]
        ord = np.argsort(np.abs(w_best[crit][idx]))
        idx = idx[ord]
        rmse = rel_rmse(true_w, w_best[crit])
        s_best = np.where(np.abs(w_best[crit]) > 1e-3)[0]
        sens, spec = sensitivity_specificity(s_best, true_s, N)
        
        print('Best fit lambdas using', crit, 'criterion:', best_lambdas[crit])
        print('Non-zero weights using', crit, 'criterion at:', idx)
        print('Weights using', crit, 'criterion:', w_best[crit][idx])
        print('Rel. RMSE:', rmse)
        print('Sensitivity:', sens)
        print('Specificity:', spec)

        rmse_list[criteria.index(crit)] = rmse
        sens_list[criteria.index(crit)] = sens
        spec_list[criteria.index(crit)] = spec

        V_est = V[idx]
        V_true = V_off_center[true_idx]
        Matching, Angle_error = match_eigenvectors(V_true, V_est, lambdas_noisy)
        
        for angle in Angle_error:
            dict[crit].append(angle)

        for i in range(3):
            angle_error_list[criteria.index(crit)][i] = np.mean([angle[i] for angle in Angle_error])

        print("Forward matching")
        for i in range(len(V_est)):
            print('Estimated eigenvector:\n', V_est[i], '\nWeight:', w_best[crit][idx[i]])
            print('Closest True eigenvector:\n', V_true[int(Matching[i])], '\nWeight:', true_w[true_idx[int(Matching[i])]])
            print('Angle_error in degrees:', Angle_error[i], '\n')

        plot_eigenvectors(V_est, V_true, Matching, best_lambdas[crit], lambdas_noisy, w_best[crit][idx], true_w[true_idx], crit)
        mega_plot(S_g/S_0, g, b, m, N, V_off_center, true_w, l1_range, l2_range, l3_range, best_lambdas[crit], lambdas_noisy, crit)
        print('-------------------------------------')

    t_plot = time() - t_plot
    print('Time taken for plotting:', t_plot, 'seconds')
    print('-------------------------------------')

    return
    
    t_cv = time()
    w_best, best_lambdas = CV_best_lambdas(S_g, S_0, g, b, m, N, V, lambdas)
    write_weights('perturbedV_and_shiftedLambdasCV.txt', w_best, best_lambdas)
    idx = np.where(np.abs(w_best) > 1e-3)[0]
    ord = np.argsort(np.abs(w_best[idx]))
    idx = idx[ord]
    rmse = rel_rmse(true_w, w_best)
    s_best = np.where(np.abs(w_best) > 1e-3)[0]
    sens, spec = sensitivity_specificity(s_best, true_s, N)

    rmse_list[-1] = rmse
    sens_list[-1] = sens
    spec_list[-1] = spec

    print('Best fit lambdas using cross-validation:', best_lambdas)
    print('Non-zero weights using cross-validation at:', idx)
    print('Weights using cross-validation:', w_best[idx])
    print('Rel. RMSE:', rmse)
    print('Sensitivity:', sens)
    print('Specificity:', spec)

    t_cv = time() - t_cv
    print('Time taken for cross-validation:', t_cv, 'seconds')
    print('-------------------------------------')

    t_plot = time()

    V_est = V[idx]
    V_true = V_off_center[true_idx]
    Matching, Angle_error = match_eigenvectors(V_true, V_est, lambdas_noisy)

    for angle in Angle_error:
        dict['CV'].append(angle)

    for i in range(3):
        angle_error_list[-1][i] = np.mean([angle[i] for angle in Angle_error])

    print("Forward matching")
    for i in range(len(V_est)):
        print('Estimated eigenvector:\n', V_est[i], '\nWeight:', w_best[idx[i]])
        print('Closest True eigenvector:\n', V_true[int(Matching[i])], '\nWeight:', true_w[true_idx[int(Matching[i])]])
        print('Angle_error in degrees:', Angle_error[i], '\n')
    
    l1, l2, l3 = best_lambdas

    plot_eigenvectors(V_est, V_true, Matching, best_lambdas, lambdas_noisy, w_best[idx], true_w[true_idx], 'CV')
    mega_plot(S_g/S_0, g, b, m, N, V, true_w, l1_range, l2_range, l3_range, best_lambdas, lambdas_noisy, 'CV')
    t_plot = time() - t_plot
    print('Time taken for plotting:', t_plot, 'seconds')
    print('-------------------------------------')
    write_table('results.txt', rmse, sens, spec, Angle_error)
    os.chdir('..')

if __name__ == '__main__':
    dict = {cr: [] for cr in criteria}
    dict['CV'] = []
    # reverse_dict = {cr: [] for cr in criteria}
    for i in range(1, n_voxels+1):
        main(i, dict)

    #plot histogram of v1, v2, v3 angle errors for all criteria
    for cr in dict.keys():
        #plot histogram of v1, v2, v3 angle errors
        v1s = [dict[cr][i][0] for i in range(len(dict[cr]))]
        v2s = [dict[cr][i][1] for i in range(len(dict[cr]))]
        v3s = [dict[cr][i][2] for i in range(len(dict[cr]))]
        plot_hist(v1s, cr, 'v1 Forward Angle Error')
        plot_hist(v2s, cr, 'v2 Forward Angle Error')
        plot_hist(v3s, cr, 'v3 Forward Angle Error')