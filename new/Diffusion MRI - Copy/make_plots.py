"""
Diffusion MRI, implementation to find the best fit Lambdas.
Using NNLS, and various information criteria
Fix a set of lambdas, followed by a 3 dimensional grid search.
---------------------------
Functions exclusively for plotting
---------------------------
@author: Ojaswi Jain
Date: 26th March 2024
"""
import matplotlib.pyplot as plt
from funcs import *
from algos import *
import numpy as np
import os
from utils import *
from gen_data import *

def plotter(x, y, x_title, y_title, folder, dotted=0, dot_title=""):
    """
    Function to plot the given x and y
    Parameters:
        x: vector
            x values
        y: vector
            y values
        x_title: string
            x axis title
        y_title: string 
            y axis title
        folder: string
            Folder to save the plot
        dotted: float (Optional)
            x value for the dotted line
        dot_title: string (Optional)
            Title of the dotted line
    Returns:
        None
    """
    plt.plot(x, y, label = y_title)
    if dotted != 0:
        plt.axvline(dotted, color='r', linestyle='--', label=dot_title)
    plt.xlabel(x_title)
    plt.xticks(x)
    plt.ylabel(y_title)
    plt.title(f'{y_title} vs {x_title}')
    os.makedirs(f'plots/{folder}/{x_title}', exist_ok=True)
    plt.savefig(f'plots/{folder}/{x_title}/{y_title}.png')
    plt.clf()
    return

def plot_with_l1(y, g, b, m, N, V, true_w, l1_range, best_lambdas, criteria):
    """
    Function to plot RMSE, Sensitivity, Specificity with respect to l1
    Parameters:
        y: vector(m, 1)
            Signal intensity  in the presence of diffusion sensitization, normalized
        g: vector(m, 3)
            Gradient directions
        b: scalar
            b-value
        m: int
            Number of gradient directions
        N: int
            Number of diffusion tensors
        V: vector(N, 3, 3)
            N sets of eigenvectors of the diffusion tensor
        true_w: vector(N, 1)
            True weights
        l1_range: vector
            Range of l1 values
        best_lambdas: vector(3, 1)
            Best fit lambdas
        criteria: string
            Criterion
    Returns:
        None
    """
    l1_true = lambdas[0]
    l2 = best_lambdas[1]
    l3 = best_lambdas[2]
    s_true = np.where(np.abs(true_w) > 1e-3)[0]
    RMSE = np.zeros(len(l1_range))
    Sensitivity = np.zeros(len(l1_range))
    Specificity = np.zeros(len(l1_range))

    if criteria == 'CV':
        cv_error = np.zeros(len(l1_range))
        for i in range(len(l1_range)):
            l1 = l1_range[i]
            lambdas = np.array([l1, l2, l3])
            A = convert_to_linear_problem(g, b, m, N, V, lambdas)
            w, cv_err = solve_with_CV(A, y)
            RMSE[i] = rel_rmse(true_w, w)
            s_best = np.where(np.abs(w) > 1e-3)[0]
            Sensitivity[i], Specificity[i] = sensitivity_specificity(s_best, s_true, N)
            cv_error[i] = cv_err

        plotter(l1_range, RMSE, 'l1', 'RMSE', criteria, l1_true, 'True l1')
        plotter(l1_range, Sensitivity, 'l1', 'Sensitivity', criteria, l1_true, 'True l1')
        plotter(l1_range, Specificity, 'l1', 'Specificity', criteria, l1_true, 'True l1')
        plotter(l1_range, cv_error, 'l1', 'Cross Validation Error', criteria, l1_true, 'True l1')

        return


    for i in range(len(l1_range)):
        l1 = l1_range[i]
        lambdas = np.array([l1, l2, l3])
        A = convert_to_linear_problem(g, b, m, N, V, lambdas)
        w_best, s_best = solve_nnls(A, y)
        RMSE[i] = rel_rmse(true_w, w_best)
        Sensitivity[i], Specificity[i] = sensitivity_specificity(s_best, s_true, N)

    plotter(l1_range, RMSE, 'l1', 'RMSE', criteria, l1_true, 'True l1')
    plotter(l1_range, Sensitivity, 'l1', 'Sensitivity', criteria, l1_true, 'True l1')
    plotter(l1_range, Specificity, 'l1', 'Specificity', criteria, l1_true, 'True l1')

def plot_with_l2(y, g, b, m, N, V, true_w, l2_range, best_lambdas, criteria):
    """
    Function to plot RMSE, Sensitivity, Specificity with respect to l2
    Parameters:
        y: vector(m, 1)
            Signal intensity  in the presence of diffusion sensitization, normalized
        g: vector(m, 3)
            Gradient directions
        b: scalar
            b-value
        m: int
            Number of gradient directions
        N: int
            Number of diffusion tensors
        V: vector(N, 3, 3)
            N sets of eigenvectors of the diffusion tensor
        true_w: vector(N, 1)
            True weights
        l2_range: vector
            Range of l2 values
        best_lambdas: vector(3, 1)
            Best fit lambdas
        criteria: string
            Criterion
    Returns:
        None
    """
    l1 = best_lambdas[0]
    l2_true = lambdas[1]
    l3 = best_lambdas[2]
    s_true = np.where(np.abs(true_w) > 1e-3)[0]
    RMSE = np.zeros(len(l2_range))
    Sensitivity = np.zeros(len(l2_range))
    Specificity = np.zeros(len(l2_range))

    if criteria == 'CV':
        cv_error = np.zeros(len(l2_range))
        for i in range(len(l2_range)):
            l2 = l2_range[i]
            lambdas = np.array([l1, l2, l3])
            A = convert_to_linear_problem(g, b, m, N, V, lambdas)
            w, cv_err = solve_with_CV(A, y)
            RMSE[i] = rel_rmse(true_w, w)
            s_best = np.where(np.abs(w) > 1e-3)[0]
            Sensitivity[i], Specificity[i] = sensitivity_specificity(s_best, s_true, N)
            cv_error[i] = cv_err

        plotter(l2_range, RMSE, 'l2', 'RMSE', criteria, l2_true, 'True l2')
        plotter(l2_range, Sensitivity, 'l2', 'Sensitivity', criteria, l2_true, 'True l2')
        plotter(l2_range, Specificity, 'l2', 'Specificity', criteria, l2_true, 'True l2')
        plotter(l2_range, cv_error, 'l2', 'Cross Validation Error', criteria, l2_true, 'True l2')

        return

    for i in range(len(l2_range)):
        l2 = l2_range[i]
        lambdas = np.array([l1, l2, l3])
        A = convert_to_linear_problem(g, b, m, N, V, lambdas)
        w_best, s_best = solve_nnls(A, y)
        RMSE[i] = rel_rmse(true_w, w_best)
        Sensitivity[i], Specificity[i] = sensitivity_specificity(s_best, s_true, N)

    plotter(l2_range, RMSE, 'l2', 'RMSE', criteria, l2_true, 'True l2')
    plotter(l2_range, Sensitivity, 'l2', 'Sensitivity', criteria, l2_true, 'True l2')
    plotter(l2_range, Specificity, 'l2', 'Specificity', criteria, l2_true, 'True l2')


def plot_with_l3(y, g, b, m, N, V, true_w, l3_range, best_lambdas, criteria):
    """
    Function to plot RMSE, Sensitivity, Specificity with respect to l3
    Parameters:
        y: vector(m, 1)
            Signal intensity  in the presence of diffusion sensitization, normalized
        g: vector(m, 3)
            Gradient directions
        b: scalar
            b-value
        m: int
            Number of gradient directions
        N: int
            Number of diffusion tensors
        V: vector(N, 3, 3)
            N sets of eigenvectors of the diffusion tensor
        true_w: vector(N, 1)
            True weights
        l3_range: vector
            Range of l3 values
        best_lambdas: vector(3, 1)
            Best fit lambdas
        criteria: string
            Criterion
    Returns:
        None
    """
    l1 = best_lambdas[0]
    l2 = best_lambdas[1]
    l3_true = lambdas[2]
    s_true = np.where(np.abs(true_w) > 1e-3)[0]
    RMSE = np.zeros(len(l3_range))
    Sensitivity = np.zeros(len(l3_range))
    Specificity = np.zeros(len(l3_range))

    if criteria == 'CV':
        cv_error = np.zeros(len(l3_range))
        for i in range(len(l3_range)):
            l3 = l3_range[i]
            lambdas = np.array([l1, l2, l3])
            A = convert_to_linear_problem(g, b, m, N, V, lambdas)
            w, cv_err = solve_with_CV(A, y)
            RMSE[i] = rel_rmse(true_w, w)
            s_best = np.where(np.abs(w) > 1e-3)[0]
            Sensitivity[i], Specificity[i] = sensitivity_specificity(s_best, s_true, N)
            cv_error[i] = cv_err

        plotter(l3_range, RMSE, 'l3', 'RMSE', criteria, l3_true, 'True l3')
        plotter(l3_range, Sensitivity, 'l3', 'Sensitivity', criteria, l3_true, 'True l3')
        plotter(l3_range, Specificity, 'l3', 'Specificity', criteria, l3_true, 'True l3')
        plotter(l3_range, cv_error, 'l3', 'Cross Validation Error', criteria, l3_true, 'True l3')

        return
    
    for i in range(len(l3_range)):
        l3 = l3_range[i]
        lambdas = np.array([l1, l2, l3])
        A = convert_to_linear_problem(g, b, m, N, V, lambdas)
        w_best, s_best = solve_nnls(A, y)
        RMSE[i] = rel_rmse(true_w, w_best)
        Sensitivity[i], Specificity[i] = sensitivity_specificity(s_best, s_true, N)

    plotter(l3_range, RMSE, 'l3', 'RMSE', criteria, l3_true, 'True l3')
    plotter(l3_range, Sensitivity, 'l3', 'Sensitivity', criteria, l3_true, 'True l3')
    plotter(l3_range, Specificity, 'l3', 'Specificity', criteria, l3_true, 'True l3')

