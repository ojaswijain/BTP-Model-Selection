import numpy as np
import matplotlib.pyplot as plt
from algos1 import *
import csv

# Function to generate a sparse signal
def generate_sparse_signal(signal_size, sparsity):
    sparse_signal = np.zeros(signal_size)
    sparse_indices = np.random.choice(signal_size, sparsity, replace=False)
    sparse_signal[sparse_indices] = np.random.randn(sparsity)
    return sparse_signal

# Function to generate a random Gaussian matrix
def generate_gaussian_matrix(num_measurements, signal_size):
    return np.random.randn(num_measurements, signal_size) / np.sqrt(num_measurements)

# Function to take measurements
def take_measurements(measurement_matrix, sparse_signal, mean = 0, var = 0):
    return np.dot(measurement_matrix, sparse_signal) + np.random.normal(mean, var, measurement_matrix.shape[0])

# Function to compute the relative RMSE
def rel_rmse(x, x_reconstructed):
    return np.linalg.norm(x - x_reconstructed) / np.linalg.norm(x)

def simulate(x, params, estimator, noise=False):
    n, p, k = params
    A = generate_gaussian_matrix(n, p)
    y = take_measurements(A, x)
    if noise:
        y = take_measurements(A, x, mean=0, var=0.01)
    if estimator == 'OMP':
        x_reconstructed, s = OMP_with_MS(A, y, k)#, criterion)
        return x_reconstructed, s
    elif estimator == 'LASSO':
        x_reconstructed, s, lams = LASSO_with_MS(A, y, k)#, criterion)
        return x_reconstructed, s, lams
    elif estimator == 'CV_LASSO':
        x_reconstructed, s, lams = LASSO_with_cv(A, y, k)
        return x_reconstructed, s, lams
    elif estimator == 'CV_OMP':
        x_reconstructed, s = OMP_with_cv(A, y, k)
        return x_reconstructed, s
    else:
        raise ValueError('Unknown estimator')

# Function to compute the sensitivity and specificity
def sensitivity_specificity(s, s_reconstructed, p):
    # convert s and s_reconstructed to sets
    s = set(s)
    s_reconstructed = set(s_reconstructed)
    TP = len(s.intersection(s_reconstructed))
    FN = len(s.difference(s_reconstructed))
    FP = len(s_reconstructed.difference(s))
    TN = p - len(s.union(s_reconstructed))
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    return sensitivity, specificity


def write_support_set(estimator, criterion, support_sets, varname):
     with open(f'support_set/{estimator}_{criterion}/{varname}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['n', 'p', 'k', 'support_set'])
        for params, s in support_sets.items():
            writer.writerow([params[0], params[1], params[2], s])
