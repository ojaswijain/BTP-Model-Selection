"""
Diffusion MRI, implementation to find the best fit Lambdas.
Using NNLS, and various information criteria
Fix a set of lambdas, followed by a 3 dimensional grid search.
---------------------------
Utility functions
---------------------------
@author: Ojaswi Jain
Date: 28th March 2024
"""
import numpy as np

def write_weights(filename, w, best_lambdas):
    """
    Function to write the weights to a file
    Parameters:
        filename: string
            Name of the file
        w: vector(N, 1)
            Weights
        best_lambdas: vector(3, 1)
            Best fit lambdas
    Returns:
        None
    """
    with open("data/"+filename, 'w') as f:
        for i in range(len(w)):
            f.write(str(w[i]) + '\n')
        f.write(str(best_lambdas[0]) + '\n')
        f.write(str(best_lambdas[1]) + '\n')
        f.write(str(best_lambdas[2]) + '\n')

def read_weights(filename):
    """
    Function to read the weights from a file
    Parameters:
        filename: string
            Name of the file
    Returns:
        w: vector(N, 1)
            Weights
        best_lambdas: vector(3, 1)
            Best fit lambdas
    """
    with open("data/"+filename, 'r') as f:
        lines = f.readlines()
        w = np.array([float(line) for line in lines[:-3]])
        best_lambdas = np.array([float(line) for line in lines[-3:]])
    return w, best_lambdas

def write_param(param, param_name):
    """
    Function to write the parameters to a file
    Parameters:
        param: vector
            Parameter value
        param_name: string
            Name of the parameter
    Returns:
        None
    """
    with open("data/"+param_name, 'w') as f:
        for i in range(len(param)):
            f.write(str(param[i]) + '\n')

def read_param(param_name):
    """
    Function to read the parameters from a file
    Parameters:
        param_name: string
            Name of the parameter
    Returns:
        param: vector
            Parameter value
    """
    with open("data/"+param_name, 'r') as f:
        lines = f.readlines()
        param = np.array([float(line) for line in lines])
    return param    