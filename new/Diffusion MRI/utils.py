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
import os

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
    os.makedirs("data", exist_ok = True)
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
    os.mkdir("data", exist_ok = True)
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

def write_table(filename, rmse, sens, spec, angle_error):
    """
    Function to write the results to a table
    Parameters:
        filename: string
            Name of the file
        rmse: numpy array
            Relative RMSE   
        sens: numpy array
            Sensitivity
        spec: numpy array
            Specificity
        angle_error: numpy array
            Angle error
    Returns:
        None
    """ 
    # Define the data arrays
    data = {
        'RMSE': rmse,
        'Sensitivity': sens,
        'Specificity': spec,
        'Mean v1 Angle Error': angle_error[:, 0],
        'Mean v2 Angle Error': angle_error[:, 1],
        'Mean v3 Angle Error': angle_error[:, 2]
    }

    # Add the model column
    model = np.array(['BIC', 'EBIC', 'EFIC', 'EBIC_R', 'CV'])

    # Stack the data arrays horizontally
    data = np.column_stack((data['RMSE'], data['Sensitivity'], data['Specificity'], 
                            data['Mean v1 Angle Error'], data['Mean v2 Angle Error'], 
                            data['Mean v3 Angle Error'], model))

    # Write the data array to a CSV file
    np.savetxt("data/"+filename, data, delimiter=',', fmt='%s')