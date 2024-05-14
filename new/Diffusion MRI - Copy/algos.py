"""
Diffusion MRI, implementation to find the best fit Lambdas.
Using NNLS, and various information criteria
Fix a set of lambdas, followed by a 3 dimensional grid search.
---------------------------
Functions to evaluate best fits
---------------------------
@author: Ojaswi Jain
Date: 21th March 2024
"""
# Imports
import numpy as np
from info_crit import *
from sklearn.model_selection import train_test_split
from funcs import *

criteria = ['BIC', 'EBIC', 'EFIC', 'EBIC_robust']

# Function to find the best fit lambdas using the given criterion
def find_best_lambdas(S_g, S_0, g, b, m, N, V, lambdas, noise = 0.1):
    """
    Function to find the best fit lambdas
    Parameters:
        S_g: vector(m, 1)  
            Signal intensity  in the presence of diffusion sensitization, direction g
        S_0: scalar 
            Signal intensity in the absence of diffusion sensitization
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
        lambdas: vector(3, 1)
            Eigenvalues of the diffusion tensor
        method: str
            Method to find the best fit lambdas
            Options: 'BIC', 'EBIC', 'AIC', 'AICc'
    Returns:
        best_lambdas: vector(3, 1)
            Best fit lambdas
        w_best: vector(N, 1)
            Best fit weights
    """
    # Initialize the best lambdas
    best_lambdas = {crit:np.zeros(3) for crit in criteria}
    # Define the lambda grid
    grid = lambda_grid(lambdas)
    # Initialize the minimum criterion
    min_criterion = {crit:np.inf for crit in criteria}
    w_best = {crit:np.zeros(N) for crit in criteria}
    
    # Iterate over the grid
    for i in range(grid.shape[1]):
        for j in range(grid.shape[1]):
            for k in range(grid.shape[1]):
                # k = j
                lambdas = np.array([grid[0, i], grid[1, j], grid[2, k]])
                A = convert_to_linear_problem(g, b, m, N, V, lambdas, noise)
                y = S_g/S_0
                w, s = solve_nnls(A, y)
                # Compute the criterion
                val_criterion = {crit: 0 for crit in criteria}
                for crit in criteria:
                    val_criterion[crit] = eval(crit)(A, y, s)
                    # Update the best lambdas
                    if val_criterion[crit] < min_criterion[crit]:
                        min_criterion[crit] = val_criterion[crit]
                        best_lambdas[crit] = lambdas
                        w_best[crit] = w
                
            z = grid.shape[1]
            percent = (z*i+j+1)*100/(z*z)
            print(percent, "% Done") 

    return w_best, best_lambdas

# Function to find the best cross-validated lambdas
def CV_best_lambdas(S_g, S_0, g, b, m, N, V, lambdas, noise = 0.1, frac = 0.2):
    """
    Function to find the best cross-validated lambdas
    Parameters:
        S_g: vector(m, 1)  
            Signal intensity  in the presence of diffusion sensitization, direction g
        S_0: scalar 
            Signal intensity in the absence of diffusion sensitization
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
        lambdas: vector(3, 1)
            Eigenvalues of the diffusion tensor
        noise: float
            Standard deviation of the noise
        frac: float
            Fraction of the data to be used for cross-validation
    Returns:
        best_lambdas: vector(3, 1)
            Best fit lambdas
        w_best: vector(N, 1)
            Best fit weights
    """
    # Initialize the best lambdas
    best_lambdas = np.zeros(3)
    # Define the lambda grid
    grid = lambda_grid(lambdas)
    # Initialize the minimum rmse
    min_rmse = np.inf
    w_best = np.zeros(N)
    # Iterate over the grid
    for i in range(grid.shape[1]):
        for j in range(grid.shape[1]):
            for k in range(grid.shape[1]):
                # k = j
                lambdas = np.array([grid[0, i], grid[1, j], grid[2, k]])
                A = convert_to_linear_problem(g, b, m, N, V, lambdas, noise)
                y = S_g/S_0
                # Split the data into training and testing
                A_train, A_valid, y_train, y_valid = train_test_split(A, y, test_size = frac, random_state=42)
                # Solve the linear problem using nnls on the training data
                w, s = solve_nnls(A_train, y_train)
                # Compute the rmse on the testing data
                rmse = rel_rmse(y_valid, A_valid @ w)
                # Update the best lambdas
                if rmse < min_rmse:
                    min_rmse = rmse
                    best_lambdas = lambdas
                    w_best = w   
            
            z = grid.shape[1]
            percent = (z*i+j+1)*100/(z*z)
            print(percent, "% Done")        

    return w_best, best_lambdas

def solve_with_CV(A, y, noise = 0.1, frac = 0.2):
    """
    Function to solve the problem using cross-validation
    Parameters:
        A: Matrix
            Matrix A
        y: vector
            Vector y
        noise: float
            Standard deviation of the noise
        frac: float
            Fraction of the data to be used for cross-validation
    Returns:
        w_best: vector(N, 1)
            Best fit weights
        rmse: float
            Relative RMSE
    """
    y_train, y_valid, A_train, A_valid = train_test_split(y, A, test_size = frac, random_state=42)
    w_train, _ = solve_nnls(A_train, y_train)
    rmse = rel_rmse(y_valid, A_valid @ w_train)
    return w_train, rmse