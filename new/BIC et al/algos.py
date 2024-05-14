"""
Created on Thu Feb 15 2024
Implementing LASSO and OMP with BIC, EBIC, EFIC
@author: Ojaswi Jain
"""

import numpy as np
from scipy.optimize import minimize
from info_crit import *

def OMP(A, y, k):
    """
    Orthogonal Matching Pursuit (OMP) algorithm for sparse recovery

    Parameters
    ----------
    A : numpy array
        The design matrix
    y : numpy array
        The target vector
    k : int
        The number of non-zero elements in the model

    Returns
    -------
    numpy array
        The support set of the model
    -------
    """
    # Initialize the support set
    s = []
    # Initialize the residual
    r = y
    # A_c = A with normalized columns
    A_c = A / np.linalg.norm(A, axis=0)
    # Perform OMP
    for _ in range(k):
        # Compute the correlation vector
        c = A_c.T @ r
        # Find the index of the feature with the maximum correlation
        j = np.argmax(np.abs(c))
        # Add the index to the support set
        s.append(j)
        # Extract the submatrix A_s
        A_s = A[:, s]
        # Compute the projection matrix P_s
        P_s = A_s @ np.linalg.inv(A_s.T @ A_s) @ A_s.T
        # Update the residual
        r = y - P_s @ y
    #save the solution
    x = np.zeros(A.shape[1])
    x[s] = np.linalg.inv(A[:, s].T @ A[:, s]) @ A[:, s].T @ y
    return x, s # Return the solution and support set

def OMP_with_MS(A, y, k, model):
    """
    Orthogonal Matching Pursuit (OMP) algorithm for sparse recovery with model selection

    Parameters
    ----------
    A : numpy array
        The design matrix
    y : numpy array
        The target vector
    k : int
        The number of non-zero elements in the model
    model : str
        The model selection criterion to use
        "BIC" for Bayesian Information Criterion
        "EBIC" for Extended Bayesian Information Criterion
        "EFIC" for Extended Focused Information Criterion
        "EBIC_robust" for Robust Extended Bayesian Information Criterion

    Returns
    -------
    numpy array
        The support set of the model
    -------
    """
    x, s = OMP(A, y, k)
    if model == "BIC":
        criterion = BIC
        param = 1
    elif model == "EBIC":
        criterion = EBIC
        param = gamma
    elif model == "EFIC":
        criterion = EFIC
        param = c
    elif model == "EBIC_robust":
        criterion = EBIC_robust
        param = zeta
    else:
        raise ValueError("Invalid model selection criterion")
    # Initialize the best criterion value
    err = []
    for j in range(1, k+1):
        s_new = s[:j]
        err.append(criterion(A, y, s_new, param))
    
    s_best = s[:np.argmin(err)+1]
    x_best = np.zeros(A.shape[1])
    x_best[s_best] = np.linalg.inv(A[:, s_best].T @ A[:, s_best]) @ A[:, s_best].T @ y

    return x_best, s_best # Return the solution and support set

def LASSO(A, y, k, lmbda):
    """
    LASSO algorithm for sparse recovery
    Constraint: No more than k non-zero elements in the model
    Parameters
    ----------
    A : numpy array
        The design matrix
    y : numpy array
        The target vector
    k : int
        The number of non-zero elements in the model
    lambda : float
        The regularization parameter

    Returns
    -------
    numpy array
        The support set of the model
    -------
    """
    n = A.shape[1]
    # Define the objective function
    def objective(x):
        count = 1e6**(k > np.count_nonzero(x)) - 1
        return 0.5/n * np.linalg.norm(A @ x - y)**2 + lmbda * np.linalg.norm(x, 1) + count
    # Initialize the solution
    x0 = np.zeros(A.shape[1])
    # Minimize the objective function
    res = minimize(objective, x0, method='L-BFGS-B')
    # Extract the support set
    s = np.where(np.abs(res.x) > 1e-6)[0]
    #save the solution
    x = res.x
    return x, s # Return the solution and support set 

def LASSO_with_MS(A, y, k, model):
    """
    LASSO algorithm for sparse recovery with model selection

    Parameters
    ----------
    A : numpy array
        The design matrix
    y : numpy array
        The target vector
    model : str
        The model selection criterion to use
        "BIC" for Bayesian Information Criterion
        "EBIC" for Extended Bayesian Information Criterion
        "EFIC" for Extended Focused Information Criterion
        "EBIC_robust" for Robust Extended Bayesian Information Criterion

    Returns
    -------
    numpy array
        The support set of the model
    -------
    """
    if model == "BIC":
        criterion = BIC
        param = 1
    elif model == "EBIC":
        criterion = EBIC
        param = gamma
    elif model == "EFIC":
        criterion = EFIC
        param = c
    elif model == "EBIC_robust":
        criterion = EBIC_robust
        param = zeta
    else:
        raise ValueError("Invalid model selection criterion")
    # Initialize the best criterion value
    err = []
    for lmbda in np.logspace(-3, 3, k):
        x, s = LASSO(A, y, k, lmbda)
        err.append(criterion(A, y, s, param))

    best_idx = np.logspace(-3, 3, k)[np.argmin(err)]
    
    return LASSO(A, y, k, best_idx)  # Return the support set with the best criterion value