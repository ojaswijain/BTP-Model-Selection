"""
Created on Thu Feb 15 2024
Implementing LASSO and OMP with BIC, EBIC, EFIC
@author: Ojaswi Jain
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from info_crit import *

criteria = ["BIC", "EBIC", "EFIC", "EBIC_robust"]

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
    x : numpy array
        The solution to the optimization problem
    s : numpy array
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
    s = np.where(np.abs(x) > 0)[0]
    return x, s # Return the solution and support set

def OMP_with_MS(A, y, k):
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
    x_best : numpy array
        The solution to the optimization problem
    s_best : numpy array
        The support set of the model
    -------
    """
    x, s = OMP(A, y, k)
    # Initialize a dictionary of lists to store the criterion values
    err = {cr: [] for cr in criteria}
    
    for j in range(1, k+1):
        s_new = s[:j]
        for cr in criteria:
            err[cr].append(eval(cr)(A, y, s_new))
    
    s_best = {cr: s[:np.argmin(err[cr])+1] for cr in criteria}
    x_best = {cr: np.zeros(A.shape[1]) for cr in criteria}

    for cr in criteria:
        x_best[cr][s_best[cr]] = np.linalg.inv(A[:, s_best[cr]].T @ A[:, s_best[cr]]) @ A[:, s_best[cr]].T @ y
    
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
    x : numpy array
        The solution to the optimization problem
    s : numpy array
        The support set of the model
    -------
    """
    
    # Initialize the LASSO model
    myLasso = Lasso(alpha=lmbda, fit_intercept=False)
    # Fit the model
    myLasso.fit(A, y)
    # Extract the solution
    x = myLasso.coef_
    #Retain only the k largest coefficients
    x[np.argsort(np.abs(x))[:-k]] = 0
    # Extract the support set
    s = np.where(np.abs(x) > 0)[0]
    return x, s # Return the solution and support set

def LASSO_with_MS(A, y, k):
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
    x_best : numpy array
        The solution to the optimization problem
    s_best : numpy array 
        The support set of the model
    best_lambda : float
        The best regularization parameter
    -------
    """
    err = {cr: [] for cr in criteria}
    for lmbda in np.logspace(-8, 0, k):
        _, s = LASSO(A, y, k, lmbda)
        for cr in criteria:
            error = eval(cr)(A, y, s)
            err[cr].append(error)

    best_lambda = {cr: np.logspace(-8, 0, k)[np.argmin(err[cr])] for cr in criteria}
    x_best = {cr: np.zeros(A.shape[1]) for cr in criteria}
    s_best = {cr: [] for cr in criteria}
    for cr in criteria:
        x_best[cr], s_best[cr] = LASSO(A, y, k, best_lambda[cr])
    return x_best, s_best, best_lambda # Return the solution and support set

def LASSO_with_cv(A, y, k, cv_frac=0.15):
    """
    LASSO algorithm for sparse recovery with cross-validation

    Parameters
    ----------
    A : numpy array
        The design matrix
    y : numpy array
        The target vector
    k : int
        The number of non-zero elements in the model
    cv_frac : float
        The fraction of the data to use for cross-validation

    Returns
    -------
    x_best : numpy array
        The solution to the optimization problem
    s_best : numpy array
        The support set of the model
    best_lambda : float
        The best regularization parameter
    -------
    """
    # Split the data into training and validation sets
    A_train, A_valid, y_train, y_valid = train_test_split(A, y, test_size=cv_frac, random_state=42)
    # Initialize the error list
    err = []
    for lmbda in np.logspace(-8, 0, k):
        x_train, _ = LASSO(A_train, y_train, k, lmbda)
        # Compute the validation error
        validation_error = np.linalg.norm(y_valid - A_valid @ x_train)**2
        err.append(validation_error)

    best_lambda = np.logspace(-8, 0, k)[np.argmin(err)]
    x_best, s_best = LASSO(A, y, k, best_lambda)
    return x_best, s_best, best_lambda # Return the solution and support set

def OMP_with_cv(A, y, k, cv_frac=0.2):
    """
    Orthogonal Matching Pursuit (OMP) algorithm for sparse recovery with cross-validation

    Parameters
    ----------
    A : numpy array
        The design matrix
    y : numpy array
        The target vector
    k : int
        The number of non-zero elements in the model
    cv_frac : float
        The fraction of the data to use for cross-validation

    Returns
    -------
    x_best : numpy array
        The solution to the optimization problem
    s_best : numpy array
        The support set of the model
    -------
    """
    # Split the data into training and validation sets
    A_train, A_valid, y_train, y_valid = train_test_split(A, y, test_size=cv_frac, random_state=42)
    # Initialize the error list
    err = []
    for k_cv in range(1, k+1):
        x_train, _ = OMP(A_train, y_train, k_cv)
        # Compute the validation error
        validation_error = np.linalg.norm(y_valid - A_valid @ x_train)**2
        err.append(validation_error)

    best_k = np.argmin(err) + 1
    x_best, s_best = OMP(A, y, best_k)
    return x_best, s_best # Return the solution and support set    