"""
Created on Thu Feb 15 2024
Implementing BIC, EBIC, EFIC
@author: Ojaswi Jain
"""

import numpy as np
import matplotlib.pyplot as plt

gamma = 0.3
c = 0.3
zeta = 0.3

def BIC(A, y, s, param=1):
    """
    Compute the BIC of a given model

    BIC = n * log(sigma^2) + k * log(n)
    
        where n is the number of samples, k is the number of non-zero elements in the model,
        and sigma^2 is the maximum likelihood estimate of the variance of the model parameters
            which is given by sigma^2 = y^T P_s y / n
            where P_s is the projection matrix onto the support set of the model
                given by P_s = A_s (A_s^T A_s)^(-1) A_s^T
                where A_s is the submatrix of A formed by the columns in the support set

    Parameters
    ----------
    A : numpy array
        The design matrix
    y : numpy array
        The target vector
    s : list
        The support set of the model
    ----------

    Returns
    -------
    float
        The BIC of the model
    -------
    """
    # n = number of samples, p = number of features, k = number of non-zero elements in the model
    n, p = A.shape
    k = len(s)
    # Extract the submatrix A_s
    A_s = A[:, s]
    # Define the projection matrix P_s
    P_s = A_s @ np.linalg.inv(A_s.T @ A_s) @ A_s.T
    # Define the projection matrix P_s_perp
    P_s_perp = np.eye(n) - P_s
    # Compute the MLE of the variance of the model parameters
    sigma2 = y.T @ P_s_perp @ y / n
    # Compute the BIC
    BIC = n * np.log(sigma2) + k * np.log(n)
    return BIC

def log_of_comb(n, k):
    """
    Compute the log of the number of combinations of n elements taken k at a time
    """
    if k >= n/2:
        k = n-k
    if k == 0:
        return 1
    ans = 0
    for i in range(n, n-k, -1):
        ans += np.log(i)
        ans -= np.log(n-i+1)
    return ans
    

def EBIC(A, y, s, gamma=0.3):
    """
    Compute the EBIC of a given model

    EBIC = BIC + 2*gamma*ln(pCk)
        where pCk is the number of combinations of p features taken k at a time
        and gamma is a penalty parameter between 0 and 1
    
    Parameters
    ----------
    A : numpy array
        The design matrix
    y : numpy array
        The target vector
    s : list
        The support set of the model
    gamma : float
        The penalty parameter
    """
    
    # n = number of samples, p = number of features, k = number of non-zero elements in the model
    n, p = A.shape
    k = len(s)
    # Compute the BIC
    BIC_val = BIC(A, y, s)
    # Compute the EBIC
    EBIC = BIC_val + 2 * gamma * log_of_comb(p, k)
    return EBIC

def EFIC(A, y, s, c=0.3):
    """
    Compute the EFIC of a given model

    EFIC = n * log(||P_s_perp y||^2) + k * log(n) 
           + log(A_s^T A_s) - (k+2) * log(||P_s y||^2) 
           + 2ck * log(p)

        where n is the number of samples, k is the number of non-zero elements in the model,
        and c is a penalty parameter between 0 and 1
        and P_s is the projection matrix onto the support set of the model
            given by P_s = A_s (A_s^T A_s)^(-1) A_s^T
            where A_s is the submatrix of A formed by the columns in the support set
            and P_s_perp = I - P_s

    Parameters
    ----------
    A : numpy array
        The design matrix
    y : numpy array
        The target vector
    s : list
        The support set of the model
    c : float
        The penalty parameter
    """
    
    # n = number of samples, p = number of features, k = number of non-zero elements in the model
    n, p = A.shape
    k = len(s)
    # Extract the submatrix A_s
    A_s = A[:, s]
    # Define the projection matrix P_s
    P_s = A_s @ np.linalg.inv(A_s.T @ A_s) @ A_s.T
    # Define the projection matrix P_s_perp
    P_s_perp = np.eye(n) - P_s
    # Compute the EFIC
    EFIC =  n * np.log(np.linalg.norm(P_s_perp @ y)**2) + k * np.log(n) + np.log(np.linalg.det(A_s.T @ A_s)) \
            - (k+2) * np.log(np.linalg.norm(P_s_perp @ y)**2) + 2 * c * k * np.log(p)
    return EFIC

def EBIC_robust(A, y, s, zeta=0.3):
    """
    Computation of the EBIC robust criterion, as proposed in the paper:
    "Robust Information Criterion for Model Selection in Sparse High-Dimensional Linear Regression Models"
    by Magnus Jansson and Prakash Borpatra Gohain

    EBIC_robust = N log(sigma_I^2) + k log(N/2*pi) + (k+2) log(sigma_0^2/sigma_I^2) + 2k* zeta * log(p)

    where N is the number of samples, k is the number of non-zero elements in the model,
    sigma_I^2 is the MLE of the variance of the model parameters under the alternative hypothesis
    sigma_0^2 = y^T y / N is the MLE of the variance of the model parameters under the null hypothesis
    and zeta is a penalty parameter between 0 and 1

    Parameters
    ----------
    A : numpy array
        The design matrix
    y : numpy array
        The target vector
    s : list
        The support set of the model
    zeta : float
        The penalty parameter

    Returns
    -------
    float
        The EBIC_robust of the model
    -------
    """

    # n = number of samples, p = number of features, k = number of non-zero elements in the model
    n, p = A.shape
    k = len(s)
    # Extract the submatrix A_s
    A_s = A[:, s]
    # Define the projection matrix P_s
    P_s = A_s @ np.linalg.inv(A_s.T @ A_s) @ A_s.T
    # Define the projection matrix P_s_perp
    P_s_perp = np.eye(n) - P_s
    # Compute the MLE of the variance of the model parameters under the null hypothesis
    sigma0 = y.T @ y / n
    # Compute the MLE of the variance of the model parameters under the alternative hypothesis
    sigmaI = y.T @ P_s_perp @ y / n
    # Compute the EBIC_robust
    EBIC_robust = n * np.log(sigmaI) + k * np.log(n / (2 * np.pi)) \
                  + (k + 2) * np.log(sigma0 / sigmaI) + 2 * zeta * k * np.log(p)
    return EBIC_robust