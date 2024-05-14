"""
Diffusion MRI, implementation to find the best fit Lambdas.
Using NNLS, and various information criteria
Fix a set of lambdas, followed by a 3 dimensional grid search.
---------------------------
Helper functions used in main code
---------------------------
@author: Ojaswi Jain
Date: 20th March 2024
"""

# import libraries
import numpy as np
from scipy.optimize import nnls
from gen_data import *
from sklearn.linear_model import Lasso

#Evaluating S_g
def evaluate_S_g(S_0, g, b, m, N, V, lambdas, w):
    """
    Function to evaluate S_g
    Parameters:
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
        w: vector(N, 1)
            Weights
        lambdas: vector(3, 1)
            Eigenvalues of the diffusion tensor
    Returns:
        S_g: vector(m, 1)
            Signal intensity  in the presence of diffusion sensitization, direction g
    """
    S_g = np.zeros(m)
    D = generate_eigenvalue_matrix(lambdas)
    for j in range(N):
        Sigma = gen_Sigma(V[j], D)
        Sigma = np.linalg.inv(Sigma)
        for i in range(m):
            g_i = g[i]
            g_i = g_i.reshape(3, 1)
            S_g[i] += w[j]*np.exp(-b*np.dot(np.dot(g_i.T, Sigma), g_i))
    return S_g*S_0

def add_noise_to_S_g(S_g, noise=0.1):
    """
    Function to add noise to S_g of mean = 0, std_dev = noise*mean(S_g)
    
    Parameters:
        S_g: vector(m, 1)  
            Signal intensity in the presence of diffusion sensitization, direction g
        noise: scalar
            Noise level
    Returns:
        S_g: vector(m, 1)
            Noisy signal intensity in the presence of diffusion sensitization, direction g
    """

    return S_g + noise*np.mean(S_g)*np.random.randn(S_g.shape[0])

def convert_to_linear_problem(g, b, m, N, V, lambdas, noise=0.1):
    """
    Function to convert the problem into a linear algebra problem
    of the form y = Aw
    
    Parameters:
        S_g: vector(m, 1)  
            Signal intensity in the presence of diffusion sensitization, direction g
        S_0: scalar 
            Signal intensity in the absence of diffusion sensitization
        g: vector(m, 3)
            Gradient direction
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

    Returns:
        A: Matrix A
        y: Vector y
    """
    # Generate D matrix
    D = generate_eigenvalue_matrix(lambdas)
    
    # Reshape g to have shape (m, 3, 1) to match with V[j]
    g = g[:, :, np.newaxis]
    # Evaluate g-transpose 
    g_T = np.transpose(g, (0, 2, 1))
    # Compute A
    A = np.zeros((m, N))
    for j in range(N):
        Sigma_inv = np.linalg.inv(gen_Sigma(V[j], D))
        res = np.exp(-b * (g_T @ Sigma_inv @ g)).squeeze()
        A[:, j] = res
        
    return A

def off_center_lambdas(lambdas, frac = 0.1):
    """
    Function to generate off-center lambdas
    Parameters:
        lambdas: vector(3, 1)
            Eigenvalues of the diffusion tensor
        frac: float
            Fraction of the eigenvalue to vary by
    Returns:
        lambdas: vector(3, 1)
            Off-center lambdas
    """
    return lambdas + frac*lambdas*np.random.randn(3)

# Function to define lambda grid
def lambda_grid(lambdas, ratio = 1.5, size = 9):
    """
    Function to define the lambda grid with lambda[i] as center
    Parameters:
        lambdas: vector(3, 1)
            Eigenvalues of the diffusion tensor
        frac: float
            Fraction of the eigenvalue to vary by
        size: int
            Number of points to vary by
    Returns:
        grid: Matrix
            Grid of lambdas
    """
    grid = np.zeros((3, size))
    for i in range(3):
        grid[i] = np.geomspace(lambdas[i]/ratio, lambdas[i]*ratio, size)
    return grid  

# Function to compute the relative RMSE
def rel_rmse(x, x_reconstructed):
    """
    Function to compute the relative RMSE
    Parameters:
        x: vector(n, 1)
            Original signal
        x_reconstructed: vector(n, 1)
            Reconstructed signal
    Returns:
        float
            Relative RMSE
    """
    return np.linalg.norm(x - x_reconstructed) / np.linalg.norm(x)  

# Solve the linear problem using nnls, return solution and support set
def solve_nnls(A, y):
    """
    Function to solve the linear problem using nnls
    Parameters:
        A: Matrix
            Design matrix
        y: vector(n, 1)
            Target vector
    Returns:
        w: vector(n, 1)
            Weights
        s: list
            Support set
    """
    try:
        w, _ = nnls(A, y, maxiter=50)
    except RuntimeError as e:
        if "Maximum number of iterations reached" in str(e):
            w = np.zeros(A.shape[1])
        else:
            raise  # Re-raise the exception if it's not the expected one

    # Null all values < 0.1*max(w)
    w = np.where(w < 0.1*np.max(w), 0, w)
    s = np.where(w > 0)[0]
    return w, s

# function for sensitivity and specificity
def sensitivity_specificity(s, s_true, N):
    """
    Function to compute the sensitivity and specificity
    Parameters:
        s: list
            Support set
        s_true: list
            True support set
        N: int
            Number of weights
    Returns:
        sensitivity: float
            Sensitivity
        specificity: float
            Specificity
    """
    # Convert the support set to a set
    s = set(s)
    s_true = set(s_true)
    # Compute the sensitivity and specificity
    TP = len(s.intersection(s_true))
    FN = len(s_true.difference(s))
    FP = len(s.difference(s_true))
    TN = N - len(s.union(s_true))
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    return sensitivity, specificity 

# Function to match the closest eigenvectors:
def match_eigenvectors(V_true, V, lambdas):
    """
    Function to match the closest eigenvector sets and compute the angle error
    Parameters:
        V_true: vector(p, 3, 3)
            True eigenvector sets
        V: vector(N, 3, 3)
            Estimated eigenvector sets
        lambdas: vector(3, 1)
            Eigenvalues of the diffusion tensor
    Returns:
        Matching: vector(N, 1)
            Matching[i] = j, if V[i] is closest to V_true[j]
        Angle_error: vector(N, 3)
            Angle error between V[i] and V_true[Matching[i]]
    """
    # Convert to absolute values for restriction to 1/8th of the sphere
    # V = np.abs(V)
    # V_true = np.abs(V_true)
    N = V.shape[0]
    p = V_true.shape[0]
    Matching = np.zeros(N)
    Angle_error = np.zeros((N, 3))
    # Iterate over the eigenvectors
    for i in range(N):
        print('here')
        min_angle_mag = np.inf
        est_vec = V[i]
        angle = np.zeros(3)
        for j in range(p):
            true_vec = V_true[j]
            for k in range(3):
                dot_prod = np.dot(est_vec[:, k], true_vec[:, k])
                print(dot_prod)
                val = np.min([np.abs(dot_prod), 1])*np.sign(dot_prod)
                #min condition added only because of machine precision errors
                # val = np.min([np.abs(np.dot(V_true[j, k], V[i, k])), 1])
                angle[k] = min(np.arccos(val), np.pi - np.arccos(val))
            angle_mag = np.linalg.norm(np.dot(lambdas, angle))
            if angle_mag < min_angle_mag:
                min_angle_mag = angle_mag
                Matching[i] = j
                Angle_error[i] = angle

    # Convert the angle error to degrees
    Angle_error = np.degrees(Angle_error)
    return Matching, Angle_error