"""
File to generate S_0, g, b, V, true_lambdas
---------------------------
@Author: Ojaswi Jain
Date: 21th March 2024
---------------------------
"""

# Imports
import numpy as np
from dipy.core.sphere import Sphere
from dipy.data import get_sphere

S_0 = 1000 # Signal intensity in the absence of diffusion sensitization
lambdas = np.array([1e-2, 3e-3, 3e-4]) # True eigenvalues of the diffusion tensor
b = 1e-3 # b-value
m = 200 # number of gradient directions
n1 = 200 # v1 options
n2 = 91 # v2 options
N = n1*n2 # number of diffusion tensors
l1, l2, l3 = lambdas
l1_range = np.geomspace(l1/1.5, l1*1.5, 9) # l1 range
l2_range = np.geomspace(l2/1.5, l2*1.5, 9) # l2 range
l3_range = np.geomspace(l3/1.5, l3*1.5, 9) # l3 range

def generate_sphere_points():
    """
    Generate N=200 points evenly distributed on the surface of a unit sphere.
    Returns:
        points: vector(N, 3)
            Points on the sphere
    """
    sphere = Sphere(xyz=get_sphere('repulsion200').vertices)
    points = sphere.vertices
    # np.random.shuffle(points)
    return points

def perturb_sphere_points(points, angle=1):
    """
    Perturb the sphere points by very small angles in theta and phi to avoid degeneracy.
    Parameters:
        points: vector(N, 3)
            Points on the sphere
        angle: float
            Angle to perturb the sphere points by
    Returns:
        perturbed_points: vector(N, 3)
            Perturbed points on the sphere
    """
    perturbed_points = np.zeros(points.shape)
    for i in range(points.shape[0]):
        theta = np.arccos(points[i, 2])
        phi = np.arctan2(points[i, 1], points[i, 0])
        # Add 1 degree of noise
        theta += np.random.rand()*angle*np.pi/180
        phi += np.random.rand()*angle*np.pi/180
        perturbed_points[i] = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    return perturbed_points

def generate_circle_points(v1, n=n2):
    """
    Given a vector v1, generate N = 90 uniformly 
    distibuted points on the circle on the plane perpendicular to v1.
    Parameters:
        v1: vector(3, 1)
            Vector on the sphere
    Returns:
        circle_points: vector(n, 3)
            Points on the circle
    """
    # Plane perpendicular to v1
    v_mid = np.random.rand(3)
    v2 = np.cross(v1, v_mid)
    v2 = v2/np.linalg.norm(v2)
    v3 = np.cross(v1, v2)
    v3 = v3/np.linalg.norm(v3)
    # Generate the circle points
    circle_points = np.zeros((n, 3))
    for i in range(n):
        theta = 2*np.pi*i/n
        circle_points[i] = np.cos(theta)*v2 + np.sin(theta)*v3
    # np.random.shuffle(circle_points)
    return circle_points

def generate_V():
    """
    Generate N = n1*n2 sets of eigenvectors of the diffusion tensor.
    Returns:
        V: vector(N, 3, 3)
            Eigenvectors of the diffusion tensor
    """
    sphere_points = generate_sphere_points()
    V = np.zeros((n1*n2, 3, 3))
    for i in range(sphere_points.shape[0]):
        v1 = sphere_points[i]
        circle_points = generate_circle_points(v1)
        for j in range(circle_points.shape[0]):
            v2 = circle_points[j]
            v3 = np.cross(v1, v2)
            V[i*n2 + j] = np.column_stack((v1, v2, v3))
    return V

def generate_off_center_V():
    """
    Generate N = n1*n2 sets of eigenvectors, none of which are
    coincident with any of the true measurement vectors.
    Returns:
        V: vector(N, 3, 3)
            Displaced eigenvectors of the diffusion tensor
    """
    v1_true = generate_sphere_points()
    perturbed_points = perturb_sphere_points(v1_true)
    V = np.zeros((n1*n2, 3, 3))
    for i in range(n1):
        v1 = perturbed_points[i]
        circle_points = generate_circle_points(v1)
        for j in range(n2):
            v2 = circle_points[j]
            v3 = np.cross(v1, v2)
            V[i*n2 + j] = np.column_stack((v1, v2, v3))
    
    return V


def generate_g(m=200):
    """
    Generate m = 200 gradient directions g.
    Parameters:
        m: int
            Number of gradient directions
    Returns:
        g: vector(m, 3)
            Gradient directions
    """
    g = np.zeros((m, 3))
    for i in range(m):
        theta = np.random.rand()*np.pi 
        phi = np.random.rand()*2*np.pi
        g[i] = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    return g

def generate_eigenvalue_matrix(lambdas):
    """
    Function to generate the eigenvalue matrix
    D = diag(lambdas)
    Parameters:
        lambdas: vector(3, 1)
            Eigenvalues of the diffusion tensor
    Returns:
        D: Matrix
            Eigenvalue matrix
    """
    D = np.diag(lambdas)
    return D

def gen_Sigma(V, D):
    """
    Generate the diffusion tensor Sigma = VDV^T
    Parameters:
        V: vector(3, 3)
            Eigenvectors of the diffusion tensor
        D: Matrix
            Eigenvalue matrix
    Returns:
        Sigma: Matrix
            Diffusion tensor
    """
    return V @ D @ V.T

# Function to generate true weights
def generate_true_weights(sparsity = 3):
    """
    Generate true weights of dim N = n1*n2
    Parameters:
        sparsity: int
            Sparsity of the true weights
    Returns:
        w: vector(N, 1)
            True weights
    """
    N = n1*n2
    w = np.zeros(N)
    s = np.random.choice(N, sparsity, replace=False)
    w[s] = abs(np.random.rand(sparsity))
    return w, s