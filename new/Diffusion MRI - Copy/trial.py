# Imports
import numpy as np
from dipy.core.sphere import Sphere
from dipy.data import get_sphere

S_0 = 100 # Signal intensity in the absence of diffusion sensitization
lambdas = np.array([1e-2, 3e-3, 3e-3]) # True eigenvalues of the diffusion tensor
b = 1 # b-value
m = 200 # number of gradient directions
n1 = 200 # v1 options
n2 = 90 # v2 options
from funcs import *

np.random.seed(1)

def generate_sphere_points():
    """
    Generate N=200 points evenly distributed on the surface of a unit sphere.
    Returns:
        points: vector(N, 3)
            Points on the sphere
    """
    sphere = Sphere(xyz=get_sphere('repulsion200').vertices)
    points = sphere.vertices
    return points

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
    # print(v2)
    v2 = v2/np.linalg.norm(v2)
    v3 = np.cross(v1, v2)
    v3 = v3/np.linalg.norm(v3)
    # Generate the circle points
    circle_points = np.zeros((n, 3))
    for i in range(n):
        theta = 2*np.pi*i/n
        circle_points[i] = np.cos(theta)*v2 + np.sin(theta)*v3
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
    for i in range(n1):
        v1 = sphere_points[i]
        circle_points = generate_circle_points(v1)
        for j in range(n2):
            v2 = circle_points[j]
            v3 = np.cross(v1, v2)
            V[i*n2 + j] = np.array([v1, v2, v3]).T

    return V

# V = generate_V()
# # choose any random set of eigenvectors
# V = V[0]
# print(V)
# print(np.linalg.norm(V, axis=1))
# print(np.linalg.norm(V, axis=0))

# #print pairwise row dot products
# for i in range(V.shape[0]):
#     for j in range(i+1, V.shape[0]):
#         print(np.dot(V[i], V[j]))

# #print pairwise column dot products
# for i in range(V.shape[1]):
#     for j in range(i+1, V.shape[1]):
#         print(np.dot(V[:,i], V[:,j]))

# v1 = generate_sphere_points()[0]
# # print(v1)
# v2 = generate_circle_points(v1)[0]
# # print(v2)
# v3 = np.cross(v1, v2)
# # print(v3)

# V = np.array([v1, v2, v3]).T
# print(V)
# print(V@V.T)
# print(V.T@V)

# D = np.diag(lambdas)
# print(D)
# print(np.linalg.inv(D))
# Sigma =V@D@V.T
# print(Sigma)
# Sigma_inv = np.linalg.inv(Sigma)
# print(Sigma_inv)

# First array
# V1 = np.array([[-0.32385683, -0.64206418, -0.69488872],
#                    [ 0.71108536,  0.31929294, -0.62642607],
#                    [ 0.62407881, -0.69699756,  0.35315724]])

# # Second array
# V2 = np.array([[-0.32385683,  0.11099549,  0.93957264],
#                    [ 0.71108536, -0.62651742,  0.31911366],
#                    [ 0.62407881,  0.77146349,  0.12397468]])

# print(V1@V1.T)
# print(V2@V2.T)
# print('---------------------------------')

# D = np.diag(lambdas)
# print(D)
# print('---------------------------------')
# Sigma1 = V1@D@V1.T
# Sigma2 = V2@D@V2.T
# print('---------------------------------')
# print(Sigma1)
# print(Sigma2)
# print('---------------------------------')
# Sigma_inv1 = np.linalg.inv(Sigma1)
# Sigma_inv2 = np.linalg.inv(Sigma2)
# print(Sigma_inv1)
# print(Sigma_inv2)
# print('---------------------------------')


grid = lambda_grid(lambdas)
for i in range(len(lambdas)):
    print(lambdas[i])
    print(grid[i])
    print('---------------------------------')

from gen_data import *
print(l1_range)
print(l2_range)
print(l3_range)