# Imports
import numpy as np
from dipy.core.sphere import Sphere
from dipy.data import get_sphere
from funcs import *
from make_plots import *
from utils import *

S_0 = 100 # Signal intensity in the absence of diffusion sensitization
lambdas = np.array([1e-2, 3e-3, 3e-3]) # True eigenvalues of the diffusion tensor
b = 1 # b-value
m = 200 # number of gradient directions
n1 = 200 # v1 options
n2 = 90 # v2 options

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

# def generate_V():
#     """
#     Generate N = n1*n2 sets of eigenvectors of the diffusion tensor.
#     Returns:
#         V: vector(N, 3, 3)
#             Eigenvectors of the diffusion tensor
#     """
#     sphere_points = generate_sphere_points()
#     V = np.zeros((n1*n2, 3, 3))
#     for i in range(n1):
#         v1 = sphere_points[i]
#         circle_points = generate_circle_points(v1)
#         for j in range(n2):
#             v2 = circle_points[j]
#             v3 = np.cross(v1, v2)
#             V[i*n2 + j] = np.array([v1, v2, v3]).T

#     return V

V = generate_V()
#make array of 3 random Vs
V_true = V[np.random.randint(n1*n2, size=2)]
V_est = V[np.random.randint(n1*n2, size=2)]
Matching, Angle_error = match_eigenvectors(V_true, V_est, lambdas)
for i in range(2):
    print('V_est:', V_est[i])
    print('Matching:', Matching[i])
    print('Corresponding V_true:', V_true[int(Matching[i])])
    print('Angle_error in degrees:', Angle_error[i])
    print('\n')

# v1_true = generate_sphere_points()
# perturbed_points = perturb_sphere_points(v1_true)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# k = 100
# ax.scatter(v1_true[:k, 0], v1_true[:k, 1], v1_true[:k, 2], c='r', marker='o')
# ax.scatter(perturbed_points[:k, 0], perturbed_points[:k, 1], perturbed_points[:k, 2], c='b', marker='o')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()





