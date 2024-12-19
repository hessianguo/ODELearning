
import math
import numpy as np

# define the lotkavolterra system
def lotkavolterra(t, x, para=[0.7,0.007,1.0,0.007]):
   """
    Parameters
    x : array-like, shape (2,)
       (x1, x2) are the quantities of prey-predator
    a = 0.7      # prey birth rate 
    b = 0.007    # prey-predator-collision rate
    c = 1        # predator death rate
    d = 0.007    # prey-predator-collision rate
    Returns
    xyz_dot : array, shape (2,)
       Values of the derivatives at *x*.
   """
   a, b, c, d = para
   x1, x2 = x
   return np.array([a*x1 - b*x1*x2, -c*x2 + d*x1*x2])


# define the Lorenz63 system
def lorenz63(t, xyz, para=[10,28,8/3]):
    """
    Parameters
    ----------
    xyz : array-like, shape (3,)
       Point of interest in three-dimensional space.
    s, r, b : float
       Parameters defining the Lorenz attractor.

    Returns
    -------
    xyz_dot : array, shape (3,)
       Values of the Lorenz attractor's partial derivatives at *xyz*.
    """
    s, r, b = para
    x, y, z = xyz
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return np.array([x_dot, y_dot, z_dot])
   
