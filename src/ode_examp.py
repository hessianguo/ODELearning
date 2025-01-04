
import math
import numpy as np

# define the lotkavolterra system
def lotkavolterra(t, xy, para=[0.7,0.007,1.0,0.007]):
   """
    Parameters
    xy : array-like, shape (2,)
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
   x1, x2 = xy
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


# Rossler system (a chaotic attractor)
def rossler(t, xyz, para=[0.2,0.2,5.7]):
    """
    Parameters
    ----------
    xyz : array-like, shape (3,)
       Point of interest in three-dimensional space.
    para=(a, r, b) : float
       Parameters defining the rossler attractor.

    Returns
    -------
    xyz_dot : array, shape (3,)
       Values of the rossler attractor's partial derivatives at *xyz*.
    """
    a, b, c = para
    x, y, z = xyz
    x_dot = -y - z
    y_dot = x + a*y
    z_dot = b + z*(x-c)
    return np.array([x_dot, y_dot, z_dot])


# Forced vibration of nonlinear pendulum.
def pendulum(t, xy, para=[10]):
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
   l = para[0]   # lenght of the pendulum
   g = 9.8
   alpha = g / l
   x1, x2 = xy
   f = lambda x: 2 * np.cos(5*x1)    # external force: f(x1)=2cos(5x_1)
   return np.array([x2, f(x1)-alpha*np.sin(x1)])


# define the Lorenz96 system. 
def lorenz96(t, x, paras=[10,8]):
   """
    Parameters
    ----------
    x: array-like, shape (n,)
       Point of interest in three-dimensional space.
    d: dimention
    F: external force. (F=8 is a common value known to cause chaotic behavior.)

    Returns
    -------
    x_dot : array, shape (,)
       Values of the Lorenz attractor's partial derivatives at *xyz*.
   """
   d = paras[0]
   F = paras[1]
   x_dot = (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F 
   return np.array(x_dot)
