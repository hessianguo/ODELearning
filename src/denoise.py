# fittig the derivative function and observation by vRKHS

import math
import numpy as np
from scipy.integrate import quad, dblquad
from scipy.linalg import inv
from scipy.special import erf

import kernel_mat

# denoise derivative function and observation data by vector-RKHS
def denoise_vrkhs(t, X_ns, kernel_type='gauss', kernel_para=(0.02,)):
    d, n1 = X_ns.shape
    n = n1 - 1
    T = t[1:]
    x0 = X_ns[:,0]
    X1_ns = X_ns[:,1:]

    XX0 = np.kron(x0, np.ones(n))
    XX0 = XX0.reshape((d,n))
    B = X1_ns - XX0    # data of x_t-x0

    if kernel_type == 'gauss':
        sigma = kernel_para[0]
    else:
        pass

    # estimate regularization parameter by L-curve
    lamb = 1e-4
    pass

    # construct kernel matrices and fit derivative
    G1 = kernel_mat.gram_int(T, kernel_type, sigma)    # nxn
    V  = kernel_mat.fit_coef(B, G1, lamb)              # dxn
    Phi = kernel_mat.da_basis(T, T, kernel_type, sigma)     # nxn
    X_dot = kernel_mat.deriv_val(Phi, V)               # dxn  

    # denoise observation data
    Phi_traj  = kernel_mat.gram_traj(T, T, kernel_type, sigma)   # fitted x_t-x_0
    X_fit = kernel_mat.deriv_val(Phi_traj, V) + np.kron(x0,np.ones(len(T))).reshape((d,-1))    # fitted x_t at points T

    return X_dot, X_fit




