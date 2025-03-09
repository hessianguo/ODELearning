# fittig the derivative function and observation by vRKHS

import math
import numpy as np
from scipy.integrate import quad, dblquad
from scipy.linalg import inv
from scipy.special import erf

import kernel_mat
from l_curve import lcurve
from tikhonov import tikh

# denoise derivative function and observation data by vector-RKHS
def denoise_vrkhs(t, X_ns, lamb, lamb_type='auto', kernel_type='gauss', kernel_para=(0.02,)):
    d, n1 = X_ns.shape
    n = n1 - 1
    T = t[1:] - t[0]
    x0 = X_ns[:,0]
    X1_ns = X_ns[:,1:]

    XX0 = np.kron(x0, np.ones(n))
    XX0 = XX0.reshape((d,n))
    B = X1_ns - XX0    # data of x_t-x0

    if kernel_type == 'gauss':
        sigma = kernel_para[0]
    else:
        pass

    # lamb = 1e-4
    # pass

    # construct kernel matrices and fit derivative
    G1 = kernel_mat.gram_int(T, kernel_type, sigma)         # nxn
    Phi = kernel_mat.da_basis(T, T, kernel_type, sigma)     # nxn

    # estimate regularization parameter by L-curve
    if lamb_type == 'pre_select':
        lamb1 = lamb
        V  = kernel_mat.fit_coef(B, G1, lamb1)              # dxn
    elif lamb_type == 'auto':
        U, s, Ut = np.linalg.svd(G1)
        reg_c, rho, eta, reg_param = lcurve(U, s, B)
        lamb1 = reg_c**2
        V = tikh(U, s, B, lamb1)
    else:
        pass
    
    X_dot = kernel_mat.deriv_val(Phi, V)               # dxn  
    # denoise observation data
    Phi_traj  = kernel_mat.gram_traj(T, T, kernel_type, sigma)   # fitted x_t-x_0
    X_fit = kernel_mat.deriv_val(Phi_traj, V) + np.kron(x0,np.ones(len(T))).reshape((d,-1))    # fitted x_t at points T

    return X_dot, X_fit, lamb1




