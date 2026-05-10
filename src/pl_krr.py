# fittig the derivative function and observation by plug-in KRR

import math
import numpy as np
from scipy.integrate import quad, dblquad
from scipy.linalg import inv
from scipy.special import erf

import kernel_mat
from l_curve import lcurve
from qoc import qoc
from tikhonov import tikh

# denoise derivative function and observation data by vector-RKHS
def pl_krr(t, X_ns, lamb, lamb_type='auto', kernel_type='gauss', kernel_para=(0.02,)):
    d, n = X_ns.shape
    T = t
    B = X_ns 

    if kernel_type == 'gauss':
        sigma = kernel_para[0]
    else:
        pass

    # lamb = 1e-4
    # pass

    # construct kernel matrices and fit derivative
    G1 = kernel_mat.gram(T, kernel_type, sigma)             # nxn
    Phi = kernel_mat.kernel_basis(T, T, kernel_type, sigma)     # nxn
    Phi_deriv = kernel_mat.deriv_basis(T, T, kernel_type, sigma) 

    # estimate regularization parameter by L-curve
    if lamb_type == 'pre_select':
        lamb1 = lamb
        V  = kernel_mat.fit_coef(B, G1, lamb1)              # dxn
    elif lamb_type == 'auto':
        U, s, Ut = np.linalg.svd(G1)
        reg_c, rho, eta, reg_param = lcurve(U, s, B)
        lamb1 = reg_c**2
        V = tikh(U, s, B, lamb1)
    elif lamb_type == 'auto-qoc':    # quasi-optimality criterion
        U, s, Ut = np.linalg.svd(G1)
        reg_c, qvals, reg_param = qoc(U, s, B)
        lamb1 = reg_c**2
        V = tikh(U, s, B, lamb1)
    else:
        pass

    # denoise observation data and derivative
    X_fit = kernel_mat.deriv_val(Phi, V)   # dxn  
    X_dot = kernel_mat.deriv_val(Phi_deriv, V) 

    return X_dot, X_fit, lamb1




