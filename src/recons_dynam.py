# reconstructing dynamics, i.e. the vector field

import numpy as np
from scipy.linalg import inv
from l_curve import lcurve
from tikhonov import tikh

# define kernel function with superparameters
def gauss(d,sigma=0.2):
    a = -d / (2*sigma**2)    # d is the square of Euclidean distance between two points
    return np.exp(a)

# build Gram matrix based on X--X point-evaluation
def gram(X, kernel='gauss', sigma=0.2):
    d, n = X.shape    # X is composed of n vectors of d-dim

    X1 = X.T
    
    One = np.ones(n).reshape((n,1))
    XX1 = np.kron(X1, One)
    XX2 = np.kron(One, X1)
    D = np.sum((XX1-XX2)**2, axis=1)    # (n^2,) array

    if kernel == 'gauss':
        func = lambda x: gauss(x,sigma)
    else:
        pass

    GG = func(D)
    G = GG.reshape((n,n))

    return G


# compute coeficients for RKHS regularization of f
def fit_coef(B, G, lamb, lamb_type='auto'):
    if lamb_type == 'pre_select':
        lamb1 = lamb
        d, n = B.shape    # B = X_dot
        G_lamb = G + lamb*np.eye(n)
        V = np.dot(B, inv(G_lamb))    # (dxn)*(nxn)-->dxn
    elif lamb_type == 'auto':
        U, s, Ut = np.linalg.svd(G)
        reg_c, rho, eta, reg_param = lcurve(U, s, B)
        lamb1 = reg_c**2
        V = tikh(U, s, B, lamb1)
    else:
        pass

    return V


# error distanse square between x and Y=[y1,...yn]
def err_mat(x, Y):
    d, n = Y.shape 
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    x1 = x.reshape((-1,d))  
    One = np.ones(n).reshape((n,1))
    X1 = np.kron(x1, One)    # (n,d) array
    Y1 = Y.T 
    D = np.sum((X1-Y1)**2, axis=1)    # (d,) array

    return D  


# recovered values of vector field f at k points X=(x1,...xk)
def vectfd(X, X_dn, X_dot, lamb, lamb_type='auto', kernel='gauss', sigma=0.2):
    G = gram(X_dn, kernel, sigma)
    V = fit_coef(X_dot, G, lamb, lamb_type)

    d, n = X_dn.shape    # Y is composed of n vectors of d-dim
    if kernel == 'gauss':
        func1 = lambda x: gauss(x,sigma)
    else:
        pass

    func2 = lambda x: err_mat(x, X_dn)
    E = list(map(func2, X.T.tolist()))   # list of k (n,) array
    E = np.array(E).T    # (n,k) array
    D = func1(E)

    Phi = func(D)
    f = np.matmul(V, Phi)    # (d,n)*(n,k)-->(d,k)
    if np.size(f,1) == 1:
        f = f.reshape(-1)

    return f




    

