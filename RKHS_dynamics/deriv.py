# fittig the derivative function using data-adaptive basis functions

import math
import numpy as np
from scipy.integrate import quad, dblquad
from scipy.linalg import inv
from scipy.special import erf

# define kernel function with superparameters
def gauss_kernel(s,t,sigma=0.2):
    d = -(s-t)**2 / (2*sigma**2)
    return np.exp(d)

# define kernel function with superparameters
def gauss_deriv_t(s,t,sigma=0.2):
    d = -(s-t)**2 / (2*sigma**2)
    l = (s-t) / (sigma**2)
    return np.exp(d)*l


# \int_{0}^{s}k(s,t)ds
def gauss_int1(s, t, sigma=0.2):
    func = lambda x: gauss_kernel(x,t,sigma)
    val = quad(func, 0, s)[0]
    return val

# \int_{0}^{t}(\int_{0}^{s}k(s,t)ds)dt
def gauss_int2(s, t, sigma=0.2):
    func = lambda x, y: gauss_kernel(x,y, sigma)
    val = dblquad(func, 0, t, lambda x: 0, lambda x: s)[0]
    return val


# antiderivative of the error function
def antierr(x):
    val = x*erf(x)+np.exp(-x**2)/np.sqrt(np.pi)
    return val

# function for generating for data-adptive basis, gauss kernel
def func_dab(s, t, sigma=0.2):
    val = 0.0
    d1 = (s-t) / (np.sqrt(2)*sigma)
    d2 = t / (np.sqrt(2)*sigma)
    c = np.sqrt(2*np.pi)*sigma / 2.0
    val = c * (erf(d1)+erf(d2))
    return val

# function for generating Gram matrix, gauss kernel
def func_gram(s, t, sigma=0.2):
    d1 = t / (np.sqrt(2)*sigma)
    d2 = s / (np.sqrt(2)*sigma)
    d3 = (s-t) / (np.sqrt(2)*sigma)
    c = np.sqrt(np.pi) * sigma**2
    val = c * (antierr(d1)+antierr(d2)-antierr(d3)-1.0/np.sqrt(np.pi))
    return val


# build Gram matrix based on point-evaluation
def gram(T, kernel='gauss', sigma=0.2):
    T = T.reshape((-1,))    # T=(t1,...,t_n), 1d array
    n = np.size(T,0)

    TT1 = np.kron(T, np.ones(n))
    TT2 = np.kron(np.ones(n), T)
    if kernel == 'gauss':
        func = lambda x, y: gauss_kernel(x,y,sigma)
    else:
        pass

    GG = map(func, TT1.tolist(), TT2.tolist())
    GG = np.array(list(GG))
    G = GG.reshape((n,n))
    G = (G+G.T) / 2

    return G

# compute point-evaluation basis functions at the given m time points T
def kernel_basis(S, T, kernel='gauss', sigma=0.2):
    S = S.reshape((-1,))
    n = np.size(S,0)
    T = T.reshape((-1,))
    m = np.size(T,0)
    if kernel == 'gauss':
        func = lambda x,y: gauss_kernel(x, y, sigma)
    else:
        pass

    SS = np.kron(S, np.ones(m))    # mn 1d array
    TT = np.kron(np.ones(n), T)    # mn ad array
    BB = map(func, SS.tolist(), TT.tolist())    # func(si,tj)
    BB = np.array(list(BB))
    Phi = BB.reshape((n,m))    # nxm 2d array, values of n basis function at m time points
    
    return Phi

# compute kernel-derivative basis functions at the given m time points T
def deriv_basis(S, T, kernel='gauss', sigma=0.2):
    S = S.reshape((-1,))
    n = np.size(S,0)
    T = T.reshape((-1,))
    m = np.size(T,0)
    if kernel == 'gauss':
        func = lambda x,y: gauss_deriv_t(x, y, sigma)
    else:
        pass

    SS = np.kron(S, np.ones(m))    # mn 1d array
    TT = np.kron(np.ones(n), T)    # mn ad array
    BB = map(func, SS.tolist(), TT.tolist())    # func(si,tj)
    BB = np.array(list(BB))
    Phi = BB.reshape((n,m))    # nxm 2d array, values of n basis function at m time points
    
    return Phi


# build Gram matrix based on integration
def gram_int(T, kernel='gauss', sigma=0.2):
    T = T.reshape((-1,))    # T=(t1,...,t_n), 1d array
    n = np.size(T,0)

    TT1 = np.kron(T, np.ones(n))
    TT2 = np.kron(np.ones(n), T)
    if kernel == 'gauss':
        # func = lambda x, y: gauss_int2(x,y,sigma)
        func = lambda x, y: func_gram(x,y,sigma)
    else:
        pass

    GG = map(func, TT1.tolist(), TT2.tolist())
    GG = np.array(list(GG))
    G = GG.reshape((n,n))
    G = (G+G.T) / 2

    return G

# compute coeficients for derivative function
def fit_coef(B, G, lamb):
    d, n = B.shape    # B=(b1,..., bn), bi=yi-x0
    # n = np.size(G,0)
    G_lamb = G + lamb*np.eye(n)
    V = np.dot(B, inv(G_lamb))    # (dxn)*(nxn)-->dxn
    return V

# compute data-adaptive basis functions at the given m time points T
def da_basis(S, T, kernel='gauss', sigma=0.2):
    S = S.reshape((-1,))
    n = np.size(S,0)
    T = T.reshape((-1,))
    m = np.size(T,0)
    if kernel == 'gauss':
        # func = lambda x,y: gauss_int1(x, y, sigma)
        func = lambda x,y: func_dab(x, y, sigma)
    else:
        pass

    SS = np.kron(S, np.ones(m))    # mn 1d array
    TT = np.kron(np.ones(n), T)    # mn ad array
    BB = map(func, SS.tolist(), TT.tolist())    # func(si,tj)
    BB = np.array(list(BB))
    Phi = BB.reshape((n,m))    # nxm 2d array, values of n basis function for derivative at m time points
    
    return Phi

# matrix for fitting trajector at the given m time points T
def gram_traj(S, T, kernel='gauss', sigma=0.2):
    S = S.reshape((-1,))
    n = np.size(S,0)
    T = T.reshape((-1,))
    m = np.size(T,0)

    if kernel == 'gauss':
        # func = lambda x, y: gauss_int2(x,y,sigma)
        func = lambda x, y: func_gram(x,y,sigma)
    else:
        pass

    SS = np.kron(S, np.ones(m))    # mn 1d array
    TT = np.kron(np.ones(n), T)    # mn ad array
    BB = map(func, SS.tolist(), TT.tolist())    # func(si,tj)
    BB = np.array(list(BB))
    Phi = BB.reshape((n,m))    # nxm 2d array, values of n basis function for trajectory at m time points

    return Phi


# compute values of fitted derivative function (or others) using the DA-basis at the observations
def deriv_val(Phi, V):
    X_dot = np.matmul(V, Phi)    # (dxn)*(nxm)-->dxm
    return X_dot
