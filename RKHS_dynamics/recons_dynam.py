
# reconstructing dynamics, i.e. the vector field

import numpy as np
from scipy.linalg import inv

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
    D = np.sum((XX1-XX2)**2, axis=1)

    if kernel == 'gauss':
        func = lambda x: gauss(x,sigma)
    else:
        pass

    # GG = map(func, D.tolist())
    # GG = np.array(list(GG))
    GG = func(D)
    G = GG.reshape((n,n))
    # G = (G+G.T) / 2

    return G


# compute coeficients for RKHS regularization
def fit_coef(B, G, lamb):
    d, n = B.shape    # B = X_dot
    # n = np.size(G,0)
    G_lamb = G + lamb*np.eye(n)
    V = np.dot(B, inv(G_lamb))    # (dxn)*(nxn)-->dxn
    return V

# recovered function of the vector field
def vecfield(x, Y, V, kernel='gauss', sigma=0.2):
    d, n = Y.shape    # Y is composed of n vectors of d-dim

    if kernel == 'gauss':
        func = lambda x: gauss(x,sigma)
    else:
        pass

    Y1 = Y.T    
    x1 = x.reshape((-1,d))
    One = np.ones(n).reshape((n,1))
    X1 = np.kron(x1, One)
    D = np.sum((X1-Y1)**2, axis=1)
    Phi = func(D)

    # V = V.reshape((d,n))
    f = np.matmul(V, Phi)    # (d,n)*(n,)-->(d,)

    return f











# %%
