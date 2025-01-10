import numpy as np



def add_noise(X, nsr, type=1):
    d, n = X.shape

    if type == 1:
        XX = np.power(X, 2)
        x1 = np.sum(XX, axis=0)
        nX = np.power(x1, 0.5)   # (n,) array, each element is the 2-norm of X_i 
        E = np.random.randn(d, n)
        EE = np.power(E, 2)
        e1 = np.sum(EE, axis=0)
        nE = np.power(e1, 0.5)
        r = nX / nE * nsr   
        e = r * E   # multiply by columns
    elif type == 2:
        e = np.random.normal(scale=nsr, size=(d,n))
    elif type == 3:
        nx = np.sqrt(np.sum(X))
        E1 = np.random.normal(scale=1, size=(d,n))
        ne = np.sqrt(np.sum(E1))
        e  = E1 / ne * nsr * nx
    else:
        pass

    X_ns = X + e

    return X_ns





