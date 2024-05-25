import numpy as np



def add_noise(X, nsr, type="white_gauss"):
    d, n = X.shape

    XX = np.power(X, 2)
    x1 = np.sum(XX, axis=0)
    nX = np.power(x1, 0.5)   # (n,) array, each element is the 2-norm of X_i 
    
    if type == 'white_gauss':
        E = np.random.randn(d, n)
    else:
        pass

    EE = np.power(E, 2)
    e1 = np.sum(EE, axis=0)
    nE = np.power(e1, 0.5)

    r = nX / nE * nsr   
    e = r * E   # multiply by columns
    X_ns = X + e

    return X_ns





