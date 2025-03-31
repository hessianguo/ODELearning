# Computes the Tikhonov regularized solution V_lamb, which is 
# V_lamb = B(G1+lamb*I_n)^{-1}

def tikh(U, s, B, lamb):
    d, n = B.shape
    U1 = B @ U
    s1 = 1. / (s+lamb)
    V_lamb = (U1*s1) @ U.T

    return V_lamb


