import numpy as np
import scipy as sp
from itertools import product
from scipy import linalg

def polynomial_basis(X, degree):
    """
    Generates polynomial basis for an n-by-d input matrix up to a given degree.
    
    Args:
        X (numpy.ndarray): n-by-d input matrix where n is the number of data points and d is the dimensionality.
        degree (int): Maximum degree of polynomial.
        
    Returns:
        numpy.ndarray: Matrix of size n-by-m containing polynomial basis evaluated at X,
                       where m is the dimensionality of the polynomial space.
    """
    if X.ndim == 1:
        d, = X.shape
        n = 1
        X = np.atleast_2d(X)
    elif X.ndim == 2:
        n, d = X.shape  # Number of data points and dimensionality
    exponents = get_exponents(d, degree)  # Get all combinations of exponents
    m = len(exponents)  # Dimensionality of the polynomial space
    
    # Initialize polynomial basis matrix
    P = np.ones((n, m))  
    
    # Calculate each polynomial term
    for j, exp in enumerate(exponents):
        P[:, j] = np.prod(X ** exp, axis=1)
    
    return P

def get_exponents(d, degree):
    """
    Generates all combinations of exponents for polynomials of d variables up to a given degree.
    
    Args:
        d (int): Number of variables.
        degree (int): Maximum degree of polynomial.
        
    Returns:
        list of tuples: Each tuple represents a combination of exponents for the polynomial terms.
    """
    exponents = []
    
    # Iterate over all possible total degrees from 0 to the given degree
    for total_degree in range(degree + 1):
        # Generate combinations of exponents that sum to total_degree
        for comb in product(range(total_degree + 1), repeat=d):
            if sum(comb) == total_degree:
                exponents.append(comb)
                
    return exponents

def sparseode(t, x, xi, degree):
    """
    Computes the time derivative of the state vector x using the sparse ODE formulation.
    
    Args:
        t (float): Time.
        x (numpy.ndarray): State vector.
        xi (numpy.ndarray): Sparse coefficient matrix.
        degree (int): Maximum degree of polynomial.
        
    Returns:
        numpy.ndarray: Time derivative of the state vector x.
    """
    n, d = xi.shape
    P = polynomial_basis(x, degree)
    dxdt = np.matmul(P, xi)
    
    return dxdt


# sparse vector field
def sparsevf(x, xi, degree):
    n, d = xi.shape
    P = polynomial_basis(x, degree)
    dx = np.matmul(P, xi)
    dx = dx.reshape((-1))
    
    return dx



def stsl(A, b, tol=1e-2, num_iter=10):
    """
    Perform sparse regression using the STSL algorithm.

    Parameters:
    A (numpy.ndarray): The input matrix of shape (m, p).
    b (numpy.ndarray): The target vector of shape (m, n).
    tol (float): The tolerance value for determining small coefficients.
    num_iter (int, optional): The number of iterations. Defaults to 10.

    Returns:
    numpy.ndarray: The sparse regression coefficients of shape (p, n).
    """
    n = b.shape[1]

    # Perform initial least squares regression
    Xi = np.linalg.lstsq(A, b, rcond=None)[0]

    for _ in range(num_iter):
        for ind in range(n):
            # Identify small coefficients
            smallinds = np.abs(Xi[:, ind]) < tol
            Xi[smallinds, ind] = 0

            biginds = ~smallinds
            if np.any(biginds):
                # Perform least squares regression with non-zero coefficients
                Xi[biginds, ind] = np.linalg.lstsq(A[:, biginds], b[:, ind], rcond=None)[0]

    return Xi