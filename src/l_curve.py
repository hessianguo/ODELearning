# Plot the L-curve and find its "corner".
# Plots the L-shaped curve of eta, the solution norm, as a function 
# of rho, the residual norm.
#
# Reference: [1] P. C. Hansen & D. P. O'Leary, "The use of the L-curve in the regularization 
# of discrete ill-posed problems", SIAM J. Sci. Comput. 14 (1993), pp. 1487-1503.

import numpy as np
from scipy.optimize import fminbound
import matplotlib.pyplot as plt


''' Computes the NEGATIVE of the curvature of L-curve
'''
def lcfun(lamb, ss, beta, xi):
    if not isinstance(lamb, np.ndarray):
        lamb = np.array([lamb])

    # Initialization
    phi = np.zeros_like(lamb)
    dphi = np.zeros_like(lamb)
    psi = np.zeros_like(lamb)
    dpsi = np.zeros_like(lamb)
    eta = np.zeros_like(lamb)
    rho = np.zeros_like(lamb)

    # Compute some intermediate quantities
    for i in range(len(lamb)):
        f = ss**2 / (ss**2 + lamb[i]**2)
        cf = 1 - f
        eta[i] = np.linalg.norm(f * xi)
        rho[i] = np.linalg.norm(cf * beta)
        
        f1 = -2 * f * cf / lamb[i]
        f2 = -f1 * (3 - 4 * f) / lamb[i]
        
        phi[i] = np.sum(f * f1 * np.abs(xi)**2)
        psi[i] = np.sum(cf * f1 * np.abs(beta)**2)
        dphi[i] = np.sum((f1**2 + f * f2) * np.abs(xi)**2)
        dpsi[i] = np.sum((-f1**2 + cf * f2) * np.abs(beta)**2)

    # Compute the first and second derivatives of eta and rho with respect to lambda
    deta = phi / eta
    drho = -psi / rho
    ddeta = dphi / eta - deta * (deta / eta)
    ddrho = -dpsi / rho - drho * (drho / rho)

    # Convert to derivatives of log(eta) and log(rho)
    dlogeta = deta / eta
    dlogrho = drho / rho
    ddlogeta = ddeta / eta - (dlogeta)**2
    ddlogrho = ddrho / rho - (dlogrho)**2

    # Let g = curvature
    g = - (dlogrho * ddlogeta - ddlogrho * dlogeta) / (dlogrho**2 + dlogeta**2)**1.5

    return g


'''
Locate the "corner" of the L-curve.
Parameters:
rho, eta, reg_param : arrays of corresponding values of ||A x - b||, ||L x ||, and the regularization parameter;
ss: the GSVD value of {G,G^{1/2}};
beta: (U\kron I_d)^T vec(B);
M: Upper bound for eta (optional)
Returns:
reg_c, rho_c, eta_c : The corner point of the L-curve
'''
def l_corner(rho, eta, reg_param, ss, beta, M=None):
    order = 4    # Order of fitting 2-D spline curve
    if len(rho) < order:
        raise ValueError('Too few data points for L-curve analysis')

    # Restrict the analysis of the L-curve according to M (if specified)
    if M is not None:
        index = eta < M
        rho = rho[index]
        eta = eta[index]
        reg_param = reg_param[index]
    
    # Compute g = - curvature of L-curve
    xi = beta / ss
    g = lcfun(reg_param, ss, beta, xi)

    # Locate the corner. If the curvature is negative everywhere, then define the leftmost point of the L-curve as the corner
    gmin = np.min(g)
    gi   = np.argmin(g)

    # Minimizer to find the corner
    reg_c = fminbound(lambda x: lcfun(x, ss, beta, xi),
                          reg_param[min(gi + 1, len(g) - 1)],  
                          reg_param[max(gi - 1, 0)],  
                          disp=False)
    kappa_max = -lcfun(reg_c, ss, beta, xi)  # Maximum curvature
    kappa_max = kappa_max[0]

    if kappa_max < 0:
        lr = len(rho)
        reg_c = reg_param[lr-1]
        rho_c = rho[lr-1]
        eta_c = eta[lr-1]
    else:
        f = (ss**2) / (ss**2 + reg_c**2)
        eta_c = np.linalg.norm(f * xi)
        rho_c = np.linalg.norm((1 - f) * beta)

    return reg_c, rho_c, eta_c 


"""
    Plot the L-curve with optional markers for regularization parameters.
    Arguments:
    rho -- Array of residual norms.
    eta -- Array of solution norms.
    marker -- Marker style for the plot (default is '-').
    reg_param -- Array of regularization parameters (optional).
"""
def plot_lc(rho, eta, marker='-', reg_param=None):
    np_points = 10  # Number of identified points

    # Initialize
    n  = len(rho)
    ni = round(n / np_points)

    # Create the plot
    plt.figure()
    if (np.max(eta) / np.min(eta) > 10 or np.max(rho) / np.min(rho) > 10):
        if reg_param is None:
            plt.loglog(rho, eta, marker)
        else:
            plt.loglog(rho, eta, marker)
            plt.plot(rho[ni-1::ni], eta[ni-1::ni], 'x')
            for k in range(ni-1, n, ni):
                plt.text(rho[k], eta[k], str(reg_param[k]))
    else:
        if reg_param is None:
            plt.plot(rho, eta, marker)
        else:
            plt.plot(rho, eta, marker)
            plt.plot(rho[ni-1::ni], eta[ni-1::ni], 'x')
            for k in range(ni-1, n, ni):
                plt.text(rho[k], eta[k], str(reg_param[k]))

    # Set the labels and title
    plt.xlabel('Residual Norm')
    plt.ylabel('Solution Norm')
    plt.title('L-curve')
    plt.show()


'''
L-curve for min||Gx-b||^2+lambda*||G^{1/2}x||^2,
where G=G1\kron I_d, b=vec(B).
G1=U\diag(s)U^T is the eigen-decomposition of G1.
'''
def lcurve(U, s, B):
    # compute the GSVD of {G,G^{1/2}}
    d, n = B.shape
    Beta = B @ U
    beta = Beta.flatten(order = 'F')    # beta=vec(Beta)=(U\kron I_d)^T vec(B)
    s1 = np.kron(s, np.ones(d))         # eigenvalues of G=G1\kron I_d
    ss = np.sqrt(s1)                    # the GSVD value of (G, G^0.5)

    # values of lambda from smallest to largest
    npoints = 200               # Number of points on the L-curve
    reg_param = np.zeros(npoints)
    eps = np.finfo(float).eps 
    smin_ratio = 1000*eps       # Smallest regularization parameter.
    reg_param[npoints-1] = np.max([s1[n*d-1], ss[0]*smin_ratio])   # smallest value of lambda
    ratio = (ss[0]/reg_param[npoints-1])**(1/(npoints-1)) 
    for i in range(npoints-2,-1,-1):
        reg_param[i] = ratio*reg_param[i+1]

    # compute the solution and residual norms
    xi = beta / ss
    s2 = ss**2
    eta = np.zeros(npoints)    # solution norm
    rho = np.zeros(npoints)    # residual norm
    for i in np.arange(npoints):
        f = s2 / (s2+reg_param[i]**2)
        eta[i] = np.linalg.norm(f * xi)
        rho[i] = np.linalg.norm((1 - f) * beta)

    # Locate the "corner" of the L-curve
    reg_c, rho_c, eta_c = l_corner(rho, eta, reg_param, ss, beta)

    # Make plot
    marker = '-'
    plot_lc(rho, eta, marker, reg_param)
    ax = plt.gca()  # Get current axis
    # Plot corner of L-curve
    plt.loglog([np.min(rho) / 100, rho_c], [eta_c, eta_c], ':r')
    plt.loglog([rho_c, rho_c], [np.min(eta) / 100, eta_c], ':r')
    plt.title(f'L-curve corner at {reg_c**2}')
    # plt.axis(ax)  # Restore the axis
    plt.axis([ax.get_xlim()[0], ax.get_xlim()[1], ax.get_ylim()[0], ax.get_ylim()[1]])

    plt.show()

    return reg_c, rho, eta, reg_param



