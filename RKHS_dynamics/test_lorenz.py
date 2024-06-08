import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sparsedynamics
from noise import add_noise
import deriv

def lorenz63(t, xyz, s=10, r=28, b=8/3):
    """
    Parameters
    ----------
    xyz : array-like, shape (3,)
       Point of interest in three-dimensional space.
    s, r, b : float
       Parameters defining the Lorenz attractor.

    Returns
    -------
    xyz_dot : array, shape (3,)
       Values of the Lorenz attractor's partial derivatives at *xyz*.
    """
    x, y, z = xyz
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return np.array([x_dot, y_dot, z_dot])

# Initial conditions and time range
x0 = [-8 , 7, 27]          
t0 = 0.001                  
tf = 100                
t = np.linspace(t0, tf, 100000)

# Parameters for sparse dynamics
polyorder = 5
nsr = 1

# Solve the ODE using solve_ivp and add noise to the solution
sol = solve_ivp(lorenz63, [t0, tf], x0, method='RK45', t_eval=t)    # compute a continuous solution
X_ns = add_noise(sol.y, nsr, type="white_gauss")

# Extract the state variables from the solution
x1, x2, x3 = sol.y

# Compute the derivative of the noisy solution at t=0
dxdt = lorenz63(0, X_ns)

# Compute the polynomial basis and sparse coefficients
A = sparsedynamics.polynomial_basis(X_ns.T, polyorder)
Xi = sparsedynamics.stsl(A, dxdt.T,  1e-2)
print(Xi)

# Define the fitted ODE function using the sparse coefficients
fitODE = lambda t, x: sparsedynamics.sparseode(t, x, Xi, polyorder)

# Solve the fitted ODE and the original ODE for comparison
t0 = 0
tf = 20
sol2 = solve_ivp(fitODE, [t0, tf], x0, dense_output=True)    # compute a continuous solution
sol1 = solve_ivp(lorenz63, [t0, tf], x0,  dense_output=True)    

# Generate time points for plotting
tt = np.linspace(t0, tf, int(1000*1.3))

# Evaluate the true and predicted solutions at the time points
z1 = sol1.sol(tt)    
z2 = sol2.sol(tt) 

# Plot the results
fig = plt.figure(figsize = (15, 6))
plt.subplot(1, 2, 1)
plt.plot(tt, z1[0], '-r', label='true x1')
plt.plot(tt, z1[1], '-g', label='true x2')
plt.plot(tt, z1[2], '-g', label='true x2')
plt.plot(tt, z2[0], '--b', label='predicted x1')
plt.plot(tt, z2[1], '--m', label='predicted x2')
plt.plot(tt, z2[2], '--m', label='predicted x2')
plt.legend()
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$x_i$', fontsize=20)
plt.subplot(1, 2, 2, projection='3d')
plt.plot(z1[0], z1[1], z1[2], '-b', label='true')
plt.plot(z2[0], z2[1], z1[2], '--r', label='predicted')
plt.xlabel('prey')
plt.ylabel('predator')
plt.legend()
plt.suptitle('Lotka-Volterra, true and prediction')
plt.tight_layout()

plt.show()

