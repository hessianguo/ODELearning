import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sparsedynamics
from noise import add_noise
import deriv

def linearfun(t, x):
    """
    Compute the derivative of the linear function.

    Parameters:
    - t: float, the independent variable.
    - x: array-like, the state variables.

    Returns:
    - array-like, the derivative of the cubic function.
    """
    x1, x2, x3 = x
    return np.array([-0.1*x1+2*x2, -2*x1-0.1*x2+2*x3, -0.3*x3])

# Initial conditions and time range
x0 = [2, 0, 1]          
t0 = 0                  
tf = 50                
t = np.linspace(t0, tf, 501)

# Parameters for sparse dynamics
polyorder = 2
nsr = 1e-3

# Solve the ODE using solve_ivp and add noise to the solution
sol = solve_ivp(linearfun, [t0, tf], x0, method='RK45', t_eval=t)    # compute a continuous solution
X_ns = add_noise(sol.y, nsr, type="white_gauss")

# Extract the state variables from the solution
x1, x2, x3 = sol.y

# Compute the derivative of the noisy solution at t=0
dxdt = linearfun(0, X_ns)

# Compute the polynomial basis and sparse coefficients
A = sparsedynamics.polynomial_basis(X_ns.T, polyorder)
Xi = sparsedynamics.stsl(A, dxdt.T,  1e-2)
print(Xi)

# Define the fitted ODE function using the sparse coefficients
fitODE = lambda t, x: sparsedynamics.sparseode(t, x, Xi, polyorder)

# Solve the fitted ODE and the original ODE for comparison
sol2 = solve_ivp(fitODE, [t0, tf*1.2], x0, dense_output=True)    # compute a continuous solution
sol1 = solve_ivp(linearfun, [t0, tf*1.3], x0,  dense_output=True)    

# Generate time points for plotting
tt = np.linspace(t0, tf*1.3, int(1000*1.3))

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

