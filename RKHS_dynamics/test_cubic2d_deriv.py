import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sparsedynamics
import deriv


def cubicfun(t, x):
    """
    Compute the derivative of a 2D cubic function.

    Parameters:
    - t: float, the independent variable.
    - x: list, the dependent variables [x1, x2].

    Returns:
    - numpy array, the derivative of the cubic function at the given point.
    """
    x1, x2 = x
    return np.array([-0.1*x1**3 + 2*x2**3, -2*x1**3 - 0.1*x2**3])

# Set initial value and time interval
x0 = [2, 0]           # Initial value y0=y(t0), a list
t0 = 0                  # Integration limits for t: start at t0=0
tf = 25                 # And finish at tf=10

sol = solve_ivp(cubicfun, [t0, tf], x0,  dense_output=True)    # Compute a continuous solution

# Sample points for plotting continuous curves
t = np.linspace(t0, tf, 1000)
z = sol.sol(t)    
x1 = z[0]
x2 = z[1]

# Get values of the derivative function at the given time points: \dot{X}=f(X(t))
func = lambda x, y: cubicfun(t, np.array([x,y]))
D1 = map(func, x1.tolist(), x2.tolist())
D1 = np.array(list(D1))    # nxd array
DX = D1.T

# Get n=200 observations at the given time points, add noise 
n1 = 2000
T1 = np.linspace(t0, tf, n1+1)
T1 = T1[1:]
# 1000 points as true observatoins by random sampling
import random
T_samp = random.sample(T1.tolist(), 1000)  
T_samp.sort()
T = np.array(T_samp)
Z = sol.sol(T)
#nsr = 1e-4    # noise level
nsr = 1e-3
from noise import add_noise
X_ns = add_noise(Z, nsr, type="white_gauss")

# Fitting derivative function using vRKHS
kernel = 'gauss'
sigma = 0.2
lamb = 1e-4
d, n = Z.shape
XX0 = np.kron(x0, np.ones(n))
XX0 = XX0.reshape((d,n))
# B  = Z - XX0
B = X_ns - XX0

G1 = deriv.gram_int(T, kernel, sigma)
V  = deriv.fit_coef(B, G1, lamb)
Phi = deriv.da_basis(T, T, kernel, sigma)
X_dot = deriv.deriv_val(Phi, V)
print(X_dot.shape)
print(X_ns.shape)

DX_num = np.zeros((d,n))
for i in np.arange(n-2):
   DX_num[:,i+1] = (X_ns[:,i+2]-X_ns[:,i]) / (T[i+2]-T[i])

# Reconstruction of the dynamics using SINDy
A = sparsedynamics.polynomial_basis(X_ns.T, 5)
Xi = sparsedynamics.stsl(A, X_dot.T, 0.001)
print(Xi)


fitODE = lambda t, x: sparsedynamics.sparseode(t, x, Xi, 5)

# Prediction by solving the reconstructed ODE model
sol2 = solve_ivp(fitODE, [t0, tf*1.2], x0, dense_output=True)    # Compute a continuous solution
sol1 = solve_ivp(cubicfun, [t0, tf*1.3], x0,  dense_output=True)    

tt = np.linspace(t0, tf*1.3, int(1000*1.3))
z1 = sol1.sol(tt)    
z2 = sol2.sol(tt) 

# Plotting the true and predicted values
fig = plt.figure(figsize = (15, 6))
plt.subplot(1, 2, 1)
plt.plot(tt, z1[0], '-r', label='true x1')
plt.plot(tt, z1[1], '-g', label='true x2')
plt.plot(tt, z2[0], '--b', label='predicted x1')
plt.plot(tt, z2[1], '--m', label='predicted x2')
plt.legend()
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$x_i$', fontsize=20)
plt.subplot(1, 2, 2)
plt.plot(z1[0], z1[1], '-b', label='true')
plt.plot(z2[0], z2[1], '--r', label='predicted')
plt.xlabel('prey')
plt.ylabel('predator')
plt.legend()
plt.suptitle('Lotka-Volterra, true and prediction')
plt.tight_layout()
plt.show()
#plt.savefig('LV_pred.png')
fig = plt.figure(figsize = (15, 6))
plt.subplot(1, 2, 1)
plt.plot(tt, z1[0], '-r', label='true x1')
plt.plot(tt, z1[1], '-g', label='true x2')
plt.plot(tt, z2[0], '--b', label='predicted x1')
plt.plot(tt, z2[1], '--m', label='predicted x2')
plt.legend()
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$x_i$', fontsize=20)
plt.subplot(1, 2, 2)
plt.plot(z1[0], z1[1], '-b', label='true')
plt.plot(z2[0], z2[1], '--r', label='predicted')
plt.xlabel('prey')
plt.ylabel('predator')
plt.legend()
plt.suptitle('Lotka-Volterra, true and prediction')
plt.tight_layout()

plt.show()
#plt.savefig('LV_pred.png')

