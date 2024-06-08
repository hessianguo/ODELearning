import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sparsedynamics
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

# Set initial value and time interval
x0 = [-8 , 7, 27]          
t0 = 0.001                  
tf = 50             
polyorder = 5

sol = solve_ivp(lorenz63, [t0, tf], x0,  dense_output=True)    # Compute a continuous solution

# Sample points for plotting continuous curves
t = np.linspace(t0, tf, 500)
z = sol.sol(t)    
x1, x2, x3 = z

# Get values of the derivative function at the given time points: \dot{X}=f(X(t))
func = lambda x, y, z: lorenz63(t, np.array([x,y,z]))
D1 = map(func, x1.tolist(), x2.tolist(), x3.tolist())
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
A = sparsedynamics.polynomial_basis(X_ns.T, polyorder)
Xi = sparsedynamics.stsl(A, X_dot.T, 0.001)
print(Xi)


fitODE = lambda t, x: sparsedynamics.sparseode(t, x, Xi, polyorder)

# Prediction by solving the reconstructed ODE model
t0 = 0
tf = 20
sol2 = solve_ivp(fitODE, [t0, tf], x0, dense_output=True)    # Compute a continuous solution
sol1 = solve_ivp(lorenz63, [t0, tf], x0,  dense_output=True)    

tt = np.linspace(t0, tf, int(1000*1.3))
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
#plt.plot(z2[0], z2[1], z2[2], '--r', label='predicted')
plt.xlabel('prey')
plt.ylabel('predator')
plt.legend()
plt.suptitle('Lotka-Volterra, true and prediction')
plt.tight_layout()

plt.show()

