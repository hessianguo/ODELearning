# %%

import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Integrator keywords for solve_ivp
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12

# define the ODE system
def lotkavolterra(t, x, a=0.7, b=0.007, c=1, d=0.007):
    """
    Parameters
    x : array-like, shape (2,)
       (x1, x2) are the quantities of prey-predator
    a = 0.7      # prey birth rate 
    b = 0.007    # prey-predator-collision rate
    c = 1        # predator death rate
    d = 0.007    # prey-predator-collision rate
    Returns
    xyz_dot : array, shape (2,)
       Values of the derivatives at *x*.
    """
    x1, x2 = x
    return [a*x1 - b*x1*x2, -c*x2 + d*x1*x2]

# set initial value and time interval
a1 = 0.7      # prey birth rate 
a2 = 0.007    # prey-predator-collision rate
a3 = 1        # predator death rate
a4 = 0.007
x0 = [70, 50]           # initial value y0=y(t0), a list
t0 = 0                  # integration limits for t: start at t0=0
tf = 20                 # and finish at tf=10

sol = solve_ivp(lotkavolterra, [t0, tf], x0, args=(a1, a2, a3, a4), dense_output=True, **integrator_keywords)    # compute a continuous solution

# sample points for plotting continuous curves
t = np.linspace(t0, tf, 1000)
z = sol.sol(t)    
x1 = z[0]
x2 = z[1]

# get values of the derivative function at the given time points: \dot{X}=f(X(t))
func = lambda x, y: lotkavolterra(t, np.array([x,y]), a1, a2, a3, a4)
D1 = map(func, x1.tolist(), x2.tolist())
D1 = np.array(list(D1))    # nxd array
DX = D1.T


# get n=200 observations at the given time points, add noise 
n1 = 4000
T1 = np.linspace(t0, tf, n1+1)
T1 = T1[1:]
# 1000 points as true observatoins by random sampling
import random
T_samp = random.sample(T1.tolist(), 2000)  
T_samp.sort()
T = np.array(T_samp)
Z = sol.sol(T)
nsr = 1e-1    # noise level
# nsr = 0
from noise import add_noise
X_ns = add_noise(Z, nsr, type="white_gauss")

# fitting derivative function using vRKHS
import deriv
kernel = 'gauss'
sigma = 0.2
lamb = 15e-1
d, n = Z.shape
XX0 = np.kron(x0, np.ones(n))
XX0 = XX0.reshape((d,n))
# B  = Z - XX0
B = X_ns - XX0

G1 = deriv.gram_int(T, kernel, sigma)
V  = deriv.fit_coef(B, G1, lamb)
Phi = deriv.da_basis(T, T, kernel, sigma)
X_dot = deriv.deriv_val(Phi, V)

# compute the denoised observation
Phi_X  = deriv.gram_traj(T, T, kernel, sigma)
X_dn = deriv.deriv_val(Phi_X,  V) + np.kron(x0,np.ones(len(T))).reshape((d,-1))   # denoised X

plt.rcParams['text.usetex'] = True

fig = plt.figure(figsize = (20, 6))
plt.subplot(1, 3, 1)
# plt.scatter(T, X_ns[0], c='purple', s = 3, label='noisy x1')
# plt.scatter(T, X_ns[1], c='orange', s = 3, label='noisy x2')
plt.plot(t, x1, '-r', label='true x1')
plt.plot(t, x2, '-g', label='true x2')
plt.plot(T, X_dn[0], '--b', label='fitted x1')
plt.plot(T, X_dn[1], '--m', label='fitted x2')
plt.legend()
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$x_i$', fontsize=20)
plt.subplot(1, 3, 2)
plt.plot(t, DX[0], '-r', label='true derivative x1')
plt.plot(t, DX[1], '-g', label='true derivative x2')
plt.plot(T, X_dot[0], '--b', label='fitted derivative x1')
plt.plot(T, X_dot[1], '--m', label='fitted derivative x2')
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$\dot{x}_i$', fontsize=20)
plt.legend()
plt.subplot(1, 3, 3)
plt.scatter(X_ns[0], X_ns[1], c='purple', s = 3, label='noisy')
plt.plot(x1, x2, '-b', label='true')
plt.plot(X_dn[0], X_dn[1], '--r', label='fitted')
plt.xlabel('prey')
plt.ylabel('predator')
plt.legend()
plt.suptitle('Lotka-Volterra, true and RKHS fitting')
plt.tight_layout()


# reconstruct vector field using fitted derivative and denoised observation
import recons_dynam
kernel2 = 'gauss'
sigma2 = 800
lamb2 = 1e-5
G2 = recons_dynam.gram(X_dn, kernel2, sigma2)
V2 = recons_dynam.fit_coef(X_dot, G2, lamb2)

f_vecfd = lambda x: recons_dynam.vecfield(x, X_dn, V2, kernel2, sigma2)    # recovered vector field

# define the reconstructed ODE model
fitODE = lambda t, x: f_vecfd(x)


# ---------plot the vector field----------------
# Creating a mesh grid for vector field
x, y = np.meshgrid(np.linspace(0, 300, 20), np.linspace(0, 300, 20))
# Directional vectors
u = a1*x - a2*x*y
v = -a3*x + a4*x*y

def f_vf(x, y):
   xy = np.array([x,y])
   return f_vecfd(xy)

XX = x.reshape(-1)
YY = y.reshape(-1)
UV1 = map(f_vf, XX.tolist(), YY.tolist())
UV = np.array(list(UV1))
U = UV[:,0].reshape(20,20)
V = UV[:,1].reshape(20,20)

# Plotting Vector Field with quiver() function
fig = plt.figure(figsize = (8, 6))
plt.quiver(x, y, u, v, color='b')
plt.quiver(x, y, U, V, color='r', linestyle='--')
plt.title('Vector Field')
# Setting boundary limits
plt.xlim(0, 310)
plt.ylim(0, 310)
# Show plot with grid
plt.grid()



# prediction by solving the reconstructed ODE model
sol2 = solve_ivp(fitODE, [t0, tf*1.5], x0, dense_output=True, **integrator_keywords)    # compute a continuous solution
sol1 = solve_ivp(lotkavolterra, [t0, tf*1.5], x0, args=(a1, a2, a3, a4), dense_output=True, **integrator_keywords)    

tt = np.linspace(t0, tf*1.5, int(1000*1.5))
z1 = sol1.sol(tt)    
z2 = sol2.sol(tt) 


# ------------- plot ------------------------


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
plt.plot(z2[0], z2[1], ':r', label='predicted')
plt.xlabel('prey')
plt.ylabel('predator')
plt.legend()
plt.suptitle('Lotka-Volterra, true and prediction')
plt.tight_layout()

plt.show()

# %%
