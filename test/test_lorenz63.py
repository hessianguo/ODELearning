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


x0 = [1, 1, 1]     # initial value y0=y(t0)
t0 = 0             # integration limits for t: start at t0=0
tf = 20            # and finish at tf=2
sol = solve_ivp(lorenz63, [t0, tf], x0, args=(10, 28, 8/3), dense_output=True, **integrator_keywords)

# sample points for plotting continuous curves
t = np.linspace(t0, tf, 10000)
z = sol.sol(t)    

# get values of the derivative function at the given time points: \dot{X}=f(X(t))
func = lambda x, y, z: lorenz63(t, np.array([x,y,z]), 10, 28, 8/3)
D1 = map(func, z[0].tolist(), z[1].tolist(), z[2].tolist())
D1 = np.array(list(D1))    # nxd array
DX = D1.T


#----------- add noise ----------------------
n1 = 5000
T1 = np.linspace(t0, tf, n1+1)
T1 = T1[1:]
# 1000 points as true observatoins by random sampling
import random
T_samp = random.sample(T1.tolist(), 2000)  
T_samp.sort()
T = np.array(T_samp)
Z = sol.sol(T)
# nsr = 1e-4    # noise level
nsr = 0
from noise import add_noise
X_ns = add_noise(Z, nsr, type="white_gauss")


# ------------- plot ------------------------
plt.rcParams['text.usetex'] = True

fig = plt.figure(figsize = (10,10))
ax = plt.axes(projection='3d')
ax.grid()
ax.scatter3D(X_ns[0], X_ns[1], X_ns[2], c='purple', s = 2)
ax.plot(z[0], z[1], z[2], 'g', lw=0.8)
ax.set_xlabel('$x_1$', fontsize=20)
ax.set_ylabel('$x_2$', fontsize=20)
ax.set_zlabel('$x_3$', fontsize=20)
ax.set_title("Lorenz63 system", fontsize=20)
plt.tight_layout()

# plot three (t, x_i) curves
fig = plt.figure(figsize = (24, 6))
plt.subplot(1, 3, 1)
plt.scatter(T, X_ns[0], c='black', s = 2, label='noisy x1')
plt.plot(t, z[0], 'orange', lw=1.0)
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$x_1$', fontsize=20)
plt.subplot(1, 3, 2)
plt.scatter(T, X_ns[1], c='black', s = 2, label='noisy x2')
plt.plot(t, z[1], 'm', lw=1.0)
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$x_2$', fontsize=20)
plt.subplot(1, 3, 3)
plt.scatter(T, X_ns[2], c='black', s = 2, label='noisy x3')
plt.plot(t, z[2], lw=1.0)
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$x_3$', fontsize=20)
plt.suptitle('Lorenz63 function curves', fontsize=20)
plt.tight_layout()

# plot three (t, \dot{x}_i) curves
fig = plt.figure(figsize = (24, 6))
plt.subplot(1, 3, 1)
plt.plot(t, DX[0], 'orange', lw=1.0)
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$\dot{x}_1$', fontsize=20)
plt.subplot(1, 3, 2)
plt.plot(t, DX[1], 'm', lw=1.0)
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$\dot{x}_2$', fontsize=20)
plt.subplot(1, 3, 3)
plt.plot(t, DX[2], lw=1.0)
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$\dot{x}_3$', fontsize=20)
plt.suptitle('Lorenz63 derivative functions', fontsize=20)
plt.tight_layout()


# fitting derivative function using vRKHS
import deriv
kernel = 'gauss'
sigma = 0.02
lamb = 1e-12
d, n = Z.shape
XX0 = np.kron(x0, np.ones(n))
XX0 = XX0.reshape((d,n))
# B  = Z - XX0
B = X_ns - XX0    # X(t_i)-X0

G1 = deriv.gram_int(T, kernel, sigma)
V  = deriv.fit_coef(B, G1, lamb)
Phi = deriv.da_basis(T, T, kernel, sigma)
X_dot = deriv.deriv_val(Phi, V)

# compute the denoised observation
Phi_X = deriv.gram_traj(T, T, kernel, sigma)
X_dn  = deriv.deriv_val(Phi_X, V) + np.kron(x0,np.ones(len(T))).reshape((d,-1))   # denoised X

# plot three (t, x_i) curves
fig = plt.figure(figsize = (24, 6))
plt.subplot(1, 3, 1)
plt.plot(t, z[0], 'orange', lw=1.0, label='true')
plt.plot(T, X_dn[0], '--b', label='fitted')
plt.legend()
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$x_1$', fontsize=20)
plt.subplot(1, 3, 2)
plt.plot(t, z[1], 'm', lw=1.0)
plt.plot(T, X_dn[1], '--g', label='fitted x2')
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$x_2$', fontsize=20)
plt.subplot(1, 3, 3)
plt.plot(t, z[2], 'blue', lw=1.0)
plt.plot(T, X_dn[2], '--r', label='fitted x3')
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$x_3$', fontsize=20)
plt.suptitle('Lorenz63 function curves', fontsize=20)
plt.tight_layout()

# plot three (t, \dot{x}_i) curves
fig = plt.figure(figsize = (24, 6))
plt.subplot(1, 3, 1)
plt.plot(t, DX[0], 'orange', lw=1.0, label='true')
plt.plot(T, X_dot[0], '--b', label='fitted')
plt.legend()
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$\dot{x}_1$', fontsize=20)
plt.subplot(1, 3, 2)
plt.plot(t, DX[1], 'm', lw=1.0)
plt.plot(T, X_dot[1], '--g', label='fitted derivative x2')
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$\dot{x}_2$', fontsize=20)
plt.subplot(1, 3, 3)
plt.plot(t, DX[2], 'blue', lw=1.0)
plt.plot(T, X_dot[2], '--r', label='fitted derivative x3')
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$\dot{x}_3$', fontsize=20)
plt.suptitle('Lorenz63 derivative functions', fontsize=20)
plt.tight_layout()

fig = plt.figure(figsize = (10,10))
ax = plt.axes(projection='3d')
ax.plot(X_dn[0], X_dn[1], X_dn[2], '--b', lw=0.8)
ax.plot(z[0], z[1], z[2], 'g', lw=1)
ax.set_xlabel('$x_1$', fontsize=20)
ax.set_ylabel('$x_2$', fontsize=20)
ax.set_zlabel('$x_3$', fontsize=20)
ax.set_title("Lorenz63 system", fontsize=20)
plt.tight_layout()

# %%
