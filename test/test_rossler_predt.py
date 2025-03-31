#%%

## Test for Rossler: denoise and learning dynamcis
import os
import sys
path1 = os.path.abspath('..')
sys.path.append(path1+'/src/')

import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import ode_examp
from gen_data import gen_observ
from denoise import denoise_vrkhs

# Integrator keywords for solve_ivp
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12


examp_type = 'rossler'
paras=[0.2,0.2,5.7]
x0 = [1, 1, 1]
time_interval = [0, 10]
t0, tf = time_interval
# pts_type = 'uniform'
pts_type = 'random'
pts_num  = 2000
nsr = 1e-2
ns_type = 2

# vector field
def f_vf(x, y, z, para=[0.2,0.2,5.7]):
    a, b, c = para
    x_dot = -y - z
    y_dot = x + a*y
    z_dot = b + z*(x-c)
    return np.array([x_dot, y_dot, z_dot])

# generata data
X_ns, X_data, T, Dx, sol = gen_observ(examp_type, paras, x0, time_interval, pts_type, pts_num, nsr, ns_type)
T1 = T[1:]

# fitting derivative and trajectory
kernel_type='gauss'
X_dot, X_fit, lamb1 = denoise_vrkhs(T, X_ns, None, 'auto', kernel_type, (0.4,))


# ------------- plot ------------------------
plt.rcParams['text.usetex'] = True

# true trajectory
fig = plt.figure(figsize = (8, 7))
plt.rc('legend', fontsize=14)
ax = plt.axes(projection='3d')
ax.plot(sol.y[0], sol.y[1], sol.y[2], c='r', lw=0.8, label='trajectory')
ax.scatter3D(X_ns[0], X_ns[1], X_ns[2], c='black', s = 1, label='noisy data')
ax.set_xlabel('$x_1$', fontsize=20)
ax.set_ylabel('$x_2$', fontsize=20)
ax.set_zlabel('$x_3$', fontsize=20)
plt.legend()
# ax.set_title("Rossler system", fontsize=20)
plt.tight_layout()


# plot true and fitted curves and derivative functions
fig = plt.figure(figsize = (20, 6))
plt.subplot(1, 3, 1)
plt.plot(T, X_data[0], '-r', label='true $x_1$')
plt.plot(T1, X_fit[0], '--c', label='fitted $x_1$')
plt.legend()
plt.subplot(1, 3, 2)
plt.plot(T, X_data[1], '-g', label='true $x_2$')
plt.plot(T1, X_fit[1], '--m', label='fitted $x_2$')
plt.legend()
plt.subplot(1, 3, 3)
plt.plot(T, X_data[2], '-b', label='true $x_3$')
plt.plot(T1, X_fit[2], '--g', label='fitted $x_3$')
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$x_i$', fontsize=20)
plt.legend()
plt.tight_layout()


fig = plt.figure(figsize = (20, 6))
plt.subplot(1, 3, 1)
plt.plot(T, Dx[0], '-r', label='true $\dot{x}_1$')
plt.plot(T1, X_dot[0], '--c', label='fitted $\dot{x}_1$')
plt.legend()
plt.subplot(1, 3, 2)
plt.plot(T, Dx[1], '-g', label='true $\dot{x}_2$')
plt.plot(T1, X_dot[1], '--m', label='fitted $\dot{x}_2$')
plt.legend()
plt.subplot(1, 3, 3)
plt.plot(T, Dx[2], '-b', label='true $\dot{x}_3$')
plt.plot(T1, X_dot[2], '--g', label='fitted $\dot{x}_3$')
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$\dot{x}_i$', fontsize=20)
plt.legend()
plt.tight_layout()


# ------------------------------------------------------------------
# learning dynamcis using {X_dot, X_fit}
from recons_dynam import fit_coef
from recons_dynam import vectfd
kernel2='gauss'
sig2 = 200  # sigma=500 is the best

V = fit_coef(X_dot, X_fit, None, 'auto','gauss', sig2)
f_recons = lambda x, y, z: vectfd([x,y,z], X_fit, V, kernel2, sig2)
# define the reconstructed ODE model
fitODE = lambda t, x: f_recons(x[0],x[1], x[2])

# ---------plot the contour error of the vector field----------------
N_pts = 50  # grid points, in each dimension
xx = np.linspace(-20, 20, N_pts)
yy = np.linspace(-20, 20, N_pts)
zz = np.linspace(0, 40, N_pts)
XX, YY, ZZ = np.meshgrid(xx, yy, zz)
XYZ_test = np.concatenate((XX.reshape(-1,1),YY.reshape(-1,1),ZZ.reshape(-1,1)), axis=1) # test data points

truth_val = map(f_vf, XYZ_test[:,0].tolist(), XYZ_test[:,1].tolist(), XYZ_test[:,2].tolist())
truth_val = np.array(list(truth_val)) 
fit_val = map(f_recons, XYZ_test[:,0].tolist(), XYZ_test[:,1].tolist(), XYZ_test[:,2].tolist())
fit_val = np.array(list(fit_val)) 

err = np.abs(fit_val-truth_val)
rel_er = np.sqrt(np.sum(err**2))/np.sqrt(np.sum(truth_val**2))
print(f'Relative L2 error of the reconstructed vector field is {rel_er}')


# make predictions and comparison
sol1 = solve_ivp(ode_examp.rossler, [t0, tf*2], x0, args=(paras,), dense_output=True, **integrator_keywords)
sol2 = solve_ivp(fitODE, [t0, tf*2], x0, dense_output=True, **integrator_keywords)    # compute a continuous solution
    
tt = np.linspace(t0, tf*2, int(pts_num*2))
z1 = sol1.sol(tt)    
z2 = sol2.sol(tt) 

fig = plt.figure(figsize = (20, 6))
plt.subplot(1, 3, 1)
plt.plot(tt, z1[0], '-r', label='true $x_1$')
plt.plot(tt, z2[0], '--c', label='predicted $x_1$')
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$x_1$', fontsize=20)
plt.axvline(tf,  color='black', ls='--')
plt.legend(loc='upper left')
plt.subplot(1, 3, 2)
plt.plot(tt, z1[1], '-g', label='true $x_2$')
plt.plot(tt, z2[1], '--m', label='predicted $x_2$')
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$x_2$', fontsize=20)
plt.axvline(tf,  color='black', ls='--')
plt.legend(loc='upper left')
plt.subplot(1, 3, 3)
plt.plot(tt, z1[2], '-b', label='true $x_3$')
plt.plot(tt, z2[2], '--g', label='predicted $x_3$')
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$x_3$', fontsize=20)
plt.axvline(tf,  color='black', ls='--')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


# %%
