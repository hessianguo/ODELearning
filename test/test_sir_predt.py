#%%

## Test for Lorenz63: denoise and learning dynamcis
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


examp_type = 'sir'
paras = [0.4, 0.04]
x0 = [900, 10, 0]
time_interval = [0, 30]
t0, tf = time_interval
# pts_type = 'uniform'
pts_type = 'random'
pts_num  = 3000
nsr = 50e-1
ns_type = 2


# vector field
def f_vf(x, y,z, para=[0.4, 0.04]):
   beta, gamma = para
   x_dot = -beta*x*y / (x+y+z)
   y_dot =  beta*x*y / (x+y+z) - gamma*y
   z_dot = gamma*y
   return np.array([x_dot, y_dot, z_dot])

# generata data
X_ns, X_data, T, Dx, sol = gen_observ(examp_type, paras, x0, time_interval, pts_type, pts_num, nsr, ns_type)
T1 = T[1:]

# fitting derivative and trajectory
kernel_type='gauss'
X_dot, X_fit, lamb1 = denoise_vrkhs(T, X_ns, None, 'auto', kernel_type, (5,))


# ------------- plot ------------------------
plt.rcParams['text.usetex'] = True

# plot true and fitted curves and derivative functions
fig = plt.figure(figsize = (20, 6))
plt.subplot(1, 3, 1)
plt.plot(T, X_ns[0], '-r', label='noise $S$')
plt.plot(T1, X_fit[0], '--c', label='fitted $S$')
plt.legend()
plt.subplot(1, 3, 2)
plt.plot(T, X_ns[1], '-g', label='noise $I$')
plt.plot(T1, X_fit[1], '--m', label='fitted $I$')
plt.legend()
plt.subplot(1, 3, 3)
plt.plot(T, X_ns[2], '-b', label='noise $R$')
plt.plot(T1, X_fit[2], '--g', label='fitted $R$')
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$x_i$', fontsize=20)
plt.legend()
plt.tight_layout()


fig = plt.figure(figsize = (20, 6))
plt.subplot(1, 3, 1)
plt.plot(T, Dx[0], '-r', label='true $\dot{S}$')
plt.plot(T1, X_dot[0], '--c', label='fitted $\dot{S}$')
plt.legend()
plt.subplot(1, 3, 2)
plt.plot(T, Dx[1], '-g', label='true $\dot{I}$')
plt.plot(T1, X_dot[1], '--m', label='fitted $\dot{I}$')
plt.legend()
plt.subplot(1, 3, 3)
plt.plot(T, Dx[2], '-b', label='true $\dot{R}$')
plt.plot(T1, X_dot[2], '--g', label='fitted $\dot{R}$')
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$\dot{x}_i$', fontsize=20)
plt.legend()
plt.tight_layout()


# ------------------------------------------------------------------
# learning dynamcis using {X_dot, X_fit}
from recons_dynam import fit_coef
from recons_dynam import vectfd
kernel2='gauss'
sig2 = 1000

V = fit_coef(X_dot, X_fit, None, 'auto','gauss', sig2)
f_recons = lambda x, y, z: vectfd([x,y,z], X_fit, V, kernel2, sig2)
# define the reconstructed ODE model
fitODE = lambda t, x: f_recons(x[0],x[1], x[2])

# # ---------plot the contour error of the vector field----------------
# N_pts = 50  # grid points, in each dimension
# xx = np.linspace(0, 900, N_pts)
# yy = np.linspace(10, 600, N_pts)
# zz = np.linspace(0, 600, N_pts)
# XX, YY, ZZ = np.meshgrid(xx, yy, zz)
# XYZ_test = np.concatenate((XX.reshape(-1,1),YY.reshape(-1,1),ZZ.reshape(-1,1)), axis=1) # test data points

# truth_val = map(f_vf, XYZ_test[:,0].tolist(), XYZ_test[:,1].tolist(), XYZ_test[:,2].tolist())
# truth_val = np.array(list(truth_val)) 
# fit_val = map(f_recons, XYZ_test[:,0].tolist(), XYZ_test[:,1].tolist(), XYZ_test[:,2].tolist())
# fit_val = np.array(list(fit_val)) 

# err = np.abs(fit_val-truth_val)
# rel_er = np.sqrt(np.sum(err**2))/np.sqrt(np.sum(truth_val**2))
# print(f'Relative L2 error of the reconstructed vector field is {rel_er}')


# make predictions and comparison
sol1 = solve_ivp(ode_examp.sir, [t0, tf*2], x0, args=(paras,), dense_output=True, **integrator_keywords)
sol2 = solve_ivp(fitODE, [t0, tf*2], x0, dense_output=True, **integrator_keywords)    # compute a continuous solution
    
tt = np.linspace(t0, tf*2, int(pts_num*2))
z1 = sol1.sol(tt)    
z2 = sol2.sol(tt) 


fig = plt.figure(figsize = (9, 7))
plt.rc('legend', fontsize=14)
plt.plot(tt, z1[0], '-r', label='true $S$')
plt.plot(tt, z2[0], '--c', label='predicted $S$')
plt.plot(tt, z1[1], '-', c='orange', label='true $I$')
plt.plot(tt, z2[1], '--b', label='predicted $I$')
plt.axvline(tf,  color='black', ls='--')
plt.plot(tt, z1[2], '-m', label='true $R$')
plt.plot(tt, z2[2], '--g', label='predicted $R$')
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$S, I, R$', fontsize=20)
plt.axvline(tf,  color='black', ls='--')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


# %%
