#%%

## Test for lotkavolterra: denoise and learning dynamcis
# import os
# import sys
# path1 = os.path.abspath('..')
# sys.path.append(path1+'/src/')

import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import ode_examp
from gen_data import gen_observ
from denoise import denoise_vrkhs
import jax.numpy as jnp
from jax import vmap

# Integrator keywords for solve_ivp
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12

examp_type = 'lotkavolterra'
paras = [0.7, 0.007, 1, 0.007]
x0 = [70, 50]
time_interval = [0, 20]
t0, tf = time_interval
# pts_type = 'uniform'
pts_type = 'random'
pts_num  = 2000
nsr = 2e-1
ns_type = 2

# vector field
def f_vf(x1, x2, para=[0.7,0.007,1.0,0.007]):
   a, b, c, d = para
   return np.array([a*x1 - b*x1*x2, -c*x2 + d*x1*x2])

# generata data
X_ns, X_data, T, Dx, sol = gen_observ(examp_type, paras, x0, time_interval, pts_type, pts_num, nsr, ns_type)
T1 = T[1:]

# fitting derivative and trajectory
kernel_type='gauss'
# X_dot, X_fit, lamb1 = denoise_vrkhs(T, X_ns, 1e-2, 'pre_select', kernel_type, (0.2,))
X_dot, X_fit, lamb1 = denoise_vrkhs(T, X_ns, None, 'auto', kernel_type, (0.3,))


# ------------- plot ------------------------
plt.rcParams['text.usetex'] = True

# true trajectory and derivative
fig = plt.figure(figsize = (15, 6))
plt.subplot(1, 2, 1)
plt.plot(T, X_data[0], '-r', label='prey')
plt.plot(T, X_data[1], '-g', label='predator')
plt.legend()
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$x_i$', fontsize=20)
# plt.savefig('predprey.pdf')
# plt.savefig('predprey.png')
plt.tight_layout()
plt.subplot(1, 2, 2)
plt.plot(T, Dx[0], '-r', label='prey')
plt.plot(T, Dx[1], '-g', label='predator')
plt.legend()
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$\dot{x}_i$', fontsize=20)
plt.suptitle('Lotka-Volterra system, trajectory and derivative')
plt.tight_layout()


# plot noise observation and fitted curves
fig = plt.figure(figsize = (15, 6))
plt.subplot(1, 2, 1)
plt.scatter(T, X_ns[0], c='b', s = 3, label='noisy x1')
plt.scatter(T, X_ns[1], c='m', s = 3, label='noisy x2')
plt.plot(T1, X_fit[0], '-r', label='fitted x1')
plt.plot(T1, X_fit[1], '-g', label='fitted x2')
plt.legend()
plt.subplot(1, 2, 2)
plt.scatter(X_ns[0], X_ns[1], c='b', s = 5, label='noisy observation')
plt.plot(X_fit[0], X_fit[1], '-r', label='fitted')
plt.xlabel('prey')
plt.ylabel('predator')
plt.legend()
plt.suptitle('Noisy observation and fitted curve')
plt.tight_layout()

# plot true and fitted curves and derivative functions
fig = plt.figure(figsize = (15, 6))
plt.subplot(1, 2, 1)
plt.plot(T, Dx[0], '-r', label='true derivative x1')
plt.plot(T, Dx[1], '-g', label='true derivative x2')
plt.plot(T1, X_dot[0], '--b', label='fitted derivative x1')
plt.plot(T1, X_dot[1], '--m', label='fitted derivative x2')
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$\dot{x}_i$', fontsize=20)
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(X_data[0], X_data[1], '-b', label='true')
plt.plot(X_fit[0], X_fit[1], '--g', label='fitted')
plt.xlabel('prey')
plt.ylabel('predator')
plt.legend()
plt.suptitle('Derivative, true and RKHS fitting')
plt.tight_layout()


#------------------------------------------------------------------
# learning dynamcis using {X_dot, X_fit}
from recons_dynam import fit_coef
from recons_dynam import vectfd
kernel2='gauss'
sig2 = 1000

V = fit_coef(X_dot, X_fit, None, 'auto','gauss', sig2)
f_recons = lambda x, y: vectfd([x,y], X_fit, V, kernel2, sig2)
# define the reconstructed ODE model
fitODE = lambda t, x: f_recons(x[0],x[1])

# ---------plot the contour error of the vector field----------------
N_pts = 50  # grid points, in each dimension
xx = np.linspace(50, 300, N_pts)
yy = np.linspace(50, 300, N_pts)
XX, YY = np.meshgrid(xx, yy)
XY_test = np.concatenate((XX.reshape(-1,1),YY.reshape(-1,1)), axis=1) # test data points

truth_val = map(f_vf, XY_test[:,0].tolist(), XY_test[:,1].tolist())
truth_val = np.array(list(truth_val)) 
fit_val = map(f_recons, XY_test[:,0].tolist(), XY_test[:,1].tolist())
fit_val = np.array(list(fit_val)) 

err = np.abs(fit_val-truth_val)
rel_er = np.sqrt(np.sum(err**2))/np.sqrt(np.sum(truth_val**2))
print(f'Relative L2 error of the reconstructed vector field is {rel_er}')


# plot contour of relative errors
import matplotlib.ticker as ticker
fmt = ticker.ScalarFormatter(useMathText=True)
fmt.set_powerlimits((0, 0))

fig = plt.figure(figsize = (15, 6))
# plt.subplot(1, 2, 1)
ax1 = fig.add_subplot(121)
err_contourf = ax1.contourf(XX, YY, err[:,0].reshape(XX.shape)/np.abs(truth_val[:,0]).reshape(XX.shape), 50, cmap=plt.cm.coolwarm)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Contour of relative errors')
fig.colorbar(err_contourf, format=fmt)
# plt.subplot(1, 2, 2)
ax2 = fig.add_subplot(122)
err_contourf = ax2.contourf(XX, YY, err[:,1].reshape(XX.shape)/np.abs(truth_val[:,1]).reshape(XX.shape), 50, cmap=plt.cm.coolwarm)
# self.XX = XX
# self.YY = YY
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Contour of relative errors')
fig.colorbar(err_contourf, format=fmt)



# make predictions and comparison
sol1 = solve_ivp(ode_examp.lotkavolterra, [t0, tf*3], x0, args=(paras,), dense_output=True, **integrator_keywords)
sol2 = solve_ivp(fitODE, [t0, tf*3], x0, dense_output=True, **integrator_keywords)    # compute a continuous solution
    
tt = np.linspace(t0, tf*3, int(pts_num*3))
z1 = sol1.sol(tt)    
z2 = sol2.sol(tt) 

fig = plt.figure(figsize = (8, 7))
plt.rc('legend', fontsize=15)
plt.plot(tt, z1[0], '-r', label='true $x_1$')
plt.plot(tt, z1[1], '-g', label='true $x_2$')
plt.plot(tt, z2[0], '--b', label='predicted $x_1$')
plt.plot(tt, z2[1], '--m', label='predicted $x_2$')
plt.axvline(tf,  color='black', ls='--')
plt.legend()
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$x_i$', fontsize=20)

fig = plt.figure(figsize = (8, 7))
plt.rc('legend', fontsize=15)
plt.plot(z1[0], z1[1], '-b', label='true')
plt.plot(z2[0], z2[1], ':r', label='predicted')
plt.xlabel('prey')
plt.ylabel('predator')
plt.legend()
plt.suptitle('Lotka-Volterra, true and prediction')
plt.tight_layout()
plt.show()


# %%
