#%%

## Test for lotkavolterra

import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from gen_data import gen_observ
from denoise import denoise_vrkhs

examp_type = 'lotkavolterra'
paras = [0.7, 0.007, 1, 0.007]
x0 = [70, 50]
time_interval = [0, 20]
# pts_type = 'uniform'
pts_type = 'random'
pts_num  = 1000
nsr = 5e-1
ns_type = 2


# generata data
X_ns, X_data, T, Dx, sol = gen_observ(examp_type, paras, x0, time_interval, pts_type, pts_num, nsr, ns_type)
T1 = T[1:]

# fitting derivative and trajectory
kernel_type='gauss'
# X_dot, X_fit, lamb1 = denoise_vrkhs(T, X_ns, 1e-2, 'pre_select', kernel_type, (0.2,))
X_dot, X_fit, lamb1 = denoise_vrkhs(T, X_ns, None, 'auto', kernel_type, (0.1,))



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


# %%
