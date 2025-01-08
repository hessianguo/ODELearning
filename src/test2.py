#%%

## Test for lorenz63

import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from gen_data import gen_observ
from denoise import denoise_vrkhs

examp_type = 'lorenz63'
paras = [10, 28, 8/3]
x0 = [1, 1, 1]
time_interval = [0, 20]
pts_type = 'uniform'
pts_num  = 2000
nsr = 5e-2
ns_type = 'white_gauss'


# generata data
X_ns, X_data, T, Dx, sol = gen_observ(examp_type, paras, x0, time_interval, pts_type, pts_num, nsr, ns_type)
T1 = T[1:]

# fitting derivative and trajectory
kernel_type='gauss'
X_dot, X_fit, lamb1 = denoise_vrkhs(T, X_ns, 1e-3, 'pre_select', kernel_type, (0.02,))
# X_dot, X_fit, lamb1 = denoise_vrkhs(T, X_ns, None, 'auto', kernel_type, (0.02,))



# ------------- plot ------------------------
plt.rcParams['text.usetex'] = True

# true trajectory and derivative
fig = plt.figure(figsize = (15, 6))
plt.subplot(1, 2, 1)
plt.plot(T, X_data[0], '-r', label='x1')
plt.plot(T, X_data[1], '-g', label='x2')
plt.plot(T, X_data[2], '-b', label='x3')
plt.legend()
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$x_i$', fontsize=20)
# plt.savefig('predprey.pdf')
# plt.savefig('predprey.png')
plt.tight_layout()
plt.subplot(1, 2, 2)
plt.plot(T, Dx[0], '-r', label='$\dot{x}_1$')
plt.plot(T, Dx[1], '-g', label='$\dot{x}_2$')
plt.plot(T, Dx[2], '-b', label='$\dot{x}_3$')
plt.legend()
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$\dot{x}_i$', fontsize=20)
plt.suptitle('Lorenz 63 system, trajectory and derivative')
plt.tight_layout()


# plot noise observation and fitted curves
fig = plt.figure(figsize = (10,10))
ax = plt.axes(projection='3d')
ax.grid()
ax.scatter3D(X_ns[0], X_ns[1], X_ns[2], c='purple', s = 2)
ax.plot(X_data[0], X_data[1], X_data[2], 'g', lw=0.8)
ax.set_xlabel('$x_1$', fontsize=20)
ax.set_ylabel('$x_2$', fontsize=20)
ax.set_zlabel('$x_3$', fontsize=20)
ax.set_title("Lorenz63, true and noisy data", fontsize=20)
plt.tight_layout()

# plot true and fitted derivative functions
fig = plt.figure(figsize = (24,6))
plt.subplot(1, 3, 1)
plt.plot(T, Dx[0], '-r', label='true derivative x1')
plt.plot(T1, X_dot[0], '--b', label='fitted derivative x1')
plt.legend()
plt.subplot(1, 3, 2)
plt.plot(T, Dx[1], '-g', label='true derivative x2')
plt.plot(T1, X_dot[1], '--m', label='fitted derivative x2')
plt.legend()
plt.subplot(1, 3, 3)
plt.plot(T, Dx[2], '-m', label='true derivative x3')
plt.plot(T1, X_dot[2], color='deepskyblue', linestyle='--', label='fitted derivative x3')
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$\dot{x}_i$', fontsize=20)
plt.legend()
plt.title('Derivative, true and RKHS fitting')
plt.tight_layout()

# plot true and fitted curves 
fig = plt.figure(figsize = (24,6))
plt.subplot(1, 3, 1)
plt.plot(T, X_data[0], '-r', label='true x1')
plt.plot(T1, X_fit[0], '--b', label='fitted x1')
plt.legend()
plt.subplot(1, 3, 2)
plt.plot(T,  X_data[1], '-g', label='true x2')
plt.plot(T1, X_fit[1], '--m', label='fitted x2')
plt.legend()
plt.subplot(1, 3, 3)
plt.plot(T, X_data[2], '-m', label='true x3')
plt.plot(T1, X_fit[2], color='deepskyblue', linestyle='--', label='fitted x3')
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$x_i$', fontsize=20)
plt.legend()
plt.title('Trajectory, true and RKHS fitting')
plt.tight_layout()

fig = plt.figure(figsize = (10,10))
ax = plt.axes(projection='3d')
ax.grid()
ax.scatter3D(X_ns[0], X_ns[1], X_ns[2], color='purple', s = 2)
ax.plot(X_fit[0], X_fit[1], X_fit[2], color='orange', lw=0.8)
ax.set_xlabel('$x_1$', fontsize=20)
ax.set_ylabel('$x_2$', fontsize=20)
ax.set_zlabel('$x_3$', fontsize=20)
ax.set_title("Lorenz63, noisy data and fitted traj", fontsize=20)
plt.tight_layout()

# %%
