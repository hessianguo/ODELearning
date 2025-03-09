#%%

## plot the trjectory with noisy data


import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from gen_data import gen_observ
from denoise import denoise_vrkhs

plt.rcParams['text.usetex'] = True

#------------------------------------------------------
# examp_type = 'lotkavolterra'
# paras = [0.7, 0.007, 1, 0.007]
# x0 = [70, 50]
# time_interval = [0, 10]
# # pts_type = 'uniform'
# pts_type = 'random'
# pts_num  = 2000
# nsr = 10e-1
# ns_type = 2
# # generata data
# X_ns, X_data, T, Dx, sol = gen_observ(examp_type, paras, x0, time_interval, pts_type, pts_num, nsr, ns_type)
# T1 = T[1:]


# # ------------- plot ------------
# fig = plt.figure(figsize = (8, 7))
# plt.rc('legend', fontsize=14)
# plt.plot(X_data[0], X_data[1], '-m', label='trajectory')
# plt.scatter(X_ns[0], X_ns[1], c='black', s = 1, label='noisy data')
# plt.xlabel('$x_1$', fontsize=25)
# plt.ylabel('$x_2$', fontsize=25)
# plt.legend()
# plt.tight_layout()
# plt.show()


#---------------------------------------------------------
# examp_type = 'lorenz63'
# paras = [10,28,8/3]
# x0 = [1, 1, 1]
# time_interval = [0, 30]
# # pts_type = 'uniform'
# pts_type = 'random'
# pts_num  = 6000
# nsr = 5e-1
# ns_type = 2
# # generata data
# X_ns, X_data, T, Dx, sol = gen_observ(examp_type, paras, x0, time_interval, pts_type, pts_num, nsr, ns_type)
# T1 = T[1:]


# # ------------- plot ------------------------
# fig = plt.figure(figsize = (8, 7))
# plt.rc('legend', fontsize=14)
# ax = plt.axes(projection='3d')
# ax.plot(sol.y[0], sol.y[1], sol.y[2], 'm', lw=0.8, label='trajectory')
# ax.scatter3D(X_ns[0], X_ns[1], X_ns[2], c='black', s = 1, label='noisy data')
# ax.set_xlabel('$x_1$', fontsize=20)
# ax.set_ylabel('$x_2$', fontsize=20)
# ax.set_zlabel('$x_3$', fontsize=20)
# plt.legend()
# # ax.set_title("Lorenz63 system", fontsize=20)
# plt.tight_layout()
# plt.show()


#------------------------------------------------------------------
# examp_type = 'rossler'
# paras = [0.2,0.2,5.7]
# x0 = [1, 1, 1]
# time_interval = [0, 50]
# # pts_type = 'uniform'
# pts_type = 'random'
# pts_num  = 5000
# nsr = 1e-1
# ns_type = 2
# # generata data
# X_ns, X_data, T, Dx, sol = gen_observ(examp_type, paras, x0, time_interval, pts_type, pts_num, nsr, ns_type)


# # ------------- plot ------------------------
# fig = plt.figure(figsize = (8, 7))
# plt.rc('legend', fontsize=14)
# ax = plt.axes(projection='3d')
# ax.plot(sol.y[0], sol.y[1], sol.y[2], 'm', lw=0.8, label='trajectory')
# ax.scatter3D(X_ns[0], X_ns[1], X_ns[2], c='black', s = 1, label='noisy data')
# ax.set_xlabel('$x_1$', fontsize=20)
# ax.set_ylabel('$x_2$', fontsize=20)
# ax.set_zlabel('$x_3$', fontsize=20)
# plt.legend()
# # ax.set_title("Lorenz63 system", fontsize=20)
# plt.tight_layout()
# plt.show()



#------------------------------------------------------------------
# examp_type = 'pendulum'
# paras = [5]
# x0 = [0, 0]
# time_interval = [0, 10]
# # pts_type = 'uniform'
# pts_type = 'random'
# pts_num  = 1000
# nsr = 1e-2
# ns_type = 2
# # generata data
# X_ns, X_data, T, Dx, sol = gen_observ(examp_type, paras, x0, time_interval, pts_type, pts_num, nsr, ns_type)
# T1 = T[1:]

# fig = plt.figure(figsize = (8, 7))
# plt.rc('legend', fontsize=14)
# plt.plot(T, X_data[0], 'r', label='$\\theta$')
# plt.scatter(T, X_ns[0], c='black', s = 1, label='noisy $\\theta$')
# plt.plot(T, X_data[1], '-b', label='$\dot{\\theta}$')
# plt.scatter(T, X_ns[1], c='purple', s = 1, label='noisy $\dot{\\theta}$')
# plt.xlabel('$t$', fontsize=25)
# plt.ylabel('$x$', fontsize=25)
# plt.legend()
# plt.tight_layout()
# plt.show()


#---------------------------------------------------------
examp_type = 'lorenz96'
paras = [5,8]
x0 = 8 * np.ones(5)
x0[0] += 0.01  # Add small perturbation to the first variable
time_interval = [0, 30]
# pts_type = 'uniform'
pts_type = 'random'
pts_num  = 8000
nsr = 1e-1
ns_type = 2
# generata data
X_ns, X_data, T, Dx, sol = gen_observ(examp_type, paras, x0, time_interval, pts_type, pts_num, nsr, ns_type)
T1 = T[1:]


# ------------- plot ------------------------
fig = plt.figure(figsize = (8, 7))
plt.rc('legend', fontsize=15)
ax = plt.axes(projection='3d')
ax.plot(sol.y[0], sol.y[1], sol.y[2], 'r', label='trajectory')
ax.scatter3D(X_ns[0], X_ns[1], X_ns[2], c='black', s = 1, label='noisy data')
ax.set_xlabel('$x_1$', fontsize=20)
ax.set_ylabel('$x_2$', fontsize=20)
ax.set_zlabel('$x_3$', fontsize=20)
plt.legend()
# ax.set_title("Lorenz63 system", fontsize=20)
plt.tight_layout()
# plt.show()

fig = plt.figure(figsize = (8, 7))
plt.rc('legend', fontsize=15)
plt.plot(sol.y[0], sol.y[1], 'b', label='trajectory')
plt.scatter(X_ns[0], X_ns[1], c='black', s = 1, label='noisy data')
plt.xlabel('$x_1$', fontsize=20)
plt.ylabel('$x_2$', fontsize=20)
plt.legend()
# ax.set_title("Lorenz63 system", fontsize=20)
plt.tight_layout()
# plt.show()

fig = plt.figure(figsize = (8, 7))
plt.rc('legend', fontsize=15)
plt.plot(sol.y[1], sol.y[2], 'g', label='trajectory')
plt.scatter(X_ns[1], X_ns[2], c='black', s = 1, label='noisy data')
plt.xlabel('$x_2$', fontsize=20)
plt.ylabel('$x_3$', fontsize=20)
plt.legend()
# ax.set_title("Lorenz63 system", fontsize=20)
plt.tight_layout()
# plt.show()

fig = plt.figure(figsize = (8, 7))
plt.rc('legend', fontsize=15)
plt.plot(sol.y[3], sol.y[4], c='orange', label='trajectory')
plt.scatter(X_ns[3], X_ns[4], c='black', s = 1, label='noisy data')
plt.xlabel('$x_4$', fontsize=25)
plt.ylabel('$x_5$', fontsize=25)
plt.legend()
# ax.set_title("Lorenz63 system", fontsize=20)
plt.tight_layout()
plt.show()

# %%


