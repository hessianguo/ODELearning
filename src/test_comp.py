#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

from gen_data import gen_observ
from denoise import denoise_vrkhs
from fd import compute_fd
from fd import compute_l2norm
from tvregdiff import TVRegDiff

examp_type = 'lorenz63'
paras = [10, 28, 8/3]
x0 = [1, 1, 1]
time_interval = [0, 4]
pts_type = 'uniform'
pts_num  = 400
nsr = 5e-0
ns_type = 2


# generata data
X_ns, X_data, T, Dx, sol = gen_observ(examp_type, paras, x0, time_interval, pts_type, pts_num, nsr, ns_type)
T1 = T[1:]


# Compute the derivative using finite difference method
X_dot_fd = compute_fd(X_ns, T)
err_fd = compute_l2norm(X_dot_fd-Dx, T)/compute_l2norm(Dx, T)

# Compute the error of the derivative using VRKHS
kernel_type='gauss'
#X_dot, X_fit, lamb1 = denoise_vrkhs(T, X_ns, 1e-4, 'pre_select', kernel_type, (0.2,))
X_dot_rkhs, X_fit, lamb1 = denoise_vrkhs(T, X_ns, None, 'auto', kernel_type, (0.06,))
err_rkhs = compute_l2norm(X_dot_rkhs-Dx[:,1:], T[1:])/compute_l2norm(Dx[:,1:], T[1:])

# Compute the error using TV regularization
dt = T[1] - T[0]
X_dot_tv_1 = TVRegDiff(X_ns[0,:], 50, 2e-1, scale='small', ep=1e-6, dx=dt, plotflag=0)
X_dot_tv_2 = TVRegDiff(X_ns[1,:], 50, 2e-1, scale='small', ep=1e-6, dx=dt, plotflag=0)
X_dot_tv_3 = TVRegDiff(X_ns[2,:], 50, 2e-1, scale='small', ep=1e-6, dx=dt, plotflag=0)

X_dot_tv = np.vstack((X_dot_tv_1, X_dot_tv_2, X_dot_tv_3))
err_tv = compute_l2norm(X_dot_tv- Dx, T)/compute_l2norm(Dx, T)
# Save the data to a .mat file
mdic = {'x_noise': X_ns, 't': T, 'x_true': X_data, 'x_dot_true': Dx}
savemat("tv.mat", mdic)

# Print the errors
print('Errors of fd, rkhs, tv:', err_fd, err_rkhs, err_tv)

# plot true and fitted derivative functions
fig = plt.figure(figsize = (24,6))
plt.subplot(1, 3, 1)
plt.plot(T, Dx[0], '-r', label='true derivative x1')
plt.plot(T1, X_dot_rkhs[0], '--b', label='fitted derivative x1')
plt.legend()
plt.subplot(1, 3, 2)
plt.plot(T, Dx[1], '-g', label='true derivative x2')
plt.plot(T1, X_dot_rkhs[1], '--m', label='fitted derivative x2')
plt.legend()
plt.subplot(1, 3, 3)
plt.plot(T, Dx[2], '-m', label='true derivative x3')
plt.plot(T1, X_dot_rkhs[2], color='deepskyblue', linestyle='--', label='fitted derivative x3')
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$\\dot{x}_i$', fontsize=20)
plt.legend()
plt.title('Derivative, true and RKHS fitting')
plt.tight_layout()
plt.show()

# plot true and fitted derivative functions using fitite difference method
fig = plt.figure(figsize = (24,6))
plt.subplot(1, 3, 1)
plt.plot(T, Dx[0], '-r', label='true derivative x1')
plt.plot(T1, X_dot_fd[0, 1:], '--b', label='fitted derivative x1')
plt.legend()
plt.subplot(1, 3, 2)
plt.plot(T, Dx[1], '-g', label='true derivative x2')
plt.plot(T1, X_dot_fd[1, 1:], '--m', label='fitted derivative x2')
plt.legend()
plt.subplot(1, 3, 3)
plt.plot(T, Dx[2], '-m', label='true derivative x3')
plt.plot(T1, X_dot_fd[2, 1:], color='deepskyblue', linestyle='--', label='fitted derivative x3')
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$\\dot{x}_i$', fontsize=20)
plt.legend()
plt.title('Derivative, true and FD fitting')
plt.tight_layout()
plt.show()

# plot true and fitted derivative functions using fitite difference method
fig = plt.figure(figsize = (24,6))
plt.subplot(1, 3, 1)
plt.plot(T, Dx[0], '-r', label='true derivative x1')
plt.plot(T1, X_dot_tv_1[1:], '--b', label='fitted derivative x1')
plt.legend()
plt.subplot(1, 3, 2)
plt.plot(T, Dx[1], '-g', label='true derivative x2')
plt.plot(T1, X_dot_tv_2[1:], '--m', label='fitted derivative x2')
plt.legend()
plt.subplot(1, 3, 3)
plt.plot(T, Dx[2], '-m', label='true derivative x3')
plt.plot(T1, X_dot_tv_3[1:], color='deepskyblue', linestyle='--', label='fitted derivative x3')
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$\\dot{x}_i$', fontsize=20)
plt.legend()
plt.title('Derivative, true and TVR fitting')
plt.tight_layout()
plt.show()
# %%
