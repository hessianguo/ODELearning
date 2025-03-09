#%%

import os
import sys
path1 = os.path.abspath('..')
sys.path.append(path1+'/src/')

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

from noise import add_noise
from denoise import denoise_vrkhs
from fd import compute_fd
from fd import compute_l2norm
from tvregdiff import TVRegDiff

time_interval = [-0.5, 0.5]
pts_type = 'uniform'
pts_num  = 101
nsr = 1e-2
ns_type = 2


# generata data
g = lambda x: np.cos(x)
dg = lambda x: -np.sin(x)
tt = np.array([-np.pi, np.pi])
T = np.linspace(tt[0], tt[1], pts_num)
X = g(T)
x0 = X[0]
X_data = np.reshape(X, (-1, pts_num))
X_ns = add_noise(X_data[:,1:], nsr, ns_type)
x0 = np.array([[x0]])
X_ns = np.hstack((x0, X_ns))
Dx = dg(T)
T1 = T[1:]


# Compute the derivative using finite difference method
X_dot_fd = compute_fd(X_ns, T)
err_fd = compute_l2norm(X_dot_fd-Dx, T)/compute_l2norm(Dx, T)

# Compute the error of the derivative using VRKHS
kernel_type='gauss'
# X_dot_rkhs, X_fit, lamb1 = denoise_vrkhs(T, X_ns, 5e-2, 'pre_select', kernel_type, (0.4,))
X_dot_rkhs, X_fit, lamb1 = denoise_vrkhs(T, X_ns, None, 'auto', kernel_type, (0.5,))
err_rkhs = compute_l2norm(X_dot_rkhs-Dx[1:], T[1:])/compute_l2norm(Dx[1:], T[1:])

# Compute the error using TV regularization
dt = T[1] - T[0]
X_dot_tv = TVRegDiff(X_ns.reshape(-1), 50, 1e-7, scale='small', ep=1e-6, dx=dt, plotflag=0)
err_tv = compute_l2norm(X_dot_tv- Dx, T)/compute_l2norm(Dx, T)
# Save the data to a .mat file
mdic = {'x_noise': X_ns, 't': T, 'x_true': X_data, 'x_dot_true': Dx}
savemat("tv.mat", mdic)

# Print the errors
print('Errors of fd, rkhs, tv:', err_fd, err_rkhs, err_tv)

# plot true and fitted derivative functions
fig = plt.figure(figsize = (24,6))
plt.plot(T, Dx, '-r', label='true derivative x1')
plt.plot(T1, X_dot_rkhs[0], '--b', label='fitted derivative x1')
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$\\dot{x}_i$', fontsize=20)
plt.legend()
plt.title('Derivative, true and RKHS fitting')
plt.tight_layout()
plt.show()

# plot true and fitted derivative functions using fitite difference method
fig = plt.figure(figsize = (24,6))
plt.plot(T, Dx, '-r', label='true derivative x1')
plt.plot(T1, X_dot_fd[0,1:], '--b', label='fitted derivative x1')
plt.ylabel('$\\dot{x}_i$', fontsize=20)
plt.legend()
plt.title('Derivative, true and FD fitting')
plt.tight_layout()
plt.show()

# plot true and fitted derivative functions using fitite difference method
fig = plt.figure(figsize = (24,6))
plt.plot(T, Dx, '-r', label='true derivative x1')
plt.plot(T, X_dot_tv, '--b', label='fitted derivative x1')
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$\\dot{x}_i$', fontsize=20)
plt.legend()
plt.title('Derivative, true and TVR fitting')
plt.tight_layout()
plt.show()
# %%
