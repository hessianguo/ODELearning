#%%

import os
import sys
path1 = os.path.abspath('..')
sys.path.append(path1+'/src/')

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

from gen_data import gen_observ
from denoise import denoise_vrkhs
from fd import compute_nonuniform_fd
from fd import compute_l2norm
from tvregdiff import TVRegDiff

examp_type = 'lorenz63'
paras = [10, 28, 8/3]
x0 = [1, 1, 1]
time_interval = [0, 30]
pts_type = 'uniform'
pts_num  = 6000
nsr = 5e-1
ns_type = 2


# generata data
X_ns, X_data, T, Dx, sol = gen_observ(examp_type, paras, x0, time_interval, pts_type, pts_num, nsr, ns_type)
T1 = T[1:]

# Compute the derivative using finite difference method
X_dot_fd = compute_nonuniform_fd(X_ns, T)
err_fd = compute_l2norm(X_dot_fd[:,1:]- Dx[:,1:], T[1:])/compute_l2norm(Dx[:,1:], T[1:])

# Compute the error of the derivative using VRKHS
kernel_type='gauss'
#X_dot_rkhs, X_fit, lamb1 = denoise_vrkhs(T, X_ns, 1e-1, 'pre_select', kernel_type, (0.04,))
X_dot_rkhs, X_fit, lamb1 = denoise_vrkhs(T, X_ns, None, 'auto', kernel_type, (0.02,))
err_rkhs = compute_l2norm(X_dot_rkhs-Dx[:,1:], T[1:])/compute_l2norm(Dx[:,1:], T[1:])

# Compute the error using TV regularization
#dt = T[1] - T[0]
# seletion of regularization parameter is crucial, the following is a good choice
# for small noise level, parameter 2e-5, for large noise level, parameter 2e-2
dt = 0.005
X_dot_tv_1 = TVRegDiff(X_ns[0,:], 50, 2e-2, scale='small', ep=1e+12, dx=dt, plotflag=0)
X_dot_tv_2 = TVRegDiff(X_ns[1,:], 50, 2e-2, scale='small', ep=1e+12, dx=dt, plotflag=0)
X_dot_tv_3 = TVRegDiff(X_ns[2,:], 50, 2e-2, scale='small', ep=1e+12, dx=dt, plotflag=0)

X_dot_tv = np.vstack((X_dot_tv_1, X_dot_tv_2, X_dot_tv_3))
err_tv = compute_l2norm(X_dot_tv[:,1:]- Dx[:,1:], T[1:])/compute_l2norm(Dx[:,1:], T[1:])

# Print the errors
print('Errors of fd, rkhs, tv:', err_fd, err_rkhs, err_tv)


# Saving the array
err = np.hstack((err_fd, err_tv, err_rkhs))

# Create a filename with the noise level formatted in scientific notation
filename = f"numerical_errors_lrz_{nsr:.2e}.txt"
np.savetxt(filename, err, fmt="%.6e", header="Numerical Errors")

# plot true and fitted derivative functions
#fig = plt.figure(figsize = (24,6))
plt.clf()
plt.plot(T, Dx[0], '--b')
plt.plot(T1, X_dot_rkhs[0], '-r')
plt.xlabel('$t$', fontsize=10)
plt.ylabel('$\\dot{x}_1$', fontsize=10)
filename = f"derivx_rkhs_lrz_nsr_{nsr:.2e}.pdf"
plt.savefig(filename, format="pdf", bbox_inches="tight")
plt.show()

# plot true and fitted derivative functions using fitite difference method
#fig = plt.figure(figsize = (24,6))
plt.plot(T, Dx[0], '--b')
plt.plot(T1, X_dot_fd[0, 1:], '-r')
plt.xlabel('$t$', fontsize=10)
plt.ylabel('$\\dot{x}_1$', fontsize=10)
filename = f"derivx_fd_lrz_nsr_{nsr:.2e}.pdf"
plt.savefig(filename, format="pdf", bbox_inches="tight")
plt.show()

# plot true and fitted derivative functions using fitite difference method
#fig = plt.figure(figsize = (24,6))
plt.plot(T, Dx[0], '--b')
plt.plot(T1, X_dot_tv_1[1:], '-r')
plt.xlabel('$t$', fontsize=10)
plt.ylabel('$\\dot{x}_1$', fontsize=10)
filename = f"derivx_tv_lrz_nsr_{nsr:.2e}.pdf"
plt.savefig(filename, format="pdf", bbox_inches="tight")
plt.show()


# plot true and fitted functions
#fig = plt.figure(figsize = (24,6))
plt.plot(T, X_data[0], '--b', label='true x1')
plt.plot(T, X_ns[0], '-r', label='nosie x1')
plt.xlabel('$t$', fontsize=10)
plt.ylabel('$x_1$', fontsize=10)
filename = f"noisedata_x_lrz_nsr_{nsr:.2e}.pdf"
plt.savefig(filename, format="pdf", bbox_inches="tight")
plt.show()
plt.plot(T, X_data[0], '--b', label='true x1')
plt.plot(T1, X_fit[0], '-g', label='fitted  x1')
plt.xlabel('$t$', fontsize=10)
plt.ylabel('$x_1$', fontsize=10)
filename = f"denoisedata_x_lrz_nsr_{nsr:.2e}.pdf"
plt.savefig(filename, format="pdf", bbox_inches="tight")
plt.show()