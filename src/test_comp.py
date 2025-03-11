import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

from gen_data import gen_observ
from denoise import denoise_vrkhs
from fd import compute_fd
from fd import compute_l2norm
from tvregdiff import TVRegDiff

examp_type = 'lorenz63'
paras = [10, 45, 8/3]
x0 = [1, 1, 1]
time_interval = [0, 30]
pts_type = 'uniform'
pts_num  = 6000
nsr = 5e-2
ns_type = 2


# generata data
X_ns, X_data, T, Dx, sol = gen_observ(examp_type, paras, x0, time_interval, pts_type, pts_num, nsr, ns_type)
T1 = T[1:]


# Compute the derivative using finite difference method
X_dot_fd = compute_fd(X_ns, T)
err_fd = compute_l2norm(X_dot_fd[:,1:]- Dx[:,1:], T[1:])/compute_l2norm(Dx[:,1:], T[1:])

# Compute the error of the derivative using VRKHS
kernel_type='gauss'
#X_dot_rkhs, X_fit, lamb1 = denoise_vrkhs(T, X_ns, 1e-1, 'pre_select', kernel_type, (0.04,))
X_dot_rkhs, X_fit, lamb1 = denoise_vrkhs(T, X_ns, None, 'auto', kernel_type, (0.04,))
err_rkhs = compute_l2norm(X_dot_rkhs-Dx[:,1:], T[1:])/compute_l2norm(Dx[:,1:], T[1:])

# Compute the error using TV regularization
#dt = T[1] - T[0]
dt = 0.005
X_dot_tv_1 = TVRegDiff(X_ns[0,:], 50, 2e-5, scale='small', ep=1e+12, dx=dt, plotflag=0)
X_dot_tv_2 = TVRegDiff(X_ns[1,:], 50, 2e-5, scale='small', ep=1e+12, dx=dt, plotflag=0)
X_dot_tv_3 = TVRegDiff(X_ns[2,:], 50, 2e-5, scale='small', ep=1e+12, dx=dt, plotflag=0)

X_dot_tv = np.vstack((X_dot_tv_1, X_dot_tv_2, X_dot_tv_3))
err_tv = compute_l2norm(X_dot_tv[:,1:]- Dx[:,1:], T[1:])/compute_l2norm(Dx[:,1:], T[1:])
# Save the data to a .mat file
"""
mdic = {'x_noise': X_ns, 't': T, 'x_true': X_data, 'x_dot_true': Dx}
savemat("tv.mat", mdic)
"""

# Print the errors
print('Errors of fd, rkhs, tv:', err_fd, err_rkhs, err_tv)



# plot true and fitted derivative functions
#fig = plt.figure(figsize = (24,6))
plt.clf()
plt.plot(T, Dx[0], '--b')
plt.plot(T1, X_dot_rkhs[0], '-r')
plt.xlabel('$t$', fontsize=10)
plt.ylabel('$\\dot{x}_1$', fontsize=10)
plt.savefig('vrkhs.pdf')
plt.show()
"""
plt.plot(T, Dx[1], '-g', label='true derivative x2')
plt.plot(T1, X_dot_rkhs[1], '--m', label='fitted derivative x2')
plt.legend()
plt.show()
plt.plot(T, Dx[2], '-m', label='true derivative x3')
plt.plot(T1, X_dot_rkhs[2], color='deepskyblue', linestyle='--', label='fitted derivative x3')
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$\\dot{x}_i$', fontsize=20)
plt.legend()
plt.title('Derivative, true and RKHS fitting')
plt.tight_layout()
plt.show()
"""

# plot true and fitted derivative functions using fitite difference method
#fig = plt.figure(figsize = (24,6))
plt.plot(T, Dx[0], '--b')
plt.plot(T1, X_dot_fd[0, 1:], '-r')
plt.xlabel('$t$', fontsize=10)
plt.ylabel('$\\dot{x}_1$', fontsize=10)
plt.savefig('fd.pdf')
plt.show()
"""
plt.plot(T, Dx[1], '-g', label='true derivative x2')
plt.plot(T1, X_dot_fd[1, 1:], '--m', label='fitted derivative x2')
plt.legend()
plt.show()
plt.plot(T, Dx[2], '-m', label='true derivative x3')
plt.plot(T1, X_dot_fd[2, 1:], color='deepskyblue', linestyle='--', label='fitted derivative x3')
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$\\dot{x}_i$', fontsize=20)
plt.legend()
plt.title('Derivative, true and FD fitting')
plt.tight_layout()
plt.show()
"""

# plot true and fitted derivative functions using fitite difference method
#fig = plt.figure(figsize = (24,6))
plt.plot(T, Dx[0], '--b')
plt.plot(T1, X_dot_tv_1[1:], '-r')
plt.xlabel('$t$', fontsize=10)
plt.ylabel('$\\dot{x}_1$', fontsize=10)
plt.savefig('tv.pdf')
plt.show()
"""
plt.plot(T, Dx[1], '-g', label='true derivative x2')
plt.plot(T1, X_dot_tv_2[1:], '--m', label='fitted derivative x2')
plt.legend()
plt.show()
plt.plot(T, Dx[2], '-m', label='true derivative x3')
plt.plot(T1, X_dot_tv_3[1:], color='deepskyblue', linestyle='--', label='fitted derivative x3')
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$\\dot{x}_i$', fontsize=20)
plt.legend()
plt.title('Derivative, true and TVR fitting')
plt.tight_layout()
plt.show()
"""


# plot true and fitted functions
#fig = plt.figure(figsize = (24,6))
plt.plot(T, X_data[0], '--b', label='true x1')
plt.plot(T, X_ns[0], '-r', label='nosie x1')
plt.xlabel('$t$', fontsize=10)
plt.ylabel('$x_1$', fontsize=10)
plt.savefig('noisedata.pdf')
plt.show()
plt.plot(T, X_data[0], '--b', label='true x1')
plt.plot(T1, X_fit[0], '-g', label='fitted  x1')
plt.xlabel('$t$', fontsize=10)
plt.ylabel('$x_1$', fontsize=10)
plt.savefig('denoisedata.pdf')
plt.show()
"""
plt.plot(T, X_ns[1], '-r', label='nosie x2')
plt.plot(T1, X_fit[1], '--m', label='fitted  x2')
plt.plot(T, X_data[1], '-g', label='true x2')
plt.legend()
plt.show()
plt.plot(T, X_ns[2], '-r', label='nosie x3')
plt.plot(T1, X_fit[2], '--b', label='fitted  x3')
plt.plot(T, X_data[2], '-g', label='true x3')
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$x_i$', fontsize=20)
plt.legend()
plt.title('Trajectory, true, noisy and RKHS fitting')
plt.show()
"""