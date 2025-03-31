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

import ode_examp
from scipy.integrate import solve_ivp

from sparsedynamics import stsl
from sparsedynamics import polynomial_basis
from sparsedynamics import sparseode, sparsevf

# Integrator keywords for solve_ivp
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12

examp_type = 'lorenz63'
paras = [10, 28, 8/3]
x0 = [1, 1, 1]
time_interval = [0, 10]
# pts_type = 'uniform'
pts_type = 'random'
pts_num  = 4000
nsr = 5e-1
ns_type = 2

# vector field
def f_vf(x1, x2, x3, para=[10,28,8/3]):
   s, r, b = para
   x_dot = s*(x2 - x1)
   y_dot = r*x1 - x2 - x1*x3
   z_dot = x1*x2 - b*x3
   return np.array([x_dot, y_dot, z_dot])

# generata data
X_ns, X_data, T, Dx, sol = gen_observ(examp_type, paras, x0, time_interval, pts_type, pts_num, nsr, ns_type)
T1 = T[1:]

# Compute the error of the derivative using VRKHS
kernel_type='gauss'
X_dot_rkhs, X_fit, lamb1 = denoise_vrkhs(T, X_ns, None, 'auto', kernel_type, (0.04,))

# Recovery the dynamics using sindy
A = polynomial_basis(X_fit.T, 2)
Xi = stsl(A, X_dot_rkhs.T, 0.1)

# Compute the error of the parameters
est_paras = np.array([Xi[2,0], Xi[3,1], -Xi[1,2]])
error = np.linalg.norm(np.array(paras) - est_paras) / np.linalg.norm(np.array(paras))
print(f'Relative error of the reconstructed parameter is {error}')

est_coeffs = Xi
coeffs = np.zeros_like(est_coeffs)
coeffs[2,0] = paras[0]
coeffs[3,0] = -paras[0]
coeffs[2,1] = -1
coeffs[3,1] = paras[1]
coeffs[7,1] = -1
coeffs[1,2] = -paras[2]
coeffs[8,2] = 1
coeff_err = np.linalg.norm(coeffs - est_coeffs)
print(coeffs)
print(est_coeffs)
print(coeff_err)

# Fit the ode using the estimated parameters
f_recons = lambda x, y, z: sparsevf(np.array([x,y,z]), Xi, 2)
fitODE   = lambda t, x: sparseode(t, x, Xi, 2)
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


# show the fitted ode
t0, tf = time_interval
sol1 = solve_ivp(ode_examp.lorenz63, [t0, tf*2], x0, args=(paras,), dense_output=True, **integrator_keywords)
sol2 = solve_ivp(fitODE, [t0, tf*2], x0, dense_output=True, **integrator_keywords)    # compute a continuous solution
    
tt = np.linspace(t0, tf*2, int(pts_num*1))
z1 = sol1.sol(tt)    
z2 = sol2.sol(tt) 

fig = plt.figure(figsize = (20, 6))
plt.subplot(1, 3, 1)
plt.plot(tt, z1[0], '-r', label='true $x_1$')
plt.plot(tt, z2[0], '--c', label='predicted $x_1$')
plt.legend()
plt.subplot(1, 3, 2)
plt.plot(tt, z1[1], '-g', label='true $x_2$')
plt.plot(tt, z2[1], '--m', label='predicted $x_2$')
plt.legend()
plt.subplot(1, 3, 3)
plt.plot(tt, z1[2], '-b', label='true $x_3$')
plt.plot(tt, z2[2], '--g', label='predicted $x_3$')
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$x_i$', fontsize=20)
plt.legend()
plt.tight_layout()
# %%
