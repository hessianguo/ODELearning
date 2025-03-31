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

examp_type = 'lotkavolterra'
paras = [0.7, 0.007, 1, 0.007]
x0 = [70, 50]
time_interval = [0, 10]
t0, tf = time_interval
# pts_type = 'uniform'
pts_type = 'random'
pts_num  = 2000
nsr = 10e-1
ns_type = 2

# vector field
def f_vf(x1, x2, para=[0.7,0.007,1.0,0.007]):
   a, b, c, d = para
   return np.array([a*x1 - b*x1*x2, -c*x2 + d*x1*x2])

# generata data
X_ns, X_data, T, Dx, sol = gen_observ(examp_type, paras, x0, time_interval, pts_type, pts_num, nsr, ns_type)
T1 = T[1:]

# Compute the error of the derivative using VRKHS
kernel_type='gauss'
X_dot_rkhs, X_fit, lamb1 = denoise_vrkhs(T, X_ns, None, 'auto', kernel_type, (0.3,))

# Recovery the dynamics using sindy
A = polynomial_basis(X_fit.T, 2)
Xi = stsl(A, X_dot_rkhs.T, 0.001)

# Compute the error of the parameters
est_paras = np.array([Xi[2,0], -Xi[4,0], -Xi[1,1], Xi[4,1]])
error = np.linalg.norm(np.array(paras) - est_paras) / np.linalg.norm(np.array(paras))
print(f'Relative error of the reconstructed parameter is {error}')

est_coeffs = Xi
coeffs = np.zeros_like(est_coeffs)
coeffs[2,0] = paras[0]
coeffs[4,0] = -paras[1]
coeffs[1,1] = -paras[2]
coeffs[4,1] = paras[3]
coeff_err = np.linalg.norm(coeffs - est_coeffs)
print(coeffs)
print(est_coeffs)
print(coeff_err)

# Fit the ode using the estimated parameters
f_recons = lambda x, y: sparsevf(np.array([x,y]), Xi, 2)
fitODE   = lambda t, x: sparseode(t, x, Xi, 2)
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


# make predictions and comparison
t0, tf = time_interval
sol1 = solve_ivp(ode_examp.lotkavolterra, [t0, tf*3], x0, args=(paras,), dense_output=True, **integrator_keywords)
sol2 = solve_ivp(fitODE, [t0, tf*3], x0, dense_output=True, **integrator_keywords)    # compute a continuous solution
    
tt = np.linspace(t0, tf*3, int(pts_num*3))
z1 = sol1.sol(tt)    
z2 = sol2.sol(tt) 

plt.rcParams['text.usetex'] = True
fig = plt.figure(figsize = (9, 7))
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
