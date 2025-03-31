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

examp_type = 'lorenz96'
d = 5
paras = [d,8]
x0 = 8 * np.ones(d)
x0[0] += 0.01  # Add small perturbation to the first variable
time_interval = [0, 30]
t0, tf = time_interval
# pts_type = 'uniform'
pts_type = 'random'
pts_num  = 8000
nsr = 1e-1
ns_type = 2

# vector field
def f_vf(x, y, z, w, r, para=[5,8]):
    xx = np.array([x,y,z,w,r])
    d = paras[0]
    F = paras[1]
    x_dot = (np.roll(xx, -1) - np.roll(xx, 2)) * np.roll(xx, 1) - xx + F 
    return np.array(x_dot)

# generata data
X_ns, X_data, T, Dx, sol = gen_observ(examp_type, paras, x0, time_interval, pts_type, pts_num, nsr, ns_type)
T1 = T[1:]

# Compute the error of the derivative using VRKHS
kernel_type='gauss'
X_dot, X_fit, lamb1 = denoise_vrkhs(T, X_ns, None, 'auto', kernel_type, (0.1,))

# Recovery the dynamics using sindy
A = polynomial_basis(X_fit.T, 2)
Xi = stsl(A, X_dot.T, 0.1)

# Compute the error of the parameters
est_paras = np.array([Xi[0,0]])
error = np.linalg.norm(np.array(paras[1]) - est_paras) / np.linalg.norm(np.array(paras))
print(f'Relative error of the reconstructed parameter is {error}')

est_coeffs = Xi
coeffs = np.zeros_like(est_coeffs)
coeffs[0,:] = paras[1]
coeffs[5,0] = -1
coeffs[12,0] = 1
coeffs[7,0] = -1
coeffs[4,1] = -1
coeffs[18,1] = 1
coeffs[16,1] = -1
coeffs[3,2] = -1
coeffs[13,2] = 1
coeffs[19,2] = -1
coeffs[2,3] = -1
coeffs[9,3] = 1
coeffs[14,3] = -1
coeffs[1,4] = -1
coeffs[17,4] = 1
coeffs[10,4] = -1
coff_err = np.linalg.norm(coeffs - est_coeffs)
print(coeffs)
print(est_coeffs)
print(coff_err)


# Fit the ode using the estimated parameters
f_recons = lambda x, y, z, w, r,: sparsevf(np.array([x,y,z,w,r]), Xi, 2)
fitODE   = lambda t, x: sparseode(t, x, Xi, 2)

N_pts = 10  # grid points, in each dimension
xx = np.linspace(-5, 12, N_pts)
yy = np.linspace(-10,10, N_pts)
zz = np.linspace(-8, 10, N_pts)
ww = np.linspace(-5, 12, N_pts)
rr = np.linspace(-6, 10, N_pts)
XX, YY, ZZ, WW, RR = np.meshgrid(xx, yy, zz, ww, rr)
XY_test = np.concatenate((XX.reshape(-1,1),YY.reshape(-1,1),ZZ.reshape(-1,1),WW.reshape(-1,1),RR.reshape(-1,1)), axis=1) # test data points

truth_val = map(f_vf, XY_test[:,0].tolist(), XY_test[:,1].tolist(), XY_test[:,2].tolist(), XY_test[:,3].tolist(), XY_test[:,4].tolist())
truth_val = np.array(list(truth_val)) 
fit_val = map(f_recons, XY_test[:,0].tolist(), XY_test[:,1].tolist(), XY_test[:,2].tolist(), XY_test[:,3].tolist(), XY_test[:,4].tolist())
fit_val = np.array(list(fit_val)) 

err = np.abs(fit_val-truth_val)
rel_er = np.sqrt(np.sum(err**2))/np.sqrt(np.sum(truth_val**2))
print(f'Relative L2 error of the reconstructed vector field is {rel_er}')


# show the fitted ode
t0, tf = time_interval
sol1 = solve_ivp(ode_examp.lorenz96, [t0, tf*2], x0, args=(paras,), dense_output=True, **integrator_keywords)
sol2 = solve_ivp(fitODE, [t0, tf*2], x0, dense_output=True, **integrator_keywords)    # compute a continuous solution
    

plt.rcParams['text.usetex'] = True

tt = np.linspace(t0, tf*2, int(pts_num*2))
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
