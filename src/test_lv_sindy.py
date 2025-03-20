import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

from gen_data import gen_observ
from denoise import denoise_vrkhs

import ode_examp
from scipy.integrate import solve_ivp

from sparsedynamics import stsl
from sparsedynamics import polynomial_basis
from sparsedynamics import sparseode

# Integrator keywords for solve_ivp
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12

examp_type = 'lotkavolterra'
paras = [0.7, 0.007, 1, 0.007]
x0 = [70, 50]
time_interval = [0, 20]
t0, tf = time_interval
# pts_type = 'uniform'
pts_type = 'random'
pts_num  = 2000
nsr = 2e-1
ns_type = 2


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
error = np.linalg.norm(np.array(paras) - est_paras)
print(error)

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
fitODE = lambda t, x: sparseode(t, x, Xi, 2)

# show the fitted ode
t0, tf = time_interval
sol1 = solve_ivp(ode_examp.lotkavolterra, [t0, tf*3], x0, args=(paras,), dense_output=True, **integrator_keywords)
sol2 = solve_ivp(fitODE, [t0, tf*3], x0, dense_output=True, **integrator_keywords)    # compute a continuous solution
    
tt = np.linspace(t0, tf*3, int(pts_num*3))
z1 = sol1.sol(tt)    
z2 = sol2.sol(tt) 
