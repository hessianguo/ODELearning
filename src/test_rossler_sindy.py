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

examp_type = 'rossler'
paras = [0.2, 0.2, 5.7]
x0 = [1, 1, 1]
time_interval = [0, 50]
pts_type = 'random'
pts_num  = 5000
nsr = 1e-2
ns_type = 2


# generata data
X_ns, X_data, T, Dx, sol = gen_observ(examp_type, paras, x0, time_interval, pts_type, pts_num, nsr, ns_type)
T1 = T[1:]

# Compute the error of the derivative using VRKHS
kernel_type='gauss'
X_dot_rkhs, X_fit, lamb1 = denoise_vrkhs(T, X_ns, None, 'auto', kernel_type, (0.2,))

# Recovery the dynamics using sindy
A = polynomial_basis(X_fit.T, 2)
Xi = stsl(A, X_dot_rkhs.T, 0.1)

# Compute the error of the parameters
est_paras = np.array([Xi[2,1], Xi[0,2], -Xi[1,2]])
error = np.linalg.norm(np.array(paras) - est_paras)
print(error)

est_coeffs = Xi
coeffs = np.zeros_like(est_coeffs)
coeffs[1,0] = -1
coeffs[2,0] = -1
coeffs[2,1] = paras[0]
coeffs[3,1] = 1
coeffs[1,2] = -paras[2]
coeffs[8,2] = 1
coeff_err = np.linalg.norm(coeffs - est_coeffs)
print(coeffs)
print(est_coeffs)
print(coeff_err)

# Fit the ode using the estimated parameters
fitODE = lambda t, x: sparseode(t, x, Xi, 2)

# show the fitted ode
t0, tf = time_interval
sol1 = solve_ivp(ode_examp.rossler, [t0, tf*3], x0, args=(paras,), dense_output=True, **integrator_keywords)
sol2 = solve_ivp(fitODE, [t0, tf*3], x0, dense_output=True, **integrator_keywords)    # compute a continuous solution
    
tt = np.linspace(t0, tf*3, int(pts_num*3))
z1 = sol1.sol(tt)    
z2 = sol2.sol(tt) 
