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

examp_type = 'lorenz96'
paras = [5, 8]
x0 = [8.01, 8, 8, 8, 8]
time_interval = [0, 10]
pts_type = 'uniform'
pts_num  = 800
nsr = 1e-1
ns_type = 2


# generata data
X_ns, X_data, T, Dx, sol = gen_observ(examp_type, paras, x0, time_interval, pts_type, pts_num, nsr, ns_type)
T1 = T[1:]

# Compute the error of the derivative using VRKHS
kernel_type='gauss'
X_dot_rkhs, X_fit, lamb1 = denoise_vrkhs(T, X_ns, None, 'auto', kernel_type, (0.05,))

# Recovery the dynamics using sindy
A = polynomial_basis(X_fit.T, 2)
Xi = stsl(A, X_dot_rkhs.T, 0.1)
print(Xi)

# Compute the error of the parameters
est_paras = np.array([Xi[1,0]])
error = np.linalg.norm(np.array(paras) - est_paras)
print(error)

est_coeffs = Xi
coeffs = np.zeros_like(est_coeffs)
coeffs[0,:] = paras[0]
coeffs[5,0] = -1
coeffs[12,0] = 1
coeffs[7,0] - -1
coeffs[4,1] = -1
coeffs[18,1] = 1
coeffs[16,1] - -1
coeffs[3,2] = -1
coeffs[13,2] = 1
coeffs[19,2] - -1
coeffs[2,3] = -1
coeffs[9,3] = 1
coeffs[14,3] - -1
coeffs[1,4] = -1
coeffs[17,4] = 1
coeffs[10,4] - -1
coff_err = np.linalg.norm(coeffs - est_coeffs)


# Fit the ode using the estimated parameters
fitODE = lambda t, x: sparseode(t, x, Xi, 2)

# show the fitted ode
t0, tf = time_interval
sol1 = solve_ivp(ode_examp.lorenz96, [t0, tf*3], x0, args=(paras,), dense_output=True, **integrator_keywords)
sol2 = solve_ivp(fitODE, [t0, tf*3], x0, dense_output=True, **integrator_keywords)    # compute a continuous solution
    
tt = np.linspace(t0, tf*3, int(pts_num*3))
z1 = sol1.sol(tt)    
z2 = sol2.sol(tt) 
