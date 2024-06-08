import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sparsedynamics
from noise import add_noise
import deriv

def cubicfun(t, x):
    x1, x2 = x
    return np.array([-0.1*x1**3 + 2*x2**3, -2*x1**3 - 0.1*x2**3])

# set initial value and time interval
x0 = [2, 0]           # initial value y0=y(t0), a list
t0 = 0                  # integration limits for t: start at t0=0
tf = 25                 # and finish at tf=10
t = np.linspace(t0, tf, 2501)
#nsr = 1e-4    # noise level
nsr = 1e-3

sol = solve_ivp(cubicfun, [t0, tf], x0, ethod='RK45', t_eval=t)    # compute a continuous solution
X_ns = add_noise(sol.y, nsr, type="white_gauss")

# sample points for plotting continuous curves
x1 = sol.y[0]
x2 = sol.y[1]


dxdt = cubicfun(0, X_ns)

# Reconstruction of the dynamics using SINDy
A = sparsedynamics.polynomial_basis(X_ns.T, 5)
Xi = sparsedynamics.stsl(A, dxdt.T,  1e-2)
print(Xi)


fitODE = lambda t, x: sparsedynamics.sparseode(t, x, Xi, 5)

# prediction by solving the reconstructed ODE model
sol2 = solve_ivp(fitODE, [t0, tf*1.2], x0, dense_output=True)    # compute a continuous solution
sol1 = solve_ivp(cubicfun, [t0, tf*1.3], x0,  dense_output=True)    

tt = np.linspace(t0, tf*1.3, int(1000*1.3))
z1 = sol1.sol(tt)    
z2 = sol2.sol(tt) 

# post processing
fig = plt.figure(figsize = (15, 6))
plt.subplot(1, 2, 1)
plt.plot(tt, z1[0], '-r', label='true x1')
plt.plot(tt, z1[1], '-g', label='true x2')
plt.plot(tt, z2[0], '--b', label='predicted x1')
plt.plot(tt, z2[1], '--m', label='predicted x2')
plt.legend()
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$x_i$', fontsize=20)
plt.subplot(1, 2, 2)
plt.plot(z1[0], z1[1], '-b', label='true')
plt.plot(z2[0], z2[1], '--r', label='predicted')
plt.xlabel('prey')
plt.ylabel('predator')
plt.legend()
plt.suptitle('Lotka-Volterra, true and prediction')
plt.tight_layout()

plt.show()
#plt.savefig('LV_pred.png')

