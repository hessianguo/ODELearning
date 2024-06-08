import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sparsedynamics
import deriv
# Integrator keywords for solve_ivp
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12

def lotkavolterra(t, x, a=0.7, b=0.007, c=1, d=0.007):
    """
    Parameters
    x : array-like, shape (2,)
       (x1, x2) are the quantities of prey-predator
    a = 0.7      # prey birth rate 
    b = 0.007    # prey-predator-collision rate
    c = 1        # predator death rate
    d = 0.007    # prey-predator-collision rate
    Returns
    xyz_dot : array, shape (2,)
       Values of the derivatives at *x*.
    """
    x1, x2 = x
    return np.array([a*x1 - b*x1*x2, -c*x2 + d*x1*x2])

# set initial value and time interval
a1 = 0.7      # prey birth rate 
a2 = 0.007    # prey-predator-collision rate
a3 = 1        # predator death rate
a4 = 0.007
x0 = [70, 50]           # initial value y0=y(t0), a list
t0 = 0                  # integration limits for t: start at t0=0
tf = 40                 # and finish at tf=10
t = np.linspace(t0, tf, 1000)
#sol = solve_ivp(lotkavolterra, [t0, tf], x0, args=(a1, a2, a3, a4), dense_output=True)    # compute a continuous solution
sol = solve_ivp(lotkavolterra, [t0, tf], x0, args=(a1, a2, a3, a4),  t_eval=t, **integrator_keywords)

# sample points for plotting continuous curves

z = sol.y    
x1 = z[0]
x2 = z[1]
nsr = 1e-2
from noise import add_noise
X_ns = add_noise(sol.y, nsr, type="white_gauss")

dxdt = lotkavolterra(0, X_ns)

# Reconstruction of the dynamics using SINDy
A = sparsedynamics.polynomial_basis(X_ns.T, 2)
Xi = sparsedynamics.stsl(A, dxdt.T, 0.0001)
#Xi2 = sparsedynamics.sparsify_dynamics(A, dxdt.T, 1e-2, 2)
print(Xi)
#print(Xi2)


fitODE = lambda t, x: sparsedynamics.sparseode(t, x, Xi, 2)

# prediction by solving the reconstructed ODE model
tt = np.linspace(t0, tf*3.5, int(1000*1.2))
sol2 = solve_ivp(fitODE, [t0, tf*3.5], x0,  t_eval=tt, **integrator_keywords)    # compute a continuous solution
sol1 = solve_ivp(lotkavolterra, [t0, tf*3.5], x0, args=(a1, a2, a3, a4), t_eval=tt, **integrator_keywords)    

z1 = sol1.y
z2 = sol2.y



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

