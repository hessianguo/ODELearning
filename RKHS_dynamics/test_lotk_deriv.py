import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sparsedynamics
import deriv

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
    return [a*x1 - b*x1*x2, -c*x2 + d*x1*x2]

# set initial value and time interval
a1 = 0.7      # prey birth rate 
a2 = 0.007    # prey-predator-collision rate
a3 = 1        # predator death rate
a4 = 0.007
x0 = [70, 50]           # initial value y0=y(t0), a list
t0 = 0                  # integration limits for t: start at t0=0
tf = 20                 # and finish at tf=10

sol = solve_ivp(lotkavolterra, [t0, tf], x0, args=(a1, a2, a3, a4), dense_output=True)    # compute a continuous solution

# sample points for plotting continuous curves
t = np.linspace(t0, tf, 1000)
z = sol.sol(t)    
x1 = z[0]
x2 = z[1]

# get values of the derivative function at the given time points: \dot{X}=f(X(t))
func = lambda x, y: lotkavolterra(t, np.array([x,y]), a1, a2, a3, a4)
D1 = map(func, x1.tolist(), x2.tolist())
D1 = np.array(list(D1))    # nxd array
DX = D1.T

# get n=200 observations at the given time points, add noise 
n1 = 2000
T1 = np.linspace(t0, tf, n1+1)
T1 = T1[1:]
# 1000 points as true observatoins by random sampling
import random
T_samp = random.sample(T1.tolist(), 1000)  
T_samp.sort()
T = np.array(T_samp)
Z = sol.sol(T)
nsr = 1e-4    # noise level
#nsr = 0.3
from noise import add_noise
X_ns = add_noise(Z, nsr, type="white_gauss")

# fitting derivative function using vRKHS
kernel = 'gauss'
sigma = 0.2
lamb = 1e-4
d, n = Z.shape
XX0 = np.kron(x0, np.ones(n))
XX0 = XX0.reshape((d,n))
# B  = Z - XX0
B = X_ns - XX0

G1 = deriv.gram_int(T, kernel, sigma)
V  = deriv.fit_coef(B, G1, lamb)
Phi = deriv.da_basis(T, T, kernel, sigma)
X_dot = deriv.deriv_val(Phi, V)
print(X_dot.shape)
print(X_ns.shape)

DX_num = np.zeros((d,n))
for i in np.arange(n-2):
   DX_num[:,i+1] = (X_ns[:,i+2]-X_ns[:,i]) / (T[i+2]-T[i])

# Reconstruction of the dynamics using SINDy
A = sparsedynamics.polynomial_basis(X_ns.T, 2)
Xi = sparsedynamics.stsl(A, X_dot.T, 5e-3)
print(Xi)


fitODE = lambda t, x: sparsedynamics.sparseode(t, x, Xi, 2)

# prediction by solving the reconstructed ODE model
sol2 = solve_ivp(fitODE, [t0, tf*1.2], x0, dense_output=True)    # compute a continuous solution
sol1 = solve_ivp(lotkavolterra, [t0, tf*1.3], x0, args=(a1, a2, a3, a4), dense_output=True)    

tt = np.linspace(t0, tf*1.3, int(1000*1.3))
z1 = sol1.sol(tt)    
z2 = sol2.sol(tt) 

tt = np.linspace(t0, tf*1.3, int(1000*1.3))
z1 = sol1.sol(tt)    
z2 = sol2.sol(tt) 


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

