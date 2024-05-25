#%%

import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# define the ODE system
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
a = 0.7      # prey birth rate 
b = 0.007    # prey-predator-collision rate
c = 1        # predator death rate
d = 0.007
x0 = [70, 50]           # initial value y0=y(t0)
t0 = 0                  # integration limits for t: start at t0=0
tf = 20                 # and finish at tf=10
# ts = np.linspace(t0, tf, 2000)  # 100 points between t0 and tf

# sol = solve_ivp(lotkavolterra, [t0, tf], x0, t_eval=ts, args=(0.7, 0.007, 1, 0.007), dense_output=True)
sol = solve_ivp(lotkavolterra, [t0, tf], x0, args=(a, b, c, d), dense_output=True)    # compute a continuous solution
# sol.y is a matrix, where each row contains the values for one degree of freedom.

# sample points for plotting continuous curves
t = np.linspace(t0, tf, 1000)
z = sol.sol(t)    
x1 = z[0]
x2 = z[1]

# ------------- plot ------------------------
plt.rcParams['text.usetex'] = True
fig = plt.figure(figsize = (8,6))
plt.plot(t, x1, '-r', label='prey')
plt.plot(t, x2, '-g', label='predator')
plt.legend()
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$x_i$', fontsize=20)
plt.title('Lotka-Volterra System')
# plt.savefig('predprey.pdf')
# plt.savefig('predprey.png')
plt.tight_layout()

# get values of the derivative function at the given time points: \dot{X}=f(X(t))
func = lambda x, y: lotkavolterra(t, np.array([x,y]), a, b, c, d)
D1 = map(func, x1.tolist(), x2.tolist())
D1 = np.array(list(D1))    # nxd array
DX = D1.T

fig = plt.figure(figsize = (8,6))
plt.plot(t, DX[0], '-r', label='prey')
plt.plot(t, DX[1], '-g', label='predator')
plt.legend()
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$\dot{x}_i$', fontsize=20)
plt.title('Derivative')
plt.tight_layout()


# get n=200 observations at the given time points, add noise 
n = 200
T = np.linspace(t0, tf, n+1)
T = T[1:]
# 60 points as true observatoins by random sampling
import random
T_samp = random.sample(T.tolist(), 60)  
T_samp.sort()
T = np.array(T_samp)

Z = sol.sol(T)
nsr = 1e-5
# nsr = 0
from noise import add_noise
X_ns = add_noise(Z, nsr, type="white_gauss")

fig = plt.figure(figsize = (8,6))
plt.scatter(X_ns[0], X_ns[1], c='b', s = 5)
plt.xlabel('prey')
plt.ylabel('predator')
plt.title('Noisy observation')
plt.plot(x1, x2, '-m')
plt.tight_layout()


# fitting derivative function using vRKHS
import deriv

kernel = 'gauss'
sigma = 0.2
lamb = 1e-14
d, n = Z.shape
XX0 = np.kron(x0, np.ones(n))
XX0 = XX0.reshape((d,n))
# B  = Z - XX0
B = X_ns - XX0

G1 = deriv.gram_int(T, lamb, kernel, sigma)
V  = deriv.fit_coef(B, G1, lamb)
Phi = deriv.da_basis(T, T, kernel, sigma)
X_dot = deriv.deriv_val(Phi, V)

fig = plt.figure(figsize = (8,6))
plt.scatter(T, X_dot[0], c='purple',  s = 5, label='prey')
plt.scatter(T, X_dot[1], c='b',  s = 5, label='predator')
plt.plot(t, DX[0], '-r', label='prey')
plt.plot(t, DX[1], '-g', label='predator')
plt.legend()
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$\dot{x}_i$', fontsize=20)
plt.title('Derivative by vRKHS')
plt.tight_layout()


# compute derivative by numerical central difference
# dt = T[1] - T[0]
DX_num = np.zeros((d,n-2))
for i in np.arange(n-2):
   DX_num[:,i] = (X_ns[:,i+2]-X_ns[:,i]) / (T[i+2]-T[i])

fig = plt.figure(figsize = (8,6))
plt.scatter(T[1:n-1], DX_num[0], c='purple',  s = 5, label='prey')
plt.scatter(T[1:n-1], DX_num[1], c='b',  s = 5, label='predator')
plt.plot(t, DX[0], '-r', label='prey')
plt.plot(t, DX[1], '-g', label='predator')
plt.legend()
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$\dot{x}_i$', fontsize=20)
plt.title('Derivative by difference')
plt.tight_layout()


plt.show()


# %%
