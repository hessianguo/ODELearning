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
x0 = [7, 5]           # initial value y0=y(t0)
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

# get values of the derivative function at the given time points: \dot{X}=f(X(t))
func = lambda x, y: lotkavolterra(t, np.array([x,y]), a, b, c, d)
D1 = map(func, x1.tolist(), x2.tolist())
D1 = np.array(list(D1))    # nxd array
DX = D1.T

# ------------- plot ------------------------
plt.rcParams['text.usetex'] = True

fig = plt.figure(figsize = (15, 6))
plt.subplot(1, 2, 1)
plt.plot(t, x1, '-r', label='prey')
plt.plot(t, x2, '-g', label='predator')
plt.legend()
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$x_i$', fontsize=20)
# plt.savefig('predprey.pdf')
# plt.savefig('predprey.png')
plt.tight_layout()
plt.subplot(1, 2, 2)
plt.plot(t, DX[0], '-r', label='prey')
plt.plot(t, DX[1], '-g', label='predator')
plt.legend()
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$\dot{x}_i$', fontsize=20)
plt.suptitle('Lotka-Volterra system, trajectory and derivative')
plt.tight_layout()

#---------------------------------------------------------------
# get n=200 observations at the given time points, add noise 
n = 2000
T = np.linspace(t0, tf, n+1)
T = T[1:]
# 60 points as true observatoins by random sampling
import random
T_samp = random.sample(T.tolist(), 1000)  
T_samp.sort()
T = np.array(T_samp)
Z = sol.sol(T)
nsr = 1e-2    # noise level
# nsr = 0
from noise import add_noise
X_ns = add_noise(Z, nsr, type="white_gauss")


# fitting derivative function using vRKHS
import deriv
kernel = 'gauss'
sigma = 0.2
lamb = 5e-4
d, n = Z.shape
XX0 = np.kron(x0, np.ones(n))
XX0 = XX0.reshape((d,n))
# B  = Z - XX0
B = X_ns - XX0

G1 = deriv.gram_int(T, kernel, sigma)
V  = deriv.fit_coef(B, G1, lamb)
Phi = deriv.da_basis(T, T, kernel, sigma)
X_dot = deriv.deriv_val(Phi, V)

# compute derivative by numerical central difference
# dt = T[1] - T[0]
DX_num = np.zeros((d,n-2))
for i in np.arange(n-2):
   DX_num[:,i] = (X_ns[:,i+2]-X_ns[:,i]) / (T[i+2]-T[i])

# plot derivative function by RKHS and central difference
fig = plt.figure(figsize = (15, 6))
plt.subplot(1, 2, 1)
plt.scatter(T, X_dot[0], c='purple',  s = 5, label='prey')
plt.scatter(T, X_dot[1], c='b',  s = 5, label='predator')
plt.plot(t, DX[0], '-r', label='prey')
plt.plot(t, DX[1], '-g', label='predator')
plt.legend()
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$\dot{x}_i$', fontsize=20)
plt.title('Derivative by vRKHS')
plt.subplot(1, 2, 2)
plt.scatter(T[1:n-1], DX_num[0], c='purple',  s = 5, label='prey')
plt.scatter(T[1:n-1], DX_num[1], c='b',  s = 5, label='predator')
plt.plot(t, DX[0], '-r', label='prey')
plt.plot(t, DX[1], '-g', label='predator')
plt.legend()
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$\dot{x}_i$', fontsize=20)
plt.title('Derivative by difference')
plt.tight_layout()

# fit trjectory and then compute analytic derivative based on kernel basis
# G1 = deriv.gram(T, kernel, sigma)
# V  = deriv.fit_coef(X_ns, G1, lamb)
# Phi_1 = deriv.kernel_basis(T, T, kernel, sigma)
# Phi_2 = deriv.deriv_basis(T, T, kernel, sigma)
# X_fit = deriv.deriv_val(Phi_1, V)
# X_dot = deriv.deriv_val(Phi_2, V)

# fig = plt.figure(figsize = (8,6))
# plt.scatter(T, X_ns[0], c='b', s = 5, label='noisy x1')
# plt.scatter(T, X_ns[1], c='m', s = 5, label='noisy x2')
# plt.plot(T, X_fit[0], '-r', label='prey')
# plt.plot(T, X_fit[1], '-g', label='predator')
# plt.legend()
# plt.title('Noisy observation')
# plt.xlabel('$t$', fontsize=20)
# plt.ylabel('$x_i$', fontsize=20)
# plt.tight_layout()

# fig = plt.figure(figsize = (8,6))
# plt.scatter(X_ns[0], X_ns[1], c='b', s = 5)
# plt.xlabel('prey')
# plt.ylabel('predator')
# plt.title('Noisy observation')
# plt.plot(X_fit[0], X_fit[1], '-m', label='fitted')
# plt.tight_layout()

# fig = plt.figure(figsize = (8,6))
# plt.scatter(T, X_dot[0], c='purple',  s = 5, label='prey')
# plt.scatter(T, X_dot[1], c='b',  s = 5, label='predator')
# plt.plot(t, DX[0], '-r', label='prey')
# plt.plot(t, DX[1], '-g', label='predator')
# plt.legend()
# plt.xlabel('$t$', fontsize=20)
# plt.ylabel('$\dot{x}_i$', fontsize=20)
# plt.title('Derivative by vRKHS')
# plt.tight_layout()

#------------------------------------------------------------
# Compute the fitted derivative function and trajectory
Phi_traj  = deriv.gram_traj(T, t, kernel, sigma)
Phi_deriv = deriv.da_basis(T,  t, kernel, sigma) 
X_fit = deriv.deriv_val(Phi_traj,  V) + np.kron(x0,np.ones(len(t))).reshape((d,-1))
dot_fit = deriv.deriv_val(Phi_deriv, V)

# plot noise observation and fitted curves
fig = plt.figure(figsize = (15, 6))
plt.subplot(1, 2, 1)
plt.scatter(T, X_ns[0], c='b', s = 3, label='noisy x1')
plt.scatter(T, X_ns[1], c='m', s = 3, label='noisy x2')
plt.plot(t, X_fit[0], '-r', label='fitted x1')
plt.plot(t, X_fit[1], '-g', label='fitted x2')
plt.legend()
plt.subplot(1, 2, 2)
plt.scatter(X_ns[0], X_ns[1], c='b', s = 5, label='observation')
plt.plot(X_fit[0], X_fit[1], '-r', label='fitted')
plt.xlabel('prey')
plt.ylabel('predator')
plt.legend()
plt.suptitle('Noisy observation and fitted curve')
plt.tight_layout()

# plot true and fitted curves and derivative functions
fig = plt.figure(figsize = (20, 6))
plt.subplot(1, 3, 1)
plt.plot(t, x1, '-r', label='true x1')
plt.plot(t, x2, '-g', label='true x1')
plt.plot(t, X_fit[0], '--b', label='fitted x1')
plt.plot(t, X_fit[1], '--m', label='fitted x2')
plt.legend()
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$x_i$', fontsize=20)
plt.subplot(1, 3, 2)
plt.plot(t, DX[0], '-r', label='true derivative x1')
plt.plot(t, DX[1], '-g', label='true derivative x2')
plt.plot(t, dot_fit[0], '--b', label='fitted derivative x1')
plt.plot(t, dot_fit[1], '--m', label='fitted derivative x2')
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$\dot{x}_i$', fontsize=20)
plt.legend()
plt.subplot(1, 3, 3)
plt.plot(x1, x2, '-b', label='true')
plt.plot(X_fit[0], X_fit[1], '--r', label='fitted')
plt.xlabel('prey')
plt.ylabel('predator')
plt.legend()
plt.suptitle('Lotka-Volterra, true and RKHS fitting')
plt.tight_layout()



plt.show()


# %%
