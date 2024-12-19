#%%

import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import rcParams

def pendulum(t, x, alpha=5.0):
    """
    Parameters
    x : array-like, shape (2,)
       (x1, x2) are the quantities of prey-predator
    alpha = g/l      
    Returns
    xyz_dot : array, shape (2,)
       Values of the derivatives at *x*.
    """
    x1, x2 = x
    return [x2, -alpha*math.sin(x1)]

x0 = [0, 1]          # initial value y0=y(t0)
t0 = 0              # integration limits for t: start at t0=0
tf = 100             # and finish at tf=2
ts = np.linspace(t0, tf, 1000)    # 100 points between t0 and tf

sol = solve_ivp(pendulum, [t0, tf], x0, t_eval=ts, args=(2,), dense_output=True)

# plot three (t, x_i) trajectory
plt.rcParams['text.usetex'] = True
rcParams.update(config)
fig = plt.figure(figsize = (15, 6))
plt.subplot(1, 2, 1)
plt.plot(sol.t, sol.y[0], 'green', lw=1.0)
plt.xlabel('$t$', fontsize=20)
plt.ylabel(r'$\theta$', fontsize=20)

plt.subplot(1, 2, 2)
plt.plot(sol.t, sol.y[1], 'orange', lw=1.0)
plt.xlabel('$t$', fontsize=20)
plt.ylabel(r'$\dot{\theta}$', fontsize=20)

# %%
