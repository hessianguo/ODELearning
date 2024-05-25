#%%

import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation


def lorenz63(t, xyz, s=10, r=28, b=8/3):
    """
    Parameters
    ----------
    xyz : array-like, shape (3,)
       Point of interest in three-dimensional space.
    s, r, b : float
       Parameters defining the Lorenz attractor.

    Returns
    -------
    xyz_dot : array, shape (3,)
       Values of the Lorenz attractor's partial derivatives at *xyz*.
    """
    x, y, z = xyz
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return np.array([x_dot, y_dot, z_dot])

x0 = [1, 1, 1]           # initial value y0=y(t0)
t0 = 0             # integration limits for t: start at t0=0
tf = 20            # and finish at tf=2
ts = np.linspace(t0, tf, 10000)  # 1000 points between t0 and tf

sol = solve_ivp(lorenz63, [t0, tf], x0, t_eval=ts, args=(10, 28, 8/3), dense_output=True)


# Plot the 3D trajectory/Phase diagram
plt.rcParams['text.usetex'] = True
fig = plt.figure(figsize = (10,10))
ax = plt.axes(projection='3d')
ax.plot(sol.y[0], sol.y[1], sol.y[2], 'g', lw=0.8)
ax.set_xlabel('$x_1$', fontsize=20)
ax.set_ylabel('$x_2$', fontsize=20)
ax.set_zlabel('$x_3$', fontsize=20)
ax.set_title("Lorenz63 system", fontsize=20)
plt.tight_layout()


# plot the 3D scatter diagram (without noise)
# add Gaussian noise to observations and plot
nsr = 1e-2
from noise import add_noise
X_ns = add_noise(sol.y, nsr, type="white_gauss")
fig = plt.figure(figsize = (10,10))
ax = plt.axes(projection='3d')
ax.grid()
# ax.scatter3D(sol.y[0], sol.y[1], sol.y[2], c = 'b', s = 1.0)
ax.scatter3D(X_ns[0], X_ns[1], X_ns[2], c='purple', s = 1)
ax.set_title('Observation', fontsize=20)
ax.set_xlabel('$x_1$', fontsize=20)
ax.set_ylabel('$x_2$', fontsize=20)
ax.set_zlabel('$x_3$', fontsize=20)
plt.tight_layout()

# plot three (x_i, x_j) 2D orbits
fig = plt.figure(figsize = (20, 6))
plt.subplot(1, 3, 1)
plt.plot(sol.y[0], sol.y[1], 'orange', lw=1.0)
plt.xlabel('$x_1$', fontsize=20)
plt.ylabel('$x_2$', fontsize=20)

plt.subplot(1, 3, 2)
plt.plot(sol.y[1], sol.y[2], 'm', lw=1.0)
plt.xlabel('$x_2$', fontsize=20)
plt.ylabel('$x_3$', fontsize=20)

plt.subplot(1, 3, 3)
plt.plot(sol.y[0], sol.y[2], lw=1.0)
plt.xlabel('$x_1$', fontsize=20)
plt.ylabel('$x_3$', fontsize=20)
plt.suptitle('Lorenz63 2D phase plane', fontsize=20)
plt.tight_layout()

# plot three (t, x_i) curves
fig = plt.figure(figsize = (24, 6))
plt.subplot(1, 3, 1)
plt.plot(sol.t, sol.y[0], 'orange', lw=1.0)
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$x_1$', fontsize=20)

plt.subplot(1, 3, 2)
plt.plot(sol.t, sol.y[1], 'm', lw=1.0)
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$x_2$', fontsize=20)

plt.subplot(1, 3, 3)
plt.plot(sol.t, sol.y[2], lw=1.0)
plt.xlabel('$t$', fontsize=20)
plt.ylabel('$x_3$', fontsize=20)
plt.suptitle('Lorenz63 projected curves', fontsize=20)
plt.tight_layout()

plt.show()



# %%
