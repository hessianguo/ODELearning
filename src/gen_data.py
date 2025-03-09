import numpy as np
from scipy.integrate import solve_ivp
import sys
from os import path
# add the relative paths into pythonpath
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import ode_examp
from noise import add_noise

# Integrator keywords for solve_ivp
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12

# Given a type of example, generate the noisy observations at some selected time 
# points, and give the true trajectory and derivative function
#   
# Inputs:
#   examp_type: type of examples, such as 'lorenz63', 'pendulum'
#   pts_type:   type of time points--uniform points or random points
#   pnt_num:    number of time points for observation
#   nsr:        noise level (ratio of the l2 norm)
#   ns_type:    noise type
#
# Outputs:
#   X_ns: noise observation
#   X: noiseless observation
#   Xd: derivative function (dicrete data)
def gen_observ(examp_type, paras, x0, time_interval, pts_type, pts_num, nsr, ns_type="white_gauss"):
    if examp_type == 'lotkavolterra':
        if len(paras) != 4:
            raise ValueError('Should be four parameters!')
        # compute a continuous solution
        sol = solve_ivp(ode_examp.lotkavolterra, time_interval, x0, args=(paras,), dense_output=True, **integrator_keywords)
    elif examp_type == 'lorenz63':
        if len(paras) != 3:
            raise ValueError('Should be three parameters!')
        # compute a continuous solution
        sol = solve_ivp(ode_examp.lorenz63, time_interval, x0, args=(paras,), dense_output=True, **integrator_keywords)
    elif examp_type == 'pendulum':
        if len(paras) != 1:
            raise ValueError('Should be one parameter!')
        # compute a continuous solution
        sol = solve_ivp(ode_examp.pendulum, time_interval, x0, args=(paras,), dense_output=True, **integrator_keywords)
    elif examp_type == 'rossler':
        if len(paras) != 3:
            raise ValueError('Should be three parameters!')
        # compute a continuous solution
        sol = solve_ivp(ode_examp.rossler, time_interval, x0, args=(paras,), dense_output=True, **integrator_keywords)
    elif examp_type == 'lorenz96':
        if len(paras) != 2:
            raise ValueError('Should be two parameters!')
        if len(x0) != paras[0]:
            raise ValueError('Dimentions are inconsistent!')
        # compute a continuous solution
        sol = solve_ivp(ode_examp.lorenz96, time_interval, x0, args=(paras,), dense_output=True, **integrator_keywords)
    else:
        pass

    # generata noiseless/noisy observations
    n = pts_num
    t0 = time_interval[0]
    tf = time_interval[1]
    if pts_type == 'uniform':
        T = np.linspace(t0, tf, n+1)
    elif pts_type == 'random':
        T = np.random.uniform(t0, tf, (n,))
        T.sort()
        if T[0] == t0:
            T.append(tf)
        else:
            tt = T.tolist()
            T = np.array(tt)

    X_data = sol.sol(T)    # noiseless observations (including the intial value at t[0])
    X_ns = add_noise(X_data[:,1:], nsr, ns_type)    # noisy observation
    x0 = X_data[:,0]
    x0 = x0[:, np.newaxis]
    X_ns = np.hstack((x0, X_ns))

    # compute a continuous derivative function
    if examp_type == 'lotkavolterra':
        func = lambda x, y: ode_examp.lotkavolterra(T, np.array([x,y]), paras)
        D1 = map(func, X_data[0].tolist(), X_data[1].tolist())
    elif examp_type == 'lorenz63':
        func = lambda x, y, z: ode_examp.lorenz63(T, np.array([x,y,z]), paras)
        D1 = map(func, X_data[0].tolist(), X_data[1].tolist(), X_data[2].tolist())
    elif examp_type == 'pendulum':
        func = lambda x, y: ode_examp.pendulum(T, np.array([x,y]), paras)
        D1 = map(func, X_data[0].tolist(), X_data[1].tolist())
    elif examp_type == 'rossler':
        func = lambda x, y, z: ode_examp.rossler(T, np.array([x,y,z]), paras)
        D1 = map(func, X_data[0].tolist(), X_data[1].tolist(), X_data[2].tolist())
    elif examp_type == 'lorenz96':
        func = lambda x: ode_examp.lorenz96(T, x, paras)
        d = paras[0]   # dimension of the system
        ll = []
        for i in np.arange(d):
            ll.append(X_data[i,:])
        D1 = map(func, ll)
    else:
        pass

    D1 = np.array(list(D1))    # (n+1)xd array
    if examp_type != 'lorenz96':
        Dx = D1.T
    else:
        Dx = D1

    return X_ns, X_data, T, Dx, sol

