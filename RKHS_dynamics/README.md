# TODO list


## 1. Generate data

define d-dim ODE system (initial value, parameter, total time, total steps for observation);

solve ODE with Scipy;

plot 2D/3D trajectory/Phase diagram;

plot curve of (t, x_i), and (x_i, x_j) 2-dim obits;

plot scatter diagram (maybe with noise), and save noise data;


## 2. Fitting derivative function

define a continuous kernel function

define two functions for constructing Gram matrix and data-adaptive basis functions

compute coefficient vector and values of data-adaptive functions 


## 3. Improve efficiency w.r.t. numerical integration

1. Consider a kernel that can write its analytic-form antiderivative,
e.g. Wendland kernel (one-one corresponds to Sobelev H^{s})

2. For Gaussian kernel, using the error function to get the (semi-) analytic form of the integral.

3. Matern kernel, how to do efficient integral computation?

4. New framework for fitting the derivative or the trajectory? 


## 4. Reconstructing vector field by vRKHS or polynomial basis

1. vRKHS: recons_dynam.py  (sensentive to the choice of kernel and superparameters)





