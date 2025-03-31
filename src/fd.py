import numpy as np

def compute_fd(x, t):
    """
    Vectorized computation of finite differences for given data points.
    Uses central differences for interior points and
    one-sided second-order differences for endpoints.
    
    Args:
        x: array-like, shape (n, d) where n is number of points and d is dimension
        t: array-like, shape (n,) timestamps
    
    Returns:
        derivatives: array-like, shape (n, d) finite differences
    """
    x = np.array(x)
    t = np.array(t)
    
    # Initialize output array
    der = np.zeros_like(x)
    
    # Compute time steps
    dt = np.diff(t)
    
    # Central differences for interior points (vectorized)
    der[:,1:-1] = (x[:,2:] - x[:,:-2]) / (dt[1:] + dt[:-1])
    
    # One-sided second-order difference for first point
    # (-3f₀ + 4f₁ - f₂)/(2h)
    der[:,0] = (-3*x[:,0] + 4*x[:,1] - x[:,2]) / (2*dt[0])
    
    # One-sided second-order difference for last point
    # (3fₙ - 4fₙ₋₁ + fₙ₋₂)/(2h)
    der[:, -1] = (3*x[:, -1] - 4*x[:, -2] + x[:, -3]) / (2*dt[-1])
    
    return der


def compute_l2norm(x, t):
    """
    Compute L2 norm in time (velocity error) between predicted and true trajectories.
    
    Args:
        x: array-like, shape (n, d) predicted trajectory
        t: array-like, shape (n,) timestamps
    
    Returns:
        norm: float, integrated L2 norm of derivatives over time
    """
    # Compute derivatives for both trajectories
    
    # Compute squared difference of derivatives at each timestep
    if x.ndim == 1:
        squared_x = x**2
    else:
        squared_x = np.sum(x**2, axis=0)
    
    # Integrate using trapezoidal rule
    dt = np.diff(t)

    error = sum((squared_x[1:] + squared_x[:-1])/2*dt)
    
    return np.sqrt(error)

def compute_nonuniform_fd(x, t, method='central'):
    """
    Compute finite differences on a nonuniform grid using various methods.
    
    Args:
        x: array-like, shape (n, d) where n is number of points and d is dimension
           or shape (n,) for 1D data
        t: array-like, shape (n,) timestamps (nonuniform grid points)
        method: str, method to use for finite differences
               'central' - central differences for interior points (default)
               'forward' - forward differences
               'backward' - backward differences
    
    Returns:
        derivatives: array-like, same shape as input x, containing the derivatives
    
    Notes:
        For central differences:
            Interior points use: f'(t[i]) ≈ (f[i+1] - f[i-1])/(t[i+1] - t[i-1])
            First point uses: f'(t[0]) ≈ (-3f[0] + 4f[1] - f[2])/(2(t[1] - t[0]))
            Last point uses: f'(t[-1]) ≈ (3f[-1] - 4f[-2] + f[-3])/(2(t[-1] - t[-2]))
            
        For forward differences:
            f'(t[i]) ≈ (f[i+1] - f[i])/(t[i+1] - t[i])
            
        For backward differences:
            f'(t[i]) ≈ (f[i] - f[i-1])/(t[i] - t[i-1])
    """
    x = np.asarray(x)
    t = np.asarray(t)
    
    if t.size < 2:
        raise ValueError("At least two points are required to compute derivatives")
    
    if x.shape[1] != t.size:
        raise ValueError(f"Number of time points ({t.size}) must match first dimension of x ({x.shape[0]})")
    
    # Make x 2D if it's 1D
    x_is_1d = x.ndim == 1
    if x_is_1d:
        x = x.reshape(-1, 1)
        
    der = np.zeros_like(x)
    
    if method == 'central':
        if t.size < 3:
            raise ValueError("At least three points are required for central differences")
            
        # Interior points
        dt_forward = t[2:] - t[1:-1]
        dt_backward = t[1:-1] - t[:-2]
        dt_central = dt_forward + dt_backward
        
        der[:, 1:-1] = (x[:, 2:] - x[:, :-2]) / dt_central.reshape(1, -1)
        
        # First point (second-order forward difference)
        h0 = t[1] - t[0]
        h1 = t[2] - t[1]
        der[:, 0] = (-3*x[:, 0] + 4*x[:, 1] - x[:, 2]) / (2*h0)
        
        # Last point (second-order backward difference)
        hn_1 = t[-1] - t[-2]
        hn_2 = t[-2] - t[-3]
        der[:, -1] = (3*x[:, -1] - 4*x[:, -2] + x[:, -3]) / (2*hn_1)
        
    elif method == 'forward':
        # Forward differences for all but last point
        dt = np.diff(t)
        der[:, :-1] = (x[:, 1:] - x[:, :-1]) / dt.reshape(1, -1)
        # Use backward difference for last point
        der[:, -1] = (x[:, -1] - x[:, -2]) / (t[-1] - t[-2])
        
    elif method == 'backward':
        # Backward differences for all but first point
        dt = np.diff(t)
        der[:, 1:] = (x[:, 1:] - x[:, :-1]) / dt.reshape(1, -1)
        # Use forward difference for first point
        der[:, 0] = (x[:, 1] - x[:, 0]) / (t[1] - t[0])
        
    else:
        raise ValueError(f"Unknown method: {method}. Use 'central', 'forward', or 'backward'")
    
    return der[:, 0] if x_is_1d else der