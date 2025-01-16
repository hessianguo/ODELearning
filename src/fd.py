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