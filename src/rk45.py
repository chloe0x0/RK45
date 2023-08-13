import numpy as np
from typing import Callable

def rk4_step(f: Callable, t: float, y: np.ndarray, h: float) -> np.ndarray:
    '''
    A single rk4 step
    '''
    k_1 = f(t, y)
    k_2 = f(t + 0.5*h, y + 0.5*h*k_1)
    k_3 = f(t + 0.5*h, y + 0.5*h*k_2)
    k_4 = f(t + h, y + h*k_3)
    y_out = y + (h/6.0)*(k_1 + 2*k_2 + 2*k_3 + k_4)
    
    return y_out

def rk45(f: Callable, t0: float, y0: np.ndarray, t_bound: float) -> np.ndarray:
    ''' Explicit Runge-Kutta integrator of order 5(4) for integrating y' = f(t,y). Aims to replicate the use of scipy.integrate.solve_ivp

    Parameters
    ----------
    f : Callable
      The right hand side of the differential equation. Must be of the form f(t,y) where t is a scalar float representing time and y is the state vector. 
      It outputs a numpy array of shape (n,)
    t0 : float
      The initial time
    y0 : np.ndarray (n,)
       Initial Values (Initial State)
    t_bound : float
       The boundary time (integration will not continue past this time)
    
    Outputs
    ----------
    trajectory : np.ndarray (n,t_bound/h)
      The Matrix representation of the systems trajectory through time.
      The ith column vector represents the state of the system at the ith time step
    '''
    # time step (delta t)
    h = 0.01
    # A single step procedure at step n:
    # need to first build the vectors k_1, k_2, k_3, k_4
    # k_1 = f(t_n,y_n)
    # k_2 = f(t_n + 0.5*h, y_n + 0.5*h*k_1)
    # k_3 = f(t_n + 0.5*h, y_n + 0.5*h*k_2)
    # k_4 = f(t_n + h, y_n + h*k_3)
    # y_{n+1} = y_n + (h/6)*(k_1 + 2*k_2 + 2*k_3 + k_4)
    n = y0.shape[0]
    # Discrete time points
    Ts = int(t_bound / h)
    t = np.linspace(t0,t_bound,Ts)
    # Matrix representation of the trajectory, each ith column vector is a state of the system y_i
    trajectory = np.zeros(shape = (n,Ts))
    y_in = y0
    # first column vector is the initial conditions 
    trajectory[:,0] = y0

    for t_i in range(Ts - 1):
        # do a single RK4 step
        step = rk4_step(f, t[t_i], y_in, h)
        # Add the state vector to the matrix
        trajectory[:,t_i+1] = step
        y_in = step

    return trajectory

