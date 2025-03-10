import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from utils import cartpole_nonlinear_dynamics

# Define your nonlinear state-transition function
def fx(state, dt, params, pid):
    derivatives = cartpole_nonlinear_dynamics(0, state, params, pid)
    return state + np.array(derivatives) * dt

# Measurement function: measure [x, theta] from state
def hx(state):
    return np.array([state[0], state[2]])

# UKF initialization function
def create_ukf(x0, params, pid, dt):
    points = MerweScaledSigmaPoints(4, alpha=1e-3, beta=2., kappa=0.)
    ukf = UnscentedKalmanFilter(dim_x=4, dim_z=2, dt=dt, fx=lambda x, dt: fx(x, dt, params, pid), hx=hx, points=points)

    ukf.x = x0
    ukf.P = np.eye(4) * 0.5      # Initial covariance (larger is more stable)
    ukf.Q = np.eye(4) * 1e-2     # Process noise (increase to stabilize)
    ukf.R = np.eye(2) * 0.05     # Measurement noise


    
    return ukf
