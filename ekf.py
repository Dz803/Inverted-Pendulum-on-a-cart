#################################
#EKF
#################################
import numpy as np
from controller import PIDController
from utils import cartpole_nonlinear_dynamics
from scipy.integrate import solve_ivp




def ekf_dynamics(state, u, params, dt, pid):
    # Compute derivatives from provided nonlinear dynamics
    derivatives = cartpole_nonlinear_dynamics(0, state, params, pid)
    # Euler integration step
    next_state = state + np.array(derivatives) * dt
    return next_state

def jacobian_f(state, u, params, dt, pid, epsilon=1e-5):
    n = len(state)
    A = np.zeros((n, n))
    f0 = ekf_dynamics(state, u, params, dt, pid)
    
    for i in range(n):
        state_perturbed = state.copy()
        state_perturbed[i] += epsilon
        f_perturbed = ekf_dynamics(state_perturbed, u, params, dt, pid)
        A[:, i] = (f_perturbed - f0) / epsilon
    return A





class ExtendedKalmanFilter:
    def __init__(self, x0, params, dt, pid):
        self.x_hat = x0
        self.P = np.eye(4) * 0.01
        self.params = params
        self.dt = dt
        self.pid = pid
        self.Q = np.eye(4)  # Process noise
        self.R = np.diag([0.05, 0.05]) ** 2  # Measurement noise covariance

    def predict(self):
        u = self.pid.compute_control(self.x_hat)
        self.x_hat = ekf_dynamics(self.x_hat, u, self.params, self.dt, self.pid)
        A = jacobian_f(self.x_hat, u, self.params, self.dt, self.pid)
        self.P = A @ self.P @ A.T + self.Q

    def update(self, z):
        # Measurement matrix H (measurement of x, theta only)
        H = np.array([[1, 0, 0, 0],
                      [0, 0, 1, 0]])

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        y = z - H @ self.x_hat

        self.x_hat += K @ y
        self.P = (np.eye(len(self.x_hat)) - K @ H) @ self.P
