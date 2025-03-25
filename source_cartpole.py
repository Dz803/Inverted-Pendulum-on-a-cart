"""
source_combined.py

Single source file combining the contents from Controller.py and utils.py:

1) Controller classes (CartpoleController, PIDController, etc.)
2) Utility functions (cartpole_nonlinear_dynamics, build_cartpole_linear_system, etc.)
"""

import numpy as np
import cvxpy as cp  # for MPC usage
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

# For LQR, NMPC and visualization.
from scipy.linalg import solve_discrete_are, expm
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.optimize as opt


#################################################
# Controllers
#################################################

class CartpoleController(ABC):
    def __init__(self, dt_model: float = 1/100., max_force: float = 100.0,
                 state_bounds: Optional[List[tuple]] = None,
                 target_state: Optional[List[float]] = None):
        self.dt = dt_model
        self.max_force = max_force
        self.state_bounds = state_bounds or [
            (-5.0, 5.0),         # cart position
            (-10.0, 10.0),       # cart velocity
            (-np.pi/2, np.pi/2), # pole angle
            (-10.0, 10.0)        # pole angular velocity
        ]
        self.control_min = -max_force
        self.control_max = max_force
        self.prev_state = None
        self.prev_error = None
        self.target_state = target_state if target_state is not None else [0.0, 0.0, 0.0, 0.0]

    @abstractmethod
    def compute_control(self, state: List[float]) -> float:
        pass

    def set_target_state(self, target: List[float]) -> None:
        self.target_state = target

    def bound_control(self, u: float) -> float:
        return np.clip(u, self.control_min, self.control_max)

    def get_state_error(self, state: List[float]) -> List[float]:
        theta_error = state[2] - self.target_state[2]
        theta_error = (theta_error + np.pi) % (2 * np.pi) - np.pi
        return [state[0] - self.target_state[0],
                state[1] - self.target_state[1],
                theta_error,
                state[3] - self.target_state[3]]

# --- PID Controller ---
class PIDController(CartpoleController):
    def __init__(self, kp: float, ki: float, kd: float,
                 dt_model: float = 1/100., max_force: float = 100.0,
                 target_state: Optional[List[float]] = None):
        super().__init__(dt_model, max_force, target_state=target_state)
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0

    def compute_control(self, state: List[float]) -> float:
        error_vec = self.get_state_error(state)
        # Focus on the pole angle error (index 2)
        pole_angle_err = error_vec[2]
        self.integral += pole_angle_err * self.dt
        derivative = ((pole_angle_err - self.prev_error[2]) / self.dt) if self.prev_error is not None else 0.0
        u = self.kp * pole_angle_err + self.ki * self.integral + self.kd * derivative
        self.prev_error = error_vec
        return self.bound_control(u)

import numpy as np
import scipy.optimize as opt

class NMPCNonlinearControllerSimplified(CartpoleController):    
    def __init__(self, horizon=5, dt_model=0.01, max_force=100.0,
                 target_state=None, Q=None, R=None, 
                 params=None):
        super().__init__(dt_model, max_force, target_state=target_state)
        self.horizon = horizon
        self.dt_model = dt_model
        self.params = params
        self.Q = Q if Q is not None else np.diag([10, 5, 100, 5])
        self.R = R if R is not None else np.array([[0.1]])
    
    def _simulate_dynamics(self, x0, u_seq):
        dt = self.dt_model
        x_current = np.array(x0, dtype=float).copy()
        traj = [x_current.copy()]
        for u in u_seq:
            # Simulate dynamics using Euler integration
            dx = cartpole_nonlinear_dynamics(0, x_current.tolist(), self.params,
                                             controller=lambda s: u)
            x_current = x_current + dt * np.array(dx)
            traj.append(x_current.copy())
        return traj

    def _cost(self, u_seq, x0):
        traj = self._simulate_dynamics(x0, u_seq)
        cost = 0.0
        target = np.array(self.target_state)
        for k in range(len(u_seq)):
            xk = traj[k]
            uk = u_seq[k]
            # Stage cost: quadratic on state error and control effort.
            cost += (xk - target).T @ self.Q @ (xk - target) + (uk**2) * self.R[0,0]
        # Terminal cost
        x_end = traj[-1]
        cost += (x_end - target).T @ self.Q @ (x_end - target)
        return cost

    def compute_control(self, state: list) -> float:
        u0 = np.zeros(self.horizon)
        
        res = opt.minimize(lambda u: self._cost(u, np.array(state)), u0,
                           method='SLSQP',
                           bounds=[(-self.max_force, self.max_force)] * self.horizon,
                           options={'maxiter': 100, 'ftol': 1e-3, 'disp': False})
        if not res.success:
            return 0.0
        # Return the first control input from the optimal sequence.
        u_opt = res.x[0]
        return self.bound_control(u_opt)

# --- LQR Controller ---
class LQRController(CartpoleController):
    def __init__(self, Ad: np.ndarray, Bd: np.ndarray, Q: np.ndarray, R: np.ndarray,
                 dt_model: float = 1/100., max_force: float = 100, target_state: Optional[List[float]] = None,
                 scale_cost: bool = False):
        super().__init__(dt_model, max_force, target_state=target_state)
        self.Ad = Ad
        self.Bd = Bd
        self.dt = dt_model
        self.target_state = np.array(target_state if target_state is not None else [0, 0, 0, 0])
        if not is_controllable(Ad, Bd):
            raise ValueError("The pair (Ad, Bd) is not controllable. LQR design will fail.")
        Qd = Q * dt_model if scale_cost else Q
        Rd = R * dt_model if scale_cost else R
        P = solve_discrete_are(Ad, Bd, Qd, Rd)
        self.K = np.linalg.inv(Bd.T @ P @ Bd + Rd) @ (Bd.T @ P @ Ad)
        self.last_update_time = 0.0
        self.u_current = 0.0
        eigvals = np.linalg.eig(Ad - Bd @ self.K)[0]
        print("Discrete LQR closed-loop eigenvalues (magnitudes):", np.abs(eigvals))

    def get_state_error(self, state: List[float]) -> np.ndarray:
        theta_err = (state[2] - self.target_state[2] + np.pi) % (2*np.pi) - np.pi
        return np.array([state[0] - self.target_state[0],
                         state[1] - self.target_state[1],
                         theta_err,
                         state[3] - self.target_state[3]])

    def compute_control(self, t, state) -> float:
        if t - self.last_update_time >= self.dt:
            x_err = self.get_state_error(state).reshape(-1, 1)
            u = -(self.K @ x_err)
            self.u_current = np.clip(u.item(), -self.max_force, self.max_force)
            self.last_update_time = t
        return self.u_current

    
#################################################
# Utility Functions
#################################################

def cartpole_nonlinear_dynamics(t, state, params, controller=None):
    """
    Nonlinear EoM for cart + uniform rod + point mass at tip.
    We'll compute l_c, I_p, I_a from (m, m_a, l).
    
    params = (M, m, m_a, l, g, b_damp, c_damp)
    """
    x, x_dot, theta, theta_dot = state
    
    M, m, m_a, l, g, b_damp, c_damp = params
    
    # compute center-of-mass, inertia, etc.
    l_c = l / 2.0
    I_p = (1.0 / 3.0) * m * (l**2)
    I_a = m_a * (l**2)
    
    M_tot = M + m + m_a
    B = m*l_c + m_a*l
    C = m*(l_c**2) + m_a*(l**2) + I_p + I_a
    D = m*g*l_c + m_a*g*l
    
    # Force from controller
    F = 0.0 if (controller is None) else controller(state)

    sin_th = np.sin(theta)
    cos_th = np.cos(theta)

    lhs = np.array([
        [M_tot,  B*cos_th],
        [B*cos_th, C]
    ])
    rhs = np.array([
        F - b_damp*x_dot + B*(theta_dot**2)*sin_th,
        -c_damp*theta_dot + D*sin_th
    ])
    ddx, ddtheta = np.linalg.solve(lhs, rhs)
    return [x_dot, ddx, theta_dot, ddtheta]

def build_cartpole_linear_system(M, m, m_a, l, g, b_damp, c_damp):
    # Compute geometric and inertial parameters
    l_c = l / 2.0
    I_p = (1.0 / 3.0) * m * (l**2)
    I_a = m_a * (l**2)
    
    M_tot = M + m + m_a
    B_val = m * l_c + m_a * l
    C_val = m * (l_c**2) + m_a * (l**2) + I_p + I_a
    D_val = m * g * l_c + m_a * g * l
    
    # Denominator for the coupled equations
    Delta = M_tot * C_val - B_val**2
    
    # Initialize A and B matrices for state: [x, x_dot, theta, theta_dot]
    A = np.zeros((4,4))
    Bmat = np.zeros((4,1))
    
    # x_dot dynamics
    A[0,1] = 1.0
    
    # Linearized ẍ equation:
    # ẍ = (C_val*(F - b_damp*x_dot) + B_val*(c_damp*theta_dot - D_val*theta)) / Delta
    A[1,1] = - (C_val * b_damp) / Delta
    A[1,2] = - (B_val * D_val) / Delta   # effect of theta
    A[1,3] = (B_val * c_damp) / Delta      # effect of theta_dot
    
    # theta_dot dynamics
    A[2,3] = 1.0
    
    # Linearized thetä equation:
    # thetä = (-B_val*(F - b_damp*x_dot) + M_tot*(- c_damp*theta_dot + D_val*theta)) / Delta
    A[3,1] = (B_val * b_damp) / Delta
    A[3,2] = (M_tot * D_val) / Delta       # effect of theta
    A[3,3] = - (M_tot * c_damp) / Delta      # effect of theta_dot
    
    # The control input enters both equations:
    Bmat[1,0] = C_val / Delta      # for ẍ
    Bmat[3,0] = - B_val / Delta      # for thetä
    
    return A, Bmat

def discretize_system(Ac, Bc, dt):
    """
    Zero-Order Hold (ZOH) discretization.
    [Ad, Bd] = exp( [Ac, Bc; 0, 0] * dt )
    """
    n = Ac.shape[0]
    r = Bc.shape[1]
    M = np.block([
        [Ac, Bc],
        [np.zeros((r,n)), np.zeros((r,r))]
    ])
    Md = expm(M*dt)
    Ad = Md[:n, :n]
    Bd = Md[:n, n:]
    return Ad, Bd

def low_pass_filter(signal, cutoff_freq, fs, order=3):
    nyquist = 0.5*fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)


def compute_settling_time(signal: np.ndarray, time: np.ndarray, threshold: float) -> Optional[float]:
    """
    Returns the earliest time when |signal| remains below threshold for the rest of the simulation.
    """
    for i in range(len(signal)):
        if np.all(np.abs(signal[i:]) < threshold):
            return time[i]
    return None

def is_controllable(A: np.ndarray, B: np.ndarray) -> bool:
    """
    Returns True if the pair (A, B) is controllable.
    """
    n = A.shape[0]
    C = B
    for i in range(1, n):
        C = np.hstack((C, np.linalg.matrix_power(A, i) @ B))
    rank_C = np.linalg.matrix_rank(C)
    return rank_C == n