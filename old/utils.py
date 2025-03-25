# utils.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation
from scipy.linalg import expm
from typing import Optional
from scipy.signal import butter, filtfilt



#################################################
#System Dynamics
#################################################

def cartpole_nonlinear_dynamics(t, state, params, controller=None):
    # state = [x, x_dot, theta, theta_dot]
    x, x_dot, theta, theta_dot = state
    M, m, m_a, l, g, b_damp, c_damp = params

    # compute center-of-mass, inertia, etc.
    l_c = l / 2
    I_p = (1.0 / 3.0) * m * (l**2)
    I_a = m_a * (l**2)
    
    # For convenience, define
    M_tot = M + m + m_a
    B = m*l_c + m_a*l
    C = m*(l_c**2) + m_a*(l**2) + I_p + I_a
    D = m*g*l_c + m_a*g*l
    
    # Force from controller or zero if none
    F = 0.0 if (controller is None) else controller(state)

    sin_th = np.sin(theta)
    cos_th = np.cos(theta)

    # Build 2x2 LHS
    lhs = np.array([
        [M_tot,  B*cos_th],
        [B*cos_th, C]
    ])

    # Build RHS
    rhs = np.array([
        F - b_damp*x_dot + B*(theta_dot**2)*sin_th,
        - c_damp*theta_dot + D*sin_th
    ])

    # Solve for ddx, ddtheta
    ddx, ddtheta = np.linalg.solve(lhs, rhs)

    return [x_dot, ddx, theta_dot, ddtheta]


def build_cartpole_linear_system(M, m, m_a, l, g, b_damp, c_damp):
    """
    Linearized system about x=0, dx=0, theta=0, dtheta=0.
    """
    # Automatically compute I_p, I_a for standard rod + point mass at tip
    I_p = (1.0 / 3.0) * m * (l ** 2)
    I_a = m_a * (l ** 2)
    l_c = l / 2
    
    M_tot = M + m + m_a
    B_ = m*l_c + m_a*l
    C_ = (m*l_c**2 + m_a*l**2 + I_p + I_a)
    D_ = (m*g*l_c + m_a*g*l)

    A = np.zeros((4, 4))
    Bmat = np.zeros((4, 1))

    # x_dot = x_dot
    A[0, 1] = 1.0

    # For the cart's linear eq: M_tot*ddx + B_*ddtheta = F - b_damp*x_dot
    # => x_ddot depends on x_dot, F, theta, etc.
    A[1, 1] = -b_damp / M_tot
    A[1, 2] = -D_ / M_tot  # small-angle torque from the pendulum

    # theta_dot = theta_dot
    A[2, 3] = 1.0

    # For pendulum eq: C_*ddtheta + B_*ddx - D_*theta = - c_damp*theta_dot
    # => ddtheta depends on theta, theta_dot, ddx
    A[3, 2] =  D_ / C_
    A[3, 3] = -c_damp / C_

    # Input F enters eq(1): ddx ~ F / M_tot
    Bmat[1, 0] = 1.0 / M_tot

    return A, Bmat

def discretize_system(Ac, Bc, dt):
    """
    Zero-Order Hold (ZOH) discretization via exact matrix exponential.
    [Ad, Bd] = exp( [Ac Bc; 0 0]*dt )
    """
    n = Ac.shape[0]
    r = Bc.shape[1]

    M = np.block([
        [Ac, Bc],
        [np.zeros((r, n)), np.zeros((r, r))]
    ])
    Md = expm(M*dt)
    Ad = Md[:n, :n]
    Bd = Md[:n, n:]
    return Ad, Bd

#################################################
#Noise and Filtering
#################################################

def apply_noise(states, noise_level=0.05):
    """
    states: shape (4, time_steps) -> [x, x_dot, theta, theta_dot]
    noise_percent: 0.05 -> 5%
    returns: noisy_states of the same shape
    """
    noisy_states = np.copy(states)
    # For each state variable, define a typical scale or compute from states
    # Suppose we just do 5% of (max-min) range or 5% of absolute value
    # Example: fixed scale approach
    scale_x = 0.6   # 1 m typical range
    scale_theta = 0.15  # ~0.2 rad range
    scale_xdot = 1.0
    scale_thetadot = 1.0

    # shape (4, time_steps)
    noisy_states[0, :] += np.random.normal(0, noise_level * scale_x, states.shape[1])
    noisy_states[1, :] += np.random.normal(0, noise_level * scale_xdot, states.shape[1])
    noisy_states[2, :] += np.random.normal(0, noise_level * scale_theta, states.shape[1])
    noisy_states[3, :] += np.random.normal(0, noise_level * scale_thetadot, states.shape[1])

    return noisy_states


def low_pass_filter(signal, cutoff_freq, fs, order=3):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)


#################################################
#Stats, Plotting, and Animation
#################################################

def compute_settling_time(signal: np.ndarray, time: np.ndarray, threshold: float) -> Optional[float]:
    for i in range(len(signal)):
        if np.all(np.abs(signal[i:]) < threshold):
            return time[i]
    return None


