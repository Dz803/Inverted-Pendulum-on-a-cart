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

def cartpole_nonlinear_dynamics(t, state, params, pid):
    x, x_dot, theta, theta_dot = state
    M, m, l, g, b, zeta, c = params

    u = pid.compute_control(state)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    denom = M + m * sin_theta**2

    x_ddot = (
        -m * g * cos_theta * sin_theta
        + theta_dot * zeta * cos_theta
        + m * l * theta_dot**2 * sin_theta
        - b * x_dot
        + u
    ) / denom

    theta_ddot = (
        (M + m) * g * sin_theta
        - m * theta_dot**2 * sin_theta * cos_theta
        - zeta * cos_theta * x_dot / l
        - c * (M + m) * theta_dot / (m * l**2)
        - cos_theta * u / l
    ) / (denom * l)

    return [x_dot, x_ddot, theta_dot, theta_ddot]


def build_cartpole_linear_system(M=0.28, m=0.075, l=0.32, g=9.81, b=0.04, zeta=0.0, c=0.015):
    A = np.array([
        [0,           1,                0,                      0],
        [0,   -b/M,        -(m*g)/M,               zeta/M    ],
        [0,           0,                0,                      1],
        [0,  b/(l*M),  (M + m)*g/(l*M),  -(c*(M + m))/(m*l**2*M)]
    ])
    B = np.array([
        [0],
        [-1/M],
        [0],
        [-1/(l*M)]
    ])
    return A, B

def discretize_system(Ac, Bc, dt):
    n = Ac.shape[0]
    M = np.block([
        [Ac, Bc],
        [np.zeros((1, n+1))]
    ])
    Md = expm(M * dt)
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


def plot_static_results(time, x_pos, theta, control_effort):
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    axs[0].plot(time, x_pos, label="Cart Position (m)")
    axs[0].axhline(0, linestyle="--", color="gray", label="Target / Origin")
    axs[0].set_ylabel("Position (m)")
    axs[0].set_title("Cart Position")
    axs[0].legend()

    axs[1].plot(time, theta, label="Pendulum Angle (rad)")
    axs[1].axhline(0, linestyle="--", color="gray", label="Upright")
    axs[1].set_ylabel("Angle (rad)")
    axs[1].set_title("Pendulum Angle")
    axs[1].legend()

    axs[2].plot(time, control_effort, label="Control Effort (N)", color='red')
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Force (N)")
    axs[2].set_title("Control Effort")
    axs[2].legend()

    plt.tight_layout()
    plt.show()


def animate_cartpole(time, x_pos, theta, dt, cart_width=0.3, cart_height=0.15, pendulum_length=0.5):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2, 2)
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title("Cart-Pole Animation")
    
    # Initialize cart and pendulum
    cart = plt.Rectangle((-cart_width/2, -cart_height/2), cart_width, cart_height, color='blue')
    ax.add_patch(cart)
    pendulum_line, = ax.plot([], [], 'ro-', lw=3)
    
    def init():
        pendulum_line.set_data([], [])
        cart.set_xy((-cart_width/2, -cart_height/2))
        return cart, pendulum_line
    
    def update(frame):
        # Update cart position
        cart_x = x_pos[frame] - cart_width / 2
        cart.set_xy((cart_x, -cart_height / 2))
        # Update pendulum position
        pendulum_x = [x_pos[frame], x_pos[frame] + pendulum_length * np.sin(theta[frame])]
        pendulum_y = [0, -pendulum_length * np.cos(theta[frame])]
        pendulum_line.set_data(pendulum_x, pendulum_y)
        return cart, pendulum_line
    
    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=range(len(time)), init_func=init, blit=True, interval=dt*1000
    )
    plt.show()

    def update(frame):
        x = x_pos[frame]
        theta_rad = theta[frame]

        cart.set_xy((x - cart_width/2, -cart_height/2))
        pendulum_x = x + pendulum_length * np.sin(theta_rad)
        pendulum_y = pendulum_length * np.cos(theta_rad)
        pendulum_line.set_data([x, pendulum_x], [0, pendulum_y])

        return cart, pendulum_line

    ani = FuncAnimation(fig, update, frames=len(time), interval=dt * 1000, blit=True)
    plt.show()


