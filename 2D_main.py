#!/usr/bin/env python3
# 2D_main_controller_option.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import argparse

# Import controllers from your 'controller.py'
from controller import (
    PIDController, 
    PolePlacementController,
    DiscretePolePlacementController,
    DiscreteLQRController   # newly added discrete LQR
)
from utils import (
    build_cartpole_linear_system,
    discretize_system,
    apply_noise,
    compute_settling_time,
    cartpole_nonlinear_dynamics,
    plot_static_results,
    animate_cartpole,
    low_pass_filter
)

def parse_args():
    parser = argparse.ArgumentParser(description="2D Cart-Pole with multiple controller options")
    parser.add_argument("--noise", action="store_true", help="Enable sensor noise & filtering")
    parser.add_argument("--animate", action="store_true", help="Enable cart-pole animation")
    # New argument: choose controller
    parser.add_argument(
        "--controller", 
        type=str,
        choices=["pid", "cpole", "dpole", "lqr", "dlqr"],
        default="pid",
        help="Controller type: 'pid', 'dpole' (discrete pole placement),  or 'dlqr' (discrete LQR)"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    dt = 1/240
    T = 5
    t_eval = np.linspace(0, T, int(T/dt))

    # Cart-pole parameters
    M, m, l, g, b, zeta, c = 0.4, 0.15, 0.5, 9.81, 0.1, 0.0, 0.015
    params = (M, m, l, g, b, zeta, c)

    # Initial Conditions
    x0 = np.array([0.0, 0.0, np.deg2rad(5), 0.0])

    # 1) Build continuous linear system
    A, B = build_cartpole_linear_system(M, m, l, g, b, zeta, c)

    # 2) Choose/instantiate the controller based on args.controller
    if args.controller == "pid":
        print("[INFO] Using PID Controller")
        ctrl = PIDController(kp=4.9, ki=0.05, kd=0.45, dt=dt)
    

    elif args.controller == "dpole":
        print("[INFO] Using DISCRETE Pole Placement Controller")
        x0 = np.array([0.0, 0.0, np.deg2rad(1), 0.0]) #smaller initial angle easier to stabilize
        Ad, Bd = discretize_system(A, B, dt)
        desired_zpoles = [ -0.5+0.1j, -0.5-0.1j, -3, -0.7 ]
        ctrl = DiscretePolePlacementController(Ad, Bd, desired_poles=desired_zpoles, dt=dt, max_force=10.0, target_state=[0, 0, 0, 0])
    
    elif args.controller == "dlqr":
        print("[INFO] Using DISCRETE LQR Controller")
        # Discretize the continuous system
        Ad, Bd = discretize_system(A, B, dt)
        # Define weighting matrices for discrete LQR
        Q = np.diag([0,1.2,2,1.5])
        R = np.array([[1e10]])
        ctrl = DiscreteLQRController(Ad, Bd, Q, R, dt=dt, max_force=5.0, target_state=[0, 0, 0, 0])
    else:
        raise ValueError("Unknown controller type!")

    # 3) Define a wrapper to pass control to the cartpole dynamics.
    def dynamics(t, state):
        # For discrete controllers, pass time as well
        if args.controller in ["dpole", "dlqr"]:
            u = ctrl.compute_control(t, state)
        else:
            u = ctrl.compute_control(state)
        # Use a dummy controller to meet the expected interface in cartpole_nonlinear_dynamics
        return cartpole_nonlinear_dynamics(t, state, params, pid=DummyController(u))

    # 4) Solve the ODE
    sol = solve_ivp(dynamics, [0, T], x0, t_eval=t_eval)
    states = sol.y

    # 5) Optional: Add noise & filtering
    if args.noise:
        print("[INFO] Applying sensor noise & low-pass filtering")
        noisy_states = apply_noise(states, noise_level=0.05)
        cutoff_freq = 1.0
        fs = 1/dt
        filtered_x = low_pass_filter(noisy_states[0], cutoff_freq, fs)
        filtered_theta = low_pass_filter(noisy_states[2], cutoff_freq, fs)

        fig, axs = plt.subplots(2, 1, figsize=(10,8))
        axs[0].plot(t_eval, noisy_states[0], alpha=0.5, label="Noisy x")
        axs[0].plot(t_eval, filtered_x, label="Filtered x")
        axs[0].legend(); axs[0].grid()

        axs[1].plot(t_eval, np.rad2deg(noisy_states[2]), alpha=0.5, label="Noisy theta (deg)")
        axs[1].plot(t_eval, np.rad2deg(filtered_theta), label="Filtered theta (deg)")
        axs[1].legend(); axs[1].grid()

        plt.tight_layout()
        plt.show()

        x_pos = filtered_x
        theta = filtered_theta
    else:
        x_pos = states[0]
        theta = states[2]

    # 6) Reconstruct control effort for plotting
    control_effort = []
    for i, t_ in enumerate(t_eval):
        st_ = states[:, i]
        if args.controller in ["dpole", "dlqr"]:
            u_ = ctrl.compute_control(t_, st_)
        else:
            u_ = ctrl.compute_control(st_)
        control_effort.append(u_)

    # 7) Compute settling time
    threshold = 0.05 * abs(x0[2])
    settling_time = compute_settling_time(np.abs(theta), t_eval, threshold)
    if settling_time:
        print(f"[RESULT] Settling time: {settling_time:.3f} s")
    else:
        print("[RESULT] System did not settle within threshold")

    # 8) Plot results
    plot_static_results(t_eval, x_pos, theta, control_effort)

    # 9) Animate if requested
    if args.animate:
        animate_cartpole(t_eval, x_pos, theta, dt)

class DummyController:
    """
    A dummy class to wrap a computed force so that cartpole_nonlinear_dynamics()
    can call compute_control() on it.
    """
    def __init__(self, force):
        self.force = force

    def compute_control(self, s):
        return self.force

if __name__ == "__main__":
    main()
