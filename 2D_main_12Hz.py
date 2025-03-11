
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
        default="dlqr",
        help="Controller type: 'pid', 'dpole' (discrete pole placement), or 'dlqr' (discrete LQR)"
    )
    # New arguments: separate simulation time step and controller sampling time
    parser.add_argument("--sim_dt", type=float, default=1/240,
                        help="Simulation time step (seconds) for ODE integration")
    parser.add_argument("--ctrl_dt", type=float, default=1/12,
                        help="Controller sampling time (seconds)")
    return parser.parse_args()

def main():
    args = parse_args()

    sim_dt = args.sim_dt    # simulation time step for integration
    ctrl_dt = args.ctrl_dt  # controller sampling time

    T = 5
    t_eval = np.linspace(0, T, int(T/sim_dt))

    # Cart-pole parameters
    M, m, l, g, b, zeta, c = 0.45, 0.12, 0.52, 9.81, 0.2, 0.0, 0.015
    params = (M, m, l, g, b, zeta, c)

    # Initial Conditions
    x0 = np.array([0.0, 0.0, np.deg2rad(5), 0.0])

    # 1) Build continuous linear system
    A, B = build_cartpole_linear_system(M, m, l, g, b, zeta, c)

    # 2) Choose/instantiate the controller based on args.controller and use ctrl_dt for the control update
    if args.controller == "pid":
        print("[INFO] Using PID Controller")



        ctrl = PIDController(kp=6, ki=0.0, kd=5, dt=ctrl_dt,max_force=1.5)



    
    elif args.controller == "dpole":
        print("[INFO] Using DISCRETE Pole Placement Controller")
        # Use a smaller initial angle for easier stabilization
        x0 = np.array([0.0, 0.0, np.deg2rad(1), 0.0])
        # Discretize the continuous system using the controller sampling time
        Ad, Bd = discretize_system(A, B, ctrl_dt)
        desired_zpoles = [ -0.5+0.1j, -0.5-0.1j, -3, -0.7 ]
        ctrl = DiscretePolePlacementController(Ad, Bd, desired_poles=desired_zpoles, dt=ctrl_dt, max_force=10.0, target_state=[0, 0, 0, 0])
    
    elif args.controller == "dlqr":
        print("[INFO] Using DISCRETE LQR Controller")
        # Discretize the continuous system using the controller sampling time
        Ad, Bd = discretize_system(A, B, ctrl_dt)
        # Define weighting matrices for discrete LQR
        Q = np.diag([0,0,10,1])
        R = np.array([[0.1]])


        ctrl = DiscreteLQRController(Ad, Bd, Q, R, dt=ctrl_dt, max_force=2.0)


    else:
        raise ValueError("Unknown controller type!")

    # 3) Define a wrapper to pass control to the cartpole dynamics.
    #    This function now uses a discrete update logic: the controller is only updated
    #    when t has advanced by one ctrl_dt interval; otherwise the last computed control is used.
    def dynamics(t, state):
        # Determine the current control update index (i.e. floor(t / ctrl_dt))
        current_index = int(np.floor(t / ctrl_dt))
        if current_index > dynamics.last_index:
            dynamics.last_index = current_index
            if args.controller in ["dpole", "dlqr"]:
                current_u = ctrl.compute_control(t, state)
            else:
                current_u = ctrl.compute_control(state)
            dynamics.current_u = current_u
        else:
            current_u = dynamics.current_u
        # Pass the control via a dummy controller to match the expected interface
        return cartpole_nonlinear_dynamics(t, state, params, pid=DummyController(current_u))

    # Initialize attributes to store the last control update index and value
    dynamics.last_index = -1
    dynamics.current_u = 0.0

    # 4) Solve the ODE with the simulation time step (using t_eval defined with sim_dt)
    sol = solve_ivp(dynamics, [0, T], x0, t_eval=t_eval)
    states = sol.y

    # 5) Optional: Add noise & filtering
    if args.noise:
        print("[INFO] Applying sensor noise & low-pass filtering")
        noisy_states = apply_noise(states, noise_level=0.05)
        cutoff_freq = 2.0
        fs = 1/sim_dt
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

    # 6) Reconstruct control effort for plotting.
    #     Here we apply the same discrete update logic to ensure the control signal is held constant between updates.
    control_effort = []
    last_index = -1
    last_u = 0.0
    for i, t_ in enumerate(t_eval):
        current_index = int(np.floor(t_ / ctrl_dt))
        if current_index > last_index:
            last_index = current_index
            if args.controller in ["dpole", "dlqr"]:
                last_u = ctrl.compute_control(t_, states[:, i])
            else:
                last_u = ctrl.compute_control(states[:, i])
        control_effort.append(last_u)

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
        animate_cartpole(t_eval, x_pos, theta, sim_dt)

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
