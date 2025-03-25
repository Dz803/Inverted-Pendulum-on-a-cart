#  /usr/bin/env python3
# interactive_cartpole_controller.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.integrate import solve_ivp
import argparse
from matplotlib.gridspec import GridSpec
from scipy import linalg

# Import controllers from your 'controller.py'
from Controller import (
    PIDController,
    ContinuousLQRController,
    DiscretePolePlacementController,
    DiscreteLQRController
)
from old.utils import (
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
    parser = argparse.ArgumentParser(
        description="2D Cart-Pole with multiple controller options and interactive parameter tuning"
    )
    parser.add_argument("--noise", action="store_true", help="Enable sensor noise & filtering")
    parser.add_argument("--animate", action="store_true", help="Enable cart-pole animation")
    parser.add_argument(
        "--controller",
        type=str,
        choices=["pid", "cpole", "dpole", "lqr", "dlqr"],
        default="pid",
        help="Controller type: 'pid', 'dpole', 'lqr' (continuous LQR), 'dlqr' (discrete LQR)"
    )
    return parser.parse_args()

class InteractiveCartpoleSimulation:
    def __init__(self, args):
        self.args = args
        self.dt = 1/240
        self.T = 10
        self.t_eval = np.linspace(0, self.T, int(self.T/self.dt))
        
        # Default Cart-pole parameters
        self.M = 0.4     # Cart mass
        self.m = 0.01    # Pendulum rod mass
        self.m_a = 0.05  # Attachment mass at rod tip
        self.l_c = 0.25  # Rod center-of-mass distance from pivot
        self.l = 0.5     # Full rod length (pivot to tip)
        self.g = 9.81
        self.b = 0.1     # Cart damping
        self.c = 0.015   # Pendulum damping
        # Moments of inertia about the pivot
        # For a uniform rod pivoted at the top: I_p = (1/3)*m*(rod length)^2
        self.I_p = (1.0 / 3.0) * self.m * (self.l**2)
        # For an attachment treated as point mass at distance l (the rod tip): I_a = m_a*l^2
        self.I_a = self.m_a * (self.l**2)
        
        # Initial conditions: [x, x_dot, theta, theta_dot]
        self.x0 = np.array([0.0, 0.0, np.deg2rad(5), 0.0])
        
        # Store simulation results for optional animation
        self.last_x_pos = None
        self.last_theta = None
        self.settling_time = None
        
        # Build the GUI and run initial simulation
        self.setup_gui()
        self.update(None)
        
        plt.tight_layout(rect=[0, 0.3, 1, 1])  # Adjust layout for sliders
        plt.show()
    
    def setup_gui(self):
        """Constructs the GUI layout: sliders, buttons, etc."""
        self.fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[3, 1], figure=self.fig)
        gs.update(top=0.9, bottom=0.35)
        
        # Axes for plots
        self.ax1 = self.fig.add_subplot(gs[0, 0])  # Cart position
        self.ax2 = self.fig.add_subplot(gs[1, 0])  # Pendulum angle
        self.ax3 = self.fig.add_subplot(gs[2, 0])  # Control effort
        self.pole_ax = self.fig.add_subplot(gs[:, 1])  # Poles / entire column
        
        # Slider geometry
        slider_height = 0.02
        slider_width = 0.30
        slider_left_col_x = 0.1
        slider_baseline_y = 0.25
        
        # System parameter sliders
        self.slider_ax_M = plt.axes([slider_left_col_x, slider_baseline_y, slider_width, slider_height])
        self.slider_ax_m = plt.axes([slider_left_col_x, slider_baseline_y - 0.05, slider_width, slider_height])
        self.slider_ax_l = plt.axes([slider_left_col_x, slider_baseline_y - 0.10, slider_width, slider_height])
        
        self.slider_M = Slider(self.slider_ax_M, 'Cart Mass (M)', 0.1, 2.0, valinit=self.M)
        self.slider_m = Slider(self.slider_ax_m, 'Pendulum Mass (m)', 0.05, 0.5, valinit=self.m)
        self.slider_l = Slider(self.slider_ax_l, 'Pendulum Length (l)', 0.1, 1.0, valinit=self.l)
        
        # Controller parameter sliders
        slider_right_col_x = 0.6
        self.controller_sliders = []
        
        if self.args.controller == "pid":
            self.slider_ax_kp = plt.axes([slider_right_col_x, slider_baseline_y, slider_width, slider_height])
            self.slider_ax_ki = plt.axes([slider_right_col_x, slider_baseline_y - 0.05, slider_width, slider_height])
            self.slider_ax_kd = plt.axes([slider_right_col_x, slider_baseline_y - 0.10, slider_width, slider_height])
            
            self.slider_kp = Slider(self.slider_ax_kp, 'Kp', 0.1, 10.0, valinit=4.9)
            self.slider_ki = Slider(self.slider_ax_ki, 'Ki', 0.0, 0.5, valinit=0.05)
            self.slider_kd = Slider(self.slider_ax_kd, 'Kd', 0.0, 2.0, valinit=0.45)
            
            self.controller_sliders = [self.slider_kp, self.slider_ki, self.slider_kd]
            
        elif self.args.controller == "lqr":
            self.slider_ax_q1 = plt.axes([slider_right_col_x, slider_baseline_y, slider_width, slider_height])
            self.slider_ax_q2 = plt.axes([slider_right_col_x, slider_baseline_y - 0.05, slider_width, slider_height])
            self.slider_ax_q3 = plt.axes([slider_right_col_x, slider_baseline_y - 0.10, slider_width, slider_height])
            self.slider_ax_q4 = plt.axes([slider_right_col_x, slider_baseline_y - 0.15, slider_width, slider_height])
            self.slider_ax_r  = plt.axes([slider_right_col_x, slider_baseline_y - 0.20, slider_width, slider_height])
            
            self.slider_q1 = Slider(self.slider_ax_q1, 'Q1 (x)', 0.1, 5.0, valinit=1.0)
            self.slider_q2 = Slider(self.slider_ax_q2, 'Q2 (ẋ)', 0.1, 5.0, valinit=0.5)
            self.slider_q3 = Slider(self.slider_ax_q3, 'Q3 (θ)', 0.1, 10.0, valinit=5.0)
            self.slider_q4 = Slider(self.slider_ax_q4, 'Q4 (θ̇)', 0.1, 5.0, valinit=0.8)
            self.slider_r  = Slider(self.slider_ax_r,  'R',      0.1, 5.0,  valinit=0.5)
            
            self.controller_sliders = [self.slider_q1, self.slider_q2, self.slider_q3, self.slider_q4, self.slider_r]
            
        elif self.args.controller == "dlqr":
            self.slider_ax_q1 = plt.axes([slider_right_col_x, slider_baseline_y, slider_width, slider_height])
            self.slider_ax_q2 = plt.axes([slider_right_col_x, slider_baseline_y - 0.05, slider_width, slider_height])
            self.slider_ax_q3 = plt.axes([slider_right_col_x, slider_baseline_y - 0.10, slider_width, slider_height])
            self.slider_ax_q4 = plt.axes([slider_right_col_x, slider_baseline_y - 0.15, slider_width, slider_height])
            self.slider_ax_r  = plt.axes([slider_right_col_x, slider_baseline_y - 0.20, slider_width, slider_height])
            
            self.slider_q1 = Slider(self.slider_ax_q1, 'Q1 (x)', 0.1, 5.0, valinit=1.0)
            self.slider_q2 = Slider(self.slider_ax_q2, 'Q2 (ẋ)', 0.1, 5.0, valinit=0.8)
            self.slider_q3 = Slider(self.slider_ax_q3, 'Q3 (θ)', 0.1, 5.0, valinit=0.7)
            self.slider_q4 = Slider(self.slider_ax_q4, 'Q4 (θ̇)', 0.1, 5.0, valinit=0.5)
            self.slider_r  = Slider(self.slider_ax_r,  'R',      1.0, 20.0, valinit=10.0)
            
            self.controller_sliders = [self.slider_q1, self.slider_q2, self.slider_q3, self.slider_q4, self.slider_r]
            
        elif self.args.controller == "dpole":
            self.slider_ax_p1_real = plt.axes([slider_right_col_x, slider_baseline_y, slider_width, slider_height])
            self.slider_ax_p1_imag = plt.axes([slider_right_col_x, slider_baseline_y - 0.05, slider_width, slider_height])
            self.slider_ax_p3      = plt.axes([slider_right_col_x, slider_baseline_y - 0.10, slider_width, slider_height])
            self.slider_ax_p4      = plt.axes([slider_right_col_x, slider_baseline_y - 0.15, slider_width, slider_height])
            
            self.slider_p1_real = Slider(self.slider_ax_p1_real, 'Pole 1 (real)', -3.0, -0.1, valinit=-0.5)
            self.slider_p1_imag = Slider(self.slider_ax_p1_imag, 'Pole 1 (imag)',  0.0,  1.0, valinit=0.1)
            self.slider_p3      = Slider(self.slider_ax_p3,      'Pole 3',        -5.0, -0.5, valinit=-3.0)
            self.slider_p4      = Slider(self.slider_ax_p4,      'Pole 4',        -3.0, -0.1, valinit=-0.7)
            
            self.controller_sliders = [self.slider_p1_real, self.slider_p1_imag, self.slider_p3, self.slider_p4]
        
        # Simulate & Animate buttons
        self.button_ax = plt.axes([0.35, 0.05, 0.1, 0.04])
        self.button = Button(self.button_ax, 'Simulate')
        self.anim_button_ax = plt.axes([0.55, 0.05, 0.1, 0.04])
        self.anim_button = Button(self.anim_button_ax, 'Animate')
        
        # Settling time display
        self.settling_time_ax = plt.axes([0.45, 0.12, 0.1, 0.04])
        self.settling_time_ax.set_axis_off()
        self.settling_time_text = self.settling_time_ax.text(
            0.5, 0.5, 'Settling Time: N/A',
            ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5)
        )
        
        # Section labels
        plt.figtext(0.1, slider_baseline_y + 0.05, "System Parameters", fontsize=12, weight='bold')
        plt.figtext(0.6, slider_baseline_y + 0.05, "Controller Parameters", fontsize=12, weight='bold')
        
        # Slider event handlers
        self.slider_M.on_changed(self.update)
        self.slider_m.on_changed(self.update)
        self.slider_l.on_changed(self.update)
        for sld in self.controller_sliders:
            sld.on_changed(self.update)
        
        # Button event handlers
        self.button.on_clicked(self.update)
        self.anim_button.on_clicked(self.animate_last_sim)
        
        # Title
        controller_names = {
            "pid": "PID Controller",
            "lqr": "Continuous LQR Controller",
            "dlqr": "Discrete LQR Controller",
            "dpole": "Discrete Pole Placement Controller"
        }
        self.fig.suptitle(
            f"Cart-Pole System with {controller_names.get(self.args.controller, 'Unknown')} - Interactive Tuning",
            fontsize=16
        )
    
    def create_controller(self):
        """
        Build the linear system with updated mass/length from sliders,
        then instantiate the selected controller type.
        """
        M = self.slider_M.val
        m = self.slider_m.val
        l = self.slider_l.val
        
        # We'll keep m_a, I_p, etc. from self.* 
        # unless you want to recalc I_p, I_a whenever l changes.
        
        A, B = build_cartpole_linear_system(
            M, m, self.m_a,
            self.l_c,
            l,       # passing 'l' as pivot->attachment distance or rod length
            self.I_p,
            self.I_a,
            self.g,
            self.b,
            self.c
        )
        
        # Store system for pole plots
        self.A = A
        self.B = B
        
        # Controller
        if self.args.controller == "pid":
            kp = self.slider_kp.val
            ki = self.slider_ki.val
            kd = self.slider_kd.val
            return PIDController(kp=kp, ki=ki, kd=kd, dt=self.dt)
        
        elif self.args.controller == "lqr":
            q1 = self.slider_q1.val
            q2 = self.slider_q2.val
            q3 = self.slider_q3.val
            q4 = self.slider_q4.val
            r  = self.slider_r.val
            
            Q = np.diag([q1, q2, q3, q4])
            R = np.array([[r]])
            ctrl = ContinuousLQRController(A, B, Q, R, max_force=8.0)
            self.K = ctrl.K  # store for pole plots
            return ctrl
        
        elif self.args.controller == "dlqr":
            q1 = self.slider_q1.val
            q2 = self.slider_q2.val
            q3 = self.slider_q3.val
            q4 = self.slider_q4.val
            r  = self.slider_r.val
            
            Q = np.diag([q1, q2, q3, q4])
            R = np.array([[r]])
            
            Ad, Bd = discretize_system(A, B, self.dt)
            self.Ad = Ad
            self.Bd = Bd
            
            ctrl = DiscreteLQRController(Ad, Bd, Q, R, dt=self.dt, max_force=4, target_state=[0, 0, 0, 0])
            self.K = ctrl.K
            return ctrl
        
        elif self.args.controller == "dpole":
            real_part = self.slider_p1_real.val
            imag_part = self.slider_p1_imag.val
            p3 = self.slider_p3.val
            p4 = self.slider_p4.val
            
            desired_zpoles = [real_part + imag_part*1j, real_part - imag_part*1j, p3, p4]
            
            Ad, Bd = discretize_system(A, B, self.dt)
            self.Ad = Ad
            self.Bd = Bd
            self.desired_poles = desired_zpoles
            
            ctrl = DiscretePolePlacementController(
                Ad, Bd, desired_poles=desired_zpoles,
                dt=self.dt, max_force=10.0, target_state=[0, 0, 0, 0]
            )
            self.K = ctrl.K
            return ctrl
    
    def calculate_closed_loop_poles(self):
        """Return the closed-loop poles for the chosen controller type."""
        if self.args.controller == "lqr":
            cl_A = self.A - np.dot(self.B, self.K)
            return linalg.eigvals(cl_A)
        elif self.args.controller in ["dlqr", "dpole"]:
            cl_A = self.Ad - np.dot(self.Bd, self.K)
            return linalg.eigvals(cl_A)
        else:
            # For PID or other controllers not mapped easily to a state matrix
            return None
    
    def plot_poles(self, poles):
        """Plot closed-loop poles in self.pole_ax."""
        self.pole_ax.clear()
        
        if poles is not None and len(poles) > 0:
            real_parts = np.real(poles)
            imag_parts = np.imag(poles)
            
            self.pole_ax.scatter(real_parts, imag_parts, marker='x', color='red', s=100)
            
            if self.args.controller in ["dlqr", "dpole"]:
                # Discrete-time => draw unit circle
                theta = np.linspace(0, 2*np.pi, 100)
                x = np.cos(theta)
                y = np.sin(theta)
                self.pole_ax.plot(x, y, 'k--', alpha=0.5)
                self.pole_ax.set_xlim(-1.5, 1.5)
                self.pole_ax.set_ylim(-1.5, 1.5)
            else:
                # Continuous-time => vertical line at Re=0
                self.pole_ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
                self.pole_ax.set_xlim(-4, 1)
                self.pole_ax.set_ylim(-2, 2)
            
            if self.settling_time is not None:
                self.settling_time_text.set_text(f"Settling Time: {self.settling_time:.3f}s")
            else:
                self.settling_time_text.set_text("Settling Time: System did not settle")
                
            self.pole_ax.grid(True)
            self.pole_ax.set_xlabel("Real Part")
            self.pole_ax.set_ylabel("Imag. Part")
            self.pole_ax.set_title("Closed-Loop Poles")
    
    def animate_last_sim(self, event):
        if self.last_x_pos is not None and self.last_theta is not None:
            print("Starting animation with data shapes:", self.last_x_pos.shape, self.last_theta.shape)
            try:
                animate_cartpole(self.t_eval, self.last_x_pos, self.last_theta, self.dt)
            except Exception as e:
                print(f"Animation error: {e}")
        else:
            print("No simulation data available")
    
    def update(self, event):
        """Re-run simulation with updated parameters and controller, then plot results."""
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.clear()
        
        # Create the current controller
        ctrl = self.create_controller()
        
        # Build the 10-parameter tuple for cartpole_nonlinear_dynamics
        params = (
            self.slider_M.val,  # M
            self.slider_m.val,  # m
            self.m_a,
            self.l_c,
            self.l,             # or self.l_a if you define that
            self.I_p,
            self.I_a,
            self.g,
            self.b,
            self.c
        )
        
        # Possibly reduce initial angle if using discrete controllers
        if self.args.controller in ["dpole", "dlqr"]:
            self.x0[2] = np.deg2rad(1)
        else:
            self.x0[2] = np.deg2rad(5)
        
        # ODE function
        def dynamics(t, state):
            # Compute control action
            if self.args.controller in ["dpole", "dlqr"]:
                u = ctrl.compute_control(t, state)
            else:
                u = ctrl.compute_control(state)
            # Then pass it to the nonlinear cartpole
            return cartpole_nonlinear_dynamics(t, state, params, controller=lambda s: u)
        
        sol = solve_ivp(dynamics, [0, self.T], self.x0, t_eval=self.t_eval)
        states = sol.y
        
        # Apply noise & filtering if requested
        if self.args.noise:
            noisy_states = apply_noise(states, noise_level=0.05)
            cutoff_freq = 1.0
            fs = 1/self.dt
            filtered_x = low_pass_filter(noisy_states[0], cutoff_freq, fs)
            filtered_theta = low_pass_filter(noisy_states[2], cutoff_freq, fs)
            x_pos = filtered_x
            theta = filtered_theta
        else:
            x_pos = states[0]
            theta = states[2]
        
        # Store for possible animation
        self.last_x_pos = x_pos
        self.last_theta = theta
        
        # Reconstruct control effort
        control_effort = []
        for i, t_ in enumerate(self.t_eval):
            st_ = states[:, i]
            if self.args.controller in ["dpole", "dlqr"]:
                u_ = ctrl.compute_control(t_, st_)
            else:
                u_ = ctrl.compute_control(st_)
            control_effort.append(u_)
        
        # Settling time
        threshold = np.deg2rad(1.2)  # 1.2° threshold
        settle_time = compute_settling_time(np.abs(theta), self.t_eval, threshold)
        
        print(f"Overall max |theta|: {np.max(np.abs(theta)):.6f} radians")
        exceed_indices = np.where(np.abs(theta) >= threshold)[0]
        if len(exceed_indices) > 0:
            last_exceed_time = self.t_eval[exceed_indices[-1]]
            print(f"Last time |theta| >= threshold: {last_exceed_time:.2f}s")
            self.settling_time = last_exceed_time + self.dt
            print(f"Settling time: {self.settling_time:.4f}s")
        else:
            print("|theta| never exceeds threshold")
            self.settling_time = 0.0
        
        # Plot
        self.ax1.plot(self.t_eval, x_pos)
        self.ax1.set_ylabel('Cart Position (m)')
        self.ax1.set_title('Cart Position vs Time')
        self.ax1.grid(True)
        
        self.ax2.plot(self.t_eval, np.rad2deg(theta))
        self.ax2.set_ylabel('Angle (degrees)')
        self.ax2.set_title('Pendulum Angle vs Time')
        self.ax2.grid(True)
        
        self.ax3.plot(self.t_eval, control_effort)
        self.ax3.set_xlabel('Time (s)')
        self.ax3.set_ylabel('Force (N)')
        self.ax3.set_title('Control Effort vs Time')
        self.ax3.grid(True)
        
        # Poles
        poles = self.calculate_closed_loop_poles()
        self.plot_poles(poles)
        
        # If system didn't settle, you can highlight it:
        if settle_time is None:
            self.ax2.text(
                0.05, 0.95, 'System did not settle',
                transform=self.ax2.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.3)
            )
        
        self.fig.canvas.draw_idle()
        
        # Animate if user asked via --animate or button
        if event is not None and self.args.animate and event == self.button:
            animate_cartpole(self.t_eval, x_pos, theta, self.dt)

def main():
    args = parse_args()
    app = InteractiveCartpoleSimulation(args)

if __name__ == "__main__":
    main()
