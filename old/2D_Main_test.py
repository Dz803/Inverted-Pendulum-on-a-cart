
"""
pid_tuning_interface.py

Demonstrates an interactive GUI for tuning PID gains on the cart-pole system.
Uses your existing PIDController, cartpole_nonlinear_dynamics, etc.,
with a consistent approach to the pendulum geometry.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.integrate import solve_ivp

# Import your PID controller and the cart-pole utilities
from Controller import PIDController
from old.utils import (
    cartpole_nonlinear_dynamics,
    plot_static_results,
    animate_cartpole
)

class PIDTuningApp:
    def __init__(self):
        #-------------------------------------------------------------
        # 1) Set up default system parameters
        #-------------------------------------------------------------
        self.M = 0.4      # cart mass
        self.m = 0.043     # rod mass
        self.m_a = 0.052   # attachment (point) mass
        self.l = 0.5      # full rod length from pivot to tip
        self.g = 9.81
        self.b_damp = 0.1
        self.c_damp = 0.015
        
        # compute center-of-mass, inertia, etc. for the nonlinear function
        l_c = self.l / 2.0
        I_p = (1.0 / 3.0) * self.m * (self.l ** 2)   # uniform rod pivoted at top
        I_a = self.m_a * (self.l ** 2)              # point mass at tip
        
        # Build the parameter tuple that matches cartpole_nonlinear_dynamics:
        #
        # def cartpole_nonlinear_dynamics(t, state, params, controller=None):
        #     # Unpacks: M, m, m_a, l_c, l_a, I_p, I_a, g, b_damp, c_damp
        #
        self.params = (
            self.M,          # M
            self.m,          # m
            self.m_a,        # m_a
            self.l,          # "l_a" => using l as the rod tip length
            self.g,
            self.b_damp,
            self.c_damp
        )
        
        #-------------------------------------------------------------
        # 2) Initialize Gains
        #-------------------------------------------------------------
        self.kp = 5.0
        self.ki = 0.0
        self.kd = 0.3
        
        #-------------------------------------------------------------
        # 3) Time/Simulation Setup
        #-------------------------------------------------------------
        self.t_start = 0.0
        self.t_end   = 8.0
        self.dt_sim  = 0.001
        
        #-------------------------------------------------------------
        # 4) Initial State: [x, x_dot, theta, theta_dot]
        #    We'll start with the pole tilted ~5 degrees from vertical
        #-------------------------------------------------------------
        self.x0 = np.array([0.0, 0.0, np.deg2rad(5.0), 0.0])
        
        #-------------------------------------------------------------
        # 5) Build the figure/GUI
        #-------------------------------------------------------------
        self.fig, (self.ax_pos, self.ax_theta, self.ax_force) = plt.subplots(3, 1, figsize=(10, 8))
        plt.subplots_adjust(left=0.1, bottom=0.35)  # more space for sliders at the bottom
        
        self.ax_pos.set_title("Cart Position vs. Time")
        self.ax_theta.set_title("Pole Angle vs. Time")
        self.ax_force.set_title("Control Effort vs. Time")
        
        # placeholders for lines
        self.line_pos, = self.ax_pos.plot([], [], label="x(t)")
        self.line_theta, = self.ax_theta.plot([], [], label="theta(t)")
        self.line_force, = self.ax_force.plot([], [], label="u(t)")
        
        for ax_ in (self.ax_pos, self.ax_theta, self.ax_force):
            ax_.grid(True)
            ax_.legend()
        
        #-------------------------------------------------------------
        # 6) Sliders for Kp, Ki, Kd
        #-------------------------------------------------------------
        slider_height = 0.03
        slider_y = 0.2
        
        self.ax_kp = plt.axes([0.1, slider_y,         0.3, slider_height])
        self.ax_ki = plt.axes([0.1, slider_y - 0.05,  0.3, slider_height])
        self.ax_kd = plt.axes([0.1, slider_y - 0.1,   0.3, slider_height])
        
        self.slider_kp = Slider(self.ax_kp, 'Kp', 0.0, 20.0, valinit=self.kp)
        self.slider_ki = Slider(self.ax_ki, 'Ki', 0.0, 1.0,  valinit=self.ki)
        self.slider_kd = Slider(self.ax_kd, 'Kd', 0.0, 5.0,  valinit=self.kd)
        
        #-------------------------------------------------------------
        # 7) Button to re-simulate
        #-------------------------------------------------------------
        self.sim_button_ax = plt.axes([0.5, 0.05, 0.1, 0.04])
        self.sim_button = Button(self.sim_button_ax, 'Simulate')
        
        #-------------------------------------------------------------
        # 8) Button to animate
        #-------------------------------------------------------------
        self.anim_button_ax = plt.axes([0.65, 0.05, 0.1, 0.04])
        self.anim_button = Button(self.anim_button_ax, 'Animate')
        
        # connect events
        self.slider_kp.on_changed(self.on_slider_change)
        self.slider_ki.on_changed(self.on_slider_change)
        self.slider_kd.on_changed(self.on_slider_change)
        
        self.sim_button.on_clicked(self.run_simulation)
        self.anim_button.on_clicked(self.animate_sim)
        
        # initial simulation
        self.run_simulation(None)
        
        plt.show()
    
    def on_slider_change(self, val):
        """
        Called when user drags the Kp, Ki, Kd sliders.
        We update local variables. We can auto-run or wait for "Simulate" press.
        """
        self.kp = self.slider_kp.val
        self.ki = self.slider_ki.val
        self.kd = self.slider_kd.val
        # Optionally, you could call self.run_simulation(None) for real-time updates.
    
    def run_simulation(self, event):
        """
        1) Read Gains from sliders
        2) Create a PIDController
        3) Solve ODE with solve_ivp
        4) Plot results
        """
        kp = self.slider_kp.val
        ki = self.slider_ki.val
        kd = self.slider_kd.val
        
        # Build a PIDController
        pid = PIDController(kp=kp, ki=ki, kd=kd, dt=1/240., max_force=8.0)
        
        # Time array
        t_eval = np.arange(self.t_start, self.t_end, self.dt_sim)
        
        def dynamics(t, state):
            # The PID gives us a force
            u_ = pid.compute_control(state)
            # Then pass that force to the cartpole ODE
            return cartpole_nonlinear_dynamics(t, state, self.params, controller=lambda s: u_)
        
        # Solve the ODE
        sol = solve_ivp(dynamics, [self.t_start, self.t_end], self.x0, t_eval=t_eval)
        
        # Gather data
        self.time = sol.t
        self.states = sol.y  # shape is (4, len(t_eval))
        
        # Re-compute control effort for plotting
        self.control_effort = []
        for i in range(len(self.time)):
            st_ = self.states[:, i]
            self.control_effort.append(pid.compute_control(st_))
        
        # Clear old plots
        self.ax_pos.cla()
        self.ax_theta.cla()
        self.ax_force.cla()
        
        self.ax_pos.set_title("Cart Position vs. Time")
        self.ax_theta.set_title("Pole Angle vs. Time")
        self.ax_force.set_title("Control Effort vs. Time")
        self.ax_pos.grid(True)
        self.ax_theta.grid(True)
        self.ax_force.grid(True)
        
        x_pos = self.states[0, :]
        theta = self.states[2, :]
        
        # Plot new results
        self.ax_pos.plot(self.time, x_pos, label="x(t)")
        self.ax_theta.plot(self.time, np.rad2deg(theta), label="theta(t) (deg)")
        self.ax_force.plot(self.time, self.control_effort, label="u(t)")
        
        self.ax_pos.legend()
        self.ax_theta.legend()
        self.ax_force.legend()
        
        self.fig.canvas.draw_idle()
    
    def animate_sim(self, event):
        """
        If we have simulation data, call animate_cartpole.
        """
        if not hasattr(self, 'states'):
            print("No simulation data yet!")
            return
        
        x_pos = self.states[0, :]
        theta = self.states[2, :]
        
        animate_cartpole(self.time, x_pos, theta, self.dt_sim,
                         cart_width=0.3, cart_height=0.15,
                         pendulum_length=self.l)  # use self.l as the rod length

if __name__ == "__main__":
    app = PIDTuningApp()
