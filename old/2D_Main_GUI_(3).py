import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button, Slider, CheckButtons
import matplotlib.patches as patches
import math
from scipy.signal import butter, filtfilt
from typing import Optional

# Import controllers and dynamics functions from your source_cartpole.py file.
from source_cartpole import (
    PIDController, NMPCNonlinearControllerSimplified, LQRController, 
    build_cartpole_linear_system, discretize_system, cartpole_nonlinear_dynamics,
    low_pass_filter, compute_settling_time
)

###############################################################################
# Simulator Classes
###############################################################################
class BaseSimulator:
    def __init__(self, T_total, dt_model, dt_sim):
        self.T_total = T_total
        self.dt_model = dt_model    # Controller update time step (e.g., 0.01 s)
        self.dt_sim = dt_sim        # Simulation integration time step (e.g., 0.001 s)
        self.steps = int(T_total / dt_sim)
        self.current_step = 0
        self.time_data = []
        self.theta_data = []
        self.x_data = []
        self.F_data = []
        self.disturbance = 0   # Disturbance amplitude (used in disturbance function)
        self.noise_amp = 0     # Noise amplitude (for measurement noise)

        # Cart-pole parameters (should match your source_cartpole.py parameters)
        self.M = 0.4
        self.m = 0.01
        self.m_a = 0.05
        self.l = 0.5
        self.g = 9.81
        self.b_damp = 0.1
        self.c_damp = 0.015
        self.params = (self.M, self.m, self.m_a, self.l, self.g, self.b_damp, self.c_damp)

        # Initial state: [x, x_dot, theta, theta_dot]
        self.state = [0.0, 0.0, math.radians(5.0), 0.0]
        self.time = 0.0

    def step(self):
            # Get the raw measurement with noise added.
        raw_state = self.state.copy()
        raw_state[2] += self.noise_amp * np.random.normal(0, 1)
        
        # Apply filtering only if filter_enabled is True.
        if filter_enabled:
            if not hasattr(self, 'filtered_state'):
                self.filtered_state = raw_state.copy()
            else:
                alpha = 0.1  # Smoothing factor: smaller alpha => more smoothing/delay.
                self.filtered_state[2] = alpha * raw_state[2] + (1 - alpha) * self.filtered_state[2]
            measured_state = self.filtered_state
        else:
            measured_state = raw_state

        # Compute the control signal using the measured state.
        if hasattr(self, 'controller'):
            try:
                u = self.controller.compute_control(measured_state)
            except TypeError:
                u = self.controller.compute_control(self.time, measured_state)
        else:
            u = 0.0

        # Add disturbance to the computed control signal.
        u_effective = u + self.disturbance

        # Compute the state derivative using the true (unfiltered) state.
        state_dot = cartpole_nonlinear_dynamics(
            self.time, self.state, self.params,
            controller=lambda s: u_effective
        )

        # Integrate with dt_sim
        new_state = [s + self.dt_sim * ds for s, ds in zip(self.state, state_dot)]
        self.state = new_state
        self.time += self.dt_sim
        self.current_step += 1
        self.time_data.append(self.time)
        self.x_data.append(self.state[0])
        self.theta_data.append(self.state[2])
        self.F_data.append(u)


    def step_batch(self, batch_size):
        for _ in range(batch_size):
            if self.current_step >= self.steps:
                break
            self.step()

    def set_disturbance(self, d):
        self.disturbance = d

    def set_noise(self, n):
        self.noise_amp = n

class PIDSimulator(BaseSimulator):
    def __init__(self, T_total, dt_model, dt_sim, Kp, Ki, Kd):
        super().__init__(T_total, dt_model, dt_sim)
        self.controller = PIDController(kp=Kp, ki=Ki, kd=Kd, dt_model=dt_model, max_force=100.0)

    def set_pid_parameters(self, Kp, Ki, Kd):
        self.controller.kp = Kp
        self.controller.ki = Ki
        self.controller.kd = Kd
        self.controller.integral = 0.0
        self.controller.prev_error = None

class LQRSimulator(BaseSimulator):
    def __init__(self, T_total, dt_model, dt_sim, Q, R, target_state):
        super().__init__(T_total, dt_model, dt_sim)
        A, B = build_cartpole_linear_system(self.M, self.m, self.m_a, self.l, self.g, self.b_damp, self.c_damp)
        Ad, Bd = discretize_system(A, B, dt_model)
        self.controller = LQRController(Ad, Bd, Q, R, dt_model, max_force=100.0, target_state=target_state)

class NMPCSimulator(BaseSimulator):
    def __init__(self, T_total, dt_model, dt_sim):
        super().__init__(T_total, dt_model, dt_sim)
        horizon = 10  # prediction horizon of 20 steps
        Q = np.diag([50, 10, 40000, 10])
        R = np.array([[0.01]])      # control cost weight
        # Use the full nonlinear NMPC controller, passing in the system parameters.
        self.controller = NMPCNonlinearControllerSimplified(horizon=horizon,
                                                    dt_model=dt_model,
                                                    max_force=100.0,
                                                    target_state=[0, 0, 0, 0],
                                                    Q=Q, R=R,
                                                    params=self.params)

# Mapping for the radio buttons.
controller_modes = {
    "PID": PIDSimulator,
    "LQR": LQRSimulator,
    "NMPC": NMPCSimulator
}
current_controller = PIDSimulator

###############################################################################
# Global Simulator Reference and Filter Toggle
###############################################################################
simulator = None
filter_enabled = False

###############################################################################
# GUI Code
###############################################################################
sim_running = False
disturb_enabled = False
noise_enabled = False

fig = plt.figure(figsize=(14, 8))
plt.subplots_adjust(left=0.05, bottom=0.02, right=0.95, top=0.92)
fig.suptitle("Controller GUI", fontsize=14)

# Text object to display settling time.
settling_text = fig.text(0.25, 0.95, "", fontsize=12)

# --- Controller Selection ---
rax = plt.axes([0.05, 0.75, 0.1, 0.15])
radio = RadioButtons(rax, ('PID', 'LQR', 'NMPC'))

###############################################################################
# PID Sliders (shown only when PID is selected)
###############################################################################
pid_kp_ax = plt.axes([0.05, 0.65, 0.12, 0.03])
slider_kp = Slider(pid_kp_ax, 'Kp', 0.0, 100.0, valinit=35)
slider_kp.label.set_position((-0.1, 0.5))
slider_kp.label.set_horizontalalignment('right')

pid_ki_ax = plt.axes([0.05, 0.60, 0.12, 0.03])
slider_ki = Slider(pid_ki_ax, 'Ki', 0.0, 100.0, valinit=0)
slider_ki.label.set_position((-0.1, 0.5))
slider_ki.label.set_horizontalalignment('right')

pid_kd_ax = plt.axes([0.05, 0.55, 0.12, 0.03])
slider_kd = Slider(pid_kd_ax, 'Kd', 0.0, 100.0, valinit=6)
slider_kd.label.set_position((-0.1, 0.5))
slider_kd.label.set_horizontalalignment('right')

###############################################################################
# LQR Target Buttons and Sliders (shown only when LQR is selected)
###############################################################################
lqr_target = [0.0, 0.0, 0.0, 0.0]

lqr_t0_ax = plt.axes([0.02, 0.665, 0.1, 0.04])
lqr_target0_button = Button(lqr_t0_ax, 'Target [0,0,0,0]')
lqr_t1_ax = plt.axes([0.12, 0.665, 0.1, 0.04])
lqr_target1_button = Button(lqr_t1_ax, 'Target [1.8,0,0,0]')

def set_target0(event):
    global lqr_target, simulator
    lqr_target = [0.0, 0.0, 0.0, 0.0]
    print("LQR target set to [0,0,0,0]")
    if simulator and isinstance(simulator, LQRSimulator):
        simulator.controller.set_target_state(lqr_target)

def set_target1(event):
    global lqr_target, simulator
    lqr_target = [1.8, 0.0, 0.0, 0.0]
    print("LQR target set to [1.8,0,0,0]")
    if simulator and isinstance(simulator, LQRSimulator):
        simulator.controller.set_target_state(lqr_target)

lqr_target0_button.on_clicked(set_target0)
lqr_target1_button.on_clicked(set_target1)

lqr_q1_ax = plt.axes([0.05, 0.60, 0.12, 0.03])
lqr_q1_slider = Slider(lqr_q1_ax, 'Q1', 0.0, 50.0, valinit=2.0)
lqr_q2_ax = plt.axes([0.05, 0.55, 0.12, 0.03])
lqr_q2_slider = Slider(lqr_q2_ax, 'Q2', 0.0, 50.0, valinit=2.0)
lqr_q3_ax = plt.axes([0.05, 0.50, 0.12, 0.03])
lqr_q3_slider = Slider(lqr_q3_ax, 'Q3', 0.0, 50.0, valinit=23.0)
lqr_q4_ax = plt.axes([0.05, 0.45, 0.12, 0.03])
lqr_q4_slider = Slider(lqr_q4_ax, 'Q4', 0.0, 50.0, valinit=5.0)
lqr_r_ax = plt.axes([0.05, 0.40, 0.12, 0.03])
lqr_r_slider = Slider(lqr_r_ax, 'R', 0.0, 10.0, valinit=1.0)

###############################################################################
# Disturbance, Noise, and Filter Check Buttons
###############################################################################
rax_disturb = plt.axes([0.05, 0.35, 0.12, 0.03])
check_disturb = CheckButtons(rax_disturb, ['Disturbance'], [False])
amp_ax = plt.axes([0.05, 0.30, 0.12, 0.03])
slider_amp = Slider(amp_ax, 'Amp', 0.0, 5.0, valinit=0.0)
slider_amp.label.set_position((-0.1, 0.5))
slider_amp.label.set_horizontalalignment('right')

rax_noise = plt.axes([0.05, 0.25, 0.12, 0.03])
check_noise = CheckButtons(rax_noise, ['Noise'], [False])
noise_ax = plt.axes([0.05, 0.20, 0.12, 0.03])
slider_noise = Slider(noise_ax, 'Amp', 0.0, 0.5, valinit=0.0)
slider_noise.label.set_position((-0.1, 0.5))
slider_noise.label.set_horizontalalignment('right')

rax_filter = plt.axes([0.05, 0.15, 0.12, 0.03])
check_filter = CheckButtons(rax_filter, ['Filter'], [False])
def filter_func(label):
    global filter_enabled
    filter_enabled = not filter_enabled
check_filter.on_clicked(filter_func)

###############################################################################
# Start and End Buttons
###############################################################################
start_ax = plt.axes([0.05, 0.05, 0.05, 0.05])
start_button = Button(start_ax, 'Start')
end_ax = plt.axes([0.12, 0.05, 0.05, 0.05])
end_button = Button(end_ax, 'End')

###############################################################################
# Plot Axes and Animation
###############################################################################
ax1 = plt.axes([0.25, 0.63, 0.45, 0.25])
ax2 = plt.axes([0.25, 0.33, 0.45, 0.25])
ax3 = plt.axes([0.25, 0.03, 0.45, 0.25])

ax_pendulum = plt.axes([0.73, 0.65, 0.22, 0.25])
ax_pendulum.set_xlim([-0.8, 0.8])
ax_pendulum.set_ylim([-0.1, 0.8])
ax_pendulum.set_xticks([])
ax_pendulum.set_yticks([])
ax_pendulum.set_title("Animation")
cart_width = 0.15
cart_height = 0.07
cart_body = patches.Rectangle((0 - cart_width/2, 0), cart_width, cart_height,
                              facecolor='gray', edgecolor='black')
cart_body.set_visible(False)
ax_pendulum.add_patch(cart_body)
pole_line, = ax_pendulum.plot([], [], 'r-', lw=3)
pendulum_tip, = ax_pendulum.plot([], [], 'bo', markersize=8)
track_line, = ax_pendulum.plot([-0.8, 0.8], [0, 0], 'k--', lw=1)

ax_disturb = plt.axes([0.73, 0.36, 0.22, 0.25])
ax_disturb.set_title("Disturbance (N)")
line_disturb, = ax_disturb.plot([], [], 'm-', lw=2)
ax_disturb.set_xlim(0, 10)
ax_disturb.set_ylim(-20, 20)

ax_noise_plot = plt.axes([0.73, 0.06, 0.22, 0.25])
ax_noise_plot.set_title("Noise (rad)")
line_noise, = ax_noise_plot.plot([], [], 'c-', lw=2)
ax_noise_plot.set_xlim(0, 10)
ax_noise_plot.set_ylim(-1, 1)

###############################################################################
# Simulation Parameters
###############################################################################
dt_model = 0.01   # Controller update time step
dt_sim   = 0.001  # Simulation integration time step
T_total  = 20.0   # Total simulation time
batch_size = 50

disturb_data = []
noise_data = []

###############################################################################
# Functions to Show/Hide Widgets Based on Controller
###############################################################################
def show_pid_sliders():
    slider_kp.ax.set_visible(True)
    slider_ki.ax.set_visible(True)
    slider_kd.ax.set_visible(True)

def hide_pid_sliders():
    slider_kp.ax.set_visible(False)
    slider_ki.ax.set_visible(False)
    slider_kd.ax.set_visible(False)

def show_lqr_widgets():
    lqr_target0_button.ax.set_visible(True)
    lqr_target1_button.ax.set_visible(True)
    lqr_q1_slider.ax.set_visible(True)
    lqr_q2_slider.ax.set_visible(True)
    lqr_q3_slider.ax.set_visible(True)
    lqr_q4_slider.ax.set_visible(True)
    lqr_r_slider.ax.set_visible(True)

def hide_lqr_widgets():
    lqr_target0_button.ax.set_visible(False)
    lqr_target1_button.ax.set_visible(False)
    lqr_q1_slider.ax.set_visible(False)
    lqr_q2_slider.ax.set_visible(False)
    lqr_q3_slider.ax.set_visible(False)
    lqr_q4_slider.ax.set_visible(False)
    lqr_r_slider.ax.set_visible(False)

# Initially show PID sliders and hide LQR widgets.
show_pid_sliders()
hide_lqr_widgets()

###############################################################################
# Radio Button Callback
###############################################################################
def radio_func(label):
    global current_controller
    current_controller = {
        "PID": PIDSimulator,
        "LQR": LQRSimulator,
        "NMPC": NMPCSimulator
    }[label]
    print("Selected controller:", label)
    if label == "PID":
        show_pid_sliders()
        hide_lqr_widgets()
    elif label == "LQR":
        hide_pid_sliders()
        show_lqr_widgets()
    else:
        hide_pid_sliders()
        hide_lqr_widgets()
    plt.draw()

radio.on_clicked(radio_func)

###############################################################################
# Disturbance, Noise, and Filter Callbacks
###############################################################################
def disturb_func(label):
    global disturb_enabled
    disturb_enabled = not disturb_enabled
check_disturb.on_clicked(disturb_func)

def noise_func(label):
    global noise_enabled
    noise_enabled = not noise_enabled
check_noise.on_clicked(noise_func)

###############################################################################
# Global Simulator Reference and Start/End Callbacks
###############################################################################
def start_simulation(event):
    global sim_running, disturb_data, noise_data, simulator
    print("Start Simulation button clicked!")
    sim_running = True
    label = radio.value_selected

    if label == "PID":
        simulator = current_controller(T_total=T_total, dt_model=dt_model, dt_sim=dt_sim,
                                        Kp=slider_kp.val, Ki=slider_ki.val, Kd=slider_kd.val)
    elif label == "LQR":
        Q = np.diag([
            lqr_q1_slider.val,
            lqr_q2_slider.val,
            lqr_q3_slider.val,
            lqr_q4_slider.val
        ])
        R = np.array([[lqr_r_slider.val]])
        simulator = current_controller(T_total=T_total, dt_model=dt_model, dt_sim=dt_sim,
                                        Q=Q, R=R, target_state=lqr_target)
    elif label == "NMPC":
        simulator = current_controller(T_total=T_total, dt_model=dt_model, dt_sim=dt_sim)
    else:
        print("Selected controller not implemented.")
        return

    cart_body.set_visible(True)
    ax1.clear(); ax2.clear(); ax3.clear(); ax_disturb.clear(); ax_noise_plot.clear()

    line1, = ax1.plot([], [], label="θ (rad)")
    ax1.axhline(0, color='r', linestyle='--', label="Desired 0")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Pole Angle (rad)")
    ax1.legend()

    line2, = ax2.plot([], [], label="x (m)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Cart Position (m)")
    ax2.legend()

    line3, = ax3.plot([], [], label="Control Force F (N)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Force F (N)")
    ax3.legend()

    line_d, = ax_disturb.plot([], [], 'm-', lw=2, label="Disturbance (N)")
    ax_disturb.set_xlabel("Time (s)")
    ax_disturb.set_ylabel("Force (N)")
    ax_disturb.legend()

    line_n, = ax_noise_plot.plot([], [], 'c-', lw=2, label="Noise (rad)")
    ax_noise_plot.set_xlabel("Time (s)")
    ax_noise_plot.set_ylabel("Noise (rad)")
    ax_noise_plot.legend()

    disturb_data = []
    noise_data = []
    num_batches = simulator.steps // batch_size

    for i in range(num_batches):
        if not sim_running:
            print("Simulation ended early.")
            break

        if label == "PID":
            simulator.set_pid_parameters(slider_kp.val, slider_ki.val, slider_kd.val)

        if disturb_enabled:
            # For disturbance, using a Gaussian value:
            disturbance = slider_amp.val * np.random.normal(0, 1)
        else:
            disturbance = 0.0

        if noise_enabled:
            # For noise, use Gaussian noise (this will be used in measurement noise)
            noise = slider_noise.val * np.random.normal(0, 1)
        else:
            noise = 0.0

        simulator.set_disturbance(disturbance)
        simulator.set_noise(noise)
        disturb_data.extend([disturbance] * batch_size)
        noise_data.extend([noise] * batch_size)
        simulator.step_batch(batch_size)
        line1.set_data(simulator.time_data, simulator.theta_data)
        line2.set_data(simulator.time_data, simulator.x_data)
        line3.set_data(simulator.time_data, simulator.F_data)
        line_d.set_data(simulator.time_data, disturb_data)
        line_n.set_data(simulator.time_data, noise_data)

        ax1.relim(); ax1.autoscale_view()
        ax2.relim(); ax2.autoscale_view()
        ax3.relim(); ax3.autoscale_view()
        ax_disturb.relim(); ax_disturb.autoscale_view()
        ax_noise_plot.relim(); ax_noise_plot.autoscale_view()

        update_pendulum(simulator.x_data[-1], simulator.theta_data[-1])
        plt.pause(0.001)

    if sim_running and simulator.current_step < simulator.steps:
        simulator.step_batch(simulator.steps - simulator.current_step)
        line1.set_data(simulator.time_data, simulator.theta_data)
        line2.set_data(simulator.time_data, simulator.x_data)
        line3.set_data(simulator.time_data, simulator.F_data)
        line_d.set_data(simulator.time_data, disturb_data)
        line_n.set_data(simulator.time_data, noise_data)
        ax1.relim(); ax1.autoscale_view()
        ax2.relim(); ax2.autoscale_view()
        ax3.relim(); ax3.autoscale_view()
        ax_disturb.relim(); ax_disturb.autoscale_view()
        ax_noise_plot.relim(); ax_noise_plot.autoscale_view()
        plt.pause(0.001)
    # After simulation, apply filtering if enabled
    if filter_enabled:
        fs = 1 / dt_sim
        cutoff = 5  # cutoff frequency in Hz (adjust as needed)
        filtered_theta = low_pass_filter(np.array(simulator.theta_data), cutoff, fs, order=3)
        simulator.theta_data = filtered_theta.tolist()
        settle_time = compute_settling_time(np.abs(filtered_theta), np.array(simulator.time_data), 0.1)
        if settle_time is None:
            settle_time_str = "Not Settled"
        else:
            settle_time_str = f"{settle_time:.2f} s"
        settling_text.set_text(f"Settling Time: {settle_time_str}")
    else:
        settling_text.set_text("Settling Time: (Filter Disabled)")

    if sim_running and simulator.current_step < simulator.steps:
        simulator.step_batch(simulator.steps - simulator.current_step)
        line1.set_data(simulator.time_data, simulator.theta_data)
        line2.set_data(simulator.time_data, simulator.x_data)
        line3.set_data(simulator.time_data, simulator.F_data)
        line_d.set_data(simulator.time_data, disturb_data)
        line_n.set_data(simulator.time_data, noise_data)
        ax1.relim(); ax1.autoscale_view()
        ax2.relim(); ax2.autoscale_view()
        ax3.relim(); ax3.autoscale_view()
        ax_disturb.relim(); ax_disturb.autoscale_view()
        ax_noise_plot.relim(); ax_noise_plot.autoscale_view()
        plt.pause(0.001)
    plt.ioff()

start_button.on_clicked(start_simulation)

def end_simulation(event):
    global sim_running
    sim_running = False
    print("End Simulation clicked.")
end_button.on_clicked(end_simulation)

###############################################################################
# Animation Update Function
###############################################################################
def update_pendulum(cart_x, theta):
    cart_body.set_xy((cart_x - cart_width/2, 0))
    current_xlim = ax_pendulum.get_xlim()
    half_width = (current_xlim[1] - current_xlim[0]) / 2.0
    if cart_x - half_width < current_xlim[0] or cart_x + half_width > current_xlim[1]:
        ax_pendulum.set_xlim(cart_x - half_width, cart_x + half_width)
    pivot_x = cart_x
    pivot_y = cart_height
    L = 0.3  # visual pendulum length
    bob_x = pivot_x + L * math.sin(theta)
    bob_y = pivot_y + L * math.cos(theta)
    pole_line.set_data([pivot_x, bob_x], [pivot_y, bob_y])
    pendulum_tip.set_data([bob_x], [bob_y])
    plt.draw()

plt.show()
