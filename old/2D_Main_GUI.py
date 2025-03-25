import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button, Slider, CheckButtons
import matplotlib.patches as patches
import math

# Import controllers and dynamics functions from your source_cartpole.py file.
from source_cartpole import PIDController, cartpole_nonlinear_dynamics, build_cartpole_linear_system, discretize_system, LQRController, PolePlacementController

###############################################################################
# Simulator Classes
###############################################################################

class BaseSimulator:
    def __init__(self, T_total, dt):
        self.T_total = T_total
        self.dt = dt
        self.steps = int(T_total / dt)
        self.current_step = 0
        self.time_data = []
        self.theta_data = []
        self.x_data = []
        self.F_data = []
        self.disturbance = 0.05
        self.noise_amp = 0.05
        
        # Cart-pole parameters (these match your source)
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
        # Compute the control force from the controller.
        if hasattr(self, 'controller'):
            try:
                # Try assuming the controller uses only the state.
                u = self.controller.compute_control(self.state)
            except TypeError:
                # If the controller expects time as well.
                u = self.controller.compute_control(self.time, self.state)
        else:
            u = 0.0

        # Add any external disturbance.
        u_effective = u + self.disturbance

        # Get the state derivative using the nonlinear dynamics.
        # We pass a lambda that always returns our constant effective force.
        state_dot = cartpole_nonlinear_dynamics(self.time, self.state, self.params, controller=lambda s: u_effective)
        
        # Simple Euler integration
        new_state = [s + self.dt * ds for s, ds in zip(self.state, state_dot)]
        
        # Optionally add noise to the pole angle (index 2)
        new_state[2] += self.noise_amp * np.random.uniform(-1, 1)
        
        self.state = new_state
        self.time += self.dt
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

# PID Simulator using the PIDController from source_cartpole.py
class PIDSimulator(BaseSimulator):
    def __init__(self, T_total, dt, Kp, Ki, Kd):
        super().__init__(T_total, dt)
        self.controller = PIDController(kp=Kp, ki=Ki, kd=Kd, dt=dt, max_force=8.0)

    def set_pid_parameters(self, Kp, Ki, Kd):
        self.controller.kp = Kp
        self.controller.ki = Ki
        self.controller.kd = Kd
        self.controller.integral = 0.0
        self.controller.prev_error = None

# Pole Placement Simulator using DiscretePolePlacementController
class PolePlacementSimulator(BaseSimulator):
    def __init__(self, T_total, dt):
        super().__init__(T_total, dt)
        A, B = build_cartpole_linear_system(self.M, self.m, self.m_a, self.l, self.g, self.b_damp, self.c_damp)
        Ad, Bd = discretize_system(A, B, dt)
        # Choose a set of desired poles (here all 0.8 for example)
        desired_poles = [0.8, 0.8, 0.8, 0.8]
        self.controller = PolePlacementController(Ad, Bd, desired_poles, dt, max_force=8.0)
    
    def step(self):
        # For discrete controllers, call with time as well.
        u = self.controller.compute_control(self.time, self.state)
        u_effective = u + self.disturbance
        state_dot = cartpole_nonlinear_dynamics(self.time, self.state, self.params, controller=lambda s: u_effective)
        new_state = [s + self.dt * ds for s, ds in zip(self.state, state_dot)]
        new_state[2] += self.noise_amp * np.random.uniform(-1, 1)
        self.state = new_state
        self.time += self.dt
        self.current_step += 1
        self.time_data.append(self.time)
        self.x_data.append(self.state[0])
        self.theta_data.append(self.state[2])
        self.F_data.append(u)

class LQRSimulator(BaseSimulator):
    def __init__(self, T_total, dt):
        super().__init__(T_total, dt)
        A, B = build_cartpole_linear_system(self.M, self.m, self.m_a, self.l, self.g, self.b_damp, self.c_damp)
        Ad, Bd = discretize_system(A, B, dt)  # Discretize the continuous matrices
        Q = np.diag([2, 2, 23.0, 5]) 
        R = np.array([[1.0]])
        self.controller = LQRController(Ad, Bd, Q, R, dt, max_force=100.0)

# NMPC Simulator (dummy placeholder; in a full implementation you’d substitute your NMPC logic)
class NMPCSimulator(PIDSimulator):
    def __init__(self, T_total, dt):
        # For now, we simply use the PID simulator as a placeholder.
        super().__init__(T_total, dt, Kp=5, Ki=0, Kd=6)

# Mapping for the radio buttons.
controller_modes = {
    "PID": PIDSimulator,
    "PolePlacement": PolePlacementSimulator,
    "LQR": LQRSimulator,
    "NMPC": NMPCSimulator
}
current_controller = PIDSimulator

###############################################################################
# GUI Code
###############################################################################

sim_running = False
disturb_enabled = False
noise_enabled = False

fig = plt.figure(figsize=(14, 8))
plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.92)
fig.suptitle("Controller GUI", fontsize=14)

# Radio buttons for controller selection.
rax = plt.axes([0.06, 0.62, 0.12, 0.15])
radio = RadioButtons(rax, ('PID', 'PolePlacement', 'LQR', 'NMPC'))

def radio_func(label):
    global current_controller
    current_controller = controller_modes[label]
    print("Selected controller:", label)
radio.on_clicked(radio_func)

# Sliders for PID gains.
ax_kp = plt.axes([0.06, 0.52, 0.12, 0.03])
ax_ki = plt.axes([0.06, 0.47, 0.12, 0.03])
ax_kd = plt.axes([0.06, 0.42, 0.12, 0.03])
slider_kp = Slider(ax_kp, 'Kp', 0.0, 100.0, valinit=5)
slider_ki = Slider(ax_ki, 'Ki', 0.0, 100.0, valinit=0)
slider_kd = Slider(ax_kd, 'Kd', 0.0, 100.0, valinit=6)

# Check and slider for disturbance.
rax_disturb = plt.axes([0.06, 0.35, 0.12, 0.03])
check_disturb = CheckButtons(rax_disturb, ['Disturbance'], [False])
def disturb_func(label):
    global disturb_enabled
    disturb_enabled = not disturb_enabled
check_disturb.on_clicked(disturb_func)
ax_amp = plt.axes([0.06, 0.30, 0.12, 0.03])
slider_amp = Slider(ax_amp, 'Amp', 0.0, 20.0, valinit=0.0)

# Check and slider for noise.
rax_noise = plt.axes([0.06, 0.23, 0.12, 0.03])
check_noise = CheckButtons(rax_noise, ['Noise'], [False])
def noise_func(label):
    global noise_enabled
    noise_enabled = not noise_enabled
check_noise.on_clicked(noise_func)
ax_noise = plt.axes([0.06, 0.18, 0.12, 0.03])
slider_noise = Slider(ax_noise, 'Noise Amp', 0.0, 0.1, valinit=0.0)

# Start and End buttons.
start_ax = plt.axes([0.06, 0.10, 0.05, 0.05])
start_button = Button(start_ax, 'Start')
end_ax = plt.axes([0.13, 0.10, 0.05, 0.05])
end_button = Button(end_ax, 'End')

# Axes for plots.
ax1 = plt.axes([0.25, 0.63, 0.45, 0.25])
ax2 = plt.axes([0.25, 0.33, 0.45, 0.25])
ax3 = plt.axes([0.25, 0.03, 0.45, 0.25])

# Animation axis for the cart-pole.
ax_pendulum = plt.axes([0.73, 0.65, 0.22, 0.25])
ax_pendulum.set_xlim([-0.8, 0.8])
ax_pendulum.set_ylim([-0.1, 0.8])
ax_pendulum.set_xticks([])
ax_pendulum.set_yticks([])
ax_pendulum.set_title("Animation")
cart_width = 0.15
cart_height = 0.07
cart_body = patches.Rectangle((0 - cart_width/2, 0), cart_width, cart_height, facecolor='gray', edgecolor='black')
cart_body.set_visible(False)
ax_pendulum.add_patch(cart_body)
pole_line, = ax_pendulum.plot([], [], 'r-', lw=3)
pendulum_tip, = ax_pendulum.plot([], [], 'bo', markersize=8)
track_line, = ax_pendulum.plot([-0.8, 0.8], [0, 0], 'k--', lw=1)

# Axes for disturbance and noise plots.
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

# Simulation parameters.
dt = 1/100
T_total = 50.0   # Adjust total simulation time as desired.
batch_size = 50

disturb_data = []
noise_data = []

def start_simulation(event):
    global sim_running, disturb_data, noise_data
    print("Start Simulation button clicked!")
    sim_running = True
    current_label = radio.value_selected
    if current_label == "PID":
        simulator = current_controller(T_total=T_total, dt=dt, Kp=slider_kp.val, Ki=slider_ki.val, Kd=slider_kd.val)
    elif current_label == "PolePlacement":
        simulator = current_controller(T_total=T_total, dt=dt)
    elif current_label == "LQR":
        simulator = current_controller(T_total=T_total, dt=dt)
    elif current_label == "NMPC":
        simulator = current_controller(T_total=T_total, dt=dt)
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
        if current_label == "PID":
            simulator.set_pid_parameters(slider_kp.val, slider_ki.val, slider_kd.val)
        if disturb_enabled:
            current_time = simulator.time_data[-1] if simulator.time_data else 0.0
            disturbance = slider_amp.val * np.sin(2 * np.pi * 0.5 * current_time)
        else:
            disturbance = 0.0
        if noise_enabled:
            noise = slider_noise.val * np.random.uniform(-1, 1)
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
    plt.ioff()

start_button.on_clicked(start_simulation)

def end_simulation(event):
    global sim_running
    sim_running = False
    print("End Simulation clicked.")
end_button.on_clicked(end_simulation)

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
    pendulum_tip.set_data(bob_x, bob_y)
    plt.draw()

plt.show()
