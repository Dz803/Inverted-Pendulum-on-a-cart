import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import place_poles

# =========================
# 1) PHYSICAL PARAMETERS
# =========================
M  = 0.28     
m  = 0.075    
l  = 0.32     
g  = 9.81     
I  = 0.006    

# ==============================
# 2) LINEARIZED SYSTEM
# ==============================
def build_cartpole_linear_system(M=0.28, m=0.075, l=0.32, g=9.81,
                                 delta=0.04, zeta=0.0, c=0.015):  
    A = np.array([
        [0,           1,                0,                      0],
        [0,   -delta/M,        -(m*g)/M,               zeta/M    ],
        [0,           0,                0,                      1],
        [0,  delta/(l*M),  (M + m)*g/(l*M),  -c / (m * l**2)]
    ])
    B = np.array([
        [0],
        [-1/M],
        [0],
        [-1/(l*M)]
    ])
    return A, B

A_lin, B_lin = build_cartpole_linear_system()

# Pole placement controller
desired_poles = np.array([ -0.5+0.1j , -0.5-0.1j, -3, -0.7 ])    
K_lin = place_poles(A_lin, B_lin, desired_poles).gain_matrix   

# ------------------------------------------------
# 3) NONLINEAR DYNAMICS (With Angle Wrapping)
# ------------------------------------------------
def dynamics_nonlinear(state, force, M, m, I, l, g):
    x, xdot, th, thdot = state
    F = force  

    # Wrap angle to [-π, π]
    while th > np.pi:
        th -= 2 * np.pi
    while th < -np.pi:
        th += 2 * np.pi

    sin_th = np.sin(th)
    cos_th = np.cos(th)
    
    A = (F + m*l*(thdot**2) * sin_th) / (M + m)
    B = (m*l*cos_th) / (M + m)

    denom = (I + m*l**2) + m*l*cos_th * B
    rhs   = m*g*l*sin_th - m*l*cos_th * A
    th_ddot = rhs / denom
    x_ddot = A - B * th_ddot

    return np.array([xdot, x_ddot, thdot, th_ddot])

# ------------------------------------------------
# 4) SIMULATION USING RK4 INTEGRATION
# ------------------------------------------------
def rk4_step(f, state, u, dt, *args):
    k1 = dt * f(state, u, *args)
    k2 = dt * f(state + 0.5 * k1, u, *args)
    k3 = dt * f(state + 0.5 * k2, u, *args)
    k4 = dt * f(state + k3, u, *args)
    return state + (k1 + 2*k2 + 2*k3 + k4) / 6

dt = 0.01  
t_final = 50.0  
time = np.arange(0, t_final, dt)

x0 = np.array([0.0, 0.0, 0.1, 0.0])  
X = x0.reshape(-1, 1)

state_history = [X.flatten()]

force_limit = 10.0  # Max force to apply

for _ in time[1:]:
    x, xdot, th, thdot = X.flatten()
    F = -K_lin @ X  
    F = float(np.clip(F, -force_limit, force_limit))  # Clip force

    X = rk4_step(dynamics_nonlinear, X.flatten(), F, dt, M, m, I, l, g).reshape(-1, 1)
    state_history.append(X.flatten())

state_history = np.array(state_history)

# ------------------------------------------------
# 5) PLOTTING RESULTS
# ------------------------------------------------
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
x_pos = state_history[:, 0]  
theta = state_history[:, 2]  
theta_deg = np.degrees(theta)  

ax[0].plot(time, x_pos, label="Cart Position (m)", color="blue")
ax[0].axhline(0, linestyle="--", color="gray", label="Equilibrium")
ax[0].legend()
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("Position (m)")
ax[0].set_title("Cart Position vs. Time")

ax[1].plot(time, theta_deg, label="Pendulum Angle (deg)", color="red")
ax[1].axhline(0, linestyle="--", color="gray", label="Upright Position")
ax[1].legend()
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Angle (degrees)")
ax[1].set_title("Pendulum Angle vs. Time")

plt.tight_layout()
plt.show()
