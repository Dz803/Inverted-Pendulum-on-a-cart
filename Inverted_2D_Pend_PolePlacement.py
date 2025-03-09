import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import place_poles

##############################################################################
# Inverted Pendulum on a Cart, with the pendulum pivot at y=0 and the rod
# extending UPWARDS for theta = 0. Angle theta is measured CLOCKWISE from
# the vertical line pointing up.
##############################################################################

# --- Physical parameters ---
m_c = 5.0    # Mass of the cart (kg)
m_p = 1.0    # Mass of the pendulum (kg)
L   = 1.0    # Length of the pendulum rod (m)
g   = 9.81   # Gravitational acceleration (m/s^2)

# ---------------------------------------------------------------------------
#    1) DERIVE THE LINEARISED STATE-SPACE MODEL AROUND THETA = 0 (UPRIGHT)
# ---------------------------------------------------------------------------
"""
We define the state vector as:
  x1 = x          (horizontal position of cart)
  x2 = dx/dt      (horizontal velocity of cart)
  x3 = theta      (pendulum angle, 0 = upright, measured clockwise)
  x4 = dtheta/dt  (angular velocity)

Input (u) is the horizontal force F on the cart.

The well-known linearised equations about theta=0 (upright) are:

  x2_dot = (1/m_c)*F - (m_p*g / m_c)*theta
  x4_dot = ((m_c + m_p)*g / (m_c * L))*theta - (1 / (m_c * L))*F

Hence, in matrix form:

    [ x1_dot ]   [ 0    1   0                                   0    ] [ x1 ]   [ 0            ]
    [ x2_dot ] = [ 0    0  -(m_p*g)/(m_c)                       0    ] [ x2 ] + [ 1/m_c        ] * u
    [ x3_dot ]   [ 0    0   0                                   1    ] [ x3 ]   [ 0            ]
    [ x4_dot ]   [ 0    0   ((m_c+m_p)*g)/(m_c*L)               0    ] [ x4 ]   [ -1/(m_c*L)   ]

"""

A = np.array([
    [0,    1,    0,                                           0],
    [0,    0, - (m_p*g)/m_c,                                   0],
    [0,    0,    0,                                           1],
    [0,    0,   ((m_c + m_p)*g)/(m_c*L),                       0]
])

B = np.array([
    [0],
    [1/m_c],
    [0],
    [-1/(m_c*L)]
])

# ---------------------------------------------------------------------------
#    2) POLE PLACEMENT TO OBTAIN FEEDBACK GAIN K
# ---------------------------------------------------------------------------
# Choose some desired negative real poles for stability & responsiveness
desired_poles = np.array([-2.0, -2.2, -2.5, -3.0])
pp = place_poles(A, B, desired_poles)
K = pp.gain_matrix

# ---------------------------------------------------------------------------
#    3) SIMULATION OF THE CLOSED-LOOP SYSTEM
# ---------------------------------------------------------------------------
# x_dot = A x + B u,    with u = -K x

# Time settings
dt = 0.01
t_final = 10.0
time = np.arange(0, t_final, dt)

# Initial state: small tilt from upright, everything else at rest
x0 = np.array([0, 0, 0, 1])  # 0.1 rad ~ 5.7 degrees clockwise
x = x0.reshape(-1, 1)

# For storage
state_history = [x.copy()]

for _ in time[1:]:
    # Control input
    u = -K @ x

    # State derivative
    x_dot = A @ x + B @ u

    # Simple Euler integration
    x = x + dt * x_dot
    state_history.append(x.copy())

state_history = np.squeeze(np.array(state_history))

# ---------------------------------------------------------------------------
#    4) ANIMATION
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlim([-2, 2])
ax.set_ylim([0, 2])  # y from 0 up to ~2 so we can see the rod above the pivot
ax.set_aspect('equal', 'box')
ax.set_title('Inverted Pendulum (Upright) via Pole Placement')

(cart_line,) = ax.plot([], [], lw=6, color='blue')
(rod_line,)  = ax.plot([], [], lw=3, color='red')

cart_width = 0.3

def init():
    cart_line.set_data([], [])
    rod_line.set_data([], [])
    return cart_line, rod_line

def animate(i):
    # Extract states
    cart_x = state_history[i, 0]
    theta  = state_history[i, 2]  # measured clockwise from up

    # Cart corners
    left  = cart_x - cart_width / 2
    right = cart_x + cart_width / 2
    y_cart = 0

    # Pendulum tip coordinates:
    # If theta = 0 => tip at (cart_x, +L).
    # For small positive theta (clockwise), the tip goes slightly to the RIGHT.
    tip_x = cart_x + L * np.sin(theta)
    tip_y = L * np.cos(theta)

    # Update cart line (just a thick horizontal segment)
    cart_line.set_data([left, right], [y_cart, y_cart])

    # Update pendulum rod
    rod_line.set_data([cart_x, tip_x], [0, tip_y])
    return cart_line, rod_line

ani = FuncAnimation(
    fig, animate, frames=len(time),
    init_func=init, blit=True, interval=dt*1000
)

plt.show()