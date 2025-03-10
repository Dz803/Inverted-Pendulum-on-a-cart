import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import place_poles

##############################################################################
#            INVERTED PENDULUM ON CART - NEW NONLINEAR EQUATIONS
#  (M + m)*x'' + m*l*cos(theta)*theta'' - m*l*(theta')^2*sin(theta) = F
#  m*l*cos(theta)*x'' + (I + m*l^2)*theta'' - m*g*l*sin(theta)      = 0
##############################################################################

# =========================
# 1) PHYSICAL PARAMETERS
# =========================
M  = 5.0     # Cart mass (kg)
m  = 1.0     # Pendulum mass (kg)
I  = 0.2     # Pendulum's moment of inertia about pivot (kg*m^2)
l  = 1.0     # Pivot-to-centre-of-mass length (m)
g  = 9.81    # Gravity (m/s^2)

# For linearization:
#  Delta = (M + m)*(I + m*l^2) - (m*l)**2
Delta = (M + m)*(I + m*l*l) - (m*l)**2

# ==============================
# 2) BUILD LINEARISED A, B
# ==============================
"""
Around theta=0 (upright):
  Let states be x1=x, x2=x_dot, x3=theta, x4=theta_dot.
  Then x_dot = A*x + B*F.

  We'll compute A, B by direct partial derivatives. For brevity, 
  a simplified final result:

    A = [[0, 1,  0, 0],
         [0, 0, -(m*g*l)/Delta,    0],
         [0, 0,  0, 1],
         [0, 0,  ((M+m)*m*g*l)/(Delta*l),  0]]

    B = [[ 0 ],
         [ (I + m*l^2)/Delta ],
         [ 0 ],
         [ -(m*l)/Delta ]]

(You can confirm signs carefully. 
Double-check that ( (M+m)*m*g*l ) / (Delta*l ) = m*g*(M+m)/Delta. )
"""

A_lin = np.array([
    [0,                   1,      0, 0],
    [0,                   0, -(m*g*l)/Delta, 0],
    [0,                   0,      0, 1],
    [0,                   0,  (m*(M+m)*g*l)/(Delta*l), 0]
], dtype=float)

B_lin = np.array([
    [0],
    [(I + m*l*l)/Delta],
    [0],
    [-(m*l)/Delta]
], dtype=float)

# -------------------------------------------
# 3) POLE PLACEMENT (LINEARISED MODEL)
# -------------------------------------------
desired_poles = np.array([-2.0, -2.2, -2.5, -3.0])
K_lin = place_poles(A_lin, B_lin, desired_poles).gain_matrix

# ------------------------------------------------
# 4) SIMULATION OF FULL NONLINEAR DYNAMICS
#    Using Euler integration & control u = -K*x
# ------------------------------------------------

dt = 0.01
t_final = 10.0
time = np.arange(0, t_final, dt)

# States: x, x_dot, theta, theta_dot
x0 = np.array([0.0, 0.0, 0.1, 0.0])  #  small tilt from upright
X = x0.reshape(-1, 1)

state_history = [X.flatten()]

def dynamics_nonlinear(state, force):
    """
    Given [x, x_dot, theta, theta_dot], compute [x_dot, x_ddot, theta_dot, theta_ddot]
    using the full nonlinear eqns with moment of inertia I included.

      (M + m)* x'' + m*l*cos(theta)* theta'' - m*l*(theta')^2 sin(theta) = F
      m*l*cos(theta)* x'' + (I + m*l^2)* theta'' - m*g*l*sin(theta)      = 0
    """
    x, xdot, th, thdot = state
    # Force = F:
    F = force

    # Solve the 2 eqns for x_ddot, th_ddot:

    # 1) (M+m)*x_ddot = F + m*l*(theta')^2 sin(theta) - m*l*cos(theta)* th_ddot
    # 2) (I + m*l^2)*th_ddot = m*g*l sin(theta) - m*l*cos(theta)* x_ddot
    #
    # We'll do it by direct substitution or iterative approach:

    # Let's define x_ddot = A - B*th_ddot, where:
    #   A = [F + m*l*(thdot^2)*sin(th)] / (M+m)
    #   B = [m*l*cos(th)] / (M+m)
    A = (F + m*l*(thdot**2)*np.sin(th)) / (M + m)
    B = (m*l*np.cos(th)) / (M + m)

    # Then from eqn (2):
    # (I + m*l^2)*th_ddot = m*g*l sin(th) - m*l*cos(th)* x_ddot
    # => th_ddot = [m*g*l sin(th) - m*l*cos(th)*(A - B*th_ddot)] / (I + m*l^2)
    # => (I + m*l^2)*th_ddot + m*l*cos(th)*B*th_ddot = m*g*l sin(th) - m*l*cos(th)*A
    # => th_ddot * [ (I + m*l^2) + m*l*cos(th)*B ] = m*g*l sin(th) - m*l*cos(th)*A

    denom = (I + m*l*l) + m*l*np.cos(th)*B  # the bracket
    rhs   = m*g*l*np.sin(th) - m*l*np.cos(th)*A
    th_ddot = rhs / denom

    # Finally x_ddot = A - B*th_ddot
    x_ddot = A - B*th_ddot

    return np.array([xdot, x_ddot, thdot, th_ddot])

for _ in time[1:]:
    # Current states
    x, xdot, th, thdot = X.flatten()

    # Control from linear state feedback
    # (Even though the system is truly nonlinear)
    F_matrix = -K_lin @ X  # shape (1,1)
    # Convert to scalar
    F = F_matrix.item()

    # Compute derivatives via the nonlinear eqns
    dX = dynamics_nonlinear((x, xdot, th, thdot), F)

    # Euler update
    X = X + dt * dX.reshape(-1, 1)
    state_history.append(X.flatten())

state_history = np.array(state_history)

# ------------------------------------------------
# 5) ANIMATION
# ------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlim([-2, 2])
ax.set_ylim([0, 2])  # so we can see the rod above the pivot
ax.set_aspect('equal', 'box')
ax.set_title('Inverted Pendulum with Inertia I, Nonlinear EOM, Pole Placement')

(cart_line,) = ax.plot([], [], lw=6, color='blue')
(rod_line,)  = ax.plot([], [], lw=3, color='red')
cart_width = 0.3

def init():
    cart_line.set_data([], [])
    rod_line.set_data([], [])
    return cart_line, rod_line

def animate(i):
    # State
    x, xdot, th, thdot = state_history[i]
    # If theta=0 => rod is upright
    # We'll measure clockwise from up, so tip is:
    tip_x = x + l*np.sin(th)
    tip_y = l*np.cos(th)

    # Cart corners
    left  = x - cart_width/2
    right = x + cart_width/2
    y_cart = 0

    # Update lines
    cart_line.set_data([left, right], [y_cart, y_cart])
    rod_line.set_data([x, tip_x], [0, tip_y])
    return cart_line, rod_line

ani = FuncAnimation(
    fig, animate, frames=len(time),
    init_func=init, blit=True, interval=dt*1000
)

plt.show()