import numpy as np
import cvxpy as cp  # for the MPC example
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

# For LQR and pole placement
from scipy.linalg import solve_continuous_are
from scipy.signal import place_poles

###############################################################################
# Base Abstract Class
###############################################################################
class CartpoleController(ABC):
    """
    Abstract base class for cartpole controllers.
    Defines the interface for different control strategies (PID, LQR, Pole Placement, MPC).
    """

    def __init__(self, 
                 dt: float = 1/240.,
                 max_force: float = 100.0,
                 state_bounds: Optional[List[Tuple[float, float]]] = None):
        """
        Initialize the controller with basic parameters.
        
        Args:
            dt: Control timestep (seconds)
            max_force: Maximum allowed force (magnitude) for the cart
            state_bounds: Optional list of (min, max) for [x, x_dot, theta, theta_dot]
        """
        self.dt = dt
        self.max_force = max_force
        self.state_bounds = state_bounds or [
            (-5.0, 5.0),         # cart position
            (-10.0, 10.0),       # cart velocity
            (-np.pi/2, np.pi/2), # pole angle
            (-10.0, 10.0)        # pole angular velocity
        ]
        
        # Control output limits
        self.control_min = -max_force
        self.control_max = max_force
        
        # State tracking
        self.prev_state = None
        self.prev_error = None

    @abstractmethod
    def compute_control(self, 
                        state: List[float]) -> float:
        """
        Compute the control output based on current state.
        
        Args:
            state: [cart_pos, cart_vel, pole_angle, pole_vel]
        
        Returns:
            Control force to apply to the cart
        """
        pass

    def bound_control(self, u: float) -> float:
        """Limit the control output to within allowed bounds."""
        return np.clip(u, self.control_min, self.control_max)

    def bound_state(self, state: List[float]) -> List[float]:
        """Limit the state to within allowed bounds (if you want to limit unrealistic states)."""
        return [
            np.clip(state[i], self.state_bounds[i][0], self.state_bounds[i][1]) 
            for i in range(len(state))
        ]

    def reset(self) -> None:
        """Reset controller internal state."""
        self.prev_state = None
        self.prev_error = None

    def get_state_error(self, 
                        state: List[float], 
                        target: List[float] = None) -> List[float]:
        """
        Calculate error from a target state. 
        By default the target is [0,0,0,0], i.e. cart at origin, pole upright.
        
        Note: For the angle error, do wrap-around to keep it in (-pi, pi).
        """
        target = target or [0.0, 0.0, 0.0, 0.0]  # [x*, xdot*, theta*, thetadot*]
        
        # Handle angle wraparound for pole angle (theta)
        theta_error = state[2] - target[2]
        theta_error = (theta_error + np.pi) % (2 * np.pi) - np.pi
        
        return [
            state[0] - target[0],  # cart position error
            state[1] - target[1],  # cart velocity error
            theta_error,           # pole angle error
            state[3] - target[3]   # pole angular velocity error
        ]

###############################################################################
# PID Controller
###############################################################################
class PIDController(CartpoleController):
    """
    Simple PID controller focusing on the pole angle (3rd state) but easily extendable
    to multi-input if you wish.
    """
    def __init__(self, 
                 kp: float, 
                 ki: float, 
                 kd: float,
                 dt: float = 1/240.,
                 max_force: float = 100.0):
        super().__init__(dt, max_force)
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0

    def compute_control(self, state: List[float]) -> float:
        """
        Compute a simple PID force based on the pole angle error only.
        
        Args:
            state: [x, xdot, theta, thetadot]
        Returns:
            float: Force to apply
        """
        error = self.get_state_error(state)
        # error = [e_x, e_xdot, e_theta, e_thetadot]
        pole_angle_err = error[2]

        # Integrator
        self.integral += pole_angle_err * self.dt
        
        # Derivative
        if self.prev_error is not None:
            derivative = (pole_angle_err - self.prev_error[2]) / self.dt
        else:
            derivative = 0.0
        
        # PID formula: u = Kp*e + Ki*int(e) + Kd*de/dt
        u = (self.kp * pole_angle_err
             + self.ki * self.integral
             + self.kd * derivative)
        
        self.prev_error = error
        return self.bound_control(u)

###############################################################################
# LQR Controller
###############################################################################
class LQRController(CartpoleController):
    """
    Linear Quadratic Regulator controller for the linearized cartpole system.
    Minimizes the integral of (x^T Q x + u^T R u) over infinite horizon.
    """
    def __init__(self, 
                 A: np.ndarray, 
                 B: np.ndarray, 
                 Q: np.ndarray, 
                 R: np.ndarray,
                 dt: float = 1/240.,
                 max_force: float = 100.0):
        """
        Args:
            A, B: continuous-time system matrices
            Q, R: weighting matrices for states and input
        """
        super().__init__(dt, max_force)
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.K = self._solve_lqr(A, B, Q, R)  # Optimal gain matrix

    def _solve_lqr(self, A, B, Q, R):
        """
        Solve the continuous-time Algebraic Riccati Equation (ARE):
            A^T P + P A - P B R^-1 B^T P + Q = 0
        Then K = R^-1 B^T P
        """
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.inv(R) @ (B.T @ P)
        return K

    def compute_control(self, state: List[float]) -> float:
        """
        LQR Control law: u = -K x
        """
        state_bounded = self.bound_state(state)
        x_vec = np.array(state_bounded).reshape(-1, 1)  # column vector
        u = -self.K @ x_vec  # shape (1,1) or (m,1)
        return self.bound_control(u.item())

###############################################################################
# Pole Placement Controller
###############################################################################
class PolePlacementController(CartpoleController):
    """
    Pole Placement controller for the cartpole system.
    We pick desired closed-loop poles and compute K to place them there.
    """
    def __init__(self, 
                 A: np.ndarray, 
                 B: np.ndarray, 
                 desired_poles: List[complex],
                 dt: float = 1/240.,
                 max_force: float = 100.0):
        super().__init__(dt, max_force)
        self.A = A
        self.B = B
        self.desired_poles = desired_poles
        self.K = self._place_poles(A, B, desired_poles)

    def _place_poles(self, A, B, poles):
        """
        place_poles returns a result object whose .gain_matrix is the K we want
        """
        result = place_poles(A, B, poles)
        return result.gain_matrix

    def compute_control(self, state: List[float]) -> float:
        """
        Control law: u = -K x
        """
        state_bounded = self.bound_state(state)
        x_vec = np.array(state_bounded).reshape(-1, 1)
        u = -self.K @ x_vec
        return self.bound_control(u.item())

###############################################################################
# MPC Controller (Minimal Example)
###############################################################################
class MPCController(CartpoleController):
    """
    Minimal linear MPC controller using cvxpy for a receding-horizon approach.
    We do a naive c2d of (A,B), set up a horizon, and solve each step.
    """
    def __init__(self, 
                 A: np.ndarray,
                 B: np.ndarray,
                 Q: np.ndarray,
                 R: np.ndarray,
                 horizon: int = 10,
                 dt: float = 1/240.,
                 max_force: float = 100.0):
        super().__init__(dt, max_force)
        self.A_c = A
        self.B_c = B
        self.Q = Q
        self.R = R
        self.horizon = horizon

    def compute_control(self, state: List[float]) -> float:
        """
        Recursively solve the finite-horizon MPC problem at each timestep.
        """
        # Convert (A,B) to discrete with a small dt
        Ad = np.eye(4) + self.dt * self.A_c
        Bd = self.dt * self.B_c

        # We only do a single-step receding horizon
        # If performance is slow, reduce horizon or pick a faster solver
        x_var = cp.Variable((4, self.horizon+1))
        u_var = cp.Variable((1, self.horizon))

        cost = 0
        constraints = []
        # initial condition
        constraints.append(x_var[:,0] == state)

        for k in range(self.horizon):
            # cost function
            cost += cp.quad_form(x_var[:,k], self.Q) + cp.quad_form(u_var[:,k], self.R)
            # system dynamics
            constraints.append(x_var[:,k+1] == Ad @ x_var[:,k] + Bd @ u_var[:,k])
            # force constraint
            constraints.append(u_var[:,k] <= self.control_max)
            constraints.append(u_var[:,k] >= self.control_min)

        # terminal cost
        cost += cp.quad_form(x_var[:,self.horizon], self.Q)

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, verbose=False)

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            # fallback if no feasible solution
            return 0.0

        # first control action
        force = u_var.value[0,0]
        # Clip in case numerical issues
        force = np.clip(force, self.control_min, self.control_max)
        return float(force)

###############################################################################
# Quick Test / Demo
###############################################################################
if __name__ == "__main__":
    """
    Example usage with some dummy A, B, Q, R, poles.
    In a real cartpole, you'd derive or approximate these from a linearization
    about the upright equilibrium.
    """
    A_dummy = np.array([[0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [0, 0, 0, 0]], dtype=float)
    B_dummy = np.array([[0],
                        [0],
                        [0],
                        [1]], dtype=float)

    Q_dummy = np.eye(4)
    R_dummy = np.array([[1]])

    # LQR
    lqr_ctrl = LQRController(A_dummy, B_dummy, Q_dummy, R_dummy)
    # Pole Placement
    desired_poles = [-2, -2.5, -3, -3.5]
    pole_ctrl = PolePlacementController(A_dummy, B_dummy, desired_poles)
    # PID
    pid_ctrl = PIDController(kp=10.0, ki=0.1, kd=1.0)
    # MPC
    mpc_ctrl = MPCController(A_dummy, B_dummy, Q_dummy, R_dummy, horizon=5)

    # Test single-step control with a random-ish state
    test_state = [0.1, 0.0, 0.2, 0.0]  # x=0.1m, xdot=0, theta=0.2rad, thetadot=0
    print("===== Controller Demo =====")
    print("LQR control output:        ", lqr_ctrl.compute_control(test_state))
    print("Pole Placement output:     ", pole_ctrl.compute_control(test_state))
    print("PID control output:        ", pid_ctrl.compute_control(test_state))
    print("MPC control output:        ", mpc_ctrl.compute_control(test_state))
