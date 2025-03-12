import numpy as np
import cvxpy as cp  # for possible MPC usage
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

# For LQR, pole placement, etc.
from scipy.linalg import solve_continuous_are
from scipy.signal import place_poles



###############################################################################
# Base Abstract Class
###############################################################################
class CartpoleController(ABC):
    """
    Abstract base class for cartpole controllers.
    Defines the interface for different control strategies (PID, LQR, Pole Placement, MPC, Nonlinear, etc.).
    
    State convention (4D):
       state = [x, x_dot, theta, theta_dot]
    where theta = 0 means the pole is UPRIGHT (if that's your chosen convention).
    """

    def __init__(self, 
                 dt: float = 1/240.,
                 max_force: float = 100.0,
                 state_bounds: Optional[List[Tuple[float, float]]] = None,
                 target_state: Optional[List[float]] = None):
        """
        Initialize the controller with basic parameters.

        Args:
            dt: Control timestep (seconds).
            max_force: Maximum allowed force (magnitude) for the cart.
            state_bounds: Optional list of (min, max) for each state dimension [x, x_dot, theta, theta_dot].
            target_state: Desired setpoint [x*, xdot*, theta*, thetadot*] for regulation.
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
        
        # For storing previous step info if needed
        self.prev_state = None
        self.prev_error = None

        # Target state
        if target_state is None:
            # default => [0, 0, 0, 0]: cart at 0, upright pole
            self.target_state = [0.0, 0.0, 0.0, 0.0]
        else:
            self.target_state = target_state

    @abstractmethod
    def compute_control(self, state: List[float]) -> float:
        """
        Compute the control output based on current state.

        Args:
            state: [x, x_dot, theta, theta_dot]

        Returns:
            Control force to apply to the cart
        """
        pass

    def set_target_state(self, target: List[float]) -> None:
        """Update the desired setpoint for this controller."""
        self.target_state = target

    def bound_control(self, u: float) -> float:
        """Limit the control output to within allowed bounds."""
        return np.clip(u, self.control_min, self.control_max)

    def bound_state(self, state: List[float]) -> List[float]:
        """Limit each state dimension to within specified bounds (if you want to clamp unrealistic states)."""
        return [
            np.clip(state[i], self.state_bounds[i][0], self.state_bounds[i][1]) 
            for i in range(len(state))
        ]

    def reset(self) -> None:
        """Reset any internal state (integral terms, prev errors, etc.)."""
        self.prev_state = None
        self.prev_error = None

    def get_state_error(self, state: List[float]) -> List[float]:
        """
        Calculate error = (state - target_state).
        For the angle dimension, we often do a wrap-around so the angle error stays in (-pi, pi).
        """
        # e.g. e_theta = (theta - theta_target) mod 2pi in (-pi, pi)
        theta_error = state[2] - self.target_state[2]
        theta_error = (theta_error + np.pi) % (2 * np.pi) - np.pi
        
        return [
            state[0] - self.target_state[0],  # x - x*
            state[1] - self.target_state[1],  # xdot - xdot*
            theta_error,                      # wrapped angle error
            state[3] - self.target_state[3]   # thetadot - thetadot*
        ]


###############################################################################
# 1) PID Controller
###############################################################################
class PIDController(CartpoleController):
    """
    Simple PID controller focusing on the pole angle (3rd state) 
    but easily extendable to multi-input if you wish.
    """
    def __init__(self, 
                 kp: float, 
                 ki: float, 
                 kd: float,
                 dt: float = 1/240.,
                 max_force: float = 100.0,
                 target_state: Optional[List[float]] = None):
        super().__init__(dt, max_force, target_state=target_state)
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0

    def compute_control(self, state: List[float]) -> float:
        """
        Compute a simple PID force based on the pole angle error 
        or whichever state dimension you want.

        Args:
            state: [x, xdot, theta, thetadot]

        Returns:
            float: Force to apply
        """
        error_vec = self.get_state_error(state)
        # e = [ex, e_xdot, e_theta, e_thetadot]
        # We'll do PID on e_theta (index=2)
        pole_angle_err = error_vec[2]

        # Integrator
        self.integral += pole_angle_err * self.dt
        
        # Derivative
        if self.prev_error is not None:
            derivative = (pole_angle_err - self.prev_error[2]) / self.dt
        else:
            derivative = 0.0
        
        # PID formula: u = Kp*e + Ki*int(e) + Kd*d(e)/dt
        u = (self.kp * pole_angle_err
             + self.ki * self.integral
             + self.kd * derivative)
        
        self.prev_error = error_vec
        return self.bound_control(u)


###############################################################################
# 2) Pole Placement Controller
###############################################################################
from scipy.linalg import solve_continuous_are

# 在controller.py中添加以下类
from scipy.linalg import solve_continuous_are

class ContinuousLQRController(CartpoleController):
    """
    Continuous-time LQR Controller using Algebraic Riccati Equation
    
    System dynamics: dx/dt = A x + B u
    Cost function: J = ∫(x.T Q x + u.T R u) dt
    """
    
    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray,
                 max_force: float = 20.0, target_state: Optional[List[float]] = None):
        
        super().__init__(dt=0, max_force=max_force, target_state=target_state)  # dt not used in continuous
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        
        # Solve Continuous-time Algebraic Riccati Equation
        self.P = solve_continuous_are(A, B, Q, R)
        
        # Compute optimal gain matrix
        self.K = np.linalg.inv(R) @ self.B.T @ self.P
        
        # Check stability
        self._verify_stability()

    def _verify_stability(self):
        """Check closed-loop eigenvalues"""
        closed_loop_A = self.A - self.B @ self.K
        print("Q, R:", self.Q, self.R)
        eigenvalues = np.linalg.eigvals(closed_loop_A)
        print("Eigenvalues of (A - B*K):", eigenvalues)
        if np.any(np.real(eigenvalues) >= 0):
            warnings.warn("Unstable closed-loop system!", RuntimeWarning)

    def compute_control(self, state: List[float]) -> float:
        """Continuous control law: u = -K(x - x_ref)"""
        error = np.array(self.get_state_error(state))
        u = -self.K @ error
        return self.bound_control(u.item())


###############################################################################
# 3) Nonlinear Controller (Partial Feedback Linearization)
###############################################################################
class NonlinearController(CartpoleController):
    """
    An example of a simple partial feedback linearization for 
    stabilizing the pole upright (theta = 0) in the cart-pole system.

    We'll assume:
       - The pole is a point mass at distance l from pivot (so moment of inertia I = m*l^2).
       - We do a PD law on theta to get a desired theta_ddot.
       - Solve the cart-pole dynamics for the needed 'u' (force) that achieves that acceleration.
    
    The continuous equations for a cart-pole (with I = m l^2) are:

       (1) (M + m) x_ddot + m l cos(theta)*theta_ddot - m l sin(theta)* (theta_dot^2) = u
       (2) (I + m l^2) theta_ddot + m l cos(theta)* x_ddot = m g l sin(theta)

    We'll define:
       desired theta_ddot = - Kp*(theta) - Kd*(theta_dot)
       solve eqn(2) for x_ddot
       then plug x_ddot, theta_ddot into eqn(1) to get u.
    """
    def __init__(self,
                 M: float,  # cart mass
                 m: float,  # pole mass
                 l: float,  # half pole length (or full pivot length if that's your URDF)
                 kp: float, # PD gains on angle
                 kd: float, 
                 dt: float = 1/240.,
                 max_force: float = 100.0,
                 target_state: Optional[List[float]] = None):
        """
        If your physical pole is truly from pivot to tip = l, 
        then moment of inertia I ~ m*l^2 for a point mass at distance l.
        Adjust if you have a distributed mass or different pivot offset.
        """
        super().__init__(dt, max_force, target_state=target_state)
        self.M = M
        self.m = m
        self.l = l
        self.I = m * (l**2)   # point mass at the tip
        self.g = 9.81
        self.kp = kp
        self.kd = kd

    def compute_control(self, state: List[float]) -> float:
        """
        1) PD on (theta, theta_dot) => desired theta_ddot
        2) Solve eqn(2) for x_ddot
        3) Plug x_ddot, theta_ddot into eqn(1) => control = u
        """
        # Unpack state
        x, xdot, theta, thetadot = state

        # PD law for angle => theta_ddot_des
        # target angle is self.target_state[2], assume 0 if we want upright
        angle_error = theta - self.target_state[2]
        angle_error = (angle_error + np.pi) % (2*np.pi) - np.pi  # wrap
        angle_rate_error = thetadot - self.target_state[3]

        theta_ddot_des = -self.kp * angle_error - self.kd * angle_rate_error

        # eqn(2): (I + m l^2)*theta_ddot + m l cos(theta)* x_ddot = m g l sin(theta)
        # solve for x_ddot:
        # x_ddot = [ m*g*l sin(theta) - (I + m*l^2)*theta_ddot_des ] / [ m*l cos(theta) ]
        # watch out if cos(theta) ~ 0 => near horizontal...
        cos_th = np.cos(theta)
        sin_th = np.sin(theta)

        # If cos_th is near zero, partial feedback linearization can blow up.
        # We'll just do a quick check or clamp for numeric safety:
        if abs(cos_th) < 1e-4:
            # fallback: or just clamp
            cos_th = np.sign(cos_th)*1e-4

        num = self.m*self.g*self.l*sin_th - (self.I + self.m*(self.l**2))*theta_ddot_des
        den = self.m*self.l*cos_th
        x_ddot_des = num / den

        # eqn(1): (M + m)* x_ddot + m l cos(theta)*theta_ddot - m l sin(theta)* (theta_dot^2) = u
        # plug in x_ddot_des and theta_ddot_des
        u = (self.M + self.m)*x_ddot_des \
            + self.m*self.l*cos_th*theta_ddot_des \
            - self.m*self.l*sin_th*(thetadot**2)

        # bound the final control
        return self.bound_control(u)




########################################################################
# Discrete Pole-Placement Controller
########################################################################
class DiscretePolePlacementController(CartpoleController):
    """
    Discrete-time pole placement on the linearized cartpole system.
    We:
      - compute (Ad, Bd) from (A, B)
      - place poles in the z-plane
      - store K
      - at each discrete step k, do u[k] = -K (x[k] - x_ref)
    For sim2real, you normally run this in a loop at dt intervals,
    but in this example, we'll still be calling it inside solve_ivp for illustration.
    """

    def __init__(self, Ad, Bd, desired_poles, dt, max_force=20.0, target_state=None):
        """
        Ad, Bd: discrete-time system (nxn, nx1)
        desired_poles: e.g. [0.9, 0.8, 0.7, 0.6]
        dt: sampling period for the discrete-time controller
        max_force: saturate output
        target_state: [x*, xdot*, theta*, thetadot*]
        """
        self.Ad = Ad
        self.Bd = Bd
        self.dt = dt
        self.max_force = max_force

        # default target
        if target_state is None:
            target_state = [0, 0, 0, 0]
        self.target_state = np.array(target_state)

        # Compute K
        placed = place_poles(Ad, Bd, desired_poles)
        self.K = placed.gain_matrix  # shape (1,4)

        # for discrete stepping
        self.last_update_time = 0.0
        self.u_current = 0.0
        self.x_error_prev = None

    def compute_control(self, t, state):
        """
        We'll only update the control every dt seconds,
        otherwise hold it (zero-order hold).
        """
        # If enough time passed to do a 'discrete step'
        if t - self.last_update_time >= self.dt:
            # compute error
            x_err = self.get_state_error(state)
            # discrete control law: u = -K e
            u = -(self.K @ x_err)
            # saturate
            u = np.clip(u.item(), -self.max_force, self.max_force)

            # store
            self.u_current = u
            self.last_update_time = t

        return self.u_current


from scipy.linalg import solve_discrete_are

class DiscreteLQRController(CartpoleController):
    """
    Discrete-time LQR controller for the cart-pole system.
    It uses the discrete algebraic Riccati equation (DARE) to compute the optimal gain.
    
    The discrete system is:
       x[k+1] = A_d x[k] + B_d u[k]
    and the control law is:
       u[k] = -K (x[k] - x_ref)
    where K is computed as:
       P = solve_discrete_are(A_d, B_d, Q, R)
       K = (B_d^T P B_d + R)^{-1} B_d^T P A_d
    """
    def __init__(self, Ad: np.ndarray, Bd: np.ndarray, Q: np.ndarray, R: np.ndarray,
                 dt: float = 1/100., max_force: float = 20.0, target_state: Optional[List[float]] = None):
        super().__init__(dt, max_force, target_state=target_state)
        self.Ad = Ad
        self.Bd = Bd
        self.dt = dt
        if target_state is None:
            target_state = [0, 0, 0,0]
        self.target_state = np.array(target_state)
        # Solve DARE to get the optimal P and then compute K.
        P = solve_discrete_are(Ad, Bd, Q, R)
        self.K = np.linalg.inv(Bd.T @ P @ Bd + R) @ (Bd.T @ P @ Ad)
        
        # For discrete stepping (sample-and-hold)
        self.last_update_time = 0.0
        self.u_current = 0.0

        eigenvalues = np.linalg.eig(Ad - Bd @ self.K)[0]
        print(np.abs(eigenvalues))  # Should all be < 1

    def compute_control(self, t, state) -> float:
        """
        Update the control every dt seconds; otherwise, hold the previous value.
        """
        if t - self.last_update_time >= self.dt:
            # Compute error: error = (state - target_state)
            x_err = np.array(self.get_state_error(state)).reshape(-1, 1)
            u = -(self.K @ x_err)
            u = np.clip(u.item(), -self.max_force, self.max_force)
            self.u_current = u
            self.last_update_time = t
        return self.u_current
