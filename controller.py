from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Optional

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
            max_force: Maximum allowed force for the cart motor
            state_bounds: Optional list of (min, max) tuples for [x, x_dot, theta, theta_dot]
        """
        self.dt = dt
        self.max_force = max_force
        self.state_bounds = state_bounds or [
            (-5.0, 5.0),    # cart position
            (-10.0, 10.0),  # cart velocity
            (-np.pi/2, np.pi/2),  # pole angle
            (-10.0, 10.0)   # pole angular velocity
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
            state: List of [cart_pos, cart_vel, pole_angle, pole_vel]
        
        Returns:
            float: Control force to apply to the cart
        """
        pass

    def bound_control(self, u: float) -> float:
        """Limit the control output to within allowed bounds."""
        return np.clip(u, self.control_min, self.control_max)

    def bound_state(self, state: List[float]) -> List[float]:
        """Limit the state to within allowed bounds."""
        return [np.clip(state[i], self.state_bounds[i][0], self.state_bounds[i][1]) 
                for i in range(len(state))]

    def reset(self) -> None:
        """Reset controller internal state."""
        self.prev_state = None
        self.prev_error = None

    def get_state_error(self, 
                       state: List[float], 
                       target: List[float] = None) -> List[float]:
        """
        Calculate error from target state.
        Default target is upright position at origin.
        """
        target = target or [0.0, 0.0, 0.0, 0.0]  # [x, x_dot, theta, theta_dot]
        
        # Handle angle wraparound for pole angle (theta)
        theta_error = state[2] - target[2]
        theta_error = (theta_error + np.pi) % (2 * np.pi) - np.pi
        
        return [
            state[0] - target[0],  # cart position error
            state[1] - target[1],  # cart velocity error
            theta_error,          # pole angle error
            state[3] - target[3]  # pole angular velocity error
        ]

# Example concrete implementations (skeletons):

class PIDController(CartpoleController):
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
        error = self.get_state_error(state)
        # Simple PID on pole angle (could be extended to full state)
        self.integral += error[2] * self.dt
        derivative = (error[2] - (self.prev_error[2] if self.prev_error else 0)) / self.dt
        
        u = (self.kp * error[2] + 
             self.ki * self.integral + 
             self.kd * derivative)
        
        self.prev_error = error
        return self.bound_control(u)


class LQRController(CartpoleController):
    def __init__(self, 
                 A: np.ndarray, 
                 B: np.ndarray, 
                 Q: np.ndarray, 
                 R: np.ndarray,
                 dt: float = 1/240.,
                 max_force: float = 100.0):
        super().__init__(dt, max_force)
        self.A = A  # System dynamics matrix
        self.B = B  # Control matrix
        self.K = self._solve_lqr(A, B, Q, R)  # Optimal gain matrix

    def _solve_lqr(self, A, B, Q, R):
        """Solve the LQR problem (implementation would use scipy or similar)"""
        # This is a placeholder - actual implementation would solve Riccati equation
        pass

    def compute_control(self, state: List[float]) -> float:
        state_bounded = self.bound_state(state)
        u = -np.dot(self.K, state_bounded)
        return self.bound_control(u[0])


class PolePlacementController(CartpoleController):
    def __init__(self, 
                 A: np.ndarray, 
                 B: np.ndarray, 
                 desired_poles: List[float],
                 dt: float = 1/240.,
                 max_force: float = 100.0):
        super().__init__(dt, max_force)
        self.K = self._place_poles(A, B, desired_poles)

    def _place_poles(self, A, B, poles):
        """Compute gain matrix for desired pole locations"""
        # Placeholder - would use control library or custom implementation
        pass

    def compute_control(self, state: List[float]) -> float:
        state_bounded = self.bound_state(state)
        u = -np.dot(self.K, state_bounded)
        return self.bound_control(u)


class MPCController(CartpoleController):
    def __init__(self, 
                 A: np.ndarray, 
                 B: np.ndarray, 
                 horizon: int,
                 Q: np.ndarray,
                 R: np.ndarray,
                 dt: float = 1/240.,
                 max_force: float = 100.0):
        super().__init__(dt, max_force)
        self.A = A
        self.B = B
        self.horizon = horizon
        self.Q = Q
        self.R = R

    def compute_control(self, state: List[float]) -> float:
        # Placeholder for MPC optimization
        # Would solve optimization problem over prediction horizon
        pass