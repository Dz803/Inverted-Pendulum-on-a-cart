# cartpole_system.py

import pybullet as p
import numpy as np

class CartpoleSystem:
    """
    Encapsulates the cartpole's physical constants, linearized system matrices,
    constraints, and a helper method to retrieve states from PyBullet.

    This is a template. You can adapt friction terms, pivot damping, etc. 
    per your exact equations and URDF details.
    """

    def __init__(self,
                 body_id: int,
                 cart_joint_index: int = 0,
                 pole_joint_index: int = 1,
                 M: float = 1.0,     # cart mass
                 m: float = 0.1,    # pole mass
                 l: float = 0.5,    # pole length
                 g: float = 9.81,
                 delta: float = 0.05,  # cart friction
                 c: float = 0.02,      # pivot friction
                 zeta: float = 0.0,    # extra pivot torque factor
                 max_force: float = 100.0):
        """
        Args:
            body_id: PyBullet body unique ID for the cartpole URDF
            cart_joint_index: which joint index is for cart translation
            pole_joint_index: which joint index is for pole rotation
            M, m, l: approximate masses/length from URDF (or manual)
            g: gravity
            delta: cart friction coefficient
            c: pivot friction
            zeta: extra coupling factor
            max_force: maximum allowed force (for saturating if needed)
        """
        self.body_id = body_id
        self.cart_joint_index = cart_joint_index
        self.pole_joint_index = pole_joint_index

        # Physical constants
        self.M = M
        self.m = m
        self.l = l
        self.g = g

        # friction/damping
        self.delta = delta
        self.c = c
        self.zeta = zeta

        self.max_force = max_force  # for optional usage

        # Build the linearized system matrices A,B using the friction terms
        # around the upright equilibrium (x=0, theta=0).
        self.A, self.B = self._build_state_space_matrices()

    def _build_state_space_matrices(self):
        """
        Return the continuous-time linearized (A,B) around upright eq:
            x=0, dot{x}=0, theta=0, dot{theta}=0, F=0

        States: [x, x_dot, theta, theta_dot]
        Input:  F
        """
        # Unpack
        M = self.M
        m = self.m
        g = self.g
        l = self.l
        delta = self.delta
        c = self.c
        zeta = self.zeta

        # From your linearization with friction:
        #   A = [[0,       1,                0,               0],
        #        [0, -delta/M,      -(m*g)/M,         zeta/M   ],
        #        [0,       0,                0,               1],
        #        [0, delta/(l*M), (M+m)*g/(l*M),  -(c*(M+m))/(m*l^2*M] ]
        #   B = [[ 0 ],
        #        [ -1/M ],
        #        [ 0 ],
        #        [ -1/(l*M) ] ]
        A = np.array([
            [0,            1,               0,                             0],
            [0,    -delta/M,       -(m*g)/M,                    zeta/M      ],
            [0,            0,               0,                             1],
            [0,  delta/(l*M), (M + m)*g/(l*M), - (c*(M + m)) / (m*l**2 * M)]
        ])
        B = np.array([
            [0],
            [-1/M],
            [0],
            [-1/(l*M)]
        ])

        return A, B

    def get_cartpole_state(self) -> np.ndarray:
        """
        Read the cartpole's current state from PyBullet.

        Returns a 1D numpy array:
            [x, x_dot, theta, theta_dot]
        """
        cart_info = p.getJointState(self.body_id, self.cart_joint_index)
        pole_info = p.getJointState(self.body_id, self.pole_joint_index)

        x_pos = cart_info[0]
        x_dot = cart_info[1]
        theta = pole_info[0]
        theta_dot = pole_info[1]

        return np.array([x_pos, x_dot, theta, theta_dot], dtype=float)

    def print_info(self):
        """Simple debug function to show constants and A,B."""
        print("\n=== Cartpole System Info ===")
        print(f"BodyID: {self.body_id}")
        print(f"cart_joint_index={self.cart_joint_index}, pole_joint_index={self.pole_joint_index}")
        print(f"M={self.M}, m={self.m}, l={self.l}, g={self.g}")
        print(f"delta={self.delta}, c={self.c}, zeta={self.zeta}")
        print("A matrix:")
        print(self.A)
        print("B matrix:")
        print(self.B)
        print("Max force:", self.max_force)
        print("===========================\n")
