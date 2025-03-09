# cartpole_system.py

import pybullet as p
import numpy as np

class CartpoleSystem:
    """
    Encapsulates the cartpole's physical constants, linearized system matrices,
    constraints, and a helper method to retrieve states from PyBullet.

    Using a 'direct approach' to interpret the cart's linear position
    and velocity from a drive wheel's angular displacement and velocity.
    """

    def __init__(self,
                 body_id: int,
                 cart_joint_index: int = 0,
                 pole_joint_index: int = 1,
                 M=0.28,  # total cart mass
                 m=0.05,  # total pole mass
                 l=0.30,
                 g=9.81,
                 delta=0.04,
                 c=0.015,
                 zeta=0.0,
                 max_force=50.0,
                 wheel_radius=0.03):
        """
        Args:
            body_id: PyBullet body unique ID for the cartpole URDF
            cart_joint_index: which joint index corresponds to the drive wheel
            pole_joint_index: which joint index is for the pole rotation
            M, m, l: approximate masses/length from URDF (or manual)
            g: gravity
            delta: cart friction coefficient
            c: pivot friction
            zeta: extra coupling factor
            max_force: maximum allowed force
            wheel_radius: radius of the drive wheel
        """
        self.body_id = body_id
        self.cart_joint_index = cart_joint_index
        self.pole_joint_index = pole_joint_index

        self.M = M
        self.m = m
        self.l = l
        self.g = g

        # friction/damping
        self.delta = delta
        self.c = c
        self.zeta = zeta

        self.max_force = max_force
        self.wheel_radius = wheel_radius  # for direct approach x = R*phi

        # Build the linearized system matrices A,B using friction terms around upright eq
        self.A, self.B = self._build_state_space_matrices()

    def _build_state_space_matrices(self):
        """
        Return the continuous-time linearized (A,B) around upright eq:
            x=0, dot{x}=0, theta=0, dot{theta}=0, F=0

        States: [x, x_dot, theta, theta_dot]
        Input:  F
        """
        M = self.M
        m = self.m
        g = self.g
        l = self.l
        delta = self.delta
        c = self.c
        zeta = self.zeta

        # Extended linearization with friction:
        # A = [[0,       1,          0,         0],
        #      [0, -delta/M,   -(m*g)/M,    zeta/M],
        #      [0,       0,          0,         1],
        #      [0, delta/(l*M), (M+m)*g/(l*M), -(c*(M+m))/(m*l^2*M] ]
        A = np.array([
            [0,            1,               0,                             0],
            [0,   -delta/M,       -(m*g)/M,                    zeta/M      ],
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
        Using the 'direct approach':
          - We interpret cart position, x, from the drive wheel's encoder angle:  x = wheel_radius * angle
          - We interpret cart velocity, x_dot, from the wheel's angular velocity: x_dot = wheel_radius * angle_dot
          - We read the pole joint to get (theta, theta_dot)

        Returns a 1D numpy array: [x, x_dot, theta, theta_dot]
        """
        # 1) Read the drive wheel joint
        wheel_info = p.getJointState(self.body_id, self.cart_joint_index)
        wheel_angle = wheel_info[0]       # revolute angle
        wheel_angular_vel = wheel_info[1] # d(angle)/dt

        # Convert to linear displacement/velocity
        x_pos = self.wheel_radius * wheel_angle
        x_dot = self.wheel_radius * wheel_angular_vel

        # 2) Read the pole revolve joint
        pole_info = p.getJointState(self.body_id, self.pole_joint_index)
        theta = pole_info[0]
        theta_dot = pole_info[1]

        # Return the 4D state
        return np.array([x_pos, x_dot, theta, theta_dot], dtype=float)

    def print_info(self):
        """Simple debug function to show constants and A,B."""
        print("\n=== Cartpole System Info ===")
        print(f"BodyID: {self.body_id}")
        print(f"cart_joint_index={self.cart_joint_index}, pole_joint_index={self.pole_joint_index}")
        print(f"M={self.M}, m={self.m}, l={self.l}, g={self.g}")
        print(f"delta={self.delta}, c={self.c}, zeta={self.zeta}")
        print(f"wheel_radius={self.wheel_radius}")
        print("A matrix:")
        print(self.A)
        print("B matrix:")
        print(self.B)
        print("Max force:", self.max_force)
        print("===========================\n")
