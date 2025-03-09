import numpy as np
import pybullet as p

import numpy as np
import pybullet as p

def get_cartpole_state(cartpoleId) -> list:
    """
    Get the current state of the cartpole system.

    Args:
        cartpoleId: The unique ID of the cartpole in the PyBullet environment.

    Returns:
        list: [left_wheel_velocity, left_wheel_torque,
               right_wheel_velocity, right_wheel_torque,
               pole_angle, pole_angular_velocity]
    """

    # Fetch joint states for the left wheel (0), right wheel (1), and pole (4)
    joint_indices = [0, 1, 4]
    joint_states = p.getJointStates(cartpoleId, joint_indices)

    # Extract the velocity and applied torque from each joint state
    left_velocity = joint_states[0][1]  # joint velocity
    left_torque = joint_states[0][3]    # applied joint torque

    right_velocity = joint_states[1][1]
    right_torque = joint_states[1][3]

    # For the pole: joint angle and angular velocity
    pole_angle = joint_states[2][0]         # joint position (angle)
    pole_angular_velocity = joint_states[2][1]

    return [left_velocity, left_torque,
            right_velocity, right_torque,
            pole_angle, pole_angular_velocity]


