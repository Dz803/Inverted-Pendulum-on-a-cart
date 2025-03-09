import numpy as np
import pybullet as p

def get_cartpole_state(cartpoleId) -> list:
    """
    Get the current state of the cartpole system.

    Args:
        cartpoleId: The unique ID of the cartpole in the PyBullet environment.

    Returns:
        list: [left_velocity, left_torque, right_velocity, right_torque, pole_angle, pole_angular_velocity]
    """

    states = p.getJointStates(cartpoleId, [0,1,4])
  
    
    return states


