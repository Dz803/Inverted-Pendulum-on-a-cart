#This is the main file that will be used to run the simulation of the cartpole in pybullet
#This file will be used to test the cartpole model and the controller that will be implemented in the future
# We will use the cartpole model that was created in the file 'my_cartpole_resized.urdf'
# The controller will be implemented in the file 'controller.py'
# The controller could consist of different types of controllers like PID, LQR, etc.


'''

The rotation of any motor, follows the right hand rule to calculate the direction of rotation.
For Euler angles, the right hand rule is used to determine the positive direction of rotation.
Specifically, the positive direction of the motor is when the thumb of the right hand points parallel to the axis of rotation
when the fingers are curled in the direction of the rotation.

We use setJointMotorControlArray() to control the two motors of the cart to balance the pole. 
The function takes in the following arguments:
- The id of the cartpole
- The indices of the joints that we want to control
- The control mode (VELOCITY_CONTROL, POSITION_CONTROL, TORQUE_CONTROL)
- The target position, velocity or torque
- The forces that we want to apply to the joints
- The maximum forces that we want to apply to the joints

'''


import pybullet as p
import pybullet_data
import time
import controller
import numpy as np
from utils import get_cartpole_state

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
planeId = p.loadURDF("plane.urdf")

startPos = [0, 0 ,0.08]
startOrientation = p.getQuaternionFromEuler([0,0,0])
cartpoleId = p.loadURDF("./cart_pole/my_cartpole.urdf",startPos,startOrientation, useFixedBase=False, flags=p.URDF_USE_SELF_COLLISION)



# Example usage in your simulation
PID = controller.PIDController(kp=1.0, ki=0.0, kd=0.0)




# Inspect joints
num_joints = p.getNumJoints(cartpoleId)
for i in range(num_joints):
    info = p.getJointInfo(cartpoleId, i)
    print(i, info[1].decode("utf-8"), info[2])  # name, jointType


# Set the motors to velocity control
rotary_joint_indices = list(range(0,5)) # The indices of the rotary joints to initialize
p.setJointMotorControlArray(cartpoleId, 
                            rotary_joint_indices, 
                            p.VELOCITY_CONTROL, 
                            forces=[0]*len(rotary_joint_indices)) # Initialize the motors to 0 velocity




#main loop 
try:
    while True:
        p.stepSimulation()
        p.setJointMotorControlArray(cartpoleId,
                                    [0,1],
                                    p.VELOCITY_CONTROL,
                                    targetVelocities=[10]*2 )
        time.sleep(1/240.)


        '''
        # Get the state of the cartpole
        state = get_cartpole_state(cartpoleId)
        control_force = PID.compute_control(state)
        p.setJointMotorControlArray(cartpoleId,
                                [0],  # just cart joint
                                p.TORQUE_CONTROL,
                                forces=[control_force])'
        '''
        print(get_cartpole_state(cartpoleId))

except KeyboardInterrupt:
    p.disconnect()
