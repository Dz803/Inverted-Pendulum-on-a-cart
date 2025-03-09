import time
import pybullet as p
import pybullet_data
import numpy as np

from cartpole_system import CartpoleSystem
from controller import LQRController

def main():
    # 1) Connect to PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    planeId = p.loadURDF("plane.urdf")

    # 2) Load 4-wheel cart-pole URDF
    startPos = [0, 0, 0.08]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    cartpoleId = p.loadURDF(
        "./cart_pole/my_cartpole.urdf",
        startPos,
        startOrientation,
        useFixedBase=False
    )

    # Increase friction for wheels/plane
    p.changeDynamics(planeId, -1, lateralFriction=1.0)
    for link_idx in range(p.getNumJoints(cartpoleId)):
        if link_idx != 4:  # Skip the pole joint if you want
            p.changeDynamics(cartpoleId, link_idx, 
                             lateralFriction=5.0,
                             rollingFriction=0.1,
                             spinningFriction=0.1)

    # (B) Identify "drive wheels"
    #   joint0 => left_front_joint
    #   joint1 => right_front_joint
    #   joint2 => left_rear_joint
    #   joint3 => right_rear_joint
    #   joint4 => pole_joint
    left_joint_index = 2
    right_joint_index = 3
    pole_joint_index = 4  # If reading angle from there

    # 3) Build CartpoleSystem
    cartpole_sys = CartpoleSystem(body_id=cartpoleId,cart_joint_index=left_joint_index,pole_joint_index=pole_joint_index,
        M=0.28,  # total cart mass
        m=0.05,  # total pole mass
        l=0.30,
        g=9.81,
        delta=0.04,
        c=0.015,
        zeta=0.0,
        max_force=50.0
    )
    cartpole_sys.print_info()

    # 4) Create LQR from cartpole_sys A,B
    A = cartpole_sys.A
    B = cartpole_sys.B

    # Smaller Q, bigger R => less aggressive control
    Q = np.diag([2,2,2,1])
    R = np.array([[0.1]])
    dt = 1/240.0

    lqr_ctrl = LQRController(A, B, Q, R, dt=dt, max_force=cartpole_sys.max_force)

    # 5) Disable default velocity controls on wheels
    num_joints = p.getNumJoints(cartpoleId)
    for j in range(num_joints):
        p.setJointMotorControl2(
            bodyUniqueId=cartpoleId,
            jointIndex=j,
            controlMode=p.VELOCITY_CONTROL,
            force=0
        )

    # We'll define a smaller motor torque bound to mimic a low-power DC motor
    maxWheelTorque = 3.0  # in Nm (example). Adjust if still flying around

    try:
        while True:
            p.stepSimulation()
            time.sleep(dt)

            # 6) Read state from cartpole_sys
            state = cartpole_sys.get_cartpole_state()

            # Single scalar "force" from LQR
            force = lqr_ctrl.compute_control(state)

            # We'll treat "force" as total horizontal effort => convert to wheel torque
            # For simplicity, do torque_left=torque_right=force/2
            torque_left = 0.5 * force
            torque_right = 0.5 * force

            # Now clamp them to a "small DC motor" torque limit
            torque_left = np.clip(torque_left, -maxWheelTorque, maxWheelTorque)
            torque_right = np.clip(torque_right, -maxWheelTorque, maxWheelTorque)

            # Apply torque to front wheels with maxForce also set
            p.setJointMotorControl2(
                bodyIndex=cartpoleId,
                jointIndex=left_joint_index,
                controlMode=p.TORQUE_CONTROL,
                force=torque_left 
            )
            p.setJointMotorControl2(
                bodyUniqueId=cartpoleId,
                jointIndex=right_joint_index,
                controlMode=p.TORQUE_CONTROL,
                force=torque_right
            )

    except KeyboardInterrupt:
        print("Exiting simulation.")
    finally:
        p.disconnect()


if __name__ == "__main__":
    main()
