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

    # (A) Example friction settings
    #  Increase friction so the wheels actually get traction
    p.changeDynamics(planeId, -1, lateralFriction=1.0)
    for link_idx in range(p.getNumJoints(cartpoleId)):
        p.changeDynamics(cartpoleId, link_idx, lateralFriction=1.0)

    # (B) Identify "drive wheels"
    #  According to URDF:
    #   joint0 => left_front_joint
    #   joint1 => right_front_joint
    #   joint2 => left_rare_joint (typo "rare"? = "rear"?)
    #   joint3 => right_rare_joint
    #   joint4 => pole_joint
    # We'll pick the FRONT wheels: joint0, joint1 as the drive wheels
    left_front_joint_index = 0
    right_front_joint_index = 1
    pole_joint_index = 4  # if we read angle from there

    # 3) Build the CartpoleSystem
    #   It's still going to guess a "linearized" A,B for your balancing
    #   We'll read total cart mass ~0.18, total pole mass ~0.05, etc.
    cartpole_sys = CartpoleSystem(
        body_id=cartpoleId,
        cart_joint_index=left_front_joint_index,  # We'll just store one for reference
        pole_joint_index=pole_joint_index,
        M=0.18,     # total cart mass ~ 0.1 + 4*0.02
        m=0.05,     # total pendulum mass 0.05, pole mass neglected
        l=0.30,     # approx length
        g=9.81,
        delta=0.04, # friction
        c=0.015,
        zeta=0.0,
        max_force=50.0
    )
    cartpole_sys.print_info()

    # 4) Create LQR from cartpole_sys A,B
    A = cartpole_sys.A
    B = cartpole_sys.B
    Q = np.diag([0.5, 0.5, 5, 0.5])  # smaller weighting
    R = np.array([[0.5]])            # bigger weighting on torque

    dt = 1/240.0

    lqr_ctrl = LQRController(A, B, Q, R, dt=dt, max_force=cartpole_sys.max_force)

    # 5) Disable default velocity controls on the 4 wheels
    num_joints = p.getNumJoints(cartpoleId)
    for j in range(num_joints):
        p.setJointMotorControl2(
            bodyUniqueId=cartpoleId,
            jointIndex=j,
            controlMode=p.VELOCITY_CONTROL,
            force=0
        )

    # 6) Main loop: measure "x, x_dot, theta, theta_dot"
    #   We'll rely on get_cartpole_state, but that might only read the
    #   "pole_joint" for angle. For "x" we might approximate from the base's x pos.
    try:
        while True:
            p.stepSimulation()
            time.sleep(dt)

            # Read state from cartpole_sys
            state = cartpole_sys.get_cartpole_state()

            # We'll interpret "x, x_dot" from the base link, or from the motion in the X direction
            # If cartpole_sys is reading them from the base link properly, great.

            force = lqr_ctrl.compute_control(state)  # a single scalar "F"

            # Now apply that as torque on the FRONT wheels:
            # e.g. half the torque to left_rear, half to right_rear
            # Convert from "force" to "wheel torque"
            # For simplicity,just do torqueLeft=torqueRight=force/2
            torque_left = force * 0.5
            torque_right = force * 0.5

            # Apply torque control
            p.setJointMotorControl2(
                bodyUniqueId=cartpoleId,
                jointIndex=left_front_joint_index,
                controlMode=p.TORQUE_CONTROL,
                force=torque_left
            )
            p.setJointMotorControl2(
                bodyUniqueId=cartpoleId,
                jointIndex=right_front_joint_index,
                controlMode=p.TORQUE_CONTROL,
                force=torque_right
            )

    except KeyboardInterrupt:
        print("Exiting simulation.")
    finally:
        p.disconnect()


if __name__ == "__main__":
    main()
