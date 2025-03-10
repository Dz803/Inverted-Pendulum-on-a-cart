import time
import pybullet as p
import pybullet_data
import numpy as np

from cartpole_system import CartpoleSystem
from controller import LQRController
import controller

def estimate_pole_length(cartpoleId, pole_link_index):
    """
    Estimate the pole length by subtracting the AABB min from AABB max 
    along the primary axis (e.g. z-axis) if your pole is oriented vertically.
    """
    aabb_min, aabb_max = p.getAABB(cartpoleId, pole_link_index)
    
    # the pole is aligned along z, then pole_length ≈ maxZ - minZ
    pole_length_z = aabb_max[2] - aabb_min[2]
    
    return pole_length_z

def main():
    # 1) Connect to PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    planeId = p.loadURDF("plane.urdf")

    # 2) Load 4-wheel cart-pole URDF
    startPos = [0, 0, 0.07]
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
        m=0.075,  # total pole mass
        l=0.3222, # pole length
        g=9.81,
        delta=0.04,
        c=0.015,
        zeta=0.0,
        max_force=50.0
    )
    cartpole_sys.print_info()

    # 4) Disable default velocity controls on wheels
    num_joints = p.getNumJoints(cartpoleId)
    for j in range(num_joints):
        p.setJointMotorControl2(
            bodyUniqueId=cartpoleId,
            jointIndex=j,
            controlMode=p.VELOCITY_CONTROL,
            force=0
        )

    # 5) Estimate wheel radius, pole length from AABB

    aabb_min, aabb_max = p.getAABB(cartpoleId, linkIndex=0)
    # radius approximation:
    diameter_x = aabb_max[0] - aabb_min[0]
    diameter_y = aabb_max[1] - aabb_min[1]
    diameter = max(diameter_x, diameter_y)
    wheel_radius = diameter / 2
    print(f"Estimated wheel radius from AABB: {wheel_radius:.4f} meters")
    print(f"Estimated pole length from AABB: {estimate_pole_length(cartpoleId, pole_link_index=4):.4f} meters")


    # 6) Create LQR from cartpole_sys A,B
    A = cartpole_sys.A
    B = cartpole_sys.B
    # In main(), when defining Q and R for LQRController:

    Q = np.diag([
        7,
        100,
        40000,
        40000
    ])
    R = np.array([[10]])

    dt = 1/240
    lqr_ctrl = LQRController(A, B, Q, R, dt=dt, max_force=cartpole_sys.max_force,target_state=np.array([1.83,0,0,0]))

    mpc_ctrl = controller.MPCController(A, B, Q, R, horizon = 10, dt=dt, max_force=cartpole_sys.max_force,target_state=np.array([-1.83,0,0,0]))

    # We'll define a smaller motor torque bound to mimic a low-power DC motor
    maxWheelTorque = 5.0  # in Nm

# 7) Run simulation
    while p.isConnected():
        p.stepSimulation()
        time.sleep(1./240.)

        #debugonly
        #time.sleep(1./5.)

        # 6) Read state from cartpole_sys
        state = cartpole_sys.get_cartpole_state()
        # Single scalar "force" from LQR
        force = lqr_ctrl.compute_control(state)
        print(f"State: x={state[0]:.3f}, x_dot={state[1]:.3f}, theta={state[2]:.3f}, theta_dot={state[3]:.3f}, Force={force:.3f}")

        # We'll treat "force" as total horizontal effort => convert to wheel torque
        # For simplicity, do torque_left=torque_right=force/2
        torque_left = 0.5 * force
        torque_right = 0.5 * force

        # Now clamp them to a "small DC motor" torque limit
        torque_left = np.clip(torque_left, -maxWheelTorque, maxWheelTorque)
        torque_right = np.clip(torque_right, -maxWheelTorque, maxWheelTorque)

        p.setJointMotorControlArray(bodyUniqueId=cartpoleId,
            jointIndices=[left_joint_index, right_joint_index],
            controlMode=p.TORQUE_CONTROL,
            forces=[torque_left, torque_right]
        )



if __name__ == "__main__":
    main()
