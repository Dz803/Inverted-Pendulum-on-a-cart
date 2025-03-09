# main.py
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
    startPos = [0, 0, 0.08]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    cartpoleId = p.loadURDF(
        "./cart_pole/my_cartpole.urdf",
        startPos,
        startOrientation,
        useFixedBase=False
    )

    # -------------------------------------------------------------------------
    # (A) Read link masses from URDF 
    # -------------------------------------------------------------------------
    # In your URDF:
    #   cart_base mass=0.1
    #   left_front_link, right_front_link, left_rare_link, right_rare_link => each 0.02
    #   pole_link=0.02, mass_link=0.05
    # We can sum them manually or confirm with getDynamicsInfo:

    total_cart_mass = 0.0
    total_pole_mass = 0.0

    # We'll store [linkName -> linkIndex].
    # Based on your URDF, the order of links is typically:
    #   linkIndex=-1 => base link (if any)
    #   linkIndex=0  => left_front_link
    #   linkIndex=1  => right_front_link
    #   linkIndex=2  => left_rare_link
    #   linkIndex=3  => right_rare_link
    #   linkIndex=4  => pole_link
    #   linkIndex=5  => mass_link
    # But let's confirm with p.getNumJoints() and so on.

    num_links = p.getNumJoints(cartpoleId)
    # We'll check each link's mass 
    for link_idx in range(-1, num_links):
        dyn = p.getDynamicsInfo(cartpoleId, link_idx)  # linkIndex=-1 => base
        mass = dyn[0]
        # The link name can be retrieved with p.getBodyInfo / p.getJointInfo / p.getLinkState if needed
        # But we can guess from your URDF structure:
        if link_idx == -1:
            # Usually the base link (cart_base?), but your URDF uses "cart_base" as link 0 in the file
            # Actually your URDF sets cart_base with <link name="cart_base"> 
            # Typically PyBullet calls that linkIndex=-1 => the "base" of the entire URDF
            # mass might be 0 or 0.1 
            pass
        else:
            # let's see if it's the 4 wheels or the pole or mass
            joint_info = p.getJointInfo(cartpoleId, link_idx)
            child_link_name = joint_info[12].decode("utf-8")
            if child_link_name in ["left_front_link", "right_front_link", "left_rare_link", "right_rare_link", "cart_base"]:
                total_cart_mass += mass
            elif child_link_name in ["pole_link", "mass_link"]:
                total_pole_mass += mass
            else:
                # default: assume cart
                total_cart_mass += mass

    # But from your URDF, cart_base is actually a link with mass=0.1
    # The 4 wheels each 0.02 => total 0.08 => cart=0.18
    # The pole link=0.02 and mass_link=0.05 => total=0.05 since pole is negligible
    # Let's do a fallback if the above doesn't match:
    if abs(total_cart_mass) < 1e-9:
        # fallback: maybe 0.18
        total_cart_mass = 0.18
    if abs(total_pole_mass) < 1e-9:
        # fallback: maybe 0.07
        total_pole_mass = 0.05

    print(f"Detected cart_mass={total_cart_mass:.3f}, pole_mass={total_pole_mass:.3f}")

    # -------------------------------------------------------------------------
    # (B) Approximate the pole length from URDF geometry or known offset
    #     In your URDF, the mass_link is ~0.29m from the pivot. Let's pick ~0.30
    # -------------------------------------------------------------------------
    estimated_pole_length = 0.30  # or measure from your CAD / URDF

    # -------------------------------------------------------------------------
    # 2) Create the cart-pole system with friction parameters
    # -------------------------------------------------------------------------
    cartpole_sys = CartpoleSystem(
        body_id=cartpoleId,
        cart_joint_index=0,      # The joint index for the cart movement?
        pole_joint_index=4,      # Actually, from your URDF, "pole_joint" might be 4
        M=total_cart_mass,
        m=total_pole_mass,
        l=estimated_pole_length,
        g=9.81,
        delta=0.04,    # example friction
        c=0.015,
        zeta=0.0,
        max_force=50.0
    )
    cartpole_sys.print_info()  # (Optional) see the system + A,B

    # 3) Build an LQR controller using the system's (A, B)
    A = cartpole_sys.A
    B = cartpole_sys.B
    Q = np.diag([1, 1, 10, 1])
    R = np.array([[0.1]])
    dt = 1/240.0

    lqr_ctrl = LQRController(
        A=A,
        B=B,
        Q=Q,
        R=R,
        dt=dt,
        max_force=cartpole_sys.max_force
    )

    # 4) Zero out the default velocity controls
    num_joints = p.getNumJoints(cartpoleId)
    for j in range(num_joints):
        p.setJointMotorControl2(
            bodyUniqueId=cartpoleId,
            jointIndex=j,
            controlMode=p.VELOCITY_CONTROL,
            force=0
        )

    # 5) Main loop: read state, compute LQR torque, apply to cart joint
    try:
        while True:
            p.stepSimulation()
            time.sleep(dt)

            # get the current [x, x_dot, theta, theta_dot]
            state = cartpole_sys.get_cartpole_state()

            # compute LQR control (u = -K x)
            force = lqr_ctrl.compute_control(state)

            # apply torque on the cart joint
            p.setJointMotorControl2(
                bodyUniqueId=cartpoleId,
                jointIndex=cartpole_sys.cart_joint_index,
                controlMode=p.TORQUE_CONTROL,
                force=force
            )

    except KeyboardInterrupt:
        print("Exiting simulation.")
    finally:
        p.disconnect()


if __name__ == "__main__":
    main()
