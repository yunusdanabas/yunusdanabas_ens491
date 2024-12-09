import mujoco
import mujoco.viewer
import numpy as np
import time

# Suppress GLFW warnings about the window positioning on Wayland
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="glfw")

# Load model and simulation
model = mujoco.MjModel.from_xml_path("rimless_wheel_model.xml")
data = mujoco.MjData(model)

# Define control parameters
desired_angle_ground = np.radians(90)  # Desired angle with respect to ground in radians
kp = 10
kd = 0.5


def get_body_angle_with_respect_to_ground(data, model, body_name):
    
    # Get the body's rotation matrix in world coordinates
    rotation_matrix = data.xmat[1].reshape(3, 3)
    #print(data.xmat[1])
    print(rotation_matrix)

    # Extract the rotation around the y-axis
    angle = np.arctan2(rotation_matrix[0, 2], rotation_matrix[2, 2])
    return angle


def control_swing_leg(data, model):
    # Get the current angle with respect to the ground
    joint_angle = data.qpos[0]
    print(f"Joint Angle (deg): {np.degrees(joint_angle):.2f}")
    angle_error = desired_angle_ground - joint_angle

    # PD control based on angle with respect to ground
    joint_velocity = data.qvel[0]  # Assuming this is the joint you want to control
    torque = kp * angle_error - kd * joint_velocity
    data.ctrl[0] = torque
    print(f"Angle Error: {np.degrees(angle_error):.2f}, Torque: {torque:.2f}")

# Viewer for visualization
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        control_swing_leg(data, model)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)