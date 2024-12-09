import mujoco
import mujoco.viewer
import numpy as np
import time

# Suppress GLFW warnings about the window positioning on Wayland
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="glfw")

# Load model and simulation
model = mujoco.MjModel.from_xml_path("/home/yunusdanabas/ENS491/MujocoTrials/XML_trials/rimless_wheel_model.xml")
data = mujoco.MjData(model)

# Define control parameters
desired_angle_ground = np.radians(90)  # Desired angle with respect to ground in radians
kp = 50
kd = 2.5

def get_body_angle_with_respect_to_ground(data, model, body_name):
    
    # Get the body's rotation matrix in world coordinates
    rotation_matrix = data.xmat[1].reshape(3, 3)
    #print(data.xmat[1])
    print(rotation_matrix)

    # Extract the rotation around the y-axis
    angle = np.arctan2(rotation_matrix[0, 2], rotation_matrix[2, 2])
    return angle

def control_swing_leg(data, model):
    # Get the body angle with respect to the ground
    current_body_angle = get_body_angle_with_respect_to_ground(data, model, "hub")
    print(f"Current Body Angle (deg): {np.degrees(current_body_angle):.2f}")
    
    # Calculate angle error based on body orientation
    angle_error = desired_angle_ground - current_body_angle
    
    # PD control to adjust the joint based on body angle error
    joint_velocity = data.qvel[0]  # Assuming this joint controls the body tilt
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
