import mujoco
import mujoco.viewer
import numpy as np
import time
import warnings

# Suppress GLFW warnings
warnings.filterwarnings("ignore", category=UserWarning, module="glfw")

# Load the model and simulation
model_path = "passive_walker.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Constants
SLOPE_ANGLE = np.radians(2)
DESIRED_HEIGHT_SLOPE = 0.05
SWING_LEG = "right"

# Leg dimensions
THIGH_LENGTH = 0.5
SHIN_LENGTH = 0.5
BASE_LEG_LENGTH = THIGH_LENGTH + SHIN_LENGTH

# Knee extension limits
MIN_KNEE_EXTENSION = -0.25
MAX_KNEE_EXTENSION = 0.25

def rotation_matrix_slope_to_global(alpha):
    return np.array([
        [np.cos(alpha), 0, np.sin(alpha)],
        [0, 1, 0],
        [-np.sin(alpha), 0, np.cos(alpha)]
    ])

def calculate_desired_knee_extension(torso_pos, foot_pos_slope, slope_angle):
    R_slope_to_global = rotation_matrix_slope_to_global(slope_angle)
    foot_pos_global = torso_pos + R_slope_to_global @ foot_pos_slope
    leg_vector = foot_pos_global - torso_pos
    leg_length = np.linalg.norm(leg_vector)
    desired_extension = leg_length - BASE_LEG_LENGTH
    desired_extension = np.clip(desired_extension, MIN_KNEE_EXTENSION, MAX_KNEE_EXTENSION)
    return desired_extension

def check_foot_contact(data, model, foot_geom_id):
    for i in range(data.ncon):
        contact = data.contact[i]
        geom_ids = [contact.geom1, contact.geom2]
        if foot_geom_id in geom_ids:
            return True
    return False

def control_phases(data, model, swing_leg):
    # Update these IDs based on your model
    left_foot_geom_id = 6
    right_foot_geom_id = 9

    left_foot_in_contact = check_foot_contact(data, model, left_foot_geom_id)
    right_foot_in_contact = check_foot_contact(data, model, right_foot_geom_id)

    if swing_leg == "right" and right_foot_in_contact:
        swing_leg = "left"
    elif swing_leg == "left" and left_foot_in_contact:
        swing_leg = "right"

    return swing_leg

def control_swing_leg(data, model, swing_leg):
    # Actuator IDs
    left_knee_actuator_id = 0
    right_knee_actuator_id = 1

    # Joint IDs and qpos addresses
    left_knee_joint_id = 4
    right_knee_joint_id = 6
    left_knee_qposadr = model.jnt_qposadr[left_knee_joint_id]
    right_knee_qposadr = model.jnt_qposadr[right_knee_joint_id]

    # Torso position
    torso_pos = data.xpos[1]

    # Desired foot positions
    foot_pos_slope_left = np.array([0, -0.075, -DESIRED_HEIGHT_SLOPE])
    foot_pos_slope_right = np.array([0, 0.075, -DESIRED_HEIGHT_SLOPE])

    if swing_leg == "right":
        desired_extension = calculate_desired_knee_extension(torso_pos, foot_pos_slope_right, SLOPE_ANGLE)
        data.ctrl[right_knee_actuator_id] = desired_extension
        current_left_knee_extension = data.qpos[left_knee_qposadr]
        data.ctrl[left_knee_actuator_id] = current_left_knee_extension
    elif swing_leg == "left":
        desired_extension = calculate_desired_knee_extension(torso_pos, foot_pos_slope_left, SLOPE_ANGLE)
        data.ctrl[left_knee_actuator_id] = desired_extension
        current_right_knee_extension = data.qpos[right_knee_qposadr]
        data.ctrl[right_knee_actuator_id] = current_right_knee_extension

# Set initial swing velocity
right_hip_joint_id = 5
right_hip_dofadr = model.jnt_dofadr[right_hip_joint_id]
initial_swing_velocity = -1.5
data.qvel[right_hip_dofadr] = initial_swing_velocity

# Initialize the simulation
duration = 60.0
time_step = model.opt.timestep
num_steps = int(duration / time_step)

# Viewer for visualization
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat = np.array([10, -2, 5])
    viewer.cam.azimuth = 60
    viewer.cam.distance = 5

    for _ in range(num_steps):
        SWING_LEG = control_phases(data, model, SWING_LEG)
        control_swing_leg(data, model, SWING_LEG)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(time_step)
