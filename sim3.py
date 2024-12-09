import mujoco
import mujoco.viewer
import numpy as np
import time
import warnings

# Suppress GLFW warnings about the window positioning on Wayland
warnings.filterwarnings("ignore", category=UserWarning, module="glfw")

# Load model and simulation
model = mujoco.MjModel.from_xml_path("/home/yunusdanabas/ENS491/MujocoTrials/XML_trials/rimless_wheel_model.xml")
data = mujoco.MjData(model)

# Define control parameters
desired_y_velocity = 10.0  # Target velocity along the y-axis for the body
kp = 5.1  # Proportional gain for velocity control
kd = 0.4  # Derivative gain for damping

def get_body_velocity_y(data, model, body_name):
    body_id = 1
    #print(data.subtree_linvel[3 * body_id + 1])
    body_velocity = data.cvel[body_id * 6]   # y-axis component
    linear_velocity = body_velocity[0: 3]  # [vx, vy, vz]
    body_velocity_y = linear_velocity[1]
    angular_velocity = data.cvel[body_id * 6 + 3: body_id * 6 + 6]  # [wx, wy, wz]
   # print("--------")
    #print(body_velocity)
    #print(linear_velocity)
    #print(body_velocity_y)
    #print("--------")
    #print(body_velocity_y)
    return float(body_velocity_y)  

def control_swing_leg_velocity(data, model):
    body_velocity_y = get_body_velocity_y(data, model, "hub")

    desired_joint_velocity = kp * (desired_y_velocity - body_velocity_y)

    joint_velocity = float(data.qvel[0])  

    velocity_error = desired_joint_velocity - joint_velocity

    torque = kp * velocity_error - kd * joint_velocity
    data.ctrl[0] = float(torque)  

    print(f"Body Y-Velocity: {body_velocity_y:.2f}")
    print(f"Joint Velocity (Current): {joint_velocity:.2f}")
    print(f"Velocity Error: {velocity_error:.2f}, Torque: {torque:.2f}")

with mujoco.viewer.launch_passive(model, data) as viewer:
    vel = np.zeros(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_QVEL))
    while viewer.is_running():
        mujoco.mj_getState(model, data, vel, mujoco.mjtState.mjSTATE_QVEL)
        print("Vel: ", vel)
        print("\n")
        control_swing_leg_velocity(data, model)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)  
