
import numpy as np
import gymnasium as gym
import robosuite as suite
from robosuite.controllers import load_controller_config


config = load_controller_config(default_controller="IK_POSE")

# create environment instance
env = suite.make(
    "Lift",
    robots=["Sawyer"],             # load a Sawyer robot and a Panda robot
    gripper_types="default",                # use default grippers per robot arm
    controller_configs=config,
    has_renderer=True,                     # no on-screen rendering
    has_offscreen_renderer=True,           # no off-screen rendering
    control_freq=20,                        # 20 hz control for applied actions
    horizon=200,                            # each episode terminates after 200 steps
    use_object_obs=True,                    # provide object observations to agent
    use_camera_obs=False,                   # don't provide image observations to agent
    reward_shaping=True,                    # use a dense reward signal for learning
)
# reset the environment
obs = env.reset()


act = np.append(obs["gripper_to_cube_pos"], [0,0,0,0])


env.render()


env.step(act)


