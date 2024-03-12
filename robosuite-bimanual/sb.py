from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import EvalCallback
import os
import time
import matplotlib.pyplot as plt
import sys
import numpy as np
import torch
import random
from datetime import datetime
import gymnasium as gym
from typing import Any, Callable
from gymnasium.spaces import Box

import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper

project_dir = os.path.dirname(os.getcwd())

models_dir = f"{project_dir}/sb_models/{int(time.time())}/"
logdir = f"{project_dir}/sb_logs/{int(time.time())}/"
print(f"logdir: {logdir}")

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

EPISODES = 200 #default = 1000
EPISODE_LENGTH = 1000


config = load_controller_config(default_controller="OSC_POSE")

# create environment instance
env = suite.make(
    "TwoArmLift",
    robots=["Sawyer","Sawyer"],             # load a Sawyer robot and a Panda robot
    gripper_types="RethinkGripper",                # use default grippers per robot arm
    has_renderer=True,                     # no on-screen rendering
    controller_configs=config,
    has_offscreen_renderer=False,           # no off-screen rendering
    control_freq=25,                        # 20 hz control for applied actions
    horizon=200,                            # each episode terminates after 200 steps
    use_object_obs=True,                    # provide object observations to agent
    use_camera_obs=False,                   # don't provide image observations to agent
    reward_shaping=True,                    # use a dense reward signal for learning
)
# reset the environment
obs = env.reset()

iters = 0

env = GymWrapper(env, keys=["gripper0_to_handle0","robot0_gripper_qpos","gripper1_to_handle1","robot1_gripper_qpos"])

replay_buffer = []
eval_callback = EvalCallback(env, best_model_save_path=models_dir,
                             log_path=logdir, eval_freq=10000,
                             deterministic=True, render=False)
n_actions = env.action_dim
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.005 * np.ones(n_actions))
noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma = 0.005 * np.ones(n_actions)) 
model = DDPG('MlpPolicy', env, action_noise=noise, verbose=1, tensorboard_log=logdir)
model.learn(total_timesteps=EPISODE_LENGTH*EPISODES, progress_bar=True, callback=eval_callback)
model.save(f"{models_dir}/sb_reachenv")

#1710278478
del model # remove to demonstrate saving and loading
env = suite.make(
    "TwoArmLift",
    robots=["Sawyer","Sawyer"],             # load a Sawyer robot and a Panda robot
    gripper_types="RethinkGripper",                # use default grippers per robot arm
    has_renderer=True,                     # no on-screen rendering
    controller_configs=config,
    has_offscreen_renderer=False,           # no off-screen rendering
    control_freq=25,                        # 20 hz control for applied actions
    horizon=300,                            # each episode terminates after 200 steps
    use_object_obs=True,                    # provide object observations to agent
    use_camera_obs=False,                   # don't provide image observations to agent
    reward_shaping=True,                    # use a dense reward signal for learning
)
# reset the environment
obs = env.reset()

iters = 0

env = GymWrapper(env, keys=["object-state"])
model = DDPG.load(f"/Users/mertarabaci/Desktop/dev/git/sb_models/sb_models/1710108740/best_model.zip")
episodes = 50
for e in range(episodes):
    obs, _ = env.reset()
    for i in range(EPISODE_LENGTH):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, terminated, info = env.step(action)
        env.render()
        if dones or terminated:
            break



print('Done!')
env.shutdown()
	
