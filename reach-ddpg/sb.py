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

class TransformObservation(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):

    def __init__(self, env: gym.Env):
        gym.ObservationWrapper.__init__(self, env)
        env.observation_space = Box(-100,100,shape=(3,))


    def observation(self, observation):
        return np.array(observation['achieved_goal'] - observation['desired_goal']) 
    

project_dir = os.path.dirname(os.getcwd())

models_dir = f"{project_dir}/sb_models/{int(time.time())}/"
logdir = f"{project_dir}/sb_logs/{int(time.time())}/"
print(f"logdir: {logdir}")

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

EPISODES = 200 #default = 1000
EPISODE_LENGTH = 300

iters = 0

env = gym.make('FetchReachDense-v2', max_episode_steps=100)
env = TransformObservation(env)

replay_buffer = []
eval_callback = EvalCallback(env, best_model_save_path=models_dir,
                             log_path=logdir, eval_freq=3000,
                             deterministic=True, render=False)
n_actions = 4
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.005 * np.ones(n_actions))
noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma = 0.005 * np.ones(n_actions)) 
model = DDPG('MlpPolicy', env, action_noise=noise, verbose=1, tensorboard_log=logdir)
model.learn(total_timesteps=EPISODE_LENGTH*EPISODES, progress_bar=True, callback=eval_callback)
model.save(f"{models_dir}/sb_reachenv")
env.shutdown()

#del model # remove to demonstrate saving and loading
env = gym.make('FetchReachDense-v2', render_mode="human", max_episode_steps=100)
env = TransformObservation(env)
model = DDPG.load(f"{models_dir}/sb_reachenv/best_model.zip")
episodes = 50
for e in range(episodes):
    obs, _ = env.reset()
    for i in range(EPISODE_LENGTH):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, terminated, info = env.step(action)
        if dones or terminated:
            break



print('Done!')
env.shutdown()
	
