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
        """Initialize the :class:`TransformObservation` wrapper with an environment and a transform function :attr:`f`.

        Args:
            env: The environment to apply the wrapper
            f: A function that transforms the observation
        """

        gym.ObservationWrapper.__init__(self, env)
        env.observation_space = Box(-100,100,shape=(3,))


    def observation(self, observation):
        """Transforms the observations with callable :attr:`f`.

        Args:
            observation: The observation to transform

        Returns:
            The transformed observation
        """
        return np.array(observation['achieved_goal'] - observation['desired_goal']) 
    

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"
print(f"logdir: {logdir}")
date_and_time = datetime.now().strftime("%Y%m%d-%H%M%S")

if not os.path.exists(models_dir):
	os.makedirs(models_dir)
        
if not os.path.exists(logdir):
	os.makedirs(logdir)

EPISODES = 200 #default = 1000
EPISODE_LENGTH = 300

iters = 0

#env = gym.make('FetchReachDense-v2', max_episode_steps=100)
#env = TransformObservation(env)

#agent_new = Agent(env)
#replay_buffer = []
#eval_log_path = '/home/mert/Desktop/Airlab/airlab_rlbench/RLBench/mertarabaci/reach-ddpg/logs'
#eval_callback = EvalCallback(env, best_model_save_path='./logs/dynamicSmall2',
#                             log_path='./logs/dynamicSmall2', eval_freq=3000,
#                             deterministic=True, render=False)
#n_actions = 4
#action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.005 * np.ones(n_actions))
#noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma = 0.005 * np.ones(n_actions)) 
#model = DDPG('MlpPolicy', env, action_noise=noise, verbose=1, tensorboard_log=eval_log_path)
#model.learn(total_timesteps=EPISODE_LENGTH*EPISODES, progress_bar=True, callback=eval_callback)
#model.save("dynamicSmall_training")
#env.shutdown()

#del model # remove to demonstrate saving and loading
env = gym.make('FetchReachDense-v2', render_mode="human", max_episode_steps=100)
env = TransformObservation(env)
model = DDPG.load("C:\\Users\\mertc\\Desktop\\dev\\gym-rl-car-racing\\logs\\dynamicSmall2\\best_model.zip")
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
	
