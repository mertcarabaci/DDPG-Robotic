import os
from ddpq import DDPG
import time
from constants import transition
from IPython.display import clear_output
import matplotlib.pyplot as plt
import torch
import numpy as np
import gymnasium as gym
from hiro import HiroAgent

project_dir = os.path.dirname(os.getcwd())

models_dir = f"{project_dir}/models/{int(time.time())}/"
logdir = f"{project_dir}/logs/{int(time.time())}/"
print(f"logdir: {logdir}")

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

n_episodes = 50000
save_model_freq = 500
save_reward_freq = 100



env = gym.make('FetchReachDense-v2', max_episode_steps=100, render_mode = 'human')
state, _ = env.reset()
all_episode_reward = []
history = {'Episode': [], 'AvgReturn': []}

agent = HiroAgent(state, env.action_space, state['achieved_goal'].shape[0])
#agent.load_models(5500, models_dir)
# Loop of episodes
for ie in range(n_episodes):
    state, _ = env.reset()
    agent.reset()
    done = False
    episode_reward = 0
    no_reward_counter = 0
    hr = 0.0

    # One-step-loop
    while not done:

        action = agent.select_action(state)

        # This will make steering much easier
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        hr += 0.1*reward

        # Models action output has a different shape for this problem
        agent.memory_push(next_state)
        agent.learn()

        state = next_state
        episode_reward += reward

        print(f"\r{ie}: Action {action} -> Reward {reward}", end="")


    all_episode_reward.append(episode_reward)
    average_result = np.array(all_episode_reward[-10:]).mean()
    print('\nLast result:', episode_reward, 'Average Step Reward:', episode_reward,'Average results:', average_result)


    if ie % save_reward_freq == 0:
        history['Episode'].append(ie)
        history['AvgReturn'].append(average_result)

    if ie % save_model_freq == 0 and ie > 0:
        clear_output()
        plt.figure(figsize=(8, 5))
        plt.plot(history['Episode'], history['AvgReturn'], 'r-')
        plt.xlabel('Episode', fontsize=16)
        plt.ylabel('AvgReturn', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(axis='y')
        plt.savefig(f"{project_dir}/ddpg_rewards.png")
        print('Saving best solution')
        agent.save_models(ie, models_dir)