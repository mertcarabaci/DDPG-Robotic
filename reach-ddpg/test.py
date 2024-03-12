import os
from ddpq import DDPG
import time
from constants import transition
from IPython.display import clear_output
import matplotlib.pyplot as plt
import torch
import numpy as np
import gymnasium as gym


project_dir = os.path.dirname(__file__)

models_dir = f"{project_dir}/models/1710265881/"
logdir = f"{project_dir}/logs/1710265881/"
print(f"logdir: {logdir}")

n_episodes = 10000
save_model_freq = 300
save_reward_freq = 100


def create_state(state):
    return np.append(state["observation"][:3], state["desired_goal"]-state["achieved_goal"])

env = gym.make('FetchReachDense-v2', max_episode_steps=100, render_mode="human")
state, _ = env.reset()
state = create_state(state)
all_episode_reward = []
history = {'Episode': [], 'AvgReturn': []}

agent = DDPG(state, env.action_space)
agent.load_models(900, models_dir)
# Loop of episodes
for ie in range(n_episodes):
    state, _ = env.reset()
    state = create_state(state)
    agent.reset()
    done = False
    episode_reward = 0
    no_reward_counter = 0

    # One-step-loop
    while not done:

        action = agent.select_action(state, training=False)

        # This will make steering much easier
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = create_state(next_state)
        done = terminated or truncated

        # Models action output has a different shape for this problem
        agent.memory.push(transition(state, action, np.array(reward), next_state))
        agent.learn()

        state = next_state
        episode_reward += reward

        print(f"{ie}: Action {action} -> Reward {reward}")


    all_episode_reward.append(episode_reward)
    average_result = np.array(all_episode_reward[-10:]).mean()
    print('Last result:', episode_reward, 'Average Step Reward:', episode_reward,'Average results:', average_result)


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