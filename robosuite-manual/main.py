import os
from ddpq import DDPG
import time
from constants import transition
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper


project_dir = os.path.dirname(__file__)

models_dir = f"{project_dir}/models/{int(time.time())}/"
logdir = f"{project_dir}/logs/{int(time.time())}/"
print(f"logdir: {logdir}")

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)


config = load_controller_config(default_controller="OSC_POSE")

# create environment instance
env = suite.make(
    "Lift",
    robots=["Sawyer"],             # load a Sawyer robot and a Panda robot
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

n_episodes = 50000
save_model_freq = 500
save_reward_freq = 100

state = env.reset()
env = GymWrapper(env, keys=["gripper_to_cube_pos","robot0_gripper_qpos","robot0_eef_pos","robot0_eef_quat"])
state, _ = env.reset()
all_episode_reward = []
history = {'Episode': [], 'AvgReturn': []}

agent = DDPG(state, env.action_space)
#agent.load_models(5500, models_dir)
# Loop of episodes
for ie in range(n_episodes):
    state, _ = env.reset()
    agent.reset()
    done = False
    episode_reward = 0
    no_reward_counter = 0

    # One-step-loop
    while not done:

        action = agent.select_action(state)

        # This will make steering much easier
        next_state, reward, terminated, truncated, info = env.step(action)
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