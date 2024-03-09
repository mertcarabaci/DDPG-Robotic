import numpy as np
import torch
import torch.nn as nn
import os
from model import ActorNet, CriticNet
from replay_buffer import ReplayMemory
from noise import NoiseGenerator
from utils import convert2tensor
from constants import DEVICE
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

class DDPG(nn.Module):
    def __init__(self, state, action_space):
        super(DDPG, self).__init__()

        self.observation_dim = state.shape[0]
        self.action_dim = action_space.shape[0]

        self.memory = ReplayMemory(state.shape, self.action_dim)
        self.action_space = action_space

        self.actor = ActorNet(self.observation_dim, self.action_dim, 9, 6).to(DEVICE)
        self.actor_target = ActorNet(self.observation_dim, self.action_dim, 9, 6).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = CriticNet(self.observation_dim, self.action_dim, 5, 4).to(DEVICE)
        self.critic_target = CriticNet(self.observation_dim, self.action_dim, 5, 4).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.gamma = 0.99
        self.actor_lr = 0.0001
        self.critic_lr = 0.001
        self.tau = 0.001
        self.memory_capacity = 60000

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.loss = nn.MSELoss(reduction='mean')

        self.noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(self.action_dim), sigma = 0.005 * np.ones(self.action_dim))
    
    def reset(self):
        self.noise.reset()

    def select_action(self, state, training=True):

        state = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        action = self.actor(state).detach()
        
        action = action.to('cpu').numpy().reshape(self.action_dim,)
        # add noise
        if training:
            action = action + self.noise()

        abs_action = abs(action).max().item()
        if abs_action > self.action_space.high[0]:
            x = abs_action/self.action_space.high[0]
            action /= x 

        return action
    
    def learn(self):
        state, action, reward, next_state = self.memory.sample()

        next_action = self.actor_target(next_state)
        y = reward + self.gamma * self.critic_target(next_state, next_action)

        self.optimizer_critic.zero_grad()
        loss_critic = self.loss(y,self.critic(state, action))
        loss_critic.backward()
        self.optimizer_critic.step()

        self.optimizer_actor.zero_grad()
        loss_actor = -self.critic(state, self.actor(state))
        loss_actor = torch.mean(loss_actor)
        loss_actor.backward()
        self.optimizer_actor.step()

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save_models(self, episode):
        if not os.path.exists(f'models_custom2/{episode}'):
            os.makedirs(f'models_custom2/{episode}')
            
        torch.save(self.actor.state_dict(), f'models_custom2/{episode}/actor.pt')
        torch.save(self.critic.state_dict(), f'models_custom2/{episode}/critic.pt')
        torch.save(self.actor_target.state_dict(), f'models_custom2/{episode}/actor_target.pt')
        torch.save(self.critic_target.state_dict(), f'models_custom2/{episode}/critic_target.pt')

    def load_models(self, episode):
        self.actor.load_state_dict(torch.load(f"models_custom2/{episode}/actor.pt")) 
        self.critic.load_state_dict(torch.load(f"models_custom2/{episode}/critic.pt")) 
        self.actor_target.load_state_dict(torch.load(f"models_custom2/{episode}/actor_target.pt")) 
        self.critic_target.load_state_dict(torch.load(f"models_custom2/{episode}/critic_target.pt")) 