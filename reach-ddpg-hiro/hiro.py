import os
import numpy as np
from ddpq import DDPG
from constants import transition

def create_state(state):
    return np.array(state['achieved_goal'] - state['desired_goal'])

class HigherController(DDPG):
    def __init__(self,state, action_space, layer):
        super(HigherController,self).__init__(state, action_space,layer)
        self.name = "higher"

class LowerController(DDPG):
    def __init__(self,state, action_space, layer):
        super(LowerController,self).__init__(state, action_space,layer)
        self.name = "lower"

class HiroAgent:
    def __init__(self, state, action_space, goal_dim):
        subgoal = np.zeros(goal_dim)
        self.higher_policy = HigherController(state=subgoal, action_space=subgoal, layer= 'higher')
        self.lower_policy = LowerController(state=subgoal, action_space=action_space, layer= 'lower')
        self.higher_reward = 0
        self.higher_state = []
        self.higher_action = []
        self.lower_action = []
        self.state = None
        self.next_state = None

    def select_action(self, state, epsilon = 0.999):
        self.state = state
        ag = state['achieved_goal']
        self.higher_state = create_state(state)
        self.higher_action = self.higher_policy.select_action(state=self.higher_state, is_lower=False)
        self.lower_state = self.higher_action
        self.lower_action =self.lower_policy.select_action(state=self.lower_state)

        return self.lower_action
    
    def memory_push(self, next_state):
        self.next_state = next_state
        higher_next = create_state(next_state)
        reward = self.calc_norm(self.state['desired_goal'], self.state['achieved_goal']) - self.calc_norm(self.next_state['desired_goal'], self.next_state['achieved_goal'])
        if (np.random.uniform(size= 1) < 0.2):
            reward = 10
            direction = create_state(self.state)
            if self.calc_norm(self.state['achieved_goal'], self.state['desired_goal']) < 0.1:
                self.higher_action = direction
            else:
                max_value = abs(direction.max()) if abs(direction.max()) > abs(direction.min()) else abs(direction.min())
                self.higher_action = direction * 0.1 / max_value
            higher_next = self.state['achieved_goal'] - self.higher_action
        
        self.higher_policy.memory.push(transition(self.higher_state, self.higher_action, reward, higher_next))
        
        next_state_lower = self.create_state_norm(next_state['achieved_goal'],self.higher_policy.select_action(state=higher_next, training=False, is_lower=False))
        reward = self.lower_reward()
        self.lower_policy.memory.push(transition(self.lower_state, self.lower_action, reward, next_state_lower))

    def reset(self):
        self.higher_policy.reset()
        self.lower_policy.reset()
        self.higher_reward = 0

    def learn(self):
        self.higher_policy.learn()
        self.lower_policy.learn()

    def save_models(self, ie, models_dir):
        self.higher_policy.save_models(ie, models_dir, policy="higher")
        self.lower_policy.save_models(ie, models_dir, policy="lower")

    def calc_norm(self, p1, p2):
        return pow(sum(pow((p1 - p2),2)),(0.5))

    def lower_reward(self):
        temp = self.state['achieved_goal'] - self.higher_action
        return - self.calc_norm(temp, self.next_state['achieved_goal'])
        
    def create_state_norm(self, ag, g):
        return np.array(ag - g) 