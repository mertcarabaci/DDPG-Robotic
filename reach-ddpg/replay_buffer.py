from constants import MAX_SIZE, BATCH_SIZE, transition
from utils import convert2tensor
import numpy as np


class ReplayMemory():
    def __init__(self, state_shape, action_shape):
        self.state = convert2tensor(np.zeros((MAX_SIZE, *state_shape)))
        self.action = convert2tensor(np.zeros((MAX_SIZE, action_shape)))
        self.reward = convert2tensor(np.zeros((MAX_SIZE, 1), dtype=float))
        self.next_state = convert2tensor(np.zeros((MAX_SIZE, *state_shape)))

        self.current_size = 0

    def push(self, transition):
        self.state[self.current_size] = convert2tensor(np.array(transition.state))
        self.action[self.current_size] = convert2tensor(transition.action)
        self.reward[self.current_size] = convert2tensor(np.array(transition.reward))
        self.next_state[self.current_size] = convert2tensor(np.array(transition.next_state))

        self.current_size = (self.current_size + 1) % MAX_SIZE

    def sample(self):
        sample_idx = np.random.randint(len(self.state), size=(min(self.current_size, BATCH_SIZE))) 
        return self.state[sample_idx], self.action[sample_idx], self.reward[sample_idx], self.next_state[sample_idx]