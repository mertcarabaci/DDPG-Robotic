import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from constants import DEVICE

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim, fc1_dim, fc2_dim, init_w=3e-3):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        #self.init_weights(init_w)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.tanh(out)
        return out
    
class CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, fc1_dim, fc2_dim, init_w=3e-3):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)#.to(DEVICE)
        self.fc2 = nn.Linear(64+action_dim, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        #self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x, u):
        out = self.fc1(x)
        out = F.relu(out)
        
        out = self.fc2(torch.cat([out, u], 1))
        out = F.relu(out)
        
        out = self.fc3(out)
        return out