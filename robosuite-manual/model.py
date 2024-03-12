import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from constants import DEVICE

class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, action_dim)


    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))

        out = self.fc4(out)
        out = F.tanh(out)
        return out
    
class CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(state_dim+action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 1)


    def forward(self, x, u):
        out = F.relu(self.fc1(torch.cat([x, u], 1)))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))

        out = self.fc4(out)
        return out