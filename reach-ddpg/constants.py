import torch
from collections import namedtuple


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SIZE = 1000
BATCH_SIZE = 64
EPS_MAX = 0.95
EPS_MIN = 0.1
EPS_DECAY = 1000
DROP = 0.2
GAMMA = 0.95
LR = 0.001


transition = namedtuple("Transition", ("state","action", "reward", "next_state"))