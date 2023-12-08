import random
from collections import deque
from typing import NamedTuple, Optional

import torch


class Transition(NamedTuple):
    state: torch.Tensor  # TODO Fix types
    action: torch.Tensor  # TODO Fix types
    next_state: Optional[torch.Tensor]  # TODO Fix types
    reward: torch.Tensor  # TODO Fix types


class ReplayMemory(object):
    def __init__(self, capacity: int):
        self.memory = deque[Transition]([], maxlen=capacity)

    def push(self, transition: Transition):
        """Save a transition"""
        self.memory.append(transition)

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
