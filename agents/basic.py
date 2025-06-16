import time

import numpy as np
import abc

class Agent(abc.ABC):
    def __init__(self, board_size, *args, **kwargs):
        self.action_dim = board_size * board_size

    @abc.abstractmethod
    def take_action(self, *args, **kwargs):
        pass

class RandomAgent(Agent):
    def __init__(self, board_size):
        super().__init__(board_size)

    def take_action(self, observation, valid_actions_mask=None):
        if valid_actions_mask is None:
            valid_actions_mask = (observation.flatten()) == 0
        valid_actions = np.where(valid_actions_mask)[0]
        if len(valid_actions) == 0:
            return np.random.choice(self.action_dim)
        return np.random.choice(valid_actions)