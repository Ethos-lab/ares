from typing import List, Optional

from art.estimators.classification import ClassifierMixin
import numpy as np
import torch
from gym import spaces


class DefenderAgent:
    def __init__(self, models: List[ClassifierMixin], probs: List[float], device: torch.device):
        self.models = models
        self.probs = probs
        self.num_models = len(models)
        self.device = device
        self.index = 0
        self.active_model = self.models[self.index]
        self.action_space = spaces.Discrete(self.num_models)
        self.observation_space = spaces.Dict({})

    def change_model(self, index: Optional[int] = None) -> int:
        if index is None:
            self.index = np.random.choice(len(self.models), p=self.probs)
        else:
            self.index = index
        self.active_model = self.models[self.index]

        return self.index

    def get_model(self, index: Optional[int] = None) -> ClassifierMixin:
        if index is None:
            return self.active_model
        return self.models[index]
