from typing import List

from art.estimators.classification import PyTorchClassifier
from gym import spaces
import numpy as np


class DefenderAgent:
    def __init__(self, classifiers: List[PyTorchClassifier], probs: List[float]):
        self.classifiers = classifiers
        self.probs = probs
        self.num_classifiers = len(classifiers)
        self.index = 0
        self.active_classifier = self.classifiers[self.index]
        self.action_space = spaces.Discrete(self.num_classifiers)
        self.observation_space = spaces.Dict({})

    def update_policy(self, observation: dict):
        pass

    def defend(self) -> PyTorchClassifier:
        self.index = np.random.choice(len(self.classifiers), p=self.probs)
        self.active_classifier = self.classifiers[self.index]

        return self.active_classifier
