from typing import Any, List, Optional

from art.estimators.classification import PyTorchClassifier
from gym import spaces
import numpy as np


class Detector:
    def __init__(self, module: Any, fn_name: str, probability: float):
        self.module = module
        self.fn_name = fn_name
        self.probability = probability

    def detect(self, x: np.ndarray) -> bool:
        p = np.random.rand()
        if p < self.probability:
            detected = getattr(self.module, self.fn_name)(x)
            return detected
        return False


class DefenderAgent:
    def __init__(self, classifiers: List[PyTorchClassifier], probs: List[float], detector: Optional[Detector]):
        self.classifiers = classifiers
        self.probs = probs
        self.num_classifiers = len(classifiers)
        self.index = 0
        self.active_classifier = self.classifiers[self.index]
        self.detector = detector
        self.action_space = spaces.Discrete(self.num_classifiers)
        self.observation_space = spaces.Dict({})

    def update_policy(self, observation: dict):
        pass

    def defend(self) -> PyTorchClassifier:
        self.index = np.random.choice(len(self.classifiers), p=self.probs)
        self.active_classifier = self.classifiers[self.index]
        return self.active_classifier

    def detect(self, x: np.ndarray) -> bool:
        if self.detector:
            return self.detector.detect(x)
        return False
