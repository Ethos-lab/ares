from typing import List, Any, Optional

from art.estimators.classification import PyTorchClassifier
from gym import spaces
import numpy as np


class Detector:
    def __init__(self, detector: Optional[Any], fn_name: Optional[str], probability: Optional[float]):
        self.detector = detector
        self.fn_name = fn_name
        self.probability = probability

    def detect(self, x: Optional[np.ndarray]) -> bool:
        if x is None or self.detector is None:
            return False

        p = np.random.rand()
        if p < self.probability:
            detected = getattr(self.detector, self.fn_name)(x)
            return detected
        return False


class DefenderAgent:
    def __init__(self, classifiers: List[PyTorchClassifier], probs: List[float], detector: Detector):
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
    
    def detect(self, x: Optional[np.ndarray]):
        is_adv = self.detector.detect(x)
        return is_adv
