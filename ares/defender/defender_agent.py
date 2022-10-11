from typing import List, Optional

from gym import spaces
import numpy as np

from ares.defender.classifiers import Classifier
from ares.defender.detector import Detector


class DefenderAgent:
    def __init__(self, classifiers: List[Classifier], probabilities: List[float], detector: Optional[Detector]):
        self.classifiers = classifiers
        self.probabilities = probabilities
        self.num_classifiers = len(classifiers)
        self.index = 0
        self.active_classifier = self.classifiers[self.index]
        self.detector = detector
        self.action_space = spaces.Discrete(self.num_classifiers)
        self.observation_space = spaces.Dict({})

    def update_policy(self, observation: dict):
        pass

    def defend(self) -> Classifier:
        self.index = np.random.choice(len(self.classifiers), p=self.probabilities)
        self.active_classifier = self.classifiers[self.index]
        self.active_classifier.reset()
        return self.active_classifier

    def detect(self, x: np.ndarray) -> bool:
        if self.detector:
            return self.detector.detect(x)
        return False
