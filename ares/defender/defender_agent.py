from typing import TYPE_CHECKING, List, Optional

from gymnasium import spaces
import numpy as np

if TYPE_CHECKING:
    from ares.defender import Classifier, Detector


class DefenderAgent:
    def __init__(self, classifiers: List["Classifier"], probabilities: List[float], detector: Optional["Detector"]):
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

    def current_defense(self) -> str:
        return self.active_classifier.name

    def defend(self) -> "Classifier":
        self.index = np.random.choice(len(self.classifiers), p=self.probabilities)
        self.active_classifier = self.classifiers[self.index]
        self.active_classifier.reset()
        return self.active_classifier

    def detect(self, x: np.ndarray) -> bool:
        if self.detector:
            return self.detector.detect(x)
        return False
