from typing import List, Optional

from art.estimators.classification import PyTorchClassifier
from gym import spaces
import numpy as np
import torch

from ares.defender.classifiers import get_classifier, get_torch_model
from ares.defender.detector import Detector, DetectorConfig, get_detector


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


def get_defender_agent(config: dict) -> DefenderAgent:
    defender_models = config["defender"]["models"]
    classifiers = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for defender_model in defender_models:
        model_file = defender_model["model_file"]
        model_name = defender_model["model_name"]
        model_params = defender_model["model_params"]
        model_state = defender_model["model_state"]
        classifier_type = defender_model["classifier_type"]
        classifier_params = defender_model["classifier_params"]
        model = get_torch_model(model_file, model_name, model_params, model_state, device)
        classifier = get_classifier(model, classifier_type, classifier_params, device)
        classifiers.append(classifier)

    probs = config["defender"].get("probabilities", None)
    if probs:
        probs = np.array(probs) / np.sum(probs)
    else:
        probs = np.ones(len(classifiers)) / len(classifiers)

    detector_args = config["defender"].get("detector", None)
    if detector_args:
        detector_config = DetectorConfig(**detector_args)
        detector = get_detector(detector_config)
    else:
        detector = None

    defender_agent = DefenderAgent(classifiers, probs, detector)
    return defender_agent
