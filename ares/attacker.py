from art.attacks.attack import EvasionAttack
from art.estimators.classification import PyTorchClassifier
import torch
import numpy as np
from gym import spaces

from ares import utils


class AttackerAgent:
    def __init__(self, attack_type: str, attack_name: str, attack_params):
        self.num_steps = 0
        self.attack_type = attack_type
        self.attack_name = attack_name
        self.attack_params = attack_params
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Dict({})

    def generate(self, image: np.ndarray, label: np.ndarray, attack: EvasionAttack) -> np.ndarray:
        with torch.enable_grad():
            image_adv = attack.generate(x=image, y=label)
        return image_adv

    def attack(self, classifier: PyTorchClassifier, image: np.ndarray, label: np.ndarray) -> np.ndarray:
        attack = self.get_attack(classifier)
        image_adv = self.generate(image, label, attack)
        self.num_steps += 1
        return image_adv

    def get_attack(self, classifier: PyTorchClassifier):
        if self.attack_type == "evasion":
            return utils.get_evasion_attack(self.attack_name, classifier, self.attack_params)
        else:
            raise Exception(f"{self.attack_type} attacks not supported")
