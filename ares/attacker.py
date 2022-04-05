from art.attacks.attack import EvasionAttack
from art.estimators.classification import PyTorchClassifier
from gym import spaces
import numpy as np

from ares import utils


class AttackerAgent:
    def __init__(self, attack_type: str, attack_name: str, attack_params):
        self.num_steps = 0
        self.attack_type = attack_type
        self.attack_name = attack_name
        self.attack_params = attack_params
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Dict({})

    def update_policy(self, observation: dict):
        pass

    def attack(self, classifier: PyTorchClassifier, image: np.ndarray, label: np.ndarray) -> np.ndarray:
        attack = self.get_attack(classifier)
        image_adv = attack.generate(x=image, y=label)
        self.num_steps += 1
        return image_adv

    def get_attack(self, classifier: PyTorchClassifier) -> EvasionAttack:
        if self.attack_type == "evasion":
            return utils.get_evasion_attack(self.attack_name, classifier, self.attack_params)
        else:
            raise Exception(f"{self.attack_type} attacks not supported")
