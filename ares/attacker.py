from typing import List, NamedTuple, Optional

import art
from art.attacks.attack import EvasionAttack, PoisoningAttack
from art.estimators.classification import PyTorchClassifier
from gym import spaces
import numpy as np


class AttackConfig(NamedTuple):
    attack_type: str
    attack_name: str
    attack_params: dict


class AttackerAgent:
    def __init__(self, attacks: List[AttackConfig], probs: List[float], evasion_prob: Optional[float]):
        self.num_steps = 0
        self.attacks = attacks
        self.probs = probs
        self.evasion_prob = evasion_prob
        self.num_attacks = len(attacks)
        self.index = 0
        self.active_attack = self.attacks[self.index]
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Dict({})

    def update_policy(self, observation: dict):
        pass

    def evade(self):
        if self.evasion_prob is not None:
            p = np.random.rand()
            return p < self.evasion_prob
        return False

    def attack(self, classifier: PyTorchClassifier, image: np.ndarray, label: np.ndarray) -> Optional[np.ndarray]:
        evade_turn = self.evade()
        if evade_turn:
            return None

        self.index = np.random.choice(len(self.attacks), p=self.probs)
        attack_config = self.attacks[self.index]
        self.active_attack = attack_config

        # only supports evasion attacks for now
        if attack_config.attack_type == "evasion":
            attack = get_evasion_attack(attack_config.attack_name, classifier, attack_config.attack_params)
            image_adv = attack.generate(x=image, y=label)
        else:
            raise Exception(f"{attack_config.attack_type} attacks not supported")

        self.num_steps += 1
        return image_adv


def get_evasion_attack(attack_name: str, classifier: PyTorchClassifier, attack_params: dict) -> EvasionAttack:
    ctor = getattr(art.attacks.evasion, attack_name)
    attack = ctor(classifier, **attack_params)
    return attack


def get_poisoning_attack(attack_name: str, classifier: PyTorchClassifier, attack_params: dict) -> PoisoningAttack:
    ctor = getattr(art.attacks.poisoning, attack_name)
    attack = ctor(classifier, **attack_params)
    return attack
