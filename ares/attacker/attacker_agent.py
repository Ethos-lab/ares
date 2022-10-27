from typing import List, Optional, Tuple

from gym import spaces
import numpy as np

from ares.attacker.attack import Attack
from ares.defender.classifiers import Classifier


class AttackerAgent:
    def __init__(self, attacks: List[Attack], probabilities: List[float], evasion_probability: Optional[float]):
        self.num_steps = 0
        self.attacks = attacks
        self.probabilities = probabilities
        self.evasion_probability = evasion_probability
        self.num_attacks = len(attacks)
        self.index = 0
        self.active_attack = self.attacks[self.index]
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Dict({})

    def update_policy(self, observation: dict):
        pass

    def current_attack(self) -> str:
        return self.active_attack.name

    def evade(self) -> bool:
        if self.evasion_probability is not None:
            p = np.random.rand()
            return p < self.evasion_probability
        return False

    def attack(self, classifier: Classifier, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        evade_turn = self.evade()
        if evade_turn:
            return x, 0, True

        self.index = np.random.choice(len(self.attacks), p=self.probabilities)
        self.active_attack = self.attacks[self.index]
        x_adv, eps = self.active_attack.generate(classifier, x, y)
        self.num_steps += 1
        return x_adv, eps, False
