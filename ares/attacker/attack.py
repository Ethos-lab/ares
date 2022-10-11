from typing import Tuple, Optional

import art
from art.attacks.attack import EvasionAttack, PoisoningAttack
from art.estimators.classification import PyTorchClassifier
import numpy as np

from ares.defender import Classifier


class Attack:
    def __init__(self, type: str, name: str, params: dict, epsilon_constraint: Optional[dict]):
        self.type = type
        self.name = name
        self.params = params
        if epsilon_constraint:
            self.epsilon_constraint = True
            self.norm = epsilon_constraint.get("norm", "inf")
            self.eps = epsilon_constraint.get("eps", 8 / 255)

    def generate(self, classifier: Classifier, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        # only supports evasion attacks for now
        if self.type == "evasion":
            attack = self.get_evasion_attack(classifier.classifier)
            x_adv = attack.generate(x=x, y=y)
        else:
            raise ValueError(f"Error loading attack: {self.type} attacks not supported")

        if self.norm == "inf":
            eps = np.linalg.norm(np.abs(x_adv - x).ravel(), ord=np.inf)
        elif self.norm in (1, "1", "l1"):
            eps = np.linalg.norm(np.abs(x_adv - x).ravel(), ord=1)
        elif self.norm in (2, "2", "l2"):
            eps = np.linalg.norm(np.abs(x_adv - x).ravel(), ord=2)
        else:
            raise ValueError(f"Error generating attack: {self.norm} norm not supported")

        # Enforce epsilon constraint
        if self.epsilon_constraint and eps > self.eps:
            raise ValueError("Error generating attack: epsilon constraint violated")

        return x_adv, eps

    def get_evasion_attack(self, classifier: PyTorchClassifier) -> EvasionAttack:
        ctor = getattr(art.attacks.evasion, self.name)
        attack = ctor(classifier, **self.params)
        return attack

    def get_poisoning_attack(self, classifier: PyTorchClassifier) -> PoisoningAttack:
        ctor = getattr(art.attacks.poisoning, self.name)
        attack = ctor(classifier, **self.params)
        return attack
