import art
from art.attacks.attack import EvasionAttack, PoisoningAttack
from art.estimators.classification import PyTorchClassifier
import numpy as np


class Attack:
    def __init__(self, type: str, name: str, params: dict):
        self.type = type
        self.name = name
        self.params = params

    def generate(self, classifier: PyTorchClassifier, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # only supports evasion attacks for now
        if self.type == "evasion":
            attack = self.get_evasion_attack(classifier)
            x_adv = attack.generate(x=x, y=y)
        else:
            raise Exception(f"{self.type} attacks not supported")

        return x_adv

    def get_evasion_attack(self, classifier: PyTorchClassifier) -> EvasionAttack:
        ctor = getattr(art.attacks.evasion, self.name)
        attack = ctor(classifier, **self.params)
        return attack

    def get_poisoning_attack(self, classifier: PyTorchClassifier) -> PoisoningAttack:
        ctor = getattr(art.attacks.poisoning, self.name)
        attack = ctor(classifier, **self.params)
        return attack
