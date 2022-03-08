from abc import ABC, abstractmethod

from art.attacks.evasion import AutoAttack, AutoProjectedGradientDescent, ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier


class Attack(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def attack(self):
        pass


class PGDAttack(Attack):
    def __init__(self, model: PyTorchClassifier) -> None:
        super().__init__()
        self.model = model

    def attack(self, **kwargs):
        return ProjectedGradientDescentPyTorch(self.model, kwargs)


class AutoPGDAttack(Attack):
    def __init__(self, model: PyTorchClassifier) -> None:
        super().__init__()
        self.model = model

    def attack(self, **kwargs):
        return AutoProjectedGradientDescent(self.model, kwargs)


class AutoAttack(Attack):
    def __init__(self, model: PyTorchClassifier) -> None:
        super().__init__()
        self.model = model

    def attack(self, **kwargs):
        return AutoAttack(self.model, kwargs)
