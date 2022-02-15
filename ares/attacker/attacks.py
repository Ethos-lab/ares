from abc import ABC, abstractmethod

class Attack:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def attack(self):
        pass


class PGDAttack(Attack):
    pass


class AutoPGDAttack(Attack):
    pass


class AutoAttack(Attack):
    pass