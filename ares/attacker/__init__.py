from art.attacks.evasion import ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier
import torch
import torch.nn as nn
from attacks import *


class AttackerAgent:
    def __init__(self, model: nn.Module, attack_type: str, device: torch.device, **model_kwargs):
        self.num_steps = 0
        self.model = PyTorchClassifier(model, model_kwargs)
        self.attack_type = attack_type
        self.device = device

    def generate(
        self, image: torch.Tensor, label: torch.Tensor, adversary: ProjectedGradientDescentPyTorch
    ) -> nn.Module:
        with torch.enable_grad():
            bx = adversary.generate(x=image.cpu().numpy(), y=label.cpu().numpy())
        image_adv = torch.tensor(bx, device=self.device)
        self.num_steps += 1
        return image_adv

    def attack(self, image: torch.Tensor, label: torch.Tensor) -> nn.Module:
        attack = self.get_attack_type()
        with torch.enable_grad():
            bx = attack.generate(x=image.cpu().numpy(), y=label.cpu().numpy())
        image_adv = torch.tensor(bx, device=self.device)
        self.num_steps += 1
        return image_adv

    def get_attack_type(self):
        if self.attack_type == "pgd":
            return PGDAttack(self.model)
        elif self.attack_type == "autopgd":
            return AutoPGDAttack(self.model)
        elif self.attack_type == "autoattack":
            return AutoAttack(self.model)
        else:
            raise Exception(f"Attack type {self.attack_type} not supported")