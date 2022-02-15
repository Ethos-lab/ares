from art.attacks.evasion import ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier
import torch
import torch.nn as nn
from attacks import *


class AttackerAgent:
    def __init__(self, model: PyTorchClassifier, attack_type: Attack, device: torch.device):
        self.num_steps = 0
        self.model = model
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

    def attack(self, attack: Attack, **kwargs):
        wrapped_art_model = self.attack_type(self.model)
        return wrapped_art_model