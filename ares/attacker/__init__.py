from art.attacks.attack import EvasionAttack
from art.estimators.classification import PyTorchClassifier
import torch
import torch.nn as nn
from gym import spaces

from ares import utils


class AttackerAgent:
    def __init__(self, attack_type: str, attack_name: str, attack_params, device: torch.device):
        self.num_steps = 0
        self.attack_type = attack_type
        self.attack_name = attack_name
        self.attack_params = attack_params
        self.device = device
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Dict({})

    def generate(self, image: torch.Tensor, label: torch.Tensor, attack: EvasionAttack) -> nn.Module:
        with torch.enable_grad():
            bx = attack.generate(x=image.cpu().numpy(), y=label.cpu().numpy())
        image_adv = torch.tensor(bx, device=self.device)
        return image_adv

    def attack(self, classifier: PyTorchClassifier, image: torch.Tensor, label: torch.Tensor) -> nn.Module:
        attack = self.get_attack(self.attack_name, classifier, self.attack_params)
        image_adv = self.generate(image, label, attack)
        self.num_steps += 1
        return image_adv

    def get_attack(self, classifier: PyTorchClassifier):
        if self.attack_type == "evasion":
            return utils.get_evasion_attack(self.attack_name, classifier, self.attack_params)
        else:
            raise Exception(f"{self.attack_type} attacks not supported")
