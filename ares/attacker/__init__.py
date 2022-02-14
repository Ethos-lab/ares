from art.attacks.evasion import ProjectedGradientDescentPyTorch
import torch
import torch.nn as nn


class AttackerAgent:
    def __init__(self, device: torch.device):
        self.num_steps = 0
        self.device = device

    def generate(
        self, image: torch.Tensor, label: torch.Tensor, adversary: ProjectedGradientDescentPyTorch
    ) -> nn.Module:
        with torch.enable_grad():
            bx = adversary.generate(x=image.cpu().numpy(), y=label.cpu().numpy())
        image_adv = torch.tensor(bx, device=self.device)
        self.num_steps += 1
        return image_adv
