from typing import List, Optional

from art.attacks.evasion import ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier
import numpy as np
import torch
import torch.nn as nn


class DefenderAgent:
    def __init__(self, models: List[nn.Module], device: torch.device):
        self.models = models
        self.num_models = len(models)
        self.device = device
        self.classifiers: List[PyTorchClassifier] = []
        self.adversaries: List[ProjectedGradientDescentPyTorch] = []
        for m in models:
            classifier = PyTorchClassifier(
                model=m,
                loss=nn.CrossEntropyLoss(),
                input_shape=(3, 32, 32),
                nb_classes=10,
                clip_values=(0, 1),
                device_type=device.type,
            )
            self.classifiers.append(classifier)
            adversary = ProjectedGradientDescentPyTorch(
                classifier,
                norm=np.inf,
                eps=8 / 255,
                eps_step=2 / 255,
                num_random_init=0,
                max_iter=1,
                targeted=False,
                batch_size=1,
                verbose=False,
            )
            self.adversaries.append(adversary)
        self.index = 0
        self.active_model = self.models[self.index]
        self.active_classifier = self.classifiers[self.index]
        self.active_adversary = self.adversaries[self.index]

    def change_model(self, index: Optional[int] = None) -> int:
        if index is None:
            self.index = np.random.choice(len(self.models))
        else:
            self.index = index
        self.active_model = self.models[self.index]
        self.active_classifier = self.classifiers[self.index]
        self.active_adversary = self.adversaries[self.index]

        return self.index

    def get_model(self, index: Optional[int] = None) -> nn.Module:
        if index is None:
            return self.active_model
        return self.models[index]

    def get_classifier(self, index: Optional[int] = None) -> PyTorchClassifier:
        if index is None:
            return self.active_classifier
        return self.classifiers[index]

    def get_adversary(self, index: Optional[int] = None) -> ProjectedGradientDescentPyTorch:
        if index is None:
            return self.active_adversary
        return self.adversaries[index]

    def generate(self, image: torch.Tensor, label: torch.Tensor, index: Optional[int] = None) -> torch.Tensor:
        adversary = self.get_adversary(index)
        with torch.enable_grad():
            bx = adversary.generate(x=image.cpu().numpy(), y=label.cpu().numpy())
        image_adv = torch.tensor(bx, device=self.device)
        return image_adv
