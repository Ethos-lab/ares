from typing import Tuple

import numpy as np
import torchvision


class Dataset:
    def __init__(self, name: str, dataroot: str):
        self.name = name
        self.dataroot = dataroot
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        if name == "cifar10":
            self.dataset = torchvision.datasets.CIFAR10(
                root=dataroot, train=False, transform=self.transforms, download=True
            )
        else:
            self.dataset = torchvision.datasets.ImageFolder(root=dataroot, transform=self.transforms)

        self.size = len(self.dataset)

    def sample(self) -> Tuple[np.ndarray, np.ndarray]:
        choice = np.random.choice(self.size)
        sample = self.dataset[choice]
        x = np.expand_dims(sample[0].numpy(), axis=0)
        y = np.array([sample[1]])
        return x, y
