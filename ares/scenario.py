from typing import List

from art.estimators.classification import PyTorchClassifier
import numpy as np
import torchvision


class ExecutionScenario:
    def __init__(self, threat_model: str, dataroot: str, random_noise: bool, num_episodes: int, num_trials: int):
        self.threat_model = threat_model
        self.dataroot = dataroot
        self.random_noise = random_noise
        self.num_episodes = num_episodes
        self.num_trials = num_trials

        transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.dataset = torchvision.datasets.CIFAR10(root=dataroot, train=False, transform=transforms, download=True)

    def get_valid_sample(self, classifiers: List[PyTorchClassifier]):
        all_correct = False
        while not all_correct:
            choice = np.random.choice(len(self.dataset))
            sample = self.dataset[choice]
            image = np.expand_dims(sample[0].numpy(), axis=0)
            label = np.array([sample[1]])
            all_correct = True
            for classifier in classifiers:
                out = classifier.predict(image)
                pred = classifier.reduce_labels(out)
                if pred != label:
                    all_correct = False

        if self.random_noise:
            noise = np.random.uniform(-8 / 255, 8 / 255, image.shape).astype(np.float32)
            image = image + noise

        return image, label
