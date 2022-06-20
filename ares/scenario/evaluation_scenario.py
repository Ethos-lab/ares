from typing import List, Tuple

from art.estimators.classification import PyTorchClassifier
import numpy as np
import torchvision


class EvaluationScenario:
    def __init__(self, threat_model: str, dataroot: str, random_noise: bool, num_episodes: int, max_rounds: int):
        self.threat_model = threat_model
        self.dataroot = dataroot
        self.random_noise = random_noise
        self.num_episodes = num_episodes
        self.max_rounds = max_rounds

        transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        # TODO: replace CIFAR10 dataset with generic dataset
        self.dataset = torchvision.datasets.CIFAR10(root=dataroot, train=False, transform=transforms, download=True)

    def get_valid_sample(self, classifiers: List[PyTorchClassifier]) -> Tuple[np.ndarray, np.ndarray]:
        all_correct = False
        while not all_correct:
            choice = np.random.choice(len(self.dataset))
            sample = self.dataset[choice]
            x = np.expand_dims(sample[0].numpy(), axis=0)
            y = np.array([sample[1]])
            all_correct = True
            for classifier in classifiers:
                out = classifier.predict(x)
                y_pred = classifier.reduce_labels(out)
                if y_pred != y:
                    all_correct = False

        if self.random_noise:
            noise = np.random.uniform(-8 / 255, 8 / 255, x.shape).astype(np.float32)
            x = x + noise

        return x, y
