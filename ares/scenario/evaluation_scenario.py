from typing import List, Tuple

import numpy as np

from ares.defender import Classifier
from ares.scenario.datasets import Dataset


class EvaluationScenario:
    def __init__(self, threat_model: str, num_episodes: int, max_rounds: int, dataset: dict):
        self.threat_model = threat_model
        self.num_episodes = num_episodes
        self.max_rounds = max_rounds
        self.random_noise = dataset.get("random_noise", False)

        name = dataset["name"]
        dataroot = dataset.get("dataroot", None)
        train_set = dataset.get("train_set", False)
        channels_first = dataset.get("channels_first", True)
        self.dataset = Dataset(name, dataroot, train_set, channels_first)

    def get_valid_sample(self, classifiers: List[Classifier]) -> Tuple[np.ndarray, np.ndarray]:
        all_correct = False
        while not all_correct:
            x, y = self.dataset.sample()
            all_correct = True
            for classifier in classifiers:
                y_pred = classifier.predict(x)
                if y_pred != y:
                    all_correct = False

        if self.random_noise:
            noise = np.random.uniform(-8 / 255, 8 / 255, x.shape).astype(np.float32)
            x = x + noise

        return x, y
