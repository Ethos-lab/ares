import importlib
import importlib.util

import numpy as np


class Detector:
    def __init__(self, file: str, name: str, function: str, params: str, probability: float):
        self.file = file
        self.name = name
        self.function = function
        self.params = params
        self.probability = probability
        self.detector = self.load()

    def load(self):
        spec = importlib.util.spec_from_file_location("module.name", self.file)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            ctor = getattr(module, self.name)
            detector = ctor(**self.params)
        else:
            raise ImportError(f"Error loading detector: cannot load {self.name} from {self.file}")

        return detector

    def reset(self):
        self.detector = self.load()

    def detect(self, x: np.ndarray) -> bool:
        p = np.random.rand()
        if p < self.probability:
            detected = getattr(self.detector, self.function)(x)
            return detected
        return False
