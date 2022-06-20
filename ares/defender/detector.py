import importlib
import importlib.util

import numpy as np


class Detector:
    def __init__(self, file: str, name: str, function: str, params: str, probability: float):
        spec = importlib.util.spec_from_file_location("module.name", file)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            ctor = getattr(module, name)
            self.module = ctor(**params)
        else:
            raise ImportError(f"Error creating detector: cannot load {name} from {file}")

        self.function = function
        self.probability = probability

    def detect(self, x: np.ndarray) -> bool:
        p = np.random.rand()
        if p < self.probability:
            detected = getattr(self.module, self.function)(x)
            return detected
        return False
