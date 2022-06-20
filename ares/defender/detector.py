import importlib
import importlib.util
from typing import Any, NamedTuple

import numpy as np


class Detector:
    def __init__(self, module: Any, fn_name: str, probability: float):
        self.module = module
        self.fn_name = fn_name
        self.probability = probability

    def detect(self, x: np.ndarray) -> bool:
        p = np.random.rand()
        if p < self.probability:
            detected = getattr(self.module, self.fn_name)(x)
            return detected
        return False


class DetectorConfig(NamedTuple):
    file: str
    name: str
    function: str
    params: dict
    probability: float = 1.0


def get_detector(detector_config: DetectorConfig) -> Detector:
    spec = importlib.util.spec_from_file_location("module.name", detector_config.file)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ctor = getattr(module, detector_config.name)
        model = ctor(**detector_config.params)
        detector = Detector(model, detector_config.function, detector_config.probability)
        return detector
    else:
        raise ImportError(f"cannot load {detector_config.name} from {detector_config.file}")
