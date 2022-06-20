import importlib
import importlib.util

import art
from art.estimators.classification import PyTorchClassifier
import numpy as np
import torch
import torch.nn as nn


class Classifier:
    def __init__(
        self,
        model_file: str,
        model_name: str,
        model_params: dict,
        model_state: str,
        classifier_type: str,
        classifier_params: dict,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        spec = importlib.util.spec_from_file_location("module.name", model_file)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            ctor = getattr(module, model_name)
            model: nn.Module = ctor(**model_params)
        else:
            raise ImportError(f"Error loading classifier: cannot load {model_name} from {model_file}")

        state_dict = torch.load(model_state, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        components = classifier_params["loss"].split(".")
        module = importlib.import_module(".".join(components[:-1]))
        loss_fn = getattr(module, components[-1])
        classifier_params["loss"] = loss_fn()

        ctor = getattr(art.estimators.classification, classifier_type)
        self.classifier: PyTorchClassifier = ctor(model, **classifier_params)

    def predict(self, x: np.ndarray, logits=False) -> np.ndarray:
        out = self.classifier.predict(x)
        if logits:
            return out
        else:
            y_pred = self.classifier.reduce_labels(out)
            return y_pred
