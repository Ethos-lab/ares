import importlib
import importlib.util

from art.estimators.classification import PyTorchClassifier
import numpy as np
import torch
import torch.nn as nn


class Classifier:
    def __init__(self, file: str, name: str, params: dict, checkpoint: str, dataset_params: dict):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        spec = importlib.util.spec_from_file_location("module.name", file)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            ctor = getattr(module, name)
            model: nn.Module = ctor(**params)
        else:
            raise ImportError(f"Error loading classifier: cannot load {name} from {file}")

        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        loss = nn.CrossEntropyLoss()
        input_shape = dataset_params["input_shape"]
        num_classes = dataset_params["num_classes"]
        clip_values = dataset_params.get("clip_values", (0, 1))
        self.classifier = PyTorchClassifier(
            model=model, loss=loss, input_shape=input_shape, nb_classes=num_classes, clip_values=clip_values
        )

    def predict(self, x: np.ndarray, logits=False) -> np.ndarray:
        out = self.classifier.predict(x)
        if logits:
            return out
        else:
            y_pred = self.classifier.reduce_labels(out)
            return y_pred
