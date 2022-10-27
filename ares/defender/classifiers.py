import importlib
import importlib.util
from typing import Optional

from art.estimators.classification import PyTorchClassifier, TensorFlowV2Classifier
import numpy as np
import torch
import torch.nn as nn


class ModelWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.queries = 0

    def reset(self):
        self.queries = 0

    def forward(self, x: torch.Tensor):
        self.queries += 1
        return self.model(x)


class Classifier(PyTorchClassifier):
    def __init__(self, file: str, name: str, params: dict, checkpoint: Optional[str], dataset_params: dict):
        self.name = name

        spec = importlib.util.spec_from_file_location("module.name", file)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            ctor = getattr(module, name)
            model: nn.Module = ctor(**params)
        else:
            raise ImportError(f"Error loading classifier: cannot load {name} from {file}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if checkpoint is not None:
            state_dict = torch.load(checkpoint, map_location=device)
            model.load_state_dict(state_dict)

        model.to(device)
        model.eval()
        model = ModelWrapper(model)

        loss = nn.CrossEntropyLoss()
        input_shape = dataset_params["input_shape"]
        num_classes = dataset_params["num_classes"]
        clip_values = dataset_params.get("clip_values", (0, 1))
        # self.classifier = PyTorchClassifier(
        #     model=self.model, loss=loss, input_shape=input_shape, nb_classes=num_classes, clip_values=clip_values
        # )
        super().__init__(model=model, loss=loss, input_shape=input_shape, nb_classes=num_classes, clip_values=clip_values)

    def reset(self):
        self.model.reset()

    def queries(self):
        return self.model.queries

    def predict(self, x: np.ndarray, batch_size: int = 128, training_mode: bool = False, logits: bool = False, **kwargs) -> np.ndarray:
        out = super().predict(x, batch_size, training_mode, **kwargs)
        if logits:
            return out
        else:
            y_pred = self.reduce_labels(out)
            return y_pred
