import importlib
import importlib.util

import art
from art.estimators.classification import PyTorchClassifier
import torch
import torch.nn as nn


def get_torch_model(
    model_file: str, model_name: str, model_params: dict, model_state: str, device: torch.device
) -> nn.Module:
    spec = importlib.util.spec_from_file_location("module.name", model_file)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ctor = getattr(module, model_name)
        model: nn.Module = ctor(**model_params)

        state_dict = torch.load(model_state, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
    else:
        raise ImportError(f"cannot load {model_name} from {model_file}")


def get_classifier(
    model: nn.Module, classifier_type: str, classifier_params: dict, device: torch.device
) -> PyTorchClassifier:
    components = classifier_params["loss"].split(".")
    module = importlib.import_module(".".join(components[:-1]))
    loss_fn = getattr(module, components[-1])
    classifier_params["loss"] = loss_fn()

    ctor = getattr(art.estimators.classification, classifier_type)
    classifier = ctor(model, device_type=device.type, **classifier_params)
    return classifier
