import importlib.util

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import art
from art.estimators.classification import PyTorchClassifier
from art.attacks import EvasionAttack


def load_torch_model(model_file: str, model_name: str, model_params: dict, model_state: str, device: torch.device) -> nn.Module:
    spec = importlib.util.spec_from_file_location("module.name", model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ctor = getattr(module, model_name)
    model = ctor(**model_params)

    state_dict = torch.load(model_state)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_classifier(model: nn.Module, classifier_type: str, classifier_params: dict, device: torch.device) -> PyTorchClassifier:
    ctor = getattr(art.estimators.classification, classifier_type)
    classifier = ctor(model, device_type=device.type, **classifier_params)
    return classifier


def get_evasion_attack(attack_name: str, classifier: PyTorchClassifier, attack_params: dict) -> EvasionAttack:
    ctor = getattr(art.attacks.evasion, attack_name)
    attack = ctor(classifier, **attack_params)
    return attack


def get_valid_sample(dataset, models, device):
    with torch.no_grad():
        all_correct = False
        while not all_correct:
            choice = np.random.choice(len(dataset))
            sample = dataset[choice]
            image = torch.stack([sample[0]]).to(device)
            label = torch.tensor([sample[1]], device=device)
            all_correct = True
            for model in models:
                model.eval()
                pred = model(image).argmax(dim=1)
                if pred[0] != label[0]:
                    all_correct = False

    return image, label