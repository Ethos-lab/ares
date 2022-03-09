import importlib
import json
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import art
from art.estimators.classification import PyTorchClassifier
from art.attacks import EvasionAttack
import torchvision

from ares import defender, attacker


def get_config(path: str):
    with open(path, 'r') as f:
        config = json.load(f)
    # TODO: validate config file
    return config


def get_torch_model(model_file: str, model_name: str, model_params: dict, model_state: str, device: torch.device) -> nn.Module:
    spec = importlib.util.spec_from_file_location("module.name", model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ctor = getattr(module, model_name)
    model = ctor(**model_params)

    state_dict = torch.load(model_state, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_classifier(model: nn.Module, classifier_type: str, classifier_params: dict, device: torch.device) -> PyTorchClassifier:
    components = classifier_params['loss'].split('.')
    module = importlib.import_module('.'.join(components[:-1]))
    loss_fn = getattr(module, components[-1])
    classifier_params['loss'] = loss_fn()

    ctor = getattr(art.estimators.classification, classifier_type)
    classifier = ctor(model, device_type=device.type, **classifier_params)
    return classifier


def get_defender_agent(config: dict, device: torch.device) -> 'defender.DefenderAgent':
    defender_models = config['defender']['models']
    classifiers = []

    for defender_model in defender_models:
        model_file = defender_model['model_file']
        model_name = defender_model['model_name']
        model_params = defender_model['model_params']
        model_state = defender_model['model_state']
        classifier_type = defender_model['classifier_type']
        classifier_params = defender_model['classifier_params']
        model = get_torch_model(model_file, model_name, model_params, model_state, device)
        classifier = get_classifier(model, classifier_type, classifier_params, device)
        classifiers.append(classifier)

    probs = config['defender'].get('probabilities', None)
    if probs:
        probs = np.array(probs) / np.sum(probs)
    else:
        probs = np.ones(len(classifiers)) / len(classifiers)

    defender_agent = defender.DefenderAgent(classifiers, probs)
    return defender_agent


def get_evasion_attack(attack_name: str, classifier: PyTorchClassifier, attack_params: dict) -> EvasionAttack:
    ctor = getattr(art.attacks.evasion, attack_name)
    attack = ctor(classifier, **attack_params)
    return attack


def get_attacker_agent(config: dict) -> 'attacker.AttackerAgent':
    attack_type = config['attacker']['attack_type'].lower()
    attack_name = config['attacker']['attack_name']
    attack_params = config['attacker']['attack_params']

    attacker_agent = attacker.AttackerAgent(attack_type, attack_name, attack_params)
    return attacker_agent


def load_dataset(path: str) -> torchvision.datasets.CIFAR10:
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(root=path, train=False, transform=transforms, download=True)
    return dataset


def get_valid_sample(dataset: torchvision.datasets.CIFAR10, classifiers: List[PyTorchClassifier]):
    with torch.no_grad():
        all_correct = False
        while not all_correct:
            choice = np.random.choice(len(dataset))
            sample = dataset[choice]
            image = np.expand_dims(sample[0].numpy(), axis=0)
            label = np.array([sample[1]])
            all_correct = True
            for classifier in classifiers:
                out = classifier.predict(image)
                pred = classifier.reduce_labels(out)
                if pred != label:
                    all_correct = False

    return image, label
