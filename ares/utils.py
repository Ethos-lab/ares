import importlib
import importlib.util
import json

import art
from art.estimators.classification import PyTorchClassifier
import numpy as np
import torch
import torch.nn as nn

from ares import attacker, defender, scenario


def get_config(path: str):
    with open(path, "r") as f:
        config = json.load(f)
    # TODO: validate config file
    return config


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


def get_detector(detector_args: dict) -> "defender.Detector":
    if not detector_args:
        return defender.Detector(None, None, None)

    detector_file = detector_args.get('detector_file', None)
    detector_name = detector_args.get('detector_name', None)
    detector_fn = detector_args.get('detector_function', None)
    detector_params = detector_args.get('detector_params', None)
    probability = detector_args.get('probability', None)

    spec = importlib.util.spec_from_file_location("module.name", detector_file)

    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ctor = getattr(module, detector_name)
        model = ctor(**detector_params)
        detector = defender.Detector(model, detector_fn, probability)
        return detector
    else:
        raise ImportError(f"cannot load {detector_name} from {detector_file}")


def get_defender_agent(config: dict, device: torch.device) -> "defender.DefenderAgent":
    defender_models = config["defender"]["models"]
    classifiers = []

    for defender_model in defender_models:
        model_file = defender_model["model_file"]
        model_name = defender_model["model_name"]
        model_params = defender_model["model_params"]
        model_state = defender_model["model_state"]
        classifier_type = defender_model["classifier_type"]
        classifier_params = defender_model["classifier_params"]
        model = get_torch_model(model_file, model_name, model_params, model_state, device)
        classifier = get_classifier(model, classifier_type, classifier_params, device)
        classifiers.append(classifier)

    probs = config["defender"].get("probabilities", None)
    if probs:
        probs = np.array(probs) / np.sum(probs)
    else:
        probs = np.ones(len(classifiers)) / len(classifiers)

    detector_args = config['defender'].get('detector', None)
    detector = get_detector(detector_args)

    defender_agent = defender.DefenderAgent(classifiers, probs, detector)
    return defender_agent


def get_attacker_agent(config: dict) -> "attacker.AttackerAgent":
    attacker_attacks = config["attacker"]["attacks"]
    attacks = []

    for attack in attacker_attacks:
        attack_type = attack["attack_type"].lower()
        attack_name = attack["attack_name"]
        attack_params = attack["attack_params"]
        attacks_config = attacker.AttackConfig(attack_type, attack_name, attack_params)
        attacks.append(attacks_config)

    probs = config["attacker"].get("probabilities", None)
    if probs:
        probs = np.array(probs) / np.sum(probs)
    else:
        probs = np.ones(len(attacks)) / len(attacks)

    evasion_prob = config["attacker"].get('evasion_prob', None)

    attacker_agent = attacker.AttackerAgent(attacks, probs, evasion_prob)
    return attacker_agent


def get_execution_scenario(config: dict) -> "scenario.ExecutionScenario":
    threat_model = config["scenario"]["threat_model"]
    dataroot = config["scenario"]["dataroot"]
    random_noise = config["scenario"]["random_noise"]
    num_episodes = config["scenario"]["num_episodes"]
    max_rounds = config["scenario"]["max_rounds"]

    execution_scenario = scenario.ExecutionScenario(threat_model, dataroot, random_noise, num_episodes, max_rounds)
    return execution_scenario
