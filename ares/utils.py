import json

import numpy as np

from ares.attacker import Attack, AttackerAgent
from ares.defender import DefenderAgent, Detector
from ares.environment import AresEnvironment
from ares.scenario import EvaluationScenario


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        config = json.load(f)
    # TODO: validate config file
    return config


def get_attacker_agent(config: dict) -> AttackerAgent:
    attacker_attacks = config["attacker"]["attacks"]
    attacks = []

    for attack_config in attacker_attacks:
        attack_type = attack_config["type"]
        attack_name = attack_config["name"]
        attack_params = attack_config["params"]
        attack = Attack(attack_type, attack_name, attack_params)
        attacks.append(attack)

    probabilities = config["attacker"].get("probabilities", None)
    if probabilities is None:
        probabilities = np.ones(len(attacks)) / len(attacks)
    else:
        probabilities = np.array(probabilities) / np.sum(probabilities)

    evasion_probability = config["attacker"].get("evasion_probability", None)

    attacker_agent = AttackerAgent(attacks, probabilities, evasion_probability)
    return attacker_agent


def get_defender_agent(config: dict) -> DefenderAgent:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    detector_args = config["defender"].get("detector", None)
    if detector_args:
        detector_file = detector_args["file"]
        detector_name = detector_args["name"]
        detector_function = detector_args.get("function", "detect")
        detector_params = detector_args.get("params", {})
        detector_probability = detector_args.get("probability", 1.0)
        detector = Detector(
            file=detector_file,
            name=detector_name,
            function=detector_function,
            params=detector_params,
            probability=detector_probability,
        )
    else:
        detector = None

    defender_agent = DefenderAgent(classifiers, probs, detector)
    return defender_agent


def get_evaluation_scenario(config: dict) -> EvaluationScenario:
    threat_model = config["scenario"]["threat_model"]
    dataroot = config["scenario"]["dataroot"]
    random_noise = config["scenario"].get("random_noise", False)
    num_episodes = config["scenario"]["num_episodes"]
    max_rounds = config["scenario"]["max_rounds"]

    execution_scenario = EvaluationScenario(threat_model, dataroot, random_noise, num_episodes, max_rounds)
    return execution_scenario


def create_environment(config: dict) -> AresEnvironment:
    attacker = get_attacker_agent(config)
    defender = get_defender_agent(config)
    scenario = get_evaluation_scenario(config)
    env = AresEnvironment(attacker, defender, scenario)
    return env
