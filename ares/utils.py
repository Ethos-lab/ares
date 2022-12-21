import json

import numpy as np

from ares.attacker import Attack, AttackerAgent
from ares.defender import Classifier, DefenderAgent, Detector
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
        attack_params = attack_config.get("params", {})
        epsilon_constraint = attack_config.get("epsilon_constraint", None)
        attack = Attack(attack_type, attack_name, attack_params, epsilon_constraint)
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
    defender_models = config["defender"]["models"]
    classifiers = []

    for model_config in defender_models:
        model_framework = model_config["framework"].lower()
        model_file = model_config["file"]
        model_name = model_config["name"]
        model_params = model_config.get("params", {})
        model_checkpoint = model_config.get("checkpoint", None)
        classifier = Classifier(
            framework=model_framework,
            file=model_file,
            name=model_name,
            params=model_params,
            checkpoint=model_checkpoint,
            dataset_params=config["scenario"]["dataset"],
        )
        classifiers.append(classifier)

    probabilities = config["defender"].get("probabilities", None)
    if probabilities is None:
        probabilities = np.ones(len(classifiers)) / len(classifiers)
    else:
        probabilities = np.array(probabilities) / np.sum(probabilities)

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

    defender_agent = DefenderAgent(classifiers, probabilities, detector)
    return defender_agent


def get_evaluation_scenario(config: dict) -> EvaluationScenario:
    threat_model = config["scenario"]["threat_model"]
    num_episodes = config["scenario"]["num_episodes"]
    max_rounds = config["scenario"]["max_rounds"]
    dataset = config["scenario"]["dataset"]

    execution_scenario = EvaluationScenario(threat_model, num_episodes, max_rounds, dataset)
    return execution_scenario


def construct(config: dict) -> AresEnvironment:
    attacker = get_attacker_agent(config)
    defender = get_defender_agent(config)
    scenario = get_evaluation_scenario(config)
    env = AresEnvironment(attacker, defender, scenario)
    return env
