from typing import List, Tuple

import gym
from gym import spaces
import numpy as np

from ares.attacker import AttackerAgent
from ares.defender import DefenderAgent
from ares.scenario import EvaluationScenario


class AresEnvironment(gym.Env):
    metadata = {"render.modes": ["console"]}

    def __init__(self, attacker: AttackerAgent, defender: DefenderAgent, scenario: EvaluationScenario):
        self.attacker = attacker
        self.defender = defender
        self.scenario = scenario
        self.n_agents = 2
        self.done = False
        self.step_count = 0
        self.queries = 0
        self.reward = 0
        self.episode_rewards: List[int] = []
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Dict({})

    def reset(self) -> dict:
        self.done = False
        self.step_count = 0
        self.queries = 0
        self.reward = 0

        x, y = self.scenario.get_valid_sample(self.defender.classifiers)
        observation = {
            "x": x,
            "y": y,
        }

        return observation

    def step(self, action: dict) -> Tuple[dict, float, bool, dict]:
        self.step_count += 1

        x: np.ndarray = action["x"]
        y: np.ndarray = action["y"]

        # defender turn
        self.defender.update_policy({})
        classifier = self.defender.defend()

        # attacker turn
        self.attacker.update_policy({})
        x_adv, eps, evaded = self.attacker.attack(classifier, x, y)
        queries = classifier.queries()
        self.queries += queries

        # run detector if not evading
        if evaded:
            detected = False
            x_adv = x
        else:
            detected = self.defender.detect(x_adv)

        # check winner
        winner = None
        y_pred = classifier.predict(x_adv)
        if detected:
            self.done = True
            winner = "defender"
        elif y_pred != y:
            self.done = True
            winner = "attacker"
        elif self.step_count >= self.scenario.max_rounds:
            self.done = True
            winner = "defender"

        observation = {
            "x": x,
            "y": y,
            "x_adv": x_adv,
            "y_pred": y_pred,
            "eps": eps,
            "evaded": evaded,
            "detected": detected,
            "winner": winner,
        }
        info = {
            "description": "",
            "step_count": self.step_count,
            "queries": self.queries,
        }
        self.reward = self.step_count
        self.episode_rewards.append(self.reward)

        return observation, self.reward, self.done, info

    def render(self, mode: str) -> None:
        return

    def close(self) -> None:
        return
