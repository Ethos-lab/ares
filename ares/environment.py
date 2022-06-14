from typing import Tuple

import gym
from gym import spaces

from ares.attacker import AttackerAgent
from ares.defender import DefenderAgent
from ares.scenario import ExecutionScenario


class AresEnv(gym.Env):
    metadata = {"render.modes": ["console"]}

    def __init__(self, attacker: AttackerAgent, defender: DefenderAgent, scenario: ExecutionScenario):
        self.n_agents = 2
        self.attacker = attacker
        self.defender = defender
        self.scenario = scenario
        self.done = False
        self.step_count = 0
        self.reward = 0
        self.episode_rewards = []
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Dict({})

    def reset(self) -> dict:
        self.done = False
        self.step_count = 0
        self.reward = 0

        image, label = self.scenario.get_valid_sample(self.defender.classifiers)
        observation = {
            "image": image,
            "label": label,
        }

        return observation

    def step(self, action: dict) -> Tuple[dict, float, bool, dict]:
        self.step_count += 1

        image = action["image"]
        label = action["label"]

        # defender turn
        self.defender.update_policy({})
        classifier = self.defender.defend()

        # attacker turn
        self.attacker.update_policy({})
        image_adv = self.attacker.attack(classifier, image, label)

        # run detector
        detected = self.defender.detect(image_adv)

        # check winner
        winner = None
        out = classifier.predict(image_adv)
        pred = classifier.reduce_labels(out)
        if detected:
            self.done = True
            winner = 'defender'
        elif pred != label:
            self.done = True
            winner = "attacker"
        elif self.step_count >= self.scenario.max_rounds:
            self.done = True
            winner = "defender"

        observation = {
            "image": image,
            "label": label,
            "image_adv": image_adv if image_adv is not None else image,
            "pred": pred,
            "winner": winner,
        }

        info = {
            "description": "",
            "step_count": self.step_count,
        }

        self.reward = self.step_count

        return observation, self.reward, self.done, info

    def render(self, mode: str) -> None:
        return

    def close(self) -> None:
        return
