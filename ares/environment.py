from typing import Tuple

import gym
from gym import spaces
import torch

from ares.attacker import AttackerAgent
from ares.defender import DefenderAgent


class AresEnv(gym.Env):
    metadata = {"render.modes": ["console"]}

    def __init__(self, attacker: AttackerAgent, defender: DefenderAgent, max_rounds):
        self.n_agents = 2
        self.attacker = attacker
        self.defender = defender
        self.max_rounds = max_rounds
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
        self.episode_rewards = []
        observation = {}
        return observation

    def step(self, action: dict) -> Tuple[dict, float, bool, dict]:
        self.step_count += 1
        observation = {}
        reward = 0
        info = {}

        image = action['image']
        label = action['label']

        # defender turn
        self.defender.change_classifier()
        classifier = self.defender.get_classifier()

        # attacker turn
        image_adv = self.attacker.attack(classifier, image, label)

        # check winner of round
        out = classifier.predict(image_adv)
        pred = classifier.reduce_labels(out)

        winner = None
        if pred != label:
            self.done = True
            winner = 'attacker'
        elif self.step_count >= self.max_rounds:
            self.done = True
            winner = 'defender'
        
        observation = {
            'image': image,
            'label': label,
            'image_adv': image_adv,
            'pred': pred,
            'winner': winner,
        }

        info = {
            'description': '',
            'step_count': self.step_count,
        }

        return observation, reward, self.done, info

    def render(self, mode: str) -> None:
        return

    def close(self) -> None:
        return
