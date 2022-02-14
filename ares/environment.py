from typing import Tuple

import gym
from gym import spaces

from ares.attacker import AttackerAgent
from ares.defender import DefenderAgent


class AresEnv(gym.Env):
    metadata = {"render.modes": ["console"]}

    def __init__(self, attacker: AttackerAgent, defender: DefenderAgent, max_rounds=10):
        self.n_agents = 2
        self.attacker = attacker
        self.defender = defender
        self.max_rounds = max_rounds
        self.done = False
        self.step_count = 0
        self.reward = 0
        self.episode_rewards = []
        self.action_space = spaces.Discrete(2)
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
        return observation, reward, self.done, info

    def render(self, mode: str) -> None:
        return

    def close(self) -> None:
        return
