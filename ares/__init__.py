import logging

import gym
from gym.envs.registration import register

from ares import environment, utils

logger = logging.getLogger(__name__)

register(id="AresEnv-v0", entry_point="ares.environment:AresEnv")


def create_env(config: dict, device) -> environment.AresEnv:
    defender_agent = utils.get_defender_agent(config, device)
    attacker_agent = utils.get_attacker_agent(config)
    execution_scenario = utils.get_execution_scenario(config)
    env = gym.make("AresEnv-v0", attacker=attacker_agent, defender=defender_agent, scenario=execution_scenario)
    return env
