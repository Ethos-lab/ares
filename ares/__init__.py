import logging

from gym.envs.registration import register

from ares import attacker, defender, scenario, utils
from ares.environment import AresEnvironment
from ares.utils import load_config

logger = logging.getLogger(__name__)

register(id="AresEnvironment-v0", entry_point="ares.environment:AresEnvironment")


def construct(config: dict, device) -> AresEnvironment:
    attacker_agent = attacker.get_attacker_agent(config)
    defender_agent = defender.get_defender_agent(config, device)
    execution_scenario = scenario.get_evaluation_scenario(config)
    env = AresEnvironment(attacker_agent, defender_agent, execution_scenario)
    return env


__all__ = ["utils", "attacker", "defender", "scenario", "load_config", "construct"]
