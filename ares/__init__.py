import logging

from gym.envs.registration import register

from ares import attacker, defender, environment, scenario
from ares.run import main
from ares.utils import construct, load_config

logger = logging.getLogger(__name__)

register(id="AresEnvironment-v0", entry_point="ares.environment:AresEnvironment")

__all__ = [
    "attacker",
    "defender",
    "scenario",
    "environment",
    "construct",
    "load_config",
    "main",
]
