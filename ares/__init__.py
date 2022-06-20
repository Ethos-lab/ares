import logging

from gym.envs.registration import register

from ares import attacker, defender, environment, scenario
from ares.run import main
from ares.utils import create_environment, load_config

logger = logging.getLogger(__name__)

register(id="AresEnvironment-v0", entry_point="ares.environment:AresEnvironment")

__all__ = [
    "attacker",
    "defender",
    "scenario",
    "environment",
    "create_environment",
    "load_config",
    "main",
]
