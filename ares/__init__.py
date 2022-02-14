import logging

from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(id="AresEnv-v0", entry_point="ares:AresEnv", kwargs={"attacker": None, "defender": None})
