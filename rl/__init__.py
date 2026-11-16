"""
IntelliSignal - Stage 2
rl package: the reinforcement-learning environment foundation.

Stage 2 builds and validates the environment that a DQN agent will use in a
later stage. **No agent, no neural network and no training exist here.**

Public interface:

    from rl import IntelliSignalEnv

    env = IntelliSignalEnv(scenario="moderate")
    observation, info = env.reset(seed=42)
    observation, reward, terminated, truncated, info = env.step(action)
    env.close()

The environment is also registered with Gymnasium, so it can be created the
standard way once a later stage needs that:

    import gymnasium as gym
    import rl                      # registers the environment id
    env = gym.make("IntelliSignal-v0", scenario="moderate")
"""

from rl.environment import IntelliSignalEnv
from rl.action import ActionController, ACTION_KEEP, ACTION_SWITCH, ACTION_NAMES
from rl.state import StateExtractor, APPROACH_ORDER, OBSERVATION_LABELS
from rl.reward import RewardFunction

__all__ = [
    "IntelliSignalEnv",
    "ActionController",
    "ACTION_KEEP",
    "ACTION_SWITCH",
    "ACTION_NAMES",
    "StateExtractor",
    "APPROACH_ORDER",
    "OBSERVATION_LABELS",
    "RewardFunction",
]

ENV_ID = "IntelliSignal-v0"


def _register() -> None:
    """Register the environment with Gymnasium (safe to call more than once)."""
    try:
        from gymnasium.envs.registration import register, registry
    except ImportError:          # gymnasium not installed - nothing to do
        return
    if ENV_ID in registry:
        return
    register(id=ENV_ID, entry_point="rl.environment:IntelliSignalEnv")


_register()
