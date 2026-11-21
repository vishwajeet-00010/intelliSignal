"""
IntelliSignal - Stage 2
rl/config.py

Loads the reinforcement-learning environment settings.

There are three layers, each one overriding the previous:

    1. the built-in defaults in this file
    2. config/rl/environment.yaml
    3. an optional "rl:" section inside the scenario file itself

The traffic scenario files from Stage 1 (config/moderate.yaml,
config/heavy.yaml) are read by the Stage 1 loader and are not modified.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict

import yaml

from utils.config_loader import PROJECT_ROOT

RL_CONFIG_FILE = os.path.join(PROJECT_ROOT, "config", "rl", "environment.yaml")

# Used when config/rl/environment.yaml is missing, and as the base that the
# file is merged onto, so a partial file is always valid.
DEFAULTS: Dict[str, Any] = {
    "environment": {
        "decision_interval": 5,
        "min_green": 5,
        "max_green": 60,
        "episode_duration": None,
        "yellow_time": None,
        "all_red_time": None,
        "warmup_time": 0,
    },
    "observation": {
        "max_queue_per_approach": 50.0,
        "max_waiting_per_approach": 1500.0,
    },
    "reward": {
        "waiting_time_weight": 1.0,
        "queue_weight": 1.0,
        "fairness_weight": 0.0,
        "scale": 0.01,
    },
}


class RLConfigError(ValueError):
    """Raised when the RL environment settings are missing or inconsistent."""


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_rl_config(scenario_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Return the effective RL settings for a scenario.

    `scenario_config` is the dictionary produced by
    utils.config_loader.load_config(). Values that are left as null in the RL
    file are filled in from it, so the RL environment automatically respects
    the same yellow / all-red clearance and the same episode length as the
    Stage 1 fixed-time controller.
    """
    config = copy.deepcopy(DEFAULTS)

    if os.path.isfile(RL_CONFIG_FILE):
        with open(RL_CONFIG_FILE, "r", encoding="utf-8") as handle:
            from_file = yaml.safe_load(handle) or {}
        if not isinstance(from_file, dict):
            raise RLConfigError("{} does not contain a YAML mapping.".format(RL_CONFIG_FILE))
        config = _deep_merge(config, from_file)

    if scenario_config:
        config = _deep_merge(config, scenario_config.get("rl", {}))
        config = _fill_from_scenario(config, scenario_config)

    validate_rl_config(config)
    return config


def _fill_from_scenario(config: Dict[str, Any],
                        scenario_config: Dict[str, Any]) -> Dict[str, Any]:
    """Replace the null placeholders with the scenario's own values."""
    env = config["environment"]
    signal = scenario_config["signal"]
    simulation = scenario_config["simulation"]

    if env.get("yellow_time") is None:
        env["yellow_time"] = float(signal["yellow"])
    if env.get("all_red_time") is None:
        env["all_red_time"] = float(signal["all_red"])
    if env.get("episode_duration") is None:
        env["episode_duration"] = float(simulation["duration"])
    return config


def validate_rl_config(config: Dict[str, Any]) -> None:
    """Raise RLConfigError if the settings cannot produce a working episode."""
    for section in ("environment", "observation", "reward"):
        if section not in config:
            raise RLConfigError("Missing '{}' section in the RL settings.".format(section))

    env = config["environment"]
    interval = float(env["decision_interval"])
    if interval <= 0:
        raise RLConfigError("environment.decision_interval must be positive.")

    yellow = env.get("yellow_time")
    all_red = env.get("all_red_time")
    if yellow is not None and all_red is not None:
        transition = float(yellow) + float(all_red)
        if transition >= interval:
            raise RLConfigError(
                "environment.decision_interval ({:.0f} s) must be longer than "
                "yellow_time + all_red_time ({:.0f} s), otherwise a switch "
                "would leave no green time in the interval.".format(interval, transition))

    if float(env["min_green"]) < 0:
        raise RLConfigError("environment.min_green must not be negative.")
    if float(env["max_green"]) <= float(env["min_green"]):
        raise RLConfigError("environment.max_green must be larger than min_green.")
    if float(env.get("warmup_time", 0)) < 0:
        raise RLConfigError("environment.warmup_time must not be negative.")

    obs = config["observation"]
    for key in ("max_queue_per_approach", "max_waiting_per_approach"):
        if float(obs[key]) <= 0:
            raise RLConfigError("observation.{} must be positive.".format(key))

    reward = config["reward"]
    for key in ("waiting_time_weight", "queue_weight", "fairness_weight", "scale"):
        if key not in reward:
            raise RLConfigError("Missing 'reward.{}'.".format(key))
    if float(reward["scale"]) <= 0:
        raise RLConfigError("reward.scale must be positive.")
