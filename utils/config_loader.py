"""
IntelliSignal - Stage 1
config_loader.py

Loads and validates a scenario configuration (YAML) from the config/ folder.

Every tunable value of the simulation (duration, signal timings, traffic
demand, vehicle parameters) lives in one YAML file per scenario so that no
timing or demand value is hard-coded inside the simulation code.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import yaml

# Absolute path of the project root (one level above utils/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")

# Keys that must be present in every scenario file
REQUIRED_SECTIONS = ("scenario", "simulation", "signal", "demand", "vehicle")

APPROACHES = ("north", "south", "east", "west")
TURNS = ("right", "straight", "left")


class ConfigError(ValueError):
    """Raised when a scenario configuration file is missing or invalid."""


def available_scenarios() -> list:
    """Return the list of scenario names that have a YAML file in config/."""
    if not os.path.isdir(CONFIG_DIR):
        return []
    names = []
    for fname in sorted(os.listdir(CONFIG_DIR)):
        if fname.endswith((".yaml", ".yml")):
            names.append(os.path.splitext(fname)[0])
    return names


def config_path(scenario: str) -> str:
    """Return the path of the YAML file belonging to `scenario`."""
    for ext in (".yaml", ".yml"):
        candidate = os.path.join(CONFIG_DIR, scenario + ext)
        if os.path.isfile(candidate):
            return candidate
    raise ConfigError(
        "No configuration file for scenario '{}'. Available scenarios: {}".format(
            scenario, ", ".join(available_scenarios()) or "(none)"
        )
    )


def load_config(scenario: str) -> Dict[str, Any]:
    """Load the YAML configuration of `scenario` and validate it."""
    path = config_path(scenario)
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ConfigError("{} does not contain a YAML mapping.".format(path))

    validate_config(config)
    # Remember where the configuration came from - useful for logging/tests.
    config["_path"] = path
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """Raise ConfigError if a required key is missing or clearly wrong."""
    for section in REQUIRED_SECTIONS:
        if section not in config:
            raise ConfigError("Missing '{}' section in configuration.".format(section))

    sim = config["simulation"]
    for key in ("duration", "step_length", "seed"):
        if key not in sim:
            raise ConfigError("Missing 'simulation.{}'.".format(key))
    if sim["duration"] <= 0:
        raise ConfigError("simulation.duration must be positive.")
    if sim["step_length"] <= 0:
        raise ConfigError("simulation.step_length must be positive.")

    sig = config["signal"]
    for key in ("ns_green", "ew_green", "yellow", "all_red"):
        if key not in sig:
            raise ConfigError("Missing 'signal.{}'.".format(key))
        if sig[key] < 0:
            raise ConfigError("signal.{} must not be negative.".format(key))
    if sig["ns_green"] <= 0 or sig["ew_green"] <= 0:
        raise ConfigError("Green times must be positive.")
    if sig["yellow"] <= 0:
        raise ConfigError("A fixed-time signal needs a positive yellow time.")

    dem = config["demand"]
    for approach in APPROACHES:
        if approach not in dem:
            raise ConfigError("Missing 'demand.{}' (vehicles per hour).".format(approach))
        if dem[approach] < 0:
            raise ConfigError("demand.{} must not be negative.".format(approach))

    ratios = dem.get("turn_ratios")
    if not isinstance(ratios, dict):
        raise ConfigError("Missing 'demand.turn_ratios' mapping.")
    for turn in TURNS:
        if turn not in ratios:
            raise ConfigError("Missing 'demand.turn_ratios.{}'.".format(turn))
    total = sum(float(ratios[t]) for t in TURNS)
    if abs(total - 1.0) > 1e-6:
        raise ConfigError(
            "demand.turn_ratios must sum to 1.0 (currently {:.4f}).".format(total)
        )

    veh = config["vehicle"]
    for key in ("max_speed", "accel", "decel", "length", "min_gap", "sigma"):
        if key not in veh:
            raise ConfigError("Missing 'vehicle.{}'.".format(key))


def cycle_time(config: Dict[str, Any]) -> float:
    """Total length of one fixed-time signal cycle, in seconds."""
    sig = config["signal"]
    return float(
        sig["ns_green"] + sig["yellow"] + sig["all_red"]
        + sig["ew_green"] + sig["yellow"] + sig["all_red"]
    )


def total_demand(config: Dict[str, Any]) -> float:
    """Total demand entering the network, in vehicles per hour."""
    dem = config["demand"]
    return float(sum(dem[a] for a in APPROACHES))
