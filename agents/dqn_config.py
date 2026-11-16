"""
IntelliSignal - Stage 3
agents/dqn_config.py

Loads the DQN and training settings.

The layering is the same idea as rl/config.py:

    1. the built-in defaults in this file
    2. config/dqn/training.yaml
    3. command-line overrides passed by training/train_dqn.py

Nothing here touches the traffic scenarios or the RL environment settings.
Every hyperparameter the training loop uses passes through `DQNConfig`, so the
value that was actually used can be written into the checkpoint metadata and
into the training history - there is no hyperparameter hidden in the code.
"""

from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from utils.config_loader import PROJECT_ROOT

DQN_CONFIG_FILE = os.path.join(PROJECT_ROOT, "config", "dqn", "training.yaml")

DEFAULTS: Dict[str, Any] = {
    "network": {
        "hidden_layers": [64, 64],
    },
    "training": {
        "episodes": 100,
        "episode_duration": None,
        "max_steps_per_episode": None,
        "batch_size": 64,
        "learning_rate": 0.001,
        "gamma": 0.9,
        "loss": "huber",
        "gradient_clip": 10.0,
        "replay_capacity": 50000,
        "learning_starts": 500,
        "train_frequency": 1,
        "target_update_frequency": 500,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": "linear",
        "epsilon_decay_steps": None,
        "exploration_fraction": 0.5,
        "epsilon_decay_rate": 0.9995,
        "seed": 42,
        "checkpoint_every": 10,
        "best_window": 5,
        "vary_episode_seed": True,
    },
}

VALID_LOSSES = ("huber", "mse")
VALID_DECAYS = ("linear", "exponential")


class DQNConfigError(ValueError):
    """Raised when the DQN settings cannot produce a working training run."""


# ---------------------------------------------------------------------------
@dataclass
class DQNConfig:
    """Every number the agent and the training loop use.

    Kept as one flat object so it can be written to JSON without surprises.
    """

    # network
    hidden_layers: List[int] = field(default_factory=lambda: [64, 64])

    # training
    episodes: int = 100
    episode_duration: Optional[float] = None
    max_steps_per_episode: Optional[int] = None
    batch_size: int = 64
    learning_rate: float = 0.001
    gamma: float = 0.9
    loss: str = "huber"
    gradient_clip: float = 10.0
    replay_capacity: int = 50000
    learning_starts: int = 500
    train_frequency: int = 1
    target_update_frequency: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: str = "linear"
    epsilon_decay_steps: Optional[int] = None
    exploration_fraction: float = 0.5
    epsilon_decay_rate: float = 0.9995
    seed: int = 42
    checkpoint_every: int = 10
    best_window: int = 5
    vary_episode_seed: bool = True

    # ------------------------------------------------------------------
    def validate(self) -> "DQNConfig":
        """Raise DQNConfigError when a setting cannot work."""
        if not self.hidden_layers or any(int(h) <= 0 for h in self.hidden_layers):
            raise DQNConfigError(
                "network.hidden_layers must be a non-empty list of positive "
                "integers, got {!r}.".format(self.hidden_layers))
        if self.episodes <= 0:
            raise DQNConfigError("training.episodes must be positive.")
        if self.batch_size <= 0:
            raise DQNConfigError("training.batch_size must be positive.")
        if not 0.0 <= self.gamma <= 1.0:
            raise DQNConfigError("training.gamma must be between 0 and 1.")
        if self.learning_rate <= 0:
            raise DQNConfigError("training.learning_rate must be positive.")
        if self.replay_capacity < self.batch_size:
            raise DQNConfigError(
                "training.replay_capacity ({}) must be at least batch_size ({})."
                .format(self.replay_capacity, self.batch_size))
        if self.learning_starts < self.batch_size:
            raise DQNConfigError(
                "training.learning_starts ({}) must be at least batch_size "
                "({}), otherwise the first mini-batch cannot be filled."
                .format(self.learning_starts, self.batch_size))
        if self.train_frequency < 1:
            raise DQNConfigError("training.train_frequency must be at least 1.")
        if self.target_update_frequency < 1:
            raise DQNConfigError(
                "training.target_update_frequency must be at least 1. "
                "Using the policy network as its own target is exactly what "
                "the target network exists to avoid.")
        if self.loss not in VALID_LOSSES:
            raise DQNConfigError("training.loss must be one of {}.".format(
                ", ".join(VALID_LOSSES)))
        if self.epsilon_decay not in VALID_DECAYS:
            raise DQNConfigError("training.epsilon_decay must be one of {}.".format(
                ", ".join(VALID_DECAYS)))
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise DQNConfigError(
                "training.epsilon_start / epsilon_end must satisfy "
                "0 <= epsilon_end <= epsilon_start <= 1.")
        if not 0.0 < self.exploration_fraction <= 1.0:
            raise DQNConfigError(
                "training.exploration_fraction must be in (0, 1].")
        if not 0.0 < self.epsilon_decay_rate < 1.0:
            raise DQNConfigError(
                "training.epsilon_decay_rate must be in (0, 1).")
        if self.best_window < 1:
            raise DQNConfigError("training.best_window must be at least 1.")
        return self

    # ------------------------------------------------------------------
    def decay_steps(self, steps_per_episode: int) -> int:
        """How many environment steps the epsilon schedule stretches over.

        When `epsilon_decay_steps` is left null it is derived from the number
        of episodes actually requested, so `--episodes 20` does not inherit a
        schedule designed for 100 episodes and stop exploring immediately.
        """
        if self.epsilon_decay_steps:
            return max(int(self.epsilon_decay_steps), 1)
        total = max(int(self.episodes) * int(steps_per_episode), 1)
        return max(int(round(total * float(self.exploration_fraction))), 1)

    # ------------------------------------------------------------------
    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["hidden_layers"] = [int(h) for h in self.hidden_layers]
        return data

    def describe(self) -> str:
        """One readable block, printed at the start of a training run."""
        return ("\n".join([
            "  hidden layers            : {}".format(self.hidden_layers),
            "  episodes                 : {}".format(self.episodes),
            "  gamma (discount)         : {}".format(self.gamma),
            "  learning rate            : {}".format(self.learning_rate),
            "  batch size               : {}".format(self.batch_size),
            "  loss                     : {}".format(self.loss),
            "  replay capacity          : {}".format(self.replay_capacity),
            "  learning starts after    : {} transitions".format(self.learning_starts),
            "  train frequency          : every {} step(s)".format(self.train_frequency),
            "  target update frequency  : every {} gradient step(s)".format(
                self.target_update_frequency),
            "  epsilon                  : {} -> {} ({})".format(
                self.epsilon_start, self.epsilon_end, self.epsilon_decay),
            "  seed                     : {}".format(self.seed),
        ]))


# ---------------------------------------------------------------------------
def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_dqn_settings(path: str = None) -> Dict[str, Any]:
    """The raw two-section dictionary (network / training), file merged on defaults."""
    config = copy.deepcopy(DEFAULTS)
    config_path = path or DQN_CONFIG_FILE
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as handle:
            from_file = yaml.safe_load(handle) or {}
        if not isinstance(from_file, dict):
            raise DQNConfigError(
                "{} does not contain a YAML mapping.".format(config_path))
        config = _deep_merge(config, from_file)
    return config


def load_dqn_config(path: str = None, **overrides) -> DQNConfig:
    """Build a validated `DQNConfig`.

    Keyword overrides win over the YAML file, which wins over the defaults.
    `None` overrides are ignored, so a command-line argument that was not
    given never erases a configured value.
    """
    settings = load_dqn_settings(path)
    flat: Dict[str, Any] = {}
    flat.update(settings.get("network", {}))
    flat.update(settings.get("training", {}))

    for key, value in overrides.items():
        if value is not None:
            flat[key] = value

    known = {f for f in DQNConfig.__dataclass_fields__}
    unknown = set(flat) - known
    if unknown:
        raise DQNConfigError(
            "Unknown DQN setting(s): {}. Known settings: {}.".format(
                ", ".join(sorted(unknown)), ", ".join(sorted(known))))

    config = DQNConfig(**{k: v for k, v in flat.items() if k in known})
    config.hidden_layers = [int(h) for h in config.hidden_layers]
    return config.validate()
