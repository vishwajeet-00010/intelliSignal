"""
IntelliSignal - the decision-making layer.

    from agents import create_agent, list_agents

    agent = create_agent("fixed_time", scenario_config=config)
    action = agent.select_action(state)

Registered agents:

    "fixed_time"    FixedTimeAgent   the Stage 1 pre-timed plan (rule-based)
    "dqn"           DQNAgent         the Stage 3 Deep Q-Network (learned)

Both are implemented. The difference is that the DQN needs trained weights:

    create_agent("dqn", scenario="moderate")

loads `models/dqn_moderate/best.pt`, and raises `DQNModelNotFoundError` if no
checkpoint is there. It never returns a randomly initialised network dressed
up as a trained controller, and it never silently substitutes the fixed-time
agent.

`DQNAgent.is_available(scenario)` is the question the dashboard asks before
offering the DQN mode: it is True only when PyTorch is installed AND a
checkpoint exists for that scenario.
"""

from agents.base_agent import (ACTION_DESCRIPTIONS, ACTION_EW_GREEN,
                               ACTION_NAMES, ACTION_NS_GREEN, ACTION_SPACE,
                               AgentNotImplementedError, ENV_ACTION_KEEP,
                               ENV_ACTION_SWITCH, PHASE_EW, PHASE_NS,
                               TrafficAgent, TrafficState,
                               phase_action_to_env_action)
from agents.dqn_agent import (DQN_IMPLEMENTED, DQNAgent,
                              DQNModelNotFoundError, DQNUnavailableError,
                              TORCH_AVAILABLE, default_model_path,
                              dqn_model_available, find_models,
                              model_directory, set_seeds)
from agents.fixed_time_agent import FixedTimeAgent

#: name -> class.
AGENT_REGISTRY = {
    "fixed_time": FixedTimeAgent,
    "dqn": DQNAgent,
}

__all__ = [
    "AGENT_REGISTRY", "create_agent", "list_agents",
    "TrafficAgent", "TrafficState", "FixedTimeAgent", "DQNAgent",
    "AgentNotImplementedError", "DQNModelNotFoundError",
    "DQNUnavailableError", "DQN_IMPLEMENTED", "TORCH_AVAILABLE",
    "dqn_model_available", "find_models", "default_model_path",
    "model_directory", "set_seeds",
    "ACTION_NS_GREEN", "ACTION_EW_GREEN", "ACTION_SPACE", "ACTION_NAMES",
    "ACTION_DESCRIPTIONS", "PHASE_NS", "PHASE_EW",
    "ENV_ACTION_KEEP", "ENV_ACTION_SWITCH", "phase_action_to_env_action",
]


def create_agent(name: str, scenario_config=None, **kwargs) -> TrafficAgent:
    """Build a registered agent by name.

    `scenario_config` is the scenario dictionary from
    `utils.config_loader.load_config()`. The fixed-time agent reads its
    timings from it; the DQN uses only the scenario NAME in it, to find the
    right trained model, and takes everything else from its checkpoint.
    """
    if name not in AGENT_REGISTRY:
        raise KeyError("Unknown agent '{}'. Registered agents: {}".format(
            name, ", ".join(sorted(AGENT_REGISTRY))))
    agent_class = AGENT_REGISTRY[name]

    if agent_class is FixedTimeAgent and scenario_config is not None:
        return FixedTimeAgent.from_config(scenario_config)

    if agent_class is DQNAgent and scenario_config is not None:
        kwargs.setdefault("scenario",
                          scenario_config.get("scenario", {}).get("name"))

    return agent_class(**kwargs)


def list_agents():
    """Every registered agent as (name, label, description, available).

    For the DQN, `available` means "PyTorch is installed and at least one
    trained model exists". A scenario-specific answer comes from
    `DQNAgent.is_available(scenario)`.
    """
    rows = []
    for name, agent_class in sorted(AGENT_REGISTRY.items()):
        checker = getattr(agent_class, "is_available", None)
        available = (bool(checker()) if callable(checker)
                     else bool(getattr(agent_class, "available", False)))
        rows.append({
            "name": name,
            "label": agent_class.name,
            "description": agent_class.description,
            "available": available,
            "is_learned": bool(getattr(agent_class, "is_learned", False)),
        })
    return rows
