"""
IntelliSignal - Stage 2
rl/sumo_rl_adapter.py

SUMO-RL compatibility layer.

What SUMO-RL is
---------------
SUMO-RL (https://github.com/LucasAlegre/sumo-rl) is a library that wraps a
SUMO simulation as a Gymnasium / PettingZoo reinforcement-learning
environment. It handles the SUMO process, the decision interval, the yellow
clearance and a default observation and reward, for one or many traffic
lights.

Why IntelliSignal has its own environment as well
-------------------------------------------------
SUMO-RL's action space for one traffic light is Discrete(number of green
phases): the agent names the green phase it wants next. The IntelliSignal
project specification calls for a two-action space instead:

    0 = keep the current phase
    1 = switch to the next phase

and for a specific 10-number observation and a specific reward. Expressing
those through SUMO-RL would mean overriding most of it, so rl/environment.py
implements them directly on TraCI. It deliberately uses SUMO-RL's timing
vocabulary and semantics - `decision_interval` is SUMO-RL's `delta_time`,
and `yellow_time`, `min_green` and `max_green` behave the same way - so the
two are conceptually interchangeable.

What this module is for
-----------------------
It proves the claim that the Stage 1 intersection is a valid SUMO-RL
environment, by building a real `sumo_rl.SumoEnvironment` on exactly the same
network and route files. The Stage 2 test suite runs it. If a later stage
wants SUMO-RL's multi-agent tooling or its built-in reward functions, this is
the entry point; nothing else has to change.

Usage
-----
    from rl.sumo_rl_adapter import make_sumo_rl_env, sumo_rl_available

    if sumo_rl_available():
        env = make_sumo_rl_env("moderate", num_seconds=120)
        observation, info = env.reset()
        env.close()
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional

from rl.config import load_rl_config
from utils.config_loader import load_config
from utils.scenario_builder import build_scenario

# SUMO input files for the adapter get their own name, so neither the Stage 1
# files nor the ones used by rl/environment.py are touched.
SUMO_RL_SUFFIX = "_sumorl"


def ensure_sumo_home() -> Optional[str]:
    """Make sure SUMO_HOME is set, because sumo_rl requires it at import time.

    A normal SUMO installation sets SUMO_HOME itself. When SUMO came from the
    `eclipse-sumo` pip package there may be no SUMO_HOME, so it is derived
    from the installed package. Returns the value, or None if SUMO cannot be
    located.
    """
    existing = os.environ.get("SUMO_HOME")
    if existing and os.path.isdir(existing):
        return existing

    try:
        import sumo
    except ImportError:
        return None

    candidate = os.path.dirname(os.path.abspath(sumo.__file__))
    if os.path.isdir(os.path.join(candidate, "tools")):
        os.environ["SUMO_HOME"] = candidate
        return candidate
    return None


def sumo_rl_available() -> bool:
    """True when sumo_rl can actually be imported in this environment."""
    ensure_sumo_home()
    try:
        import sumo_rl  # noqa: F401
    except Exception:
        return False
    return True


def sumo_rl_version() -> Optional[str]:
    if not sumo_rl_available():
        return None
    import sumo_rl
    return getattr(sumo_rl, "__version__", None)


def scenario_files(scenario: str, num_seconds: float) -> Dict[str, str]:
    """Generate the SUMO network / route files for the SUMO-RL environment.

    Reuses the Stage 1 builder, so the demand and the network are identical to
    the ones the fixed-time baseline used.
    """
    config = copy.deepcopy(load_config(scenario))
    config["scenario"]["name"] = scenario + SUMO_RL_SUFFIX
    config["simulation"]["duration"] = float(num_seconds)
    return build_scenario(config)


def make_sumo_rl_env(scenario: str = "moderate",
                     num_seconds: float = 300.0,
                     use_gui: bool = False) -> Any:
    """Build a single-agent sumo_rl.SumoEnvironment on the Stage 1 network.

    The timing settings are taken from config/rl/environment.yaml, so the
    SUMO-RL environment and the IntelliSignal environment use the same
    decision interval, yellow time, minimum green and maximum green.
    """
    if not sumo_rl_available():
        raise RuntimeError(
            "sumo_rl is not importable. Install it with 'pip install sumo-rl' "
            "and make sure SUMO_HOME points at your SUMO installation.")

    from sumo_rl import SumoEnvironment

    scenario_config = load_config(scenario)
    rl_config = load_rl_config(scenario_config)
    env_cfg = rl_config["environment"]
    files = scenario_files(scenario, num_seconds)

    return SumoEnvironment(
        net_file=files["net"],
        route_file=files["routes"],
        use_gui=use_gui,
        num_seconds=float(num_seconds),
        delta_time=int(env_cfg["decision_interval"]),
        yellow_time=int(env_cfg["yellow_time"]),
        min_green=int(env_cfg["min_green"]),
        max_green=int(env_cfg["max_green"]),
        single_agent=True,
        sumo_seed=int(scenario_config["simulation"]["seed"]),
        sumo_warnings=False,
    )
