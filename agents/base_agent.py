"""
IntelliSignal - Dashboard stage
agents/base_agent.py

The decision-making layer's interface.

IntelliSignal is built in layers:

    presentation   Streamlit dashboard          (dashboard/)
    decision       THIS LAYER                   (agents/)
    environment    Gymnasium RL environment     (rl/)
    communication  TraCI                        (SUMO's python API)
    simulation     SUMO                         (sumo/)
    evaluation     metrics                      (utils/metrics.py)

Anything that decides which traffic-signal phase should be green implements
`TrafficAgent`:

    TrafficAgent
       |-- FixedTimeAgent   the Stage 1 fixed-time plan      (rule-based)
       |-- DQNAgent         learned Q-values, Stage 3        (neural network)

Stage 3 added the second implementation and needed no change to this
interface: `DQNAgent` answers `select_action` like any other agent. The one
addition is `TrafficState.observation()`, because a learned agent reads the
environment's full per-approach observation rather than the NS/EW summary -
see the note on that method.

A DQN agent is only usable when trained weights exist. `DQNAgent` raises
`DQNModelNotFoundError` when asked to control a signal without them; it never
falls back to the fixed-time controller while calling itself a DQN.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------
# An agent names the phase that should be green. This is kept here, in one
# place, rather than scattered through the code.
ACTION_NS_GREEN = 0
ACTION_EW_GREEN = 1

ACTION_SPACE: Tuple[int, ...] = (ACTION_NS_GREEN, ACTION_EW_GREEN)
ACTION_NAMES: Dict[int, str] = {
    ACTION_NS_GREEN: "NS GREEN",
    ACTION_EW_GREEN: "EW GREEN",
}
ACTION_DESCRIPTIONS: Dict[int, str] = {
    ACTION_NS_GREEN: "Give green to the North-South movement",
    ACTION_EW_GREEN: "Give green to the East-West movement",
}

# Phase indices used by the simulation and by rl/action.py. They line up with
# the action values above by design: action i asks for phase i.
PHASE_NS = 0
PHASE_EW = 1


# ---------------------------------------------------------------------------
# Two conventions, and the bridge between them
# ---------------------------------------------------------------------------
# This module uses PHASE SELECTION: "which phase should be green".
# The Stage 2 RL environment (rl/action.py) uses KEEP / SWITCH: "hold the
# current phase, or change it". Both are common in the literature and each is
# natural in its own place - an agent reasons about which direction to serve,
# while the environment has to translate that into a safe signal change with
# a yellow clearance.
#
# Rather than let the two conventions drift apart, the translation lives here
# and is tested. A future DQN can be written in either convention and plugged
# in through this function.
ENV_ACTION_KEEP = 0
ENV_ACTION_SWITCH = 1


def phase_action_to_env_action(desired_phase: int, current_phase: int) -> int:
    """Translate 'I want phase X green' into the RL environment's keep/switch.

    >>> phase_action_to_env_action(PHASE_NS, PHASE_NS)
    0
    >>> phase_action_to_env_action(PHASE_EW, PHASE_NS)
    1
    """
    if desired_phase not in ACTION_SPACE:
        raise ValueError(
            "Unknown phase {}. Valid phases are {}.".format(desired_phase, ACTION_SPACE))
    return ENV_ACTION_KEEP if desired_phase == current_phase else ENV_ACTION_SWITCH


# ---------------------------------------------------------------------------
# Traffic state
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrafficState:
    """What an agent is allowed to look at when it decides.

    Deliberately small and readable, and deliberately the same quantities the
    Stage 2 RL observation is built from, so a DQN reading `rl/state.py` and a
    rule reading this object are reasoning about the same intersection.

        ns_queue                 halting vehicles on North + South approaches
        ew_queue                 halting vehicles on East + West approaches
        ns_waiting               current waiting time on N + S, vehicle-seconds
        ew_waiting               current waiting time on E + W, vehicle-seconds
        current_phase            PHASE_NS or PHASE_EW
        time_since_phase_change  seconds the current green has been running
        simulation_time          SUMO's clock, in seconds
    """

    ns_queue: float = 0.0
    ew_queue: float = 0.0
    ns_waiting: float = 0.0
    ew_waiting: float = 0.0
    current_phase: int = PHASE_NS
    time_since_phase_change: float = 0.0
    simulation_time: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- convenience ----------------------------------------------------
    @property
    def total_queue(self) -> float:
        return self.ns_queue + self.ew_queue

    @property
    def total_waiting(self) -> float:
        return self.ns_waiting + self.ew_waiting

    def as_vector(self) -> Tuple[float, ...]:
        """The state as a plain tuple, in a fixed documented order."""
        return (
            self.ns_queue,
            self.ew_queue,
            self.ns_waiting,
            self.ew_waiting,
            float(self.current_phase),
            self.time_since_phase_change,
        )

    VECTOR_LABELS: Tuple[str, ...] = (
        "ns_queue", "ew_queue", "ns_waiting", "ew_waiting",
        "current_phase", "time_since_phase_change",
    )

    # -- the DQN's own input, carried alongside -------------------------
    def observation(self):
        """The Stage 2 10-number observation, when the caller supplied one.

        The four NS/EW fields above are a readable SUMMARY: they add North to
        South and East to West, so the per-approach detail the Q-network reads
        cannot be reconstructed from them. The learned agent therefore needs
        the environment's own observation vector, and it travels in `extra`
        under the key "observation" - put there by `from_env_info()`.

        Returns None when no observation was supplied, which is the normal
        case for a rule-based agent that does not need one.
        """
        return self.extra.get("observation")

    # -- bridge from the Stage 2 RL environment -------------------------
    @classmethod
    def from_env_info(cls, info: Dict[str, Any], observation=None) -> "TrafficState":
        """Build a TrafficState from the info dict of `rl.IntelliSignalEnv`.

        This is the seam the DQN stage uses: the RL environment already reports
        per-approach queues and waiting times, and this turns them into the
        NS / EW pairs an agent reasons about.

        `observation` is the 10-number vector the same `step()` returned. Pass
        it whenever a learned agent may see this state - it is stored in
        `extra["observation"]` and read back by `observation()`. Rule-based
        agents ignore it, so passing it is always safe.
        """
        queues = info.get("queue_per_approach", {})
        waiting = info.get("waiting_per_approach", {})
        extra: Dict[str, Any] = {}
        if observation is not None:
            extra["observation"] = observation
        if "min_green" in info:
            extra["min_green"] = float(info["min_green"])
        if "max_green" in info:
            extra["max_green"] = float(info["max_green"])
        return cls(
            ns_queue=float(queues.get("north", 0.0) + queues.get("south", 0.0)),
            ew_queue=float(queues.get("east", 0.0) + queues.get("west", 0.0)),
            ns_waiting=float(waiting.get("north", 0.0) + waiting.get("south", 0.0)),
            ew_waiting=float(waiting.get("east", 0.0) + waiting.get("west", 0.0)),
            current_phase=int(info.get("phase", PHASE_NS)),
            time_since_phase_change=float(info.get("green_elapsed", 0.0)),
            simulation_time=float(info.get("simulation_time", 0.0)),
            extra=extra,
        )


# ---------------------------------------------------------------------------
# The interface
# ---------------------------------------------------------------------------
class TrafficAgent(ABC):
    """Anything that decides which phase should be green.

    A subclass must implement `select_action`. Everything else is optional.
    """

    #: Short name shown in the user interface.
    name: str = "Traffic agent"

    #: One line describing how the agent decides.
    description: str = ""

    #: False for agents that are declared but not implemented yet.
    available: bool = True

    #: True once a learning algorithm is behind the decisions. Used by the
    #: dashboard so that it never describes a rule-based controller as AI.
    is_learned: bool = False

    # ------------------------------------------------------------------
    @abstractmethod
    def select_action(self, state: TrafficState) -> int:
        """Return ACTION_NS_GREEN or ACTION_EW_GREEN for the given state."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear any internal state at the start of an episode. Optional."""

    def explain(self, state: TrafficState) -> Dict[str, Any]:
        """Why the agent chose what it chose.

        The dashboard shows this next to the signal, so a panel member can see
        the reasoning rather than only the outcome. The default implementation
        reports the action and the state it was based on; a DQN will override
        it to add its Q-values.
        """
        action = self.select_action(state)
        return {
            "agent": self.name,
            "action": action,
            "action_name": ACTION_NAMES[action],
            "is_learned": self.is_learned,
            "state": {
                "NS queue": state.ns_queue,
                "EW queue": state.ew_queue,
                "NS waiting": state.ns_waiting,
                "EW waiting": state.ew_waiting,
                "current phase": ACTION_NAMES[state.current_phase],
                "time in phase": state.time_since_phase_change,
            },
        }

    # ------------------------------------------------------------------
    def env_action(self, state: TrafficState) -> int:
        """The agent's choice expressed as the RL environment's keep/switch."""
        return phase_action_to_env_action(self.select_action(state),
                                          state.current_phase)

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        return "<{} name={!r} available={}>".format(
            type(self).__name__, self.name, self.available)


class AgentNotImplementedError(NotImplementedError):
    """Raised when a declared-but-unbuilt agent is asked to decide."""
