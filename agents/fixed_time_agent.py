"""
IntelliSignal - Dashboard stage
agents/fixed_time_agent.py

The Stage 1 fixed-time controller, expressed through the `TrafficAgent`
interface.

This is a real, working implementation - not a placeholder. It reproduces the
signal plan that produced the Stage 1 results:

    North-South green -> yellow -> East-West green -> yellow -> repeat

and it takes its timings from the scenario YAML file, so it cannot drift away
from the plan the simulation actually runs.

Its purpose here is twofold:

  1. It proves the agent interface is usable, because a second implementation
     (the DQN) will slot into exactly the same place.
  2. It makes the difference explicit. This agent never looks at the traffic:
     `select_action` ignores every queue and waiting time in the state and
     decides purely from the clock. That is precisely the limitation the
     adaptive controller is meant to remove, and it is easy to point at
     during a demonstration.
"""

from __future__ import annotations

from typing import Any, Dict

from agents.base_agent import (ACTION_EW_GREEN, ACTION_NAMES, ACTION_NS_GREEN,
                               PHASE_EW, PHASE_NS, TrafficAgent, TrafficState)


class FixedTimeAgent(TrafficAgent):
    """Serves each direction for a fixed number of seconds, in turn."""

    name = "Fixed-Time Controller"
    description = ("Pre-timed signal plan. Each direction receives a "
                   "pre-defined green duration regardless of traffic.")
    available = True
    is_learned = False

    def __init__(self, ns_green: float = 30.0, ew_green: float = 30.0,
                 yellow: float = 4.0, all_red: float = 0.0):
        self.ns_green = float(ns_green)
        self.ew_green = float(ew_green)
        self.yellow = float(yellow)
        self.all_red = float(all_red)

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, scenario_config: Dict[str, Any]) -> "FixedTimeAgent":
        """Build the agent from a scenario YAML dictionary."""
        signal = scenario_config["signal"]
        return cls(ns_green=signal["ns_green"], ew_green=signal["ew_green"],
                   yellow=signal["yellow"], all_red=signal.get("all_red", 0.0))

    # ------------------------------------------------------------------
    @property
    def cycle_time(self) -> float:
        """One full cycle, in seconds."""
        return (self.ns_green + self.yellow + self.all_red
                + self.ew_green + self.yellow + self.all_red)

    # ------------------------------------------------------------------
    def select_action(self, state: TrafficState) -> int:
        """Keep the current phase until its green time is used up, then swap.

        Note what is *not* used: `state.ns_queue`, `state.ew_queue`,
        `state.ns_waiting` and `state.ew_waiting` are all ignored. A fixed-time
        controller cannot react to traffic - that is the whole point of the
        baseline.
        """
        green_time = (self.ns_green if state.current_phase == PHASE_NS
                      else self.ew_green)
        if state.time_since_phase_change >= green_time:
            return ACTION_EW_GREEN if state.current_phase == PHASE_NS else ACTION_NS_GREEN
        return ACTION_NS_GREEN if state.current_phase == PHASE_NS else ACTION_EW_GREEN

    # ------------------------------------------------------------------
    def explain(self, state: TrafficState) -> Dict[str, Any]:
        action = self.select_action(state)
        green_time = (self.ns_green if state.current_phase == PHASE_NS
                      else self.ew_green)
        remaining = max(green_time - state.time_since_phase_change, 0.0)
        return {
            "agent": self.name,
            "action": action,
            "action_name": ACTION_NAMES[action],
            "is_learned": False,
            "reason": ("Green time for this direction is used up, so the "
                       "signal changes." if remaining <= 0 else
                       "{:.0f} s of the {:.0f} s green remain, so the signal "
                       "is held.".format(remaining, green_time)),
            "uses_traffic_state": False,
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
    def phase_at(self, simulation_time: float) -> int:
        """Which direction has green at a given time in the cycle.

        Used by the dashboard to describe the configured plan without needing
        a running simulation.
        """
        position = float(simulation_time) % self.cycle_time
        return PHASE_NS if position < (self.ns_green + self.yellow + self.all_red) else PHASE_EW

    def timeline(self):
        """The configured plan as (label, seconds) segments, for display."""
        segments = [("NS green", self.ns_green), ("Yellow", self.yellow)]
        if self.all_red > 0:
            segments.append(("All red", self.all_red))
        segments += [("EW green", self.ew_green), ("Yellow", self.yellow)]
        if self.all_red > 0:
            segments.append(("All red", self.all_red))
        return segments
