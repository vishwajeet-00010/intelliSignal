"""
IntelliSignal - Stage 2
rl/action.py

Defines WHAT THE AGENT CAN DO, and turns an action into real signal states.

Action space: Discrete(2)

    0 = KEEP    keep the direction that currently has green
    1 = SWITCH  end the current green and give green to the other direction

There are only two green phases at this intersection:

    phase 0 : North-South green   (state GGGgrrrrGGGgrrrr)
    phase 1 : East-West   green   (state rrrrGGGgrrrrGGGg)

Exactly what each action does to the traffic light
--------------------------------------------------
Every decision covers `decision_interval` seconds of simulation. The
controller turns the action into a list of (signal state, seconds) segments
that together fill that interval exactly.

  action = 0 (KEEP)
        the current green state is held for the whole interval
        -> [(current green, decision_interval)]

  action = 1 (SWITCH), when allowed
        the mandatory clearance is served first, then the other direction
        gets the rest of the interval
        -> [(current yellow, yellow_time),
            (all red,        all_red_time),      # omitted when all_red = 0
            (other green,    decision_interval - yellow_time - all_red_time)]

Safety rules - a switch is never instantaneous and never rapid
--------------------------------------------------------------
  * A switch always passes through the yellow (and, if configured, all-red)
    state taken from the Stage 1 signal design. The green of one direction
    can never turn directly into the green of the other.
  * min_green: a switch requested before the current green has run for
    min_green seconds is refused and treated as KEEP. The refusal is
    reported in the info dictionary, it is not hidden.
  * max_green: a KEEP requested after the current green has already run for
    max_green seconds is overridden into a SWITCH. This is the fairness
    guarantee - no approach can be held at red forever.
"""

from __future__ import annotations

from typing import List, Tuple

from utils.scenario_builder import (ALL_RED, EW_GREEN, EW_YELLOW,
                                    NS_GREEN, NS_YELLOW)

ACTION_KEEP = 0
ACTION_SWITCH = 1
ACTION_NAMES = {ACTION_KEEP: "KEEP", ACTION_SWITCH: "SWITCH"}

NUM_ACTIONS = 2

# Phase index -> signal state string (16 characters, one per controlled link)
PHASE_NS = 0
PHASE_EW = 1
GREEN_STATE = {PHASE_NS: NS_GREEN, PHASE_EW: EW_GREEN}
YELLOW_STATE = {PHASE_NS: NS_YELLOW, PHASE_EW: EW_YELLOW}
PHASE_NAMES = {PHASE_NS: "North-South green", PHASE_EW: "East-West green"}

NUM_PHASES = 2


class ActionController:
    """Translates an action into the signal states SUMO has to display.

    The controller holds no SUMO connection; it only produces the plan. The
    environment executes that plan, which keeps the timing logic testable
    without starting a simulator.
    """

    def __init__(self, decision_interval: float, yellow_time: float,
                 all_red_time: float, min_green: float, max_green: float,
                 step_length: float = 1.0):
        self.decision_interval = float(decision_interval)
        self.yellow_time = float(yellow_time)
        self.all_red_time = float(all_red_time)
        self.min_green = float(min_green)
        self.max_green = float(max_green)
        self.step_length = float(step_length)

        transition = self.yellow_time + self.all_red_time
        if transition >= self.decision_interval:
            raise ValueError(
                "decision_interval ({:.0f} s) must be longer than "
                "yellow + all-red ({:.0f} s).".format(
                    self.decision_interval, transition))

        self.phase = PHASE_NS        # the intersection starts North-South green
        self.green_elapsed = 0.0     # seconds the current green has been on
        self.switch_count = 0

    # ------------------------------------------------------------------
    def reset(self, phase: int = PHASE_NS) -> None:
        self.phase = phase
        self.green_elapsed = 0.0
        self.switch_count = 0

    # ------------------------------------------------------------------
    def resolve(self, action: int) -> Tuple[int, dict]:
        """Decide what will actually happen, before touching the simulation.

        Returns the effective action and a small dictionary explaining why it
        differs from the requested one, if it does.
        """
        info = {
            "requested_action": int(action),
            "switch_blocked": False,
            "forced_switch": False,
        }

        if action == ACTION_SWITCH:
            if self.green_elapsed < self.min_green:
                info["switch_blocked"] = True
                return ACTION_KEEP, info
            return ACTION_SWITCH, info

        # action == KEEP
        if self.green_elapsed >= self.max_green:
            info["forced_switch"] = True
            return ACTION_SWITCH, info
        return ACTION_KEEP, info

    # ------------------------------------------------------------------
    def plan(self, effective_action: int) -> List[Tuple[str, int]]:
        """Build the (signal state, number of simulation steps) segments that
        fill exactly one decision interval."""
        total_steps = self._steps(self.decision_interval)

        if effective_action == ACTION_KEEP:
            return [(GREEN_STATE[self.phase], total_steps)]

        yellow_steps = self._steps(self.yellow_time)
        all_red_steps = self._steps(self.all_red_time)
        green_steps = total_steps - yellow_steps - all_red_steps

        other = self.other_phase()
        segments = [(YELLOW_STATE[self.phase], yellow_steps)]
        if all_red_steps > 0:
            segments.append((ALL_RED, all_red_steps))
        segments.append((GREEN_STATE[other], green_steps))
        return [(state, steps) for state, steps in segments if steps > 0]

    # ------------------------------------------------------------------
    def commit(self, effective_action: int) -> None:
        """Update the controller's own state after the plan has been run."""
        if effective_action == ACTION_SWITCH:
            self.phase = self.other_phase()
            self.switch_count += 1
            # Only the green part of the interval counted as green time.
            self.green_elapsed = (self.decision_interval
                                  - self.yellow_time - self.all_red_time)
        else:
            self.green_elapsed += self.decision_interval

    # ------------------------------------------------------------------
    def other_phase(self) -> int:
        return PHASE_EW if self.phase == PHASE_NS else PHASE_NS

    def current_state(self) -> str:
        return GREEN_STATE[self.phase]

    def phase_name(self) -> str:
        return PHASE_NAMES[self.phase]

    def _steps(self, seconds: float) -> int:
        return int(round(float(seconds) / self.step_length))
