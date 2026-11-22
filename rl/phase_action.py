"""
IntelliSignal
rl/phase_action.py

A second action space: ONE DECISION PER GREEN, choosing that green's length.

Why a second action space exists
--------------------------------
The Stage 2 action space is Discrete(2) - KEEP or SWITCH - evaluated every
`decision_interval` seconds. It is a fine formulation, and it is the one the
deployed moderate-traffic model uses. On heavy traffic it has three problems
that together kept every learned policy behind a plain fixed-time plan:

1. **The optimum is barely expressible.** The best fixed plan measured for
   heavy traffic is north-south 7 s against east-west 6 s. With KEEP/SWITCH
   the green is a whole number of decision intervals, so the two directions
   can only differ by a whole interval; a one-second difference needs a
   one-second interval.

2. **A one-second interval destroys the credit assignment.** It produces
   ~1100 decisions per 1800 s episode, almost all of them KEEP, each with an
   almost identical reward. The signal the agent needs - "this green should
   have been one second shorter" - is spread over a hundred near-identical
   transitions.

3. **The transitions stop being Markov.** Once the clearance is served
   outside the interval, KEEP advances 1 s and SWITCH advances 1 + yellow +
   all-red seconds. A single discount factor and a per-interval mean reward
   charge those two decisions the same, which makes switching look
   systematically cheaper than it is.

Choosing a duration fixes all three. The decision is "how long should this
green be", taken once, with the whole green's consequence attached to it:

    action i  ->  [ (green of the current phase, durations[i] seconds),
                    (yellow of the current phase, yellow_time),
                    (all red, all_red_time) ]            # omitted when 0
                  then the phase changes

7 s north-south against 6 s east-west is then two actions, not a timing
pattern that has to be held across eleven consecutive decisions. An episode
has roughly 170 decisions instead of 1100, and each one owns its outcome.

Safety is unchanged and is still structural
-------------------------------------------
Every green is followed by the Stage 1 yellow clearance before the other
direction is released - the menu cannot express a green-to-green transition.
`min_green` and `max_green` bound the menu itself, so a green shorter than
the minimum or longer than the maximum is not a thing the agent can ask for,
rather than a request that has to be refused. Because every action ends the
phase, no approach can be starved: the other direction is served next, always.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from utils.scenario_builder import (ALL_RED, EW_GREEN, EW_YELLOW,
                                    NS_GREEN, NS_YELLOW)

PHASE_NS = 0
PHASE_EW = 1
NUM_PHASES = 2

GREEN_STATE = {PHASE_NS: NS_GREEN, PHASE_EW: EW_GREEN}
YELLOW_STATE = {PHASE_NS: NS_YELLOW, PHASE_EW: EW_YELLOW}
PHASE_NAMES = {PHASE_NS: "North-South green", PHASE_EW: "East-West green"}

#: The default menu. It is dense where the measured optimum lives (4-12 s)
#: and sparse above it, so the agent can still choose a long green for a
#: surge without spending most of its action space on durations that the
#: measured delay surface rules out.
DEFAULT_DURATIONS: Tuple[int, ...] = (4, 5, 6, 7, 8, 9, 10, 12, 15, 20)


class PhaseDurationController:
    """Turns a duration choice into the signal states SUMO has to display.

    Holds no SUMO connection: it only produces the plan, so the timing logic
    is testable without starting a simulator.
    """

    def __init__(self, durations: Sequence[float] = DEFAULT_DURATIONS,
                 yellow_time: float = 4.0, all_red_time: float = 0.0,
                 min_green: float = 4.0, max_green: float = 60.0,
                 step_length: float = 1.0):
        self.durations = tuple(float(d) for d in durations)
        if not self.durations:
            raise ValueError("durations must not be empty.")
        if len(set(self.durations)) != len(self.durations):
            raise ValueError(
                "durations must not repeat a value: {!r}".format(durations))
        if list(self.durations) != sorted(self.durations):
            raise ValueError(
                "durations must be given in increasing order: {!r}".format(
                    durations))

        self.yellow_time = float(yellow_time)
        self.all_red_time = float(all_red_time)
        self.min_green = float(min_green)
        self.max_green = float(max_green)
        self.step_length = float(step_length)

        if self.yellow_time < 0 or self.all_red_time < 0:
            raise ValueError("the clearance times must not be negative.")
        if self.durations[0] < self.min_green:
            raise ValueError(
                "the shortest duration ({:g} s) is below min_green ({:g} s). "
                "The menu must not contain a green the safety rules forbid."
                .format(self.durations[0], self.min_green))
        if self.durations[-1] > self.max_green:
            raise ValueError(
                "the longest duration ({:g} s) is above max_green ({:g} s)."
                .format(self.durations[-1], self.max_green))

        self.phase = PHASE_NS       # the intersection starts North-South green
        self.green_elapsed = 0.0    # length of the green currently being served
        self.switch_count = 0
        self.last_green = {PHASE_NS: 0.0, PHASE_EW: 0.0}

    # ------------------------------------------------------------------
    @property
    def num_actions(self) -> int:
        return len(self.durations)

    # ------------------------------------------------------------------
    def reset(self, phase: int = PHASE_NS) -> None:
        self.phase = int(phase)
        self.green_elapsed = 0.0
        self.switch_count = 0
        self.last_green = {PHASE_NS: 0.0, PHASE_EW: 0.0}

    # ------------------------------------------------------------------
    def duration_for(self, action: int) -> float:
        """The green duration an action index asks for, in seconds."""
        action = int(action)
        if not 0 <= action < len(self.durations):
            raise ValueError(
                "Invalid action {}. Valid actions are 0..{} (durations {})."
                .format(action, len(self.durations) - 1,
                        ", ".join("{:g}".format(d) for d in self.durations)))
        return self.durations[action]

    # ------------------------------------------------------------------
    def plan(self, action: int) -> List[Tuple[str, int]]:
        """The (signal state, number of simulation steps) segments to run.

        Green first, then the mandatory clearance. The clearance is part of
        the decision, which is what makes the decision's cost honest: a short
        green pays the same four seconds of clearance as a long one, so the
        agent sees the real price of changing the signal often.
        """
        green_steps = self._steps(self.duration_for(action))
        yellow_steps = self._steps(self.yellow_time)
        all_red_steps = self._steps(self.all_red_time)

        segments = [(GREEN_STATE[self.phase], green_steps),
                    (YELLOW_STATE[self.phase], yellow_steps),
                    (ALL_RED, all_red_steps)]
        return [(state, steps) for state, steps in segments if steps > 0]

    # ------------------------------------------------------------------
    def decision_seconds(self, action: int) -> float:
        """How much simulated time one decision covers, clearance included.

        This is the transition's duration, and it is what the semi-Markov
        discount and the integrated reward are both computed from.
        """
        return (self.duration_for(action) + self.yellow_time
                + self.all_red_time)

    # ------------------------------------------------------------------
    def commit(self, action: int) -> None:
        """Update the controller's state after the plan has been run."""
        served = self.duration_for(action)
        self.last_green[self.phase] = served
        self.green_elapsed = served
        self.phase = self.other_phase()
        self.switch_count += 1

    # ------------------------------------------------------------------
    def other_phase(self) -> int:
        return PHASE_EW if self.phase == PHASE_NS else PHASE_NS

    def current_state(self) -> str:
        return GREEN_STATE[self.phase]

    def phase_name(self) -> str:
        return PHASE_NAMES[self.phase]

    def _steps(self, seconds: float) -> int:
        return int(round(float(seconds) / self.step_length))

    def describe(self) -> str:
        return ("duration menu {} s, clearance {:g} s, "
                "cycle {:g}-{:g} s".format(
                    "/".join("{:g}".format(d) for d in self.durations),
                    self.yellow_time + self.all_red_time,
                    2 * (self.durations[0] + self.yellow_time + self.all_red_time),
                    2 * (self.durations[-1] + self.yellow_time + self.all_red_time)))

    def __repr__(self) -> str:
        return "<PhaseDurationController {}>".format(self.describe())
