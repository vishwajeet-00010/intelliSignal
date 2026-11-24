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
                 step_length: float = 1.0, profile=None):
        # A profile supplies the menu and the limits together, so a
        # controller cannot be built with a menu its own limits forbid.
        self.profile = profile
        if profile is not None:
            durations = profile.durations
            min_green = profile.min_green
            max_green = profile.max_green
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

        #: Controller-local clock, in simulated seconds. The controller has
        #: no SUMO connection, so it counts the time its own plans occupy.
        self.clock = 0.0
        #: Completed red spells, per group, for the compliance report.
        self.red_history = {PHASE_NS: [], PHASE_EW: []}
        #: One entry per decision: what was asked for, what was executed,
        #: when the phase ran, and whether anything intervened.
        self.decision_log = []

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
        self.clock = 0.0
        self.red_history = {PHASE_NS: [], PHASE_EW: []}
        self.decision_log = []

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
    @property
    def clearance(self) -> float:
        return self.yellow_time + self.all_red_time

    def max_red_for(self, group: int) -> float:
        """The longest continuous red this group may be held for.

        It is `max_red_factor` times that group's own preceding completed
        green. Before the group has been green, there is no preceding
        green to multiply, so the profile's minimum green stands in - the
        strictest reading, and one that cannot be gamed by opening the
        episode on a very short green.
        """
        if self.profile is None:
            return float("inf")
        previous = self.last_green.get(group, 0.0)
        if previous <= 0.0:
            return self.profile.initial_max_red_seconds()
        return self.profile.max_red_seconds(previous)

    def resolve(self, action: int) -> Tuple[int, dict]:
        """Decide what will actually run, before the simulation is touched.

        The menu already keeps every green inside `min_green`/`max_green`,
        so the only rule that can intervene here is the red limit. On this
        two-group junction the waiting group's red will be exactly this
        green plus the clearance, so a green is shortened when

            green + clearance  >  max_red_factor x the waiting group's
                                  own preceding green

        If even the shortest green on the menu would breach the limit, the
        shortest green runs and the breach is recorded rather than hidden:
        the alternative would be a green below the minimum, which is the
        one thing the profile is there to prevent.
        """
        action = int(action)
        requested = self.duration_for(action)
        info = {
            "requested_action": action,
            "requested_green": requested,
            "executed_action": action,
            "executed_green": requested,
            "intervention": "",
            "red_limit": self.max_red_for(self.other_phase()),
        }
        limit = info["red_limit"]
        if limit == float("inf"):
            return action, info

        allowed_green = limit - self.clearance
        if requested <= allowed_green + 1e-9:
            return action, info

        feasible = [index for index, value in enumerate(self.durations)
                    if value <= allowed_green + 1e-9]
        if feasible:
            chosen = max(feasible)
            info.update({
                "executed_action": chosen,
                "executed_green": self.durations[chosen],
                "intervention": (
                    "green shortened from {:g} s to {:g} s: holding the "
                    "other group would have exceeded its {:g} s red limit"
                    .format(requested, self.durations[chosen], limit)),
            })
            return chosen, info

        info["intervention"] = (
            "red limit of {:g} s cannot be met without a green below the "
            "{:g} s minimum; the minimum green ran and the limit was "
            "exceeded".format(limit, self.min_green))
        return 0, dict(info, executed_action=0,
                       executed_green=self.durations[0])

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
    def commit(self, action: int, served: float = None,
               truncated: bool = False, info: dict = None) -> None:
        """Update the controller's state after the plan has been run.

        `served` is the green that actually ran, which is shorter than the
        green that was asked for when the episode ended mid-phase. That
        case is recorded as a truncation, never as a completed short green.
        """
        requested = self.duration_for(action)
        served = requested if served is None else float(served)
        waiting = self.other_phase()

        # The waiting group's red ends now and its length was exactly this
        # green plus the clearance.
        red_seconds = served + self.clearance
        limit = self.max_red_for(waiting)
        self.red_history[waiting].append({
            "started_at": self.clock - self.clearance,
            "seconds": red_seconds,
            "limit": limit,
            "within_limit": bool(red_seconds <= limit + 1e-9),
        })

        entry = {
            "decision": len(self.decision_log),
            "phase": self.phase,
            "phase_name": PHASE_NAMES[self.phase],
            "started_at": self.clock,
            "ended_at": self.clock + served + self.clearance,
            "requested_green": requested,
            "served_green": served,
            "clearance": self.clearance,
            "other_group_red": red_seconds,
            "red_limit": limit,
            "red_within_limit": bool(red_seconds <= limit + 1e-9),
            "truncated_by_episode_end": bool(truncated),
        }
        entry.update(info or {})
        self.decision_log.append(entry)

        self.clock += served + self.clearance
        self.last_green[self.phase] = served
        self.green_elapsed = served
        self.phase = waiting
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

    # ------------------------------------------------------------------
    def compliance(self) -> dict:
        """Did the run obey its own timing rules? Measured, not assumed."""
        completed = [e for e in self.decision_log
                     if not e["truncated_by_episode_end"]]
        greens = [e["served_green"] for e in completed]
        red_breaches = [e for e in completed if not e["red_within_limit"]]
        below_min = [g for g in greens if g < self.min_green - 1e-9]
        above_max = [g for g in greens if g > self.max_green + 1e-9]
        return {
            "profile": self.profile.name if self.profile else "none",
            "decisions": len(self.decision_log),
            "completed_phases": len(completed),
            "truncated_phases": len(self.decision_log) - len(completed),
            "min_green_seen": min(greens) if greens else None,
            "max_green_seen": max(greens) if greens else None,
            "greens_below_min": len(below_min),
            "greens_above_max": len(above_max),
            "red_limit_breaches": len(red_breaches),
            "interventions": sum(1 for e in self.decision_log
                                 if e.get("intervention")),
            "compliant": (not below_min and not above_max
                          and not red_breaches),
        }

    def describe(self) -> str:
        return ("duration menu {} s, clearance {:g} s, "
                "cycle {:g}-{:g} s".format(
                    "/".join("{:g}".format(d) for d in self.durations),
                    self.yellow_time + self.all_red_time,
                    2 * (self.durations[0] + self.yellow_time + self.all_red_time),
                    2 * (self.durations[-1] + self.yellow_time + self.all_red_time)))

    def __repr__(self) -> str:
        return "<PhaseDurationController {}>".format(self.describe())
