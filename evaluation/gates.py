"""
IntelliSignal
evaluation/gates.py

Shared, testable definitions for the quantitative gates used in experiment
reports.

Why this module exists
----------------------
Experiment reports repeatedly need to answer "is the learned policy actually
state-dependent, or has it collapsed into a fixed rhythm?".  The natural
summary is how often the controller switches phase, expressed as a fraction of
how often it *could* switch.

`output/dqn_retraining_20261120/build_report.py` computed that fraction as

    switch_count / decisions

and passed the gate when the value was below 0.90.  That test can never fail.
A switch leaves the new green with

    green_elapsed = decision_interval - yellow_time - all_red_time

seconds already served (see `rl/action.py::ActionController.commit`).  With the
project's settings that is 5 - 4 - 0 = 1 s, which is below `min_green`, so the
very next decision is refused by `ActionController.resolve`.  A switch is
therefore possible on at most every second decision and the ratio is bounded
above by 0.5.  The gate reported PASS for policies that were in fact switching
on 95-99% of the opportunities available to them.

The correct denominator is the number of switches the safety layer would allow
a maximally impatient controller to make.  `max_possible_switches` computes it
from the timing settings rather than assuming a constant, so the figure stays
correct if the cadence changes.

Nothing here reads a file or runs a simulation, which keeps it unit-testable.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

# The project default, used only when a row does not carry its own timings.
DEFAULT_DECISION_INTERVAL = 5.0
DEFAULT_YELLOW_TIME = 4.0
DEFAULT_ALL_RED_TIME = 0.0
DEFAULT_MIN_GREEN = 5.0

#: A policy switching on at least this fraction of its opportunities is
#: treated as saturated: it is following a fixed rhythm, not the traffic.
SWITCH_SATURATION_THRESHOLD = 0.90


# ---------------------------------------------------------------------------
def decisions_blocked_after_switch(decision_interval: float,
                                   yellow_time: float,
                                   all_red_time: float,
                                   min_green: float) -> int:
    """How many decisions are refused immediately after a switch.

    Mirrors `rl/action.py`: a switch sets
    `green_elapsed = decision_interval - yellow_time - all_red_time`, and each
    subsequent KEEP adds another `decision_interval`.  Count how many decisions
    pass before `green_elapsed` reaches `min_green`.
    """
    interval = float(decision_interval)
    if interval <= 0:
        raise ValueError("decision_interval must be positive")
    elapsed = interval - float(yellow_time) - float(all_red_time)
    blocked = 0
    # A real controller cannot be blocked forever: max_green forces a switch.
    # The loop is bounded defensively so a misconfiguration cannot hang a report.
    while elapsed < float(min_green) and blocked < 1000:
        elapsed += interval
        blocked += 1
    return blocked


# ---------------------------------------------------------------------------
def max_possible_switches(decisions: float,
                          decision_interval: float = DEFAULT_DECISION_INTERVAL,
                          yellow_time: float = DEFAULT_YELLOW_TIME,
                          all_red_time: float = DEFAULT_ALL_RED_TIME,
                          min_green: float = DEFAULT_MIN_GREEN) -> float:
    """The most switches the safety layer permits over `decisions` decisions.

    One switch consumes its own decision plus every decision refused while the
    new green serves its minimum, so the ceiling is

        decisions / (1 + decisions_blocked_after_switch)

    With the project defaults (5 s cadence, 4 s yellow, 5 s minimum green) one
    decision is blocked after each switch, giving `decisions / 2`.
    """
    count = float(decisions)
    if count < 0:
        raise ValueError("decisions must not be negative")
    blocked = decisions_blocked_after_switch(
        decision_interval, yellow_time, all_red_time, min_green)
    return count / (1.0 + blocked)


# ---------------------------------------------------------------------------
def switch_saturation(switch_count: float, decisions: float,
                      decision_interval: float = DEFAULT_DECISION_INTERVAL,
                      yellow_time: float = DEFAULT_YELLOW_TIME,
                      all_red_time: float = DEFAULT_ALL_RED_TIME,
                      min_green: float = DEFAULT_MIN_GREEN) -> float:
    """Switches as a fraction of the switches that were actually available.

    1.0 means the controller switched at every opportunity, i.e. it is running
    the shortest cycle the safety rules allow and is not reacting to traffic.
    """
    ceiling = max_possible_switches(decisions, decision_interval, yellow_time,
                                    all_red_time, min_green)
    if ceiling <= 0:
        return 0.0
    return float(switch_count) / ceiling


# ---------------------------------------------------------------------------
def timings_from_row(row: Dict[str, Any],
                     fallback: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Read the timing settings a result row carries, falling back to defaults.

    Result CSVs written before this module existed do not record the cadence.
    Passing the experiment's known settings as `fallback` keeps those rows
    usable without pretending the row contained the values.
    """
    defaults = {
        "decision_interval": DEFAULT_DECISION_INTERVAL,
        "yellow_time": DEFAULT_YELLOW_TIME,
        "all_red_time": DEFAULT_ALL_RED_TIME,
        "min_green": DEFAULT_MIN_GREEN,
    }
    if fallback:
        defaults.update({key: float(value) for key, value in fallback.items()
                         if key in defaults})
    for key in list(defaults):
        if row.get(key) not in (None, ""):
            defaults[key] = float(row[key])
    return defaults


# ---------------------------------------------------------------------------
def switch_saturation_for_rows(rows: Iterable[Dict[str, Any]],
                               fallback: Optional[Dict[str, float]] = None) -> float:
    """Mean switch saturation over result rows, each using its own timings."""
    values = []
    for row in rows:
        timings = timings_from_row(row, fallback)
        values.append(switch_saturation(
            float(row["switch_count"]), float(row["decisions"]), **timings))
    if not values:
        return 0.0
    return sum(values) / len(values)


# ---------------------------------------------------------------------------
def switch_saturation_gate(rows: Iterable[Dict[str, Any]],
                           threshold: float = SWITCH_SATURATION_THRESHOLD,
                           fallback: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """The gate row a report should print.

    Returns the measured value, the threshold, and PASS/FAIL.  A saturated
    policy FAILS: it has collapsed onto the minimum cycle and more training
    cannot help it, because the action space is the binding constraint.
    """
    rows = list(rows)
    value = switch_saturation_for_rows(rows, fallback)
    return {
        "label": "Switches / maximum possible switches below {:.2f}".format(threshold),
        "value": value,
        "threshold": float(threshold),
        "status": "PASS" if value < float(threshold) else "FAIL",
        "runs": len(rows),
    }
