"""Tests for serving the yellow/all-red clearance outside the decision interval.

Why the mode exists
-------------------
With the clearance taken out of the interval, a switch leaves
`decision_interval - yellow - all_red` seconds of green already served, so the
reachable green durations are

    {interval - clearance, 2*interval - clearance, 3*interval - clearance, ...}

and the two directions can only ever differ by a whole interval. Measured on
heavy traffic over confirmatory seeds 6101-6120, the best fixed plan gives
north-south 7 s and east-west 6 s at 5.754 s mean waiting, against 12.638 s for
the best symmetric plan. A one-second difference between the directions is
unreachable unless the interval is one second, which the old rule forbids
because the interval had to exceed the 4 s clearance.

Serving the clearance outside the interval makes greens whole multiples of the
interval, so a one-second interval becomes legal and 7/6 becomes expressible.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.action import (ACTION_KEEP, ACTION_SWITCH,  # noqa: E402
                       ActionController)

YELLOW = 4.0
ALL_RED = 0.0


def controller(interval, min_green, outside, max_green=60.0):
    return ActionController(interval, YELLOW, ALL_RED, min_green, max_green,
                            clearance_outside_interval=outside)


def green_ladder(ctrl, count=8):
    """The green durations reachable by keeping for 0, 1, 2 ... intervals."""
    ctrl.commit(ACTION_SWITCH)
    ladder = []
    for _ in range(count):
        ladder.append(ctrl.green_elapsed)
        ctrl.commit(ACTION_KEEP)
    return ladder


# ---------------------------------------------------------------------------
# The default is unchanged
# ---------------------------------------------------------------------------
def test_default_keeps_the_clearance_inside_the_interval():
    ctrl = controller(5, 5, outside=False)
    assert ctrl.clearance_outside_interval is False
    # A switch serves 5 - 4 = 1 s of green, so greens step 1, 6, 11, ...
    assert green_ladder(ctrl, 4) == [1.0, 6.0, 11.0, 16.0]


def test_default_still_refuses_an_interval_shorter_than_the_clearance():
    with pytest.raises(ValueError, match="must be longer than"):
        controller(3, 5, outside=False)


def test_default_switch_plan_fills_exactly_one_interval():
    ctrl = controller(5, 5, outside=False)
    plan = ctrl.plan(ACTION_SWITCH)
    assert sum(steps for _, steps in plan) == 5


# ---------------------------------------------------------------------------
# The new mode
# ---------------------------------------------------------------------------
def test_clearance_outside_allows_an_interval_shorter_than_the_clearance():
    ctrl = controller(1, 5, outside=True)
    assert ctrl.clearance_outside_interval is True


def test_clearance_outside_still_rejects_a_non_positive_interval():
    with pytest.raises(ValueError, match="must be positive"):
        controller(0, 5, outside=True)


def test_greens_become_whole_multiples_of_the_interval():
    ctrl = controller(1, 5, outside=True)
    assert green_ladder(ctrl, 8) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]


def test_a_switch_costs_the_interval_plus_the_clearance():
    ctrl = controller(1, 5, outside=True)
    plan = ctrl.plan(ACTION_SWITCH)
    assert sum(steps for _, steps in plan) == 1 + int(YELLOW)
    # ... and a keep still costs exactly one interval.
    assert sum(steps for _, steps in ctrl.plan(ACTION_KEEP)) == 1


def test_the_switch_plan_serves_yellow_before_the_new_green():
    ctrl = controller(1, 5, outside=True)
    states = [state for state, _ in ctrl.plan(ACTION_SWITCH)]
    assert "y" in states[0], "the clearance must come first"
    assert "y" not in states[-1], "the new green must come last"


def test_min_green_is_honoured_at_one_second_resolution():
    """The point of the mode: 6 s and 7 s greens are both reachable."""
    ctrl = controller(1, 5, outside=True)
    ctrl.commit(ACTION_SWITCH)                 # green_elapsed = 1
    for _ in range(4):
        assert ctrl.resolve(ACTION_SWITCH)[0] == ACTION_KEEP, "blocked below min_green"
        ctrl.commit(ACTION_KEEP)
    assert ctrl.green_elapsed == 5.0
    assert ctrl.resolve(ACTION_SWITCH)[0] == ACTION_SWITCH, "allowed at min_green"
    ctrl.commit(ACTION_KEEP)
    assert ctrl.green_elapsed == 6.0
    assert ctrl.resolve(ACTION_SWITCH)[0] == ACTION_SWITCH
    ctrl.commit(ACTION_KEEP)
    assert ctrl.green_elapsed == 7.0, "a 7 s green is reachable, unlike before"


def test_max_green_still_forces_a_switch():
    ctrl = controller(1, 5, outside=True, max_green=10.0)
    ctrl.commit(ACTION_SWITCH)
    for _ in range(9):
        ctrl.commit(ACTION_KEEP)
    assert ctrl.green_elapsed == 10.0
    effective, info = ctrl.resolve(ACTION_KEEP)
    assert effective == ACTION_SWITCH and info["forced_switch"]


def test_the_two_directions_can_differ_by_one_second():
    """7 s north-south against 6 s east-west is the measured best heavy plan."""
    ctrl = controller(1, 5, outside=True)
    completed = []
    for target in (7, 6, 7, 6):
        ctrl.commit(ACTION_SWITCH)             # green_elapsed = 1
        while ctrl.green_elapsed < target:
            ctrl.commit(ACTION_KEEP)
        completed.append(ctrl.green_elapsed)
    assert completed == [7.0, 6.0, 7.0, 6.0]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def test_config_defaults_to_the_old_behaviour():
    from rl.config import DEFAULTS
    assert DEFAULTS["environment"]["clearance_outside_interval"] is False


def test_config_allows_a_short_interval_only_in_the_new_mode():
    from rl.config import RLConfigError, load_rl_config

    scenario = {
        "signal": {"yellow": 4, "all_red": 0},
        "simulation": {"duration": 1800, "step_length": 1.0},
        "rl": {"environment": {"decision_interval": 1, "min_green": 5,
                               "clearance_outside_interval": True}},
    }
    config = load_rl_config(scenario)
    assert config["environment"]["decision_interval"] == 1

    scenario["rl"]["environment"]["clearance_outside_interval"] = False
    with pytest.raises(RLConfigError, match="must be longer than"):
        load_rl_config(scenario)
