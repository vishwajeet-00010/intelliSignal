"""Tests for evaluation/gates.py - the switch-saturation gate.

These tests pin the behaviour that the original gate got wrong: dividing by
the number of decisions instead of by the number of switches the safety layer
actually allows.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.gates import (  # noqa: E402
    SWITCH_SATURATION_THRESHOLD, decisions_blocked_after_switch,
    max_possible_switches, switch_saturation, switch_saturation_for_rows,
    switch_saturation_gate, timings_from_row)


# ---------------------------------------------------------------------------
# The ceiling itself
# ---------------------------------------------------------------------------
def test_one_decision_is_blocked_with_project_settings():
    # 5 s cadence, 4 s yellow: a switch leaves 1 s served, below min_green 5 s.
    assert decisions_blocked_after_switch(5, 4, 0, 5) == 1


def test_stage3_settings_also_block_one_decision():
    # The historical 10 s cadence left 6 s served, below min_green 10 s.
    assert decisions_blocked_after_switch(10, 4, 0, 10) == 1


def test_a_long_minimum_green_blocks_several_decisions():
    # 5 s cadence, 4 s yellow, 20 s minimum: 1, 6, 11, 16, 21 -> 4 refusals.
    assert decisions_blocked_after_switch(5, 4, 0, 20) == 4


def test_no_decision_is_blocked_when_the_interval_already_serves_min_green():
    # A 10 s interval with 1 s clearance serves 9 s, which meets a 5 s minimum.
    assert decisions_blocked_after_switch(10, 1, 0, 5) == 0


def test_ceiling_is_half_the_decisions_under_project_settings():
    assert max_possible_switches(360) == 180.0
    assert max_possible_switches(180, 10, 4, 0, 10) == 90.0


def test_ceiling_rejects_a_non_positive_interval():
    with pytest.raises(ValueError):
        max_possible_switches(360, decision_interval=0)


def test_ceiling_rejects_negative_decisions():
    with pytest.raises(ValueError):
        max_possible_switches(-1)


# ---------------------------------------------------------------------------
# The ratio
# ---------------------------------------------------------------------------
def test_switching_at_every_opportunity_saturates_at_one():
    assert switch_saturation(180, 360) == pytest.approx(1.0)


def test_the_old_denominator_cannot_reach_the_threshold():
    """Regression guard for the defect this module replaces.

    switch_count / decisions is bounded above by 0.5 whenever a decision is
    blocked after each switch, so a 0.90 threshold on that quantity can never
    fail.  The corrected ratio reaches 1.0 for the same behaviour.
    """
    switches, decisions = 180, 360
    broken_ratio = switches / decisions
    assert broken_ratio < SWITCH_SATURATION_THRESHOLD      # always passed
    assert switch_saturation(switches, decisions) >= SWITCH_SATURATION_THRESHOLD


@pytest.mark.parametrize("switches,expected", [(178.8, 0.993), (171.0, 0.950)])
def test_stage4_incumbents_are_measured_as_saturated(switches, expected):
    """The Stage 4 incumbents sit on the switching ceiling."""
    value = switch_saturation(switches, 360)
    assert value == pytest.approx(expected, abs=5e-4)
    assert value >= SWITCH_SATURATION_THRESHOLD


def test_an_unsaturated_policy_scores_below_the_threshold():
    # 120 switches over 360 decisions is an 11 s green: two thirds of the ceiling.
    assert switch_saturation(120, 360) == pytest.approx(2.0 / 3.0)


# ---------------------------------------------------------------------------
# Row handling
# ---------------------------------------------------------------------------
def test_a_row_overrides_the_fallback_timings():
    row = {"decision_interval": 10, "yellow_time": 4, "min_green": 10}
    timings = timings_from_row(row, fallback={"decision_interval": 5})
    assert timings["decision_interval"] == 10.0
    assert timings["min_green"] == 10.0


def test_blank_row_values_fall_back_rather_than_crashing():
    timings = timings_from_row({"decision_interval": ""}, fallback=None)
    assert timings["decision_interval"] == 5.0


def test_rows_without_timings_use_the_supplied_fallback():
    rows = [{"switch_count": 90, "decisions": 180}]
    value = switch_saturation_for_rows(
        rows, fallback={"decision_interval": 10, "min_green": 10})
    assert value == pytest.approx(1.0)


def test_empty_rows_score_zero_rather_than_dividing_by_zero():
    assert switch_saturation_for_rows([]) == 0.0


# ---------------------------------------------------------------------------
# The gate row
# ---------------------------------------------------------------------------
def test_gate_fails_a_saturated_policy():
    rows = [{"switch_count": 171.0, "decisions": 360}]
    gate = switch_saturation_gate(rows)
    assert gate["status"] == "FAIL"
    assert gate["value"] == pytest.approx(0.95)
    assert gate["runs"] == 1
    assert "maximum possible" in gate["label"]


def test_gate_passes_a_genuinely_adaptive_policy():
    rows = [{"switch_count": 100, "decisions": 360}]
    assert switch_saturation_gate(rows)["status"] == "PASS"
