"""Tests for the operational timing profiles and their enforcement.

What these protect:

  * a profile cannot be built that contradicts itself
  * the red limit is applied where it binds, and the arithmetic that says
    it cannot bind on this junction is stated rather than assumed
  * a green below the minimum or above the maximum is not expressible
  * a phase cut off by the end of the episode is recorded as truncated,
    never as a completed short green
  * the observation no longer pins at 1.0, which is what made the model
    emit the shortest green on its menu regardless of the traffic
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.phase_action import (DEFAULT_DURATIONS, PHASE_EW,  # noqa: E402
                             PHASE_NS, PhaseDurationController)
from rl.phase_extend_env import END, EXTEND, extension_ladder  # noqa: E402
from rl.phase_state import (OBSERVATION_LABELS,  # noqa: E402
                            PhaseStateExtractor, SATURATION_CLIP,
                            SATURATION_SOFT, extractor_from_contract)
from rl.timing_profile import (PROFILES, RESPONSIVE, STANDARD,  # noqa: E402
                               TimingProfile, compatible, get_profile,
                               profile_for_durations)


# ===========================================================================
# The profiles themselves
# ===========================================================================
def test_both_profiles_are_self_consistent():
    for name, profile in PROFILES.items():
        assert profile.durations[0] >= profile.min_green, name
        assert profile.durations[-1] <= profile.max_green, name
        assert list(profile.durations) == sorted(profile.durations), name


def test_the_standard_profile_is_the_one_that_was_asked_for():
    profile = get_profile(STANDARD)
    assert profile.min_green == 15.0
    assert profile.max_green == 30.0
    assert profile.max_red_factor == 4.0
    assert min(profile.durations) == 15.0
    assert max(profile.durations) == 30.0


def test_the_responsive_profile_is_what_the_old_models_were_trained_with():
    profile = get_profile(RESPONSIVE)
    assert tuple(profile.durations) == tuple(float(d) for d in DEFAULT_DURATIONS)


@pytest.mark.parametrize("kwargs", [
    {"min_green": 0},
    {"max_green": 3},                      # below min_green
    {"durations": (3, 10)},                # below min_green
    {"durations": (10, 99)},               # above max_green
    {"durations": (10, 8)},                # not sorted
    {"durations": (8, 8)},                 # repeated
    {"max_red_factor": 0},
])
def test_an_impossible_profile_is_refused(kwargs):
    base = dict(name="x", min_green=5.0, max_green=20.0,
                durations=(5, 10, 20), max_red_factor=4.0)
    base.update(kwargs)
    with pytest.raises(ValueError):
        TimingProfile(**base)


def test_an_unknown_profile_name_says_what_is_available():
    with pytest.raises(ValueError, match="Unknown timing profile"):
        get_profile("nonsense")


def test_a_checkpoint_menu_can_be_matched_back_to_its_profile():
    assert profile_for_durations(get_profile(STANDARD).durations) == STANDARD
    assert profile_for_durations(DEFAULT_DURATIONS) == RESPONSIVE
    assert profile_for_durations((1, 2, 3)) is None


def test_an_old_short_green_model_is_not_compatible_with_the_new_profile():
    """A 4 s-green policy under a 15 s minimum is invalid, not conservative."""
    standard = get_profile(STANDARD)
    assert not compatible(standard, DEFAULT_DURATIONS)
    assert compatible(standard, standard.durations)


# ===========================================================================
# The red limit
# ===========================================================================
def test_the_red_limit_can_never_bind_under_the_standard_profile():
    """Stated with its arithmetic, because a rule that never fires must
    not be presented as an improvement."""
    analysis = get_profile(STANDARD).red_limit_analysis(yellow=4.0)
    # A group's red is exactly the other group's green plus the clearance.
    assert analysis["worst_case_red_seconds"] == 34.0      # 30 + 4
    assert analysis["tightest_limit_seconds"] == 60.0      # 4 x 15
    assert analysis["can_bind"] is False
    assert "can never bind" in analysis["explanation"]


def test_the_red_limit_can_bind_under_the_responsive_profile():
    analysis = get_profile(RESPONSIVE).red_limit_analysis(yellow=4.0)
    assert analysis["worst_case_red_seconds"] == 24.0      # 20 + 4
    assert analysis["tightest_limit_seconds"] == 16.0      # 4 x 4
    assert analysis["can_bind"] is True


def test_a_green_is_shortened_when_it_would_breach_the_red_limit():
    controller = PhaseDurationController(profile=get_profile(RESPONSIVE),
                                         yellow_time=4.0)
    controller.commit(0)                      # north-south gets 4 s
    assert controller.max_red_for(PHASE_NS) == 16.0

    executed, info = controller.resolve(len(controller.durations) - 1)  # 20 s
    assert info["requested_green"] == 20.0
    assert info["executed_green"] == 12.0     # 12 + 4 clearance = 16 s
    assert "shortened" in info["intervention"]


def test_nothing_is_shortened_under_the_standard_profile():
    controller = PhaseDurationController(profile=get_profile(STANDARD),
                                         yellow_time=4.0)
    controller.commit(0)
    executed, info = controller.resolve(len(controller.durations) - 1)
    assert info["executed_green"] == 30.0
    assert info["intervention"] == ""


def test_the_first_red_has_an_explicit_limit():
    """No preceding green to multiply, so the minimum green stands in."""
    profile = get_profile(STANDARD)
    controller = PhaseDurationController(profile=profile, yellow_time=4.0)
    assert controller.max_red_for(PHASE_EW) == profile.initial_max_red_seconds()
    assert controller.max_red_for(PHASE_EW) == 60.0


def test_without_a_profile_nothing_is_constrained():
    controller = PhaseDurationController(durations=(4, 20), yellow_time=4.0)
    assert controller.max_red_for(PHASE_EW) == float("inf")
    executed, info = controller.resolve(1)
    assert info["intervention"] == ""


# ===========================================================================
# Compliance and truncation
# ===========================================================================
def test_a_truncated_phase_is_not_counted_as_a_short_green():
    profile = get_profile(STANDARD)
    controller = PhaseDurationController(profile=profile, yellow_time=4.0)
    for index in (0, 2, 4):
        executed, info = controller.resolve(index)
        controller.commit(executed, info=info)
    # The episode ends six seconds into a 15 s green.
    controller.commit(0, served=6.0, truncated=True)

    report = controller.compliance()
    assert report["truncated_phases"] == 1
    assert report["completed_phases"] == 3
    assert report["greens_below_min"] == 0, (
        "6 s was the end of the run, not a green the controller chose")
    assert report["min_green_seen"] >= profile.min_green
    assert report["compliant"] is True


def test_the_decision_log_records_request_execution_and_timing():
    controller = PhaseDurationController(profile=get_profile(RESPONSIVE),
                                         yellow_time=4.0)
    controller.commit(0)
    executed, info = controller.resolve(len(controller.durations) - 1)
    controller.commit(executed, info=info)

    entry = controller.decision_log[-1]
    for key in ("phase", "started_at", "ended_at", "requested_green",
                "served_green", "other_group_red", "red_limit",
                "red_within_limit", "truncated_by_episode_end"):
        assert key in entry, key
    assert entry["requested_green"] == 20.0
    assert entry["served_green"] == 12.0
    assert entry["red_within_limit"] is True
    assert entry["ended_at"] > entry["started_at"]


def test_red_is_measured_from_entering_red_to_the_next_green():
    """One group's red is the other group's green plus the clearance."""
    controller = PhaseDurationController(profile=get_profile(STANDARD),
                                         yellow_time=4.0, all_red_time=0.0)
    executed, info = controller.resolve(0)        # 15 s north-south green
    controller.commit(executed, info=info)
    record = controller.red_history[PHASE_EW][-1]
    assert record["seconds"] == 19.0              # 15 green + 4 yellow
    assert record["within_limit"] is True


# ===========================================================================
# Candidate B's ladder
# ===========================================================================
def test_the_extension_ladder_starts_at_the_minimum_and_stops_at_the_maximum():
    ladder = extension_ladder(get_profile(STANDARD), 3.0)
    assert ladder[0] == 15.0
    assert ladder[-1] == 30.0
    assert all(15.0 <= g <= 30.0 for g in ladder)


def test_a_short_decision_interval_is_not_a_short_green():
    """The lock is the whole point of Candidate B."""
    ladder = extension_ladder(get_profile(STANDARD), 3.0)
    assert min(ladder) == 15.0, (
        "asking a question every 3 s must not make a 3 s green reachable")


# ===========================================================================
# The observation that caused the regression
# ===========================================================================
def _raw(queue_per_approach: float, waiting: float = 0.0):
    queues = {a: float(queue_per_approach)
              for a in ("north", "east", "south", "west")}
    waits = {a: float(waiting) for a in queues}
    return {"approach_queues": queues, "approach_waiting": waits,
            "queues": {0: queues["north"] + queues["south"],
                       1: queues["east"] + queues["west"]},
            "waiting": {0: waits["north"] + waits["south"],
                        1: waits["east"] + waits["west"]},
            "vehicles": {0: queues["north"] + queues["south"],
                         1: queues["east"] + queues["west"]}}


def test_clipping_makes_different_congestion_levels_identical():
    """The confirmed defect, pinned so it cannot come back unnoticed."""
    extractor = PhaseStateExtractor(max_queue_per_direction=7.0,
                                    max_queue_per_approach=4.0)
    eight = extractor.observe(_raw(8), PHASE_NS, 0, 0)
    sixty = extractor.observe(_raw(60), PHASE_NS, 0, 0)
    assert eight[0] == sixty[0] == 1.0
    assert eight[7] == sixty[7] == 1.0
    assert np.allclose(eight, sixty), (
        "eight and sixty halting vehicles produce the same observation")


def test_soft_saturation_keeps_them_apart():
    extractor = PhaseStateExtractor(max_queue_per_direction=7.0,
                                    max_queue_per_approach=4.0,
                                    saturation=SATURATION_SOFT)
    readings = [extractor.observe(_raw(n), PHASE_NS, 0, 0)[0]
                for n in (1, 4, 8, 16, 32, 64)]
    assert readings == sorted(readings), "must be increasing"
    assert len(set(readings)) == len(readings), "and never equal"
    assert max(readings) < 1.0, "and never pinned"


def test_soft_saturation_still_lands_inside_the_declared_space():
    extractor = PhaseStateExtractor(saturation=SATURATION_SOFT)
    observation = extractor.observe(_raw(10_000, 10_000), PHASE_NS, 999, 999)
    assert observation.min() >= 0.0 and observation.max() <= 1.0
    assert observation.shape == (len(OBSERVATION_LABELS),)


def test_a_value_at_the_scale_reads_one_half():
    extractor = PhaseStateExtractor(max_queue_per_direction=10.0,
                                    saturation=SATURATION_SOFT)
    # served direction = north + south = 10 vehicles, at the scale.
    assert extractor.observe(_raw(5), PHASE_NS, 0, 0)[0] == pytest.approx(0.5)


def test_an_old_checkpoint_keeps_the_scaling_it_was_trained_with():
    """A contract with no saturation key predates the soft mode."""
    old = extractor_from_contract({"max_queue_per_direction": 7.0})
    assert old.saturation == SATURATION_CLIP
    assert old.observation_version == 1

    new = extractor_from_contract(
        PhaseStateExtractor(saturation=SATURATION_SOFT).contract())
    assert new.saturation == SATURATION_SOFT
    assert new.observation_version == 2


def test_the_contract_carries_the_version():
    contract = PhaseStateExtractor(saturation=SATURATION_SOFT).contract()
    assert contract["observation_version"] == 2.0
    assert contract["saturation"] == SATURATION_SOFT
