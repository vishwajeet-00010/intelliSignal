"""
Tests for `simulation/engine.py`.

The engine is the single path from "this traffic, this controller" to a
measured result. Everything the dashboard shows comes through it, so
what these tests protect is the property the whole product rests on:
two controllers on the same traffic are measured the same way, and the
numbers on the screen are the numbers the simulation produced.
"""

import json
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl.timing_profile import get_profile                        # noqa: E402
from simulation import engine                                    # noqa: E402
from simulation.engine import (CONTROLLER_ORDER, CONTROLLERS,     # noqa: E402
                               OPERATING_PROFILE, PRESET_ORDER, PRESETS,
                               RunResult, compare, configured_plan,
                               fleet_summary, green_value, load_run,
                               model_status, preset_workload, run_controller,
                               save_run, saved_runs)

PROFILE = get_profile(OPERATING_PROFILE)
SEED = 90101
DURATION = 180.0


@pytest.fixture(scope="module")
def traffic():
    return preset_workload("offpeak", DURATION)


# ---------------------------------------------------------------------------
# The traffic it offers
# ---------------------------------------------------------------------------
def test_every_preset_builds_and_is_described():
    for key in PRESET_ORDER:
        preset = PRESETS[key]
        assert preset.label and preset.headline and preset.detail
        workload = preset_workload(key, DURATION)
        assert workload.total_vehicles > 0
        assert workload.traffic_model == "indian_urban"


def test_a_preset_is_the_same_traffic_every_time():
    first = preset_workload("typical", DURATION)
    second = preset_workload("typical", DURATION)
    assert first.workload_id == second.workload_id


def test_demonstration_traffic_is_not_the_traffic_the_model_was_judged_on():
    """A demonstration must never accidentally be run on the evidence."""
    from training.workload_library import split

    demo = {preset_workload(key, 900.0).workload_id for key in PRESET_ORDER}
    for name in ("train", "validation", "test"):
        evidence = {w.workload_id for w in split(name, duration_s=900.0)}
        assert not (demo & evidence), \
            "a preset shares its traffic with the {} split".format(name)


def test_heavier_presets_really_are_heavier():
    light = preset_workload("offpeak", 900.0).total_vehicles
    busy = preset_workload("typical", 900.0).total_vehicles
    peak = preset_workload("peak", 900.0).total_vehicles
    assert light < busy < peak


# ---------------------------------------------------------------------------
# Running a controller
# ---------------------------------------------------------------------------
def test_a_fixed_plan_run_reports_everything_a_page_needs(traffic):
    result = run_controller(traffic, "fixed_time", seed=SEED,
                            plan={"ns": 22, "ew": 15})
    assert result.controller == "fixed_time"
    assert "22" in result.label and "15" in result.label
    assert result.metrics["requested"] == traffic.total_vehicles
    assert 0 < result.metrics["completed"] <= result.metrics["requested"]
    assert result.metrics["delay_per_requested"] > 0
    assert result.series and len(result.series) == int(DURATION)
    assert result.replay["steps"] == int(DURATION)
    assert result.audit["compliant"] is True
    assert result.greens, "the timeline must have the greens in it"
    assert result.detail["ns_green"] == 22 and result.detail["ew_green"] == 15


def test_the_audit_is_independent_of_what_the_controller_claims(traffic):
    """Both are reported; they are produced by different code."""
    result = run_controller(traffic, "fixed_time", seed=SEED,
                            plan={"ns": 15, "ew": 15})
    assert result.compliance.get("compliant") is True
    assert result.audit.get("compliant") is True
    assert result.audit["shortest_green"] >= PROFILE.min_green
    assert result.audit["longest_green"] <= PROFILE.max_green
    # the audit counts greens off the signal, the controller off its log
    assert result.audit["greens_completed"] >= 1


def test_every_controller_runs_and_obeys_the_timing_rules(traffic):
    for key in CONTROLLER_ORDER:
        if key == "dqn" and not model_status().get("usable"):
            continue
        result = run_controller(traffic, key, seed=SEED)
        assert result.metrics["completed"] > 0, key
        assert result.audit["compliant"] is True, \
            "{} broke the timing rules: {}".format(
                key, result.audit["violations"])
        assert result.metrics["collisions"] == 0, key
        assert result.trustworthy is True, key


def test_a_run_without_recording_skips_the_replay_but_still_measures(traffic):
    result = run_controller(traffic, "fixed_time", seed=SEED, record=False)
    assert result.replay == {}
    assert result.audit == {}
    assert result.metrics["delay_per_requested"] > 0


def test_an_unknown_controller_is_refused_by_name(traffic):
    with pytest.raises(ValueError) as error:
        run_controller(traffic, "telepathy", seed=SEED)
    assert "telepathy" in str(error.value)


def test_the_configured_plan_is_pulled_onto_the_profile_menu(traffic):
    plan = configured_plan(traffic)
    assert plan["ns"] in PROFILE.durations
    assert plan["ew"] in PROFILE.durations


def test_the_same_traffic_and_seed_give_the_same_result(traffic):
    first = run_controller(traffic, "fixed_time", seed=SEED,
                           plan={"ns": 20, "ew": 20}, record=False)
    second = run_controller(traffic, "fixed_time", seed=SEED,
                            plan={"ns": 20, "ew": 20}, record=False)
    assert first.delay == second.delay
    assert first.completed == second.completed


# ---------------------------------------------------------------------------
# Comparing
# ---------------------------------------------------------------------------
def test_a_comparison_is_against_the_fixed_plan(traffic):
    runs = [run_controller(traffic, key, seed=SEED, record=False)
            for key in ("fixed_time", "actuated")]
    table = compare(runs)
    assert table["baseline"] == "fixed_time"
    rows = {row["controller"]: row for row in table["rows"]}
    assert rows["fixed_time"]["change_percent"] == 0.0
    assert "change_percent" in rows["actuated"]
    assert [row["delay"] for row in table["rows"]] == \
        sorted(row["delay"] for row in table["rows"]), \
        "the best controller is listed first"


def test_runs_on_different_traffic_cannot_be_put_in_one_table(traffic):
    other = preset_workload("typical", DURATION)
    runs = [run_controller(traffic, "fixed_time", seed=SEED, record=False),
            run_controller(other, "fixed_time", seed=SEED, record=False)]
    with pytest.raises(ValueError) as error:
        compare(runs)
    assert "different traffic" in str(error.value)


def test_an_empty_comparison_is_empty_rather_than_broken():
    assert compare([])["rows"] == []


# ---------------------------------------------------------------------------
# Saving and reloading
# ---------------------------------------------------------------------------
def test_a_saved_run_reloads_with_the_same_numbers(traffic, tmp_path):
    result = run_controller(traffic, "fixed_time", seed=SEED,
                            plan={"ns": 18, "ew": 18})
    save_run(result, directory=str(tmp_path))
    back = load_run(traffic.workload_id, "fixed_time", SEED,
                    directory=str(tmp_path))
    assert back is not None
    assert back.delay == result.delay
    assert back.waiting == result.waiting
    assert back.audit["compliant"] == result.audit["compliant"]
    assert len(back.series) == len(result.series)
    assert back.replay["steps"] == result.replay["steps"]

    listed = saved_runs(directory=str(tmp_path))
    assert len(listed) == 1
    assert listed[0]["workload_id"] == traffic.workload_id


def test_a_run_that_was_never_saved_is_absent_rather_than_invented(tmp_path):
    assert load_run("nothing", "fixed_time", 1, directory=str(tmp_path)) is None
    assert saved_runs(directory=str(tmp_path / "nowhere")) == []


def test_a_damaged_saved_run_is_skipped(tmp_path):
    (tmp_path / "broken_fixed_time_1.json").write_text("{not json",
                                                       encoding="utf-8")
    assert saved_runs(directory=str(tmp_path)) == []


# ---------------------------------------------------------------------------
# What the pages read without running anything
# ---------------------------------------------------------------------------
def test_the_model_status_lists_what_it_checked():
    status = model_status()
    assert status["profile"] == OPERATING_PROFILE
    assert status["checks"], "the page shows these, so they must exist"
    for check in status["checks"]:
        assert check["check"] and check["detail"]
        assert isinstance(check["ok"], bool)
    if status["usable"]:
        assert all(check["ok"] for check in status["checks"])
        assert status["describe"]["summary"]
    else:
        assert status["reason"]


def test_an_unregistered_profile_is_reported_as_unusable_with_a_reason():
    status = model_status("responsive")
    if not status["registered"]:
        assert status["usable"] is False
        assert "No model" in status["reason"]
        assert status["checks"][0]["ok"] is False


def test_the_fleet_is_reported_with_its_startup_delay():
    fleet = fleet_summary("indian_urban")
    names = {row["vehicle"] for row in fleet["rows"]}
    assert names == {"motorcycle", "car", "auto", "bus"}
    assert 0.5 < fleet["mean_pcu"] < 0.7
    assert fleet["mean_startup_delay"] > 0.5
    assert sum(row["share"] for row in fleet["rows"]) == pytest.approx(1.0)


def test_a_short_green_is_mostly_spent_starting_the_queue():
    assert green_value(4)["share"] < 0.35
    assert green_value(15)["share"] > 0.75
    assert green_value(30)["share"] > green_value(15)["share"]
    assert green_value(4)["lost"] == green_value(30)["lost"], \
        "the loss is the same whatever the green; that is the point"


def test_every_controller_the_dashboard_offers_is_explained():
    for key in CONTROLLER_ORDER:
        controller = CONTROLLERS[key]
        assert controller.label and controller.headline and controller.detail
        assert len(controller.detail) > 60, \
            "a visitor should learn what it does, not just its name"


# ===========================================================================
# A saved run must be attributable
# ===========================================================================
def test_a_model_run_records_which_network_produced_it(traffic, tmp_path):
    """A registry entry is a pointer, and pointers move.

    Without the fingerprint of the files that actually ran, a result
    from last month says "DQN" and means nothing.
    """
    if not model_status().get("usable"):
        pytest.skip("no validated model is registered")

    result = run_controller(traffic, "dqn", seed=SEED, record=False)
    identity = result.detail.get("model") or {}
    assert identity.get("paths"), "the run must name its checkpoints"
    assert all(m["sha256"] for m in identity["members"]), \
        "and fingerprint every one of them"
    assert identity["action_space"] in ("duration_menu", "lock_extend")
    assert result.matches_model(model_status()) is True

    save_run(result, directory=str(tmp_path))
    back = load_run(traffic.workload_id, "dqn", SEED, directory=str(tmp_path))
    assert back.matches_model(model_status()) is True

    # A run produced by some other network must not pass as this one.
    back.detail["model"]["members"] = [{"path": "x.pt", "sha256": "0" * 64}]
    assert back.matches_model(model_status()) is False


def test_a_run_knows_which_version_of_the_metric_measured_it(traffic):
    from utils.delay_accounting import METRICS_VERSION

    result = run_controller(traffic, "fixed_time", seed=SEED, record=False)
    assert result.detail["metrics_version"] == METRICS_VERSION
    assert result.stale_metrics is False
    result.detail["metrics_version"] = METRICS_VERSION - 1
    assert result.stale_metrics is True, (
        "numbers from an older accounting are not comparable and the run "
        "has to be able to say so")


def test_an_unaudited_run_is_not_called_trustworthy(traffic):
    """Not audited is not the same as clean."""
    result = run_controller(traffic, "fixed_time", seed=SEED, record=False)
    assert result.audit == {}
    assert result.trustworthy is False

    audited = run_controller(traffic, "fixed_time", seed=SEED, record=True)
    assert audited.audit["compliant"] is True
    assert audited.trustworthy is True


def test_runs_on_different_seeds_cannot_be_put_in_one_table(traffic):
    runs = [run_controller(traffic, "fixed_time", seed=SEED, record=False),
            run_controller(traffic, "actuated", seed=SEED + 1, record=False)]
    with pytest.raises(ValueError) as error:
        compare(runs)
    assert "different SUMO seeds" in str(error.value)


def test_the_environment_check_reports_what_is_installed():
    from simulation.engine import environment_status

    status = environment_status()
    names = {check["check"] for check in status["checks"]}
    assert "SUMO is installed" in names
    assert "the junction is built" in names
    assert status["ready"] == all(c["ok"] for c in status["checks"])
