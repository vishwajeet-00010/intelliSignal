"""Tests for the fair-comparison harness and the profile model registry.

The harness exists so that a comparison is fair by construction. These
tests check the construction, not the result:

  * every controller is held to the same profile
  * both measurements are produced for every controller, and cross-checked
  * a checkpoint that does not fit a profile is refused rather than run
  * a model selected on the test split cannot be promoted
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agents.phase_dqn_agent as phase_agent  # noqa: E402
from agents.phase_dqn_agent import PhaseDQNAgent, PhaseModelError  # noqa: E402
from custom_traffic.spec import CustomSpec  # noqa: E402
from custom_traffic.resolver import resolve  # noqa: E402
from evaluation.harness import (WORKER, aggregate, fixed_policy,  # noqa: E402
                                max_pressure_policy, run_actuated,
                                run_policy)
from evaluation.promote_profile_model import main as promote  # noqa: E402
from rl.phase_state import PhaseStateExtractor, SATURATION_SOFT  # noqa: E402
from rl.timing_profile import get_profile  # noqa: E402
from training.workload_library import split, sumo_seeds  # noqa: E402

STANDARD = get_profile("standard")


def tiny_workload(duration=200.0, seed=5151):
    spec = CustomSpec.empty(seed=seed)
    spec.name = "harness_test"
    spec.duration_s = duration
    for approach in ("north", "south", "east", "west"):
        lane = spec.lane(approach, 0)
        lane.enabled = True
        lane.count = 12
    resolution = resolve(spec)
    assert resolution.ok
    return resolution.workload


# ===========================================================================
# The workload library
# ===========================================================================
def test_the_three_splits_share_no_traffic():
    sets = {name: {w.workload_id for w in split(name, 600.0)}
            for name in ("train", "validation", "test")}
    assert not sets["train"] & sets["validation"]
    assert not sets["train"] & sets["test"]
    assert not sets["validation"] & sets["test"]


def test_the_three_splits_share_no_sumo_seeds():
    seeds = {name: set(sumo_seeds(name, 40))
             for name in ("train", "validation", "test")}
    assert not seeds["train"] & seeds["validation"]
    assert not seeds["train"] & seeds["test"]
    assert not seeds["validation"] & seeds["test"]


def test_training_covers_light_moderate_and_heavy():
    names = [w.name for w in split("train", 600.0)]
    for intensity in ("light", "moderate", "heavy"):
        assert any(name.endswith(intensity) for name in names), intensity


def test_validation_and_test_also_span_every_intensity():
    """Otherwise a model is selected and judged on one kind of load."""
    for name in ("validation", "test"):
        names = [w.name for w in split(name, 600.0)]
        for intensity in ("light", "moderate", "heavy"):
            assert any(n.endswith(intensity) for n in names), (name, intensity)


def test_validation_and_test_use_different_shapes():
    validation = {w.name for w in split("validation", 600.0)}
    test = {w.name for w in split("test", 600.0)}
    assert not validation & test


def test_the_library_covers_the_shapes_the_audit_asked_for():
    names = " ".join(w.name for w in split("train", 600.0))
    for shape in ("balanced", "directional", "lane_skew", "burst", "ramp",
                  "turn_heavy"):
        assert shape in names, shape


# ===========================================================================
# The harness
# ===========================================================================
def test_the_worker_token_keeps_parallel_runs_apart():
    """Two processes evaluating the same workload must not share filenames."""
    assert WORKER.startswith("p")
    assert str(os.getpid()) in WORKER


@pytest.mark.parametrize("controller", ["fixed", "max-pressure"])
def test_a_run_produces_both_measurements_and_a_cross_check(controller):
    workload = tiny_workload()
    policy = (fixed_policy(STANDARD, 15, 15) if controller == "fixed"
              else max_pressure_policy(STANDARD))
    run = run_policy(workload, policy, STANDARD, 7001, controller=controller,
                     tag="_t")

    assert run.summary["average_waiting_time"] >= 0
    assert run.delay["metrics_version"] >= 2
    assert run.delay["requested"] == workload.total_vehicles
    assert run.crosscheck["completed_agree"] is True, (
        "the live collector and SUMO's trip information must agree on how "
        "many vehicles finished")
    assert run.compliance["profile"] == "standard"


def test_every_controller_is_held_to_the_same_profile():
    workload = tiny_workload()
    runs = [
        run_policy(workload, fixed_policy(STANDARD, 30, 30), STANDARD, 7001,
                   controller="fixed", tag="_p1"),
        run_policy(workload, max_pressure_policy(STANDARD), STANDARD, 7001,
                   controller="pressure", tag="_p2"),
        run_actuated(workload, STANDARD, 7001, gap_seconds=2.0, tag="_p3"),
    ]
    for run in runs:
        assert run.compliant, run.controller
        completed = [g for g in run.greens][:-1]      # the last may be cut off
        for green in completed:
            assert STANDARD.min_green - 1e-6 <= green <= STANDARD.max_green + 1e-6, (
                "{} produced a {} s green".format(run.controller, green))


def test_the_actuated_benchmark_is_not_given_freedom_the_others_lack():
    workload = tiny_workload()
    run = run_actuated(workload, STANDARD, 7001, gap_seconds=1.0, tag="_a1")
    assert run.compliance["greens_below_min"] == 0
    assert run.compliance["greens_above_max"] == 0


def test_aggregate_reports_spread_and_compliance():
    workload = tiny_workload()
    runs = [run_policy(workload, fixed_policy(STANDARD, 15, 15), STANDARD,
                       seed, controller="fixed", tag="_g")
            for seed in (7001, 7002)]
    summary = aggregate(runs)
    assert summary["n"] == 2
    assert "sd_total_delay" in summary
    assert summary["all_compliant"] is True


def test_an_empty_aggregate_is_empty_not_a_crash():
    assert aggregate([]) == {}


# ===========================================================================
# The profile registry
# ===========================================================================
@pytest.fixture
def registry(tmp_path, monkeypatch):
    path = tmp_path / "active_models.json"
    path.write_text(json.dumps({
        "active_models": {"heavy": {"path": "keep/me.pt"}},
        "phase_duration_models": {"heavy": {"path": "keep/too.pt"}},
    }), encoding="utf-8")
    monkeypatch.setattr(phase_agent, "REGISTRY_PATH", str(path))
    monkeypatch.setattr(phase_agent, "PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr("evaluation.promote_profile_model.REGISTRY_PATH",
                        str(path))
    monkeypatch.setattr("evaluation.promote_profile_model.PROJECT_ROOT",
                        str(tmp_path))
    return path


def make_checkpoint(tmp_path, durations, name="best.pt"):
    agent = PhaseDQNAgent(
        observation_size=10, num_actions=len(durations),
        hidden_layers=(8, 8), durations=durations,
        observation_contract=PhaseStateExtractor(
            saturation=SATURATION_SOFT).contract())
    path = tmp_path / "models" / name
    agent.save(str(path))
    return str(path)


def results_file(tmp_path, learned, baseline):
    path = tmp_path / "cmp.json"
    path.write_text(json.dumps({
        "split": "test",
        "profile": {"name": "standard"},
        "table": [
            {"controller": "DQN candidate B", "mean_total_delay": learned,
             "mean_completed": 500, "all_compliant": True},
            {"controller": "fixed ns30/ew15", "mean_total_delay": baseline,
             "mean_completed": 490},
        ],
        "per_workload": {
            "DQN candidate B": [{"workload": "a", "delay": learned}],
            "fixed ns30/ew15": [{"workload": "a", "delay": baseline}],
        },
    }), encoding="utf-8")
    return str(path)


def test_a_fitting_model_that_wins_is_registered(registry, tmp_path):
    model = make_checkpoint(tmp_path, STANDARD.durations)
    result = results_file(tmp_path, 15.9, 22.5)
    promote(["--profile", "standard", "--model", model, "--result", result,
             "--controller", "candidate B", "--selected-on", "validation"])

    payload = json.loads(registry.read_text(encoding="utf-8"))
    record = payload["profile_models"]["standard"]
    assert record["selected_on"] == "validation"
    assert record["measurement"]["mean_total_delay"] == 15.9
    assert record["profile"]["min_green"] == 15.0
    # The two older sections are untouched.
    assert payload["active_models"] == {"heavy": {"path": "keep/me.pt"}}
    assert payload["phase_duration_models"] == {"heavy": {"path": "keep/too.pt"}}


def test_a_model_outside_the_profile_is_refused(registry, tmp_path):
    model = make_checkpoint(tmp_path, (4, 5, 6, 7), name="old.pt")
    result = results_file(tmp_path, 15.9, 22.5)
    with pytest.raises(SystemExit, match="outside the standard profile"):
        promote(["--profile", "standard", "--model", model,
                 "--result", result, "--controller", "candidate B"])


def test_a_model_that_loses_is_refused(registry, tmp_path):
    model = make_checkpoint(tmp_path, STANDARD.durations)
    result = results_file(tmp_path, 30.0, 22.5)
    with pytest.raises(SystemExit, match="refusing to promote"):
        promote(["--profile", "standard", "--model", model,
                 "--result", result, "--controller", "candidate B"])


def test_a_model_selected_on_the_test_split_is_refused(registry, tmp_path):
    model = make_checkpoint(tmp_path, STANDARD.durations)
    result = results_file(tmp_path, 15.9, 22.5)
    with pytest.raises(SystemExit, match="selected on the test split"):
        promote(["--profile", "standard", "--model", model,
                 "--result", result, "--controller", "candidate B",
                 "--selected-on", "test"])


def test_promotion_without_a_measurement_is_refused(registry, tmp_path):
    model = make_checkpoint(tmp_path, STANDARD.durations)
    with pytest.raises(SystemExit, match="without a measurement"):
        promote(["--profile", "standard", "--model", model])


def test_a_registered_profile_model_loads_back(registry, tmp_path):
    model = make_checkpoint(tmp_path, STANDARD.durations)
    result = results_file(tmp_path, 15.9, 22.5)
    promote(["--profile", "standard", "--model", model, "--result", result,
             "--controller", "candidate B"])
    restored = phase_agent.load_registered_profile_model("standard")
    assert list(restored.durations) == list(STANDARD.durations)


def test_an_unregistered_profile_says_so(registry):
    with pytest.raises(PhaseModelError, match="No model is registered"):
        phase_agent.load_registered_profile_model("responsive")


# ===========================================================================
def test_the_real_registry_has_a_standard_profile_model():
    """What this work promoted, and the claim it was promoted on.

    The registry file is read directly rather than through the module, so
    this test cannot disturb the monkeypatched registry another test file
    is using.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "models", "active_models.json"),
              encoding="utf-8") as handle:
        payload = json.load(handle)
    record = (payload.get("profile_models") or {}).get("standard")
    assert record is not None, "the standard profile model must be registered"
    assert record["selected_on"] == "validation", (
        "a model selected on the test split would be selected on the "
        "numbers it is reported with")
    measurement = record["measurement"]
    assert measurement["mean_total_delay"] < measurement["baseline_total_delay"]
    assert measurement["paired"]["wins"] == measurement["paired"]["workloads"]
    assert min(record["durations"]) >= 15.0
    assert os.path.isfile(os.path.join(root, record["path"])), record["path"]
