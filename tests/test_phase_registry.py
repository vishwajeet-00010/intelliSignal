"""Tests for promoting and loading a duration-choosing model.

The project has promoted a model against a mislabelled baseline once already
(`output/dqn_stage4_20261120/CORRECTION.md`). These tests pin the two guards
that would have caught it:

  * a promotion is refused unless the recorded measurement actually shows the
    model beating its fixed-time baseline
  * a promoted checkpoint is refused at load time if the file no longer
    matches the sha256 that was promoted
"""

import json
import os
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agents.phase_dqn_agent as phase_agent  # noqa: E402
from agents.phase_dqn_agent import (PhaseDQNAgent,  # noqa: E402
                                    PhaseModelError, REGISTRY_SECTION)
from evaluation.promote_phase_model import (main as promote_main,  # noqa: E402
                                            read_measurement, sha256_of)
from rl.phase_state import OBSERVATION_SIZE, PhaseStateExtractor  # noqa: E402


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A registry file and a checkpoint, isolated from the real ones."""
    registry = tmp_path / "active_models.json"
    registry.write_text(json.dumps({
        "format_version": 1,
        "active_models": {"heavy": {"path": "models/old/best.pt"}},
    }), encoding="utf-8")
    monkeypatch.setattr(phase_agent, "REGISTRY_PATH", str(registry))
    monkeypatch.setattr("evaluation.promote_phase_model.REGISTRY_PATH",
                        str(registry))
    monkeypatch.setattr(phase_agent, "PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr("evaluation.promote_phase_model.PROJECT_ROOT",
                        str(tmp_path))

    agent = PhaseDQNAgent(observation_size=OBSERVATION_SIZE, num_actions=4,
                          hidden_layers=(8, 8), durations=(4, 6, 8, 10),
                          observation_contract=PhaseStateExtractor().contract())
    model = tmp_path / "models" / "best.pt"
    agent.save(str(model))
    return {"registry": registry, "model": model, "root": tmp_path,
            "agent": agent}


def results_file(path, learned, baseline):
    payload = [
        {"controller": "dqn_phase (duration DQN)", "family": "phase_duration",
         "mean_waiting_time": learned, "sd_waiting_time": 0.3,
         "mean_throughput": 1468.0, "seeds": [6101, 6102]},
        {"controller": "fixed_ns7_ew6", "family": "fixed_time_tuned_alias",
         "mean_waiting_time": baseline, "sd_waiting_time": 0.4,
         "mean_throughput": 1466.5},
    ]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return str(path)


# ---------------------------------------------------------------------------
def test_a_model_that_beats_its_baseline_is_registered(sandbox):
    result = results_file(sandbox["root"] / "r.json", 5.151, 5.754)
    promote_main(["--scenario", "heavy", "--model", str(sandbox["model"]),
                  "--result", result])

    payload = json.loads(sandbox["registry"].read_text(encoding="utf-8"))
    record = payload[REGISTRY_SECTION]["heavy"]
    assert record["sha256"] == sha256_of(str(sandbox["model"]))
    assert record["durations"] == [4.0, 6.0, 8.0, 10.0]
    assert record["measurement"]["mean_waiting_time"] == 5.151
    assert record["measurement"]["baseline_mean_waiting_time"] == 5.754
    assert record["controller_family"] == "phase_duration"


def test_promotion_does_not_touch_the_keep_switch_section(sandbox):
    """`active_models` is read by code that expects a different checkpoint."""
    result = results_file(sandbox["root"] / "r.json", 5.151, 5.754)
    promote_main(["--scenario", "heavy", "--model", str(sandbox["model"]),
                  "--result", result])
    payload = json.loads(sandbox["registry"].read_text(encoding="utf-8"))
    assert payload["active_models"] == {"heavy": {"path": "models/old/best.pt"}}


def test_a_model_that_loses_to_its_baseline_is_refused(sandbox):
    result = results_file(sandbox["root"] / "r.json", 6.900, 5.754)
    with pytest.raises(SystemExit, match="not better than"):
        promote_main(["--scenario", "heavy", "--model", str(sandbox["model"]),
                      "--result", result])
    payload = json.loads(sandbox["registry"].read_text(encoding="utf-8"))
    assert REGISTRY_SECTION not in payload


def test_promotion_without_any_measurement_is_refused(sandbox):
    with pytest.raises(SystemExit, match="without a measurement"):
        promote_main(["--scenario", "heavy", "--model", str(sandbox["model"])])


def test_force_overrides_the_gate_but_must_be_asked_for(sandbox):
    result = results_file(sandbox["root"] / "r.json", 6.900, 5.754)
    promote_main(["--scenario", "heavy", "--model", str(sandbox["model"]),
                  "--result", result, "--force"])
    payload = json.loads(sandbox["registry"].read_text(encoding="utf-8"))
    assert REGISTRY_SECTION in payload


def test_the_baseline_is_found_whatever_the_plan_is_called(tmp_path):
    """A scenario whose optimum is not ns=7/ew=6 must still find its baseline."""
    path = tmp_path / "moderate.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump([
            {"controller": "dqn_phase (duration DQN)",
             "family": "phase_duration", "mean_waiting_time": 2.237},
            {"controller": "fixed_ns6_ew4", "family": "fixed_time_tuned_alias",
             "mean_waiting_time": 2.761},
        ], handle)
    measurement = read_measurement(str(path))
    assert measurement["baseline_mean_waiting_time"] == 2.761


# ---------------------------------------------------------------------------
def test_a_registered_model_loads_with_its_menu_and_scales(sandbox):
    result = results_file(sandbox["root"] / "r.json", 5.151, 5.754)
    promote_main(["--scenario", "heavy", "--model", str(sandbox["model"]),
                  "--result", result])

    restored = phase_agent.load_registered_phase_model("heavy")
    assert restored.durations == [4.0, 6.0, 8.0, 10.0]
    observation = np.full(OBSERVATION_SIZE, 0.3, dtype=np.float32)
    assert np.allclose(restored.q_values(observation),
                       sandbox["agent"].q_values(observation))


def test_a_checkpoint_that_changed_since_promotion_is_refused(sandbox):
    result = results_file(sandbox["root"] / "r.json", 5.151, 5.754)
    promote_main(["--scenario", "heavy", "--model", str(sandbox["model"]),
                  "--result", result])

    # Someone retrains and leaves a different file at the promoted path.
    other = PhaseDQNAgent(observation_size=OBSERVATION_SIZE, num_actions=4,
                          hidden_layers=(8, 8), seed=99,
                          durations=(4, 6, 8, 10))
    other.save(str(sandbox["model"]))

    with pytest.raises(PhaseModelError, match="does not match its recorded"):
        phase_agent.load_registered_phase_model("heavy")


def test_an_unregistered_scenario_says_so(sandbox):
    with pytest.raises(PhaseModelError, match="No duration-choosing model"):
        phase_agent.load_registered_phase_model("moderate")


def test_a_registered_but_missing_file_says_so(sandbox):
    result = results_file(sandbox["root"] / "r.json", 5.151, 5.754)
    promote_main(["--scenario", "heavy", "--model", str(sandbox["model"]),
                  "--result", result])
    os.remove(str(sandbox["model"]))
    with pytest.raises(PhaseModelError, match="is missing"):
        phase_agent.load_registered_phase_model("heavy")


# ---------------------------------------------------------------------------
def test_the_real_registry_has_both_scenarios_promoted():
    """The models this work promoted are registered and loadable."""
    import importlib
    importlib.reload(phase_agent)
    for scenario, expected_wait in (("heavy", 5.151), ("moderate", 2.237)):
        record = phase_agent.registered_phase_model(scenario)
        assert record is not None, "{} is not promoted".format(scenario)
        measured = record["measurement"]["mean_waiting_time"]
        baseline = record["measurement"]["baseline_mean_waiting_time"]
        assert measured == pytest.approx(expected_wait, abs=0.001)
        assert measured < baseline, (
            "a promoted model must beat the fixed plan it was measured "
            "against: {} got {} against {}".format(scenario, measured, baseline))
        agent = phase_agent.load_registered_phase_model(scenario)
        assert agent.num_actions == len(agent.durations)
