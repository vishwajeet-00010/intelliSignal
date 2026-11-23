"""Tests for the Hybrid controller.

The decision rule and the budget logic are tested without SUMO, by replacing
the two measurement functions with ones that return numbers chosen for the
case under test. That is deliberate: a test of "the DQN must lose a tie"
should not depend on how a simulation happened to turn out that day. One
end-to-end test at the bottom does run SUMO, on a deliberately small
workload, to check the pieces fit together.
"""

import json
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from custom_traffic.resolver import resolve, workload_from_scenario  # noqa: E402
from custom_traffic.spec import CustomSpec                           # noqa: E402
from rl.phase_action import (DEFAULT_DURATIONS, PHASE_EW,            # noqa: E402
                             PHASE_NS, PhaseDurationController)
from simulation import hybrid, hybrid_job                            # noqa: E402
from simulation.hybrid import (FixedPlan, HybridDecision,            # noqa: E402
                               HybridLimits, RunMetrics, SELECTED_DQN,
                               SELECTED_FIXED, judge, load_decision,
                               save_decision, tune_fixed_plan)
from utils.scenario_builder import ALL_RED, EW_YELLOW, NS_YELLOW     # noqa: E402


def metrics(controller, wait, completed=190, pending=0, requested=200,
            seed=7001):
    return RunMetrics(controller=controller, seed=seed, requested=requested,
                      entered=requested - pending, completed=completed,
                      still_on_road=requested - pending - completed,
                      waiting_to_enter=pending, mean_waiting_time=wait,
                      mean_queue_length=2.0, max_queue_length=9, teleports=0)


def small_workload(duration=300.0, seed=6101):
    spec = CustomSpec.empty(seed=seed)
    spec.name = "hybrid_test"
    spec.duration_s = duration
    for approach in ("north", "south"):
        lane = spec.lane(approach, 0)
        lane.enabled = True
        lane.count = 30
    for approach in ("east", "west"):
        lane = spec.lane(approach, 0)
        lane.enabled = True
        lane.count = 20
    resolution = resolve(spec)
    assert resolution.ok, [str(i) for i in resolution.errors]
    return resolution.workload


# ===========================================================================
# The decision rule
# ===========================================================================
def test_the_dqn_is_selected_only_when_it_wins_on_all_three_counts():
    ok, reason = judge([metrics("dqn", 5.0, completed=190)],
                       [metrics("fixed", 8.0, completed=188)])
    assert ok
    assert "5.00" in reason and "8.00" in reason


def test_a_tie_goes_to_fixed_time():
    ok, reason = judge([metrics("dqn", 8.0)], [metrics("fixed", 8.0)])
    assert not ok
    assert "tie goes to fixed time" in reason.lower()


def test_lower_waiting_does_not_win_if_fewer_vehicles_complete():
    ok, reason = judge([metrics("dqn", 5.0, completed=150)],
                       [metrics("fixed", 8.0, completed=188)])
    assert not ok
    assert "fewer journeys" in reason


def test_lower_waiting_does_not_win_if_more_are_left_waiting_to_enter():
    ok, reason = judge([metrics("dqn", 5.0, pending=40)],
                       [metrics("fixed", 8.0, pending=10)])
    assert not ok
    assert "waiting to enter" in reason


def test_no_comparison_means_no_selection():
    ok, reason = judge([], [])
    assert not ok
    assert "nothing was selected on" in reason


def test_mismatched_seed_counts_are_refused():
    ok, reason = judge([metrics("dqn", 5.0)],
                       [metrics("fixed", 8.0), metrics("fixed", 8.0)])
    assert not ok
    assert "same seeds" in reason


# ===========================================================================
# Limits
# ===========================================================================
def test_the_three_limits_have_the_documented_defaults():
    limits = HybridLimits()
    assert limits.wall_clock_s == 300.0      # 5 minutes
    assert limits.max_episodes == 30
    assert limits.patience == 3


@pytest.mark.parametrize("kwargs", [
    {"wall_clock_s": 0}, {"max_episodes": -1}, {"patience": 0}])
def test_an_impossible_limit_is_refused(kwargs):
    with pytest.raises(ValueError):
        HybridLimits(**kwargs).validate()


# ===========================================================================
# Safe transitions
# ===========================================================================
def test_a_green_never_becomes_the_other_green_without_a_clearance():
    """Whichever controller is in force, the yellow is not skippable."""
    controller = PhaseDurationController(durations=DEFAULT_DURATIONS,
                                         yellow_time=4.0, all_red_time=0.0)
    for action in range(controller.num_actions):
        states = [state for state, _ in controller.plan(action)]
        assert states[-1] in (NS_YELLOW, EW_YELLOW, ALL_RED), (
            "a decision must end in the clearance, so the next green cannot "
            "follow a green directly")


def test_the_controller_in_force_changes_only_at_a_phase_boundary():
    """A decision is one whole green plus its clearance - nothing partial."""
    controller = PhaseDurationController(durations=(5, 10), yellow_time=4.0,
                                         all_red_time=0.0)
    assert controller.decision_seconds(0) == 9.0
    assert controller.decision_seconds(1) == 14.0
    before = controller.phase
    controller.commit(0)
    assert controller.phase != before, "the phase changes once, at the end"


# ===========================================================================
# The job: selection, fallback, timeout, cancellation
# ===========================================================================
class FakeAgent:
    durations = list(DEFAULT_DURATIONS)
    observation_contract = {}
    model_path = ""

    def act(self, observation, training=False):
        return 0

    def save(self, path, extra=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("fake")
        return path


@pytest.fixture
def staged(monkeypatch, tmp_path):
    """Hybrid with its SUMO measurements replaced by scripted numbers."""
    monkeypatch.setattr(hybrid, "HYBRID_DIR", str(tmp_path / "hybrid"))
    monkeypatch.setattr(hybrid_job, "HYBRID_DIR", str(tmp_path / "hybrid"))

    state = {"fixed_wait": 8.0, "dqn_waits": [], "trained": 0}

    def fake_fixed(workload, plan, seed, tag="_x"):
        return metrics("fixed_time", state["fixed_wait"], seed=seed)

    def fake_dqn(workload, agent, seed, tag="_x"):
        # A check runs the policy once per validation seed. The scripted
        # value belongs to the CHECK, not to the seed, so it is drawn on the
        # first seed and reused for the rest.
        if seed == hybrid_job.VALIDATION_SEEDS[0]:
            waits = state["dqn_waits"]
            state["current"] = waits.pop(0) if waits else 99.0
            state.setdefault("used", []).append(state["current"])
        return metrics("dqn", state.get("current", 99.0), seed=seed)

    monkeypatch.setattr(hybrid_job, "run_fixed_plan", fake_fixed)
    monkeypatch.setattr(hybrid_job, "run_dqn_policy", fake_dqn)
    monkeypatch.setattr(hybrid_job, "tune_fixed_plan",
                        lambda workload, start, seed, **kw: (
                            FixedPlan(7, 6, 4, 0, "tuned here",
                                      mean_waiting_time=state["fixed_wait"],
                                      measured_on=[seed]), True))
    return state


def run_job(workload, limits, **kwargs):
    job = hybrid_job.HybridJob(workload, limits,
                              FixedPlan(30, 30, 4, 0, "configured"), **kwargs)
    job.start()
    deadline = time.time() + 60
    while job.alive and time.time() < deadline:
        time.sleep(0.05)
    assert not job.alive, "the job did not finish"
    return job


def test_an_existing_model_that_already_wins_skips_retraining(staged,
                                                              monkeypatch):
    staged["dqn_waits"] = [5.0]            # one check, and it wins
    monkeypatch.setattr(hybrid_job.HybridJob, "_existing_agent",
                        lambda self: (FakeAgent(), "registered model"))

    job = run_job(small_workload(), HybridLimits(wall_clock_s=30,
                                                 max_episodes=30))
    decision = job.decision
    assert decision.selected == SELECTED_DQN
    assert decision.episodes_trained == 0
    assert decision.stopped_because == "the existing model already qualified"
    assert len(decision.checks) == 1
    assert "selected the DQN" in decision.headline()


def test_the_search_gives_up_after_the_patience_limit(staged, monkeypatch):
    """Three consecutive failed checks and fixed time keeps the signal."""
    staged["dqn_waits"] = [12.0, 11.0, 10.0, 9.5, 9.0]
    monkeypatch.setattr(hybrid_job.HybridJob, "_existing_agent",
                        lambda self: (None, ""))
    monkeypatch.setattr(hybrid_job.HybridJob, "_train_episode",
                        lambda self, env, agent, schedule, step, seed: step + 1)
    monkeypatch.setattr(hybrid_job.HybridJob, "_train_and_validate",
                        _patience_only)

    job = run_job(small_workload(), HybridLimits(wall_clock_s=30,
                                                 max_episodes=30, patience=3))
    decision = job.decision
    assert decision.selected == SELECTED_FIXED
    assert "3 consecutive checks" in decision.stopped_because
    assert len(decision.checks) == 3
    assert "selected tuned fixed time" in decision.headline()


def _patience_only(self, plan, fixed_runs, checks, notes, tuning_complete):
    """Validation loop with the training replaced by nothing."""
    failures = 0
    for index in range(10):
        check = self._check(len(checks), index + 1, FakeAgent(), fixed_runs)
        checks.append(check)
        if check.qualified:
            return HybridDecision(
                selected=SELECTED_DQN, reason=check.reason,
                workload_id=self.workload.workload_id, fixed_plan=plan,
                checks=checks, validation_complete=True)
        failures += 1
        if failures >= self.limits.patience:
            return self._fixed_decision(
                plan, notes,
                "the candidate failed {} consecutive checks".format(failures),
                tuning_complete, True, checks, episodes=index + 1)
    return self._fixed_decision(plan, notes, "ran out", tuning_complete,
                                True, checks)


def test_a_zero_episode_budget_never_trains(staged, monkeypatch):
    monkeypatch.setattr(hybrid_job.HybridJob, "_existing_agent",
                        lambda self: (None, ""))
    job = run_job(small_workload(),
                  HybridLimits(wall_clock_s=30, max_episodes=0))
    assert job.decision.selected == SELECTED_FIXED
    assert "episode limit is zero" in job.decision.stopped_because
    assert job.decision.episodes_trained == 0


def test_running_out_of_time_during_training_is_reported_honestly(
        staged, monkeypatch):
    """The deadline cancels the episode in progress and says so."""
    staged["dqn_waits"] = [99.0] * 10
    monkeypatch.setattr(hybrid_job.HybridJob, "_existing_agent",
                        lambda self: (None, ""))

    def slow_episode(self, env, agent, schedule, step, seed):
        time.sleep(0.4)
        return step + 1

    monkeypatch.setattr(hybrid_job.HybridJob, "_train_episode", slow_episode)
    monkeypatch.setattr(hybrid_job.HybridJob, "_build_training_env",
                        lambda self: (None, FakeAgent(), None), raising=False)

    job = hybrid_job.HybridJob(
        small_workload(), HybridLimits(wall_clock_s=1.0, max_episodes=30),
        FixedPlan(30, 30, 4, 0, "configured"))
    job.start()
    deadline = time.time() + 60
    while job.alive and time.time() < deadline:
        time.sleep(0.05)

    decision = job.decision
    assert decision is not None
    assert decision.selected == SELECTED_FIXED
    assert "time limit" in decision.stopped_because.lower()


def test_cancelling_stops_the_job_and_marks_the_decision(staged, monkeypatch):
    staged["dqn_waits"] = [99.0] * 20
    monkeypatch.setattr(hybrid_job.HybridJob, "_existing_agent",
                        lambda self: (None, ""))

    def slow_episode(self, env, agent, schedule, step, seed):
        time.sleep(0.2)
        return step + 1

    monkeypatch.setattr(hybrid_job.HybridJob, "_train_episode", slow_episode)

    job = hybrid_job.HybridJob(
        small_workload(), HybridLimits(wall_clock_s=60, max_episodes=30),
        FixedPlan(30, 30, 4, 0, "configured"))
    job.start()
    time.sleep(0.5)
    job.cancel()
    assert not job.alive
    assert job.state == "cancelled"


# ===========================================================================
# One job per workload
# ===========================================================================
def test_pressing_run_twice_does_not_start_a_second_worker(staged,
                                                           monkeypatch):
    monkeypatch.setattr(hybrid_job.HybridJob, "_existing_agent",
                        lambda self: (None, ""))
    workload = small_workload()
    limits = HybridLimits(wall_clock_s=20, max_episodes=0)
    plan = FixedPlan(30, 30, 4, 0, "configured")
    try:
        first = hybrid_job.start_job(workload, limits, plan)
        second = hybrid_job.start_job(workload, limits, plan)
        assert first is second
        assert hybrid_job.get_job(workload.workload_id) is first
    finally:
        hybrid_job.cancel_job(workload.workload_id)
        hybrid_job.forget_job(workload.workload_id)


def test_cancelling_an_unknown_job_is_not_an_error():
    assert hybrid_job.cancel_job("nothing-here") is False


# ===========================================================================
# Remembering a decision
# ===========================================================================
def test_a_decision_is_reused_only_for_the_same_workload(tmp_path,
                                                         monkeypatch):
    monkeypatch.setattr(hybrid, "HYBRID_DIR", str(tmp_path))
    decision = HybridDecision(
        selected=SELECTED_FIXED, reason="measured", workload_id="abc123",
        fixed_plan=FixedPlan(7, 6, 4, 0, "tuned here", 5.5, [7001]))
    save_decision(decision)

    restored = load_decision("abc123")
    assert restored is not None
    assert restored.selected == SELECTED_FIXED
    assert restored.fixed_plan.ns_green == 7
    assert load_decision("different") is None, (
        "an edited workload must never be answered with the old result")


def test_a_decision_reports_an_unfinished_search_rather_than_a_number():
    decision = HybridDecision(
        selected=SELECTED_FIXED, reason="", workload_id="x",
        fixed_plan=FixedPlan(30, 30, 4, 0, "configured"),
        stopped_because="the time limit was reached during tuning",
        tuning_complete=False, validation_complete=False)
    payload = decision.to_dict()
    assert payload["tuning_complete"] is False
    assert payload["validation_complete"] is False
    assert "No comparison completed" in decision.headline() or \
        "time limit" in decision.headline()


# ===========================================================================
# End to end, with SUMO
# ===========================================================================
@pytest.mark.parametrize("duration", [240.0])
def test_a_real_short_search_produces_a_defensible_decision(duration,
                                                            tmp_path,
                                                            monkeypatch):
    """The whole path, on a small workload, with a tight budget."""
    monkeypatch.setattr(hybrid, "HYBRID_DIR", str(tmp_path / "hybrid"))
    monkeypatch.setattr(hybrid_job, "HYBRID_DIR", str(tmp_path / "hybrid"))

    workload = small_workload(duration=duration, seed=4242)
    job = hybrid_job.HybridJob(
        workload, HybridLimits(wall_clock_s=90, max_episodes=0),
        hybrid.configured_plan_for(workload))
    job.start()
    deadline = time.time() + 300
    while job.alive and time.time() < deadline:
        time.sleep(0.5)
    assert not job.alive, "the job did not finish inside 300 s"

    decision = job.decision
    assert decision is not None, job.error
    assert decision.selected in (SELECTED_DQN, SELECTED_FIXED)
    assert decision.fixed_plan is not None
    assert decision.fixed_plan.mean_waiting_time is not None, (
        "the plan that was selected must have been measured")
    assert decision.headline()
    # Whatever it chose, it must not claim a comparison it did not run.
    if not decision.checks:
        assert not decision.validation_complete
