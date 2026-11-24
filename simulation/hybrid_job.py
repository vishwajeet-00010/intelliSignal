"""
IntelliSignal
simulation/hybrid_job.py

The background worker behind the Hybrid control mode.

`simulation/hybrid.py` holds the measurement and the decision rule, which are
pure and testable without SUMO. This file holds the part that has to deal
with time, threads and a browser: starting the search, respecting the
deadline, cancelling cleanly, and surviving a page refresh.

Three rules that are easy to get wrong in a dashboard:

  * **One job per workload.** Pressing Run twice must not start a second
    worker - two of them would run SUMO over the same generated files.
    `start_job` hands back the job already running.
  * **A refresh reconnects.** The job lives in this module, not in the
    Streamlit session, so reloading the page finds it again.
  * **Cancelling cleans up.** The worker stops at the next safe point and
    the generated SUMO inputs are removed, so a cancelled run leaves nothing
    behind for the next one to trip over.
"""

from __future__ import annotations

import os
import statistics
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from custom_traffic.builder import cleanup_workload_inputs
from custom_traffic.workload import ResolvedWorkload
from simulation.hybrid import (HYBRID_DIR, SELECTED_DQN, SELECTED_FIXED,
                               FixedPlan, HybridDecision, HybridLimits,
                               RunMetrics, ValidationCheck, judge,
                               run_dqn_policy, run_fixed_plan, tune_fixed_plan)

#: Seeds used only to validate. They never appear in training, so a candidate
#: cannot be selected on traffic it was fitted to.
VALIDATION_SEEDS = (7001, 7002)

#: Seeds used only for training episodes.
TRAINING_SEED_BASE = 8000

#: How long one decision of the learned controller takes, roughly, used only
#: to size the exploration schedule.
SECONDS_PER_DECISION = 12.0


class HybridJob:
    """Runs the search in a background thread, against a deadline."""

    def __init__(self, workload: ResolvedWorkload, limits: HybridLimits,
                 configured_plan: FixedPlan, validate_every: int = 5,
                 validation_seeds: Sequence[int] = VALIDATION_SEEDS,
                 profile=None):
        from rl.timing_profile import get_profile
        #: Every plan Hybrid tunes and every candidate it trains is held to
        #: this profile, so its fallback is a plan the profile allows.
        self.profile = get_profile(profile) if profile is not None else None
        self.workload = workload
        self.limits = limits.validate()
        self.configured_plan = configured_plan
        self.validate_every = max(int(validate_every), 1)
        self.validation_seeds = [int(seed) for seed in validation_seeds]

        self.started_at: Optional[float] = None
        self.deadline: Optional[float] = None
        self.state = "idle"       # idle | running | done | cancelled | failed
        self.phase = "not started"
        self.messages: List[str] = []
        self.decision: Optional[HybridDecision] = None
        self.error = ""

        self._cancel = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def elapsed_s(self) -> float:
        return 0.0 if self.started_at is None else time.time() - self.started_at

    @property
    def remaining_s(self) -> float:
        if self.deadline is None:
            return float(self.limits.wall_clock_s)
        return max(self.deadline - time.time(), 0.0)

    def _say(self, message: str) -> None:
        with self._lock:
            self.messages.append("[{:>5.1f}s] {}".format(self.elapsed_s, message))

    def _should_stop(self) -> bool:
        return self._cancel.is_set()

    def _out_of_time(self) -> bool:
        return (self._cancel.is_set()
                or (self.deadline is not None and time.time() >= self.deadline))

    # ------------------------------------------------------------------
    def start(self) -> "HybridJob":
        if self.alive:
            return self
        self.started_at = time.time()
        self.deadline = self.started_at + float(self.limits.wall_clock_s)
        self.state = "running"
        self._cancel.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True,
            name="hybrid-{}".format(self.workload.workload_id))
        self._thread.start()
        return self

    def cancel(self, timeout: float = 30.0) -> None:
        """Stop at the next safe point, then remove the generated inputs."""
        self._cancel.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        if self.state == "running":
            self.state = "cancelled"
        cleanup_workload_inputs(self.workload)

    # ------------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        """A snapshot the dashboard can render without blocking."""
        with self._lock:
            messages = list(self.messages[-12:])
        return {
            "state": self.state, "phase": self.phase,
            "elapsed_s": round(self.elapsed_s, 1),
            "remaining_s": round(self.remaining_s, 1),
            "limit_s": float(self.limits.wall_clock_s),
            "messages": messages, "error": self.error,
            "workload_id": self.workload.workload_id,
            "decision": self.decision.to_dict() if self.decision else None,
        }

    # ==================================================================
    def _run(self) -> None:
        try:
            self.decision = self._search()
            self.state = "cancelled" if self._cancel.is_set() else "done"
            self.phase = "finished"
            from simulation.hybrid import save_decision
            if self.decision and not self.decision.cancelled:
                save_decision(self.decision)
        except Exception as exc:                 # pragma: no cover - defensive
            self.error = "{}: {}".format(type(exc).__name__, exc)
            self.state = "failed"
            self.phase = "failed"
            self._say("failed: {}".format(self.error))
            self.decision = HybridDecision(
                selected=SELECTED_FIXED,
                reason="The background search failed, so the configured "
                       "fixed plan stayed in force. {}".format(self.error),
                workload_id=self.workload.workload_id,
                fixed_plan=self.configured_plan,
                stopped_because="the background search raised an error",
                elapsed_s=self.elapsed_s)
        finally:
            cleanup_workload_inputs(self.workload)

    # ------------------------------------------------------------------
    def _search(self) -> HybridDecision:
        notes: List[str] = []

        self.phase = "tuning the fixed-time plan"
        self._say("tuning north-south and east-west green independently")
        plan, tuning_complete = tune_fixed_plan(
            self.workload, self.configured_plan, seed=self.validation_seeds[0],
            deadline=self.deadline, should_stop=self._should_stop,
            on_progress=self._say, profile=self.profile)
        if not tuning_complete:
            notes.append("Tuning did not finish inside the time limit; the "
                         "best plan measured so far was used.")
        self._say("best measured plan: {} at {:.2f} s".format(
            plan.label(), plan.mean_waiting_time or float("nan")))

        if self._out_of_time():
            return self._fixed_decision(
                plan, notes, "the time limit was reached during tuning",
                tuning_complete, False, [])

        self.phase = "measuring the tuned plan"
        fixed_runs = [run_fixed_plan(self.workload, plan, seed)
                      for seed in self.validation_seeds]
        if self._out_of_time():
            return self._fixed_decision(
                plan, notes,
                "the time limit was reached before any DQN comparison "
                "completed", tuning_complete, False, [])

        checks: List[ValidationCheck] = []
        agent, source = self._existing_agent()
        if agent is not None:
            self.phase = "testing the existing DQN"
            self._say("evaluating the registered model: {}".format(source))
            check = self._check(0, 0, agent, fixed_runs)
            checks.append(check)
            if check.qualified:
                return HybridDecision(
                    selected=SELECTED_DQN, reason=check.reason,
                    workload_id=self.workload.workload_id, fixed_plan=plan,
                    checks=checks, elapsed_s=self.elapsed_s,
                    stopped_because="the existing model already qualified",
                    tuning_complete=tuning_complete, validation_complete=True,
                    model_path=self._save_candidate(agent, "existing"),
                    model_source=source, notes=notes)
            self._say("the existing model did not qualify; training a candidate")
        else:
            notes.append("No compatible registered phase-duration model, so a "
                         "candidate was trained from scratch.")

        if self.limits.max_episodes <= 0:
            return self._fixed_decision(
                plan, notes,
                "the episode limit is zero, so no candidate was trained",
                tuning_complete, bool(checks), checks)

        return self._train_and_validate(plan, fixed_runs, checks, notes,
                                        tuning_complete)

    # ------------------------------------------------------------------
    def _existing_agent(self):
        """A registered phase-duration model for this workload's scenario."""
        try:
            from agents.phase_dqn_agent import (load_registered_phase_model,
                                                load_registered_profile_model,
                                                registered_phase_model,
                                                registered_profile_model)
            if self.profile is not None:
                record = registered_profile_model(self.profile.name)
                if not record:
                    return None, ""
                return (load_registered_profile_model(self.profile.name),
                        "{} [profile {}]".format(
                            record.get("label", "registered model"),
                            self.profile.name))

            record = registered_phase_model(self.workload.base_scenario)
            if not record:
                return None, ""
            agent = load_registered_phase_model(self.workload.base_scenario)
            return agent, "{} [{}]".format(
                record.get("label", "registered model"),
                self.workload.base_scenario)
        except Exception as exc:
            self._say("could not load a registered model: {}".format(exc))
            return None, ""

    # ------------------------------------------------------------------
    def _check(self, index: int, episode: int, agent,
               fixed_runs: Sequence[RunMetrics]) -> ValidationCheck:
        """Freeze the candidate and compare it with the tuned plan."""
        dqn_runs = [run_dqn_policy(self.workload, agent, seed,
                                   profile=self.profile)
                    for seed in self.validation_seeds]
        qualified, reason = judge(dqn_runs, fixed_runs)
        check = ValidationCheck(
            index=index, episode=episode, elapsed_s=round(self.elapsed_s, 1),
            seeds=list(self.validation_seeds),
            dqn_waiting=statistics.fmean(m.mean_waiting_time for m in dqn_runs),
            fixed_waiting=statistics.fmean(m.mean_waiting_time
                                           for m in fixed_runs),
            dqn_throughput=statistics.fmean(m.completed for m in dqn_runs),
            fixed_throughput=statistics.fmean(m.completed for m in fixed_runs),
            dqn_waiting_to_enter=statistics.fmean(m.waiting_to_enter
                                                  for m in dqn_runs),
            fixed_waiting_to_enter=statistics.fmean(m.waiting_to_enter
                                                    for m in fixed_runs),
            qualified=qualified, reason=reason)
        self._say("check {}: DQN {:.2f} s against fixed {:.2f} s - {}".format(
            index, check.dqn_waiting, check.fixed_waiting,
            "qualifies" if qualified else "does not qualify"))
        return check

    # ------------------------------------------------------------------
    def _train_and_validate(self, plan: FixedPlan,
                            fixed_runs: Sequence[RunMetrics],
                            checks: List[ValidationCheck], notes: List[str],
                            tuning_complete: bool) -> HybridDecision:
        import torch

        from agents.phase_dqn_agent import EpsilonSchedule, PhaseDQNAgent
        from rl.phase_action import DEFAULT_DURATIONS
        from rl.phase_environment import PhaseDurationEnv
        from rl.phase_state import PhaseStateExtractor

        torch.set_num_threads(1)
        self.phase = "training a candidate"

        durations = (self.profile.durations if self.profile is not None
                     else DEFAULT_DURATIONS)
        extractor = PhaseStateExtractor(
            max_green_seconds=float(max(durations)))
        env = PhaseDurationEnv(
            workload=self.workload, durations=durations,
            profile=self.profile.name if self.profile is not None else None,
            state_extractor=extractor, gamma_per_second=0.99,
            reward_scale=0.005, tag="_hybtrain")
        agent = PhaseDQNAgent(
            observation_size=int(env.observation_space.shape[0]),
            num_actions=int(env.action_space.n), hidden_layers=(128, 128),
            gamma_per_second=0.99, batch_size=128, learning_starts=400,
            seed=TRAINING_SEED_BASE,
            scenario=self.workload.base_scenario,
            durations=list(env.controller.durations),
            observation_contract=extractor.contract())

        per_episode = max(int(self.workload.duration_s / SECONDS_PER_DECISION), 1)
        schedule = EpsilonSchedule(
            start=1.0, end=0.05,
            decay_steps=max(int(self.limits.max_episodes * per_episode * 0.6), 1))

        consecutive_failures = sum(1 for check in checks if not check.qualified)
        episodes = 0
        global_step = 0
        stopped = ""
        validation_complete = bool(checks)

        try:
            while episodes < self.limits.max_episodes:
                if self._out_of_time():
                    stopped = ("the {:.0f} s time limit was reached during "
                               "training".format(self.limits.wall_clock_s))
                    notes.append("The episode running at the deadline was "
                                 "cancelled and is not counted.")
                    break

                episodes += 1
                self.phase = "training episode {} of {}".format(
                    episodes, self.limits.max_episodes)
                global_step = self._train_episode(
                    env, agent, schedule, global_step,
                    seed=TRAINING_SEED_BASE + episodes)

                due = (episodes % self.validate_every == 0
                       or episodes == self.limits.max_episodes)
                if not due:
                    continue
                if self._out_of_time():
                    stopped = ("the {:.0f} s time limit was reached before "
                               "the next check".format(self.limits.wall_clock_s))
                    break

                self.phase = "validating after episode {}".format(episodes)
                check = self._check(len(checks), episodes, agent, fixed_runs)
                checks.append(check)
                validation_complete = True

                if check.qualified:
                    return HybridDecision(
                        selected=SELECTED_DQN, reason=check.reason,
                        workload_id=self.workload.workload_id,
                        fixed_plan=plan, checks=checks,
                        episodes_trained=episodes, elapsed_s=self.elapsed_s,
                        stopped_because="the candidate qualified",
                        tuning_complete=tuning_complete,
                        validation_complete=True,
                        model_path=self._save_candidate(agent, "trained"),
                        model_source="trained for this workload", notes=notes)

                consecutive_failures += 1
                if consecutive_failures >= self.limits.patience:
                    stopped = ("the candidate failed {} consecutive checks"
                               .format(consecutive_failures))
                    break
            else:
                stopped = "the {} episode limit was reached".format(
                    self.limits.max_episodes)
        finally:
            env.close()
            env.cleanup_generated_inputs()

        if self._cancel.is_set():
            stopped = "the job was cancelled"

        return self._fixed_decision(plan, notes, stopped, tuning_complete,
                                    validation_complete, checks,
                                    episodes=episodes)

    # ------------------------------------------------------------------
    def _train_episode(self, env, agent, schedule, global_step: int,
                       seed: int) -> int:
        """One training episode. No MetricsCollector - nothing reads it."""
        from agents.phase_dqn_agent import n_step_transitions

        observation, _ = env.reset(seed=int(seed))
        env.step_callback = None
        pending: List[Dict[str, Any]] = []
        n_step = 2

        while True:
            if self._out_of_time():
                break
            agent.set_epsilon(schedule.value(global_step))
            action = agent.act(observation, training=True)
            next_observation, reward, terminated, truncated, info = env.step(action)
            global_step += 1

            pending.append({"state": observation, "action": action,
                            "reward": reward, "next_state": next_observation,
                            "done": terminated,
                            "duration": float(info["duration"])})
            while len(pending) >= n_step:
                agent.remember(**n_step_transitions(pending, n_step,
                                                    agent.gamma_per_second))
                pending.pop(0)
            agent.learn()

            observation = next_observation
            if terminated or truncated:
                break

        while pending:
            agent.remember(**n_step_transitions(pending, len(pending),
                                                agent.gamma_per_second))
            pending.pop(0)
        return global_step

    # ------------------------------------------------------------------
    def _save_candidate(self, agent, origin: str) -> str:
        """Keep the qualifying checkpoint, filed under the workload id."""
        directory = os.path.join(HYBRID_DIR, self.workload.workload_id)
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "candidate.pt")
        agent.save(path, extra={
            "selection": "qualified against the tuned fixed plan",
            "workload_id": self.workload.workload_id, "origin": origin})
        return os.path.relpath(path, PROJECT_ROOT).replace("\\", "/")

    # ------------------------------------------------------------------
    def _fixed_decision(self, plan: FixedPlan, notes: List[str], stopped: str,
                        tuning_complete: bool, validation_complete: bool,
                        checks: List[ValidationCheck],
                        episodes: int = 0) -> HybridDecision:
        reason = (checks[-1].reason if checks else
                  "No completed DQN comparison, so the best measured fixed "
                  "plan stayed in force.")
        return HybridDecision(
            selected=SELECTED_FIXED, reason=reason,
            workload_id=self.workload.workload_id, fixed_plan=plan,
            checks=checks, episodes_trained=episodes,
            elapsed_s=self.elapsed_s, stopped_because=stopped,
            tuning_complete=tuning_complete,
            validation_complete=validation_complete,
            cancelled=self._cancel.is_set(), notes=notes)


# ===========================================================================
# One job per workload, so a refresh reconnects instead of starting another
# ===========================================================================
_JOBS: Dict[str, HybridJob] = {}
_JOBS_LOCK = threading.Lock()


def get_job(workload_id: str) -> Optional[HybridJob]:
    with _JOBS_LOCK:
        return _JOBS.get(workload_id)


def start_job(workload: ResolvedWorkload, limits: HybridLimits,
              configured_plan: FixedPlan, **kwargs) -> HybridJob:
    """Start the search, or hand back the job already running for it."""
    with _JOBS_LOCK:
        existing = _JOBS.get(workload.workload_id)
        if existing is not None and (existing.alive or existing.state == "done"):
            return existing
        job = HybridJob(workload, limits, configured_plan, **kwargs)
        _JOBS[workload.workload_id] = job
    return job.start()


def cancel_job(workload_id: str) -> bool:
    job = get_job(workload_id)
    if job is None:
        return False
    job.cancel()
    return True


def forget_job(workload_id: str) -> None:
    with _JOBS_LOCK:
        _JOBS.pop(workload_id, None)


def active_jobs() -> List[str]:
    with _JOBS_LOCK:
        return [key for key, job in _JOBS.items() if job.alive]
