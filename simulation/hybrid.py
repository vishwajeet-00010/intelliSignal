"""
IntelliSignal
simulation/hybrid.py

The Hybrid controller: give the DQN a fixed amount of time to prove itself.

What happens when the user presses Run
--------------------------------------
    t = 0      traffic starts moving under a valid fixed-time plan
               (a validated tuned plan if one matches this workload,
                otherwise the plan configured for the base scenario)
    t = 0..    in the background, and against a deadline:
                 1. tune north-south and east-west green independently
                 2. try an existing phase-duration DQN on this workload
                 3. if it does not already qualify, train a candidate,
                    freezing and validating it at intervals
    deadline   whatever has not finished is abandoned and said so

The point of the deadline is that a demonstration cannot wait. The default
budget is five minutes of WALL-CLOCK time, which is not the same thing as
simulated traffic time - a 1800 s simulation runs in about fifteen seconds.

The decision rule
-----------------
The DQN is selected only when, across matched validation runs, it has

    * lower mean waiting time per vehicle, AND
    * at least equal mean throughput, AND
    * no increase in vehicles still waiting to enter

Anything else selects the best fixed plan that was actually measured. A tie
goes to fixed time: a learned controller has to earn its place, and "no
worse" is not "better".

Three ways the search ends early:
    * the DQN qualifies      -> stop, save the checkpoint for this workload
    * `patience` consecutive failed checks -> stop, select fixed time
    * the deadline or the episode limit    -> stop, select fixed time

What "matched" means
--------------------
Both controllers are handed the same explicit vehicle list, the same SUMO
seed, the same simulated duration and the same `utils.metrics.MetricsCollector`.
With a custom workload the arrivals are written out vehicle by vehicle, so
this is exact rather than merely reproducible. Training seeds and validation
seeds never overlap.
"""

from __future__ import annotations

import os
import statistics
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from custom_traffic.builder import build_workload, cleanup_workload_inputs
from custom_traffic.workload import ResolvedWorkload
from simulation.run_simulation import prepare_sumo_imports, sumo_binary
from utils.metrics import MetricsCollector

#: Where Hybrid keeps its own candidates and decisions, away from the
#: registered Moderate/Heavy models.
HYBRID_DIR = os.path.join(PROJECT_ROOT, "results", "hybrid")

#: Green durations the tuner will try, in seconds.
TUNE_GREENS: Tuple[int, ...] = (4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30)

SELECTED_DQN = "dqn"
SELECTED_FIXED = "fixed_time"


# ===========================================================================
@dataclass
class HybridLimits:
    """The three limits the user can edit under Advanced."""

    wall_clock_s: float = 300.0      # 5 minutes, measured from Run
    max_episodes: int = 30           # additional training episodes
    patience: int = 3                # consecutive failed validation checks

    def validate(self) -> "HybridLimits":
        if self.wall_clock_s <= 0:
            raise ValueError("The time limit must be more than 0 seconds.")
        if self.max_episodes < 0:
            raise ValueError("The episode limit cannot be negative.")
        if self.patience < 1:
            raise ValueError("Patience must be at least 1 check.")
        return self


@dataclass
class RunMetrics:
    """One measured run, with the vehicle accounting spelled out."""

    controller: str
    seed: int
    requested: int
    entered: int
    completed: int
    still_on_road: int
    waiting_to_enter: int
    mean_waiting_time: float
    mean_queue_length: float
    max_queue_length: float
    teleports: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FixedPlan:
    """A fixed-time plan and where it came from."""

    ns_green: float
    ew_green: float
    yellow: float
    all_red: float
    source: str                      # "configured" | "tuned here" | "validated"
    mean_waiting_time: Optional[float] = None
    measured_on: List[int] = field(default_factory=list)

    @property
    def cycle(self) -> float:
        return self.ns_green + self.ew_green + 2 * (self.yellow + self.all_red)

    def signal(self) -> Dict[str, float]:
        return {"ns_green": self.ns_green, "ew_green": self.ew_green,
                "yellow": self.yellow, "all_red": self.all_red}

    def label(self) -> str:
        return "ns {:.0f} s / ew {:.0f} s ({})".format(
            self.ns_green, self.ew_green, self.source)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["cycle"] = self.cycle
        return data


@dataclass
class ValidationCheck:
    """One frozen comparison of the candidate against the tuned plan."""

    index: int
    episode: int
    elapsed_s: float
    seeds: List[int]
    dqn_waiting: float
    fixed_waiting: float
    dqn_throughput: float
    fixed_throughput: float
    dqn_waiting_to_enter: float
    fixed_waiting_to_enter: float
    qualified: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HybridDecision:
    """What Hybrid chose, and everything needed to defend the choice."""

    selected: str                    # SELECTED_DQN | SELECTED_FIXED
    reason: str
    workload_id: str
    fixed_plan: Optional[FixedPlan] = None
    checks: List[ValidationCheck] = field(default_factory=list)
    episodes_trained: int = 0
    elapsed_s: float = 0.0
    stopped_because: str = ""
    tuning_complete: bool = False
    validation_complete: bool = False
    model_path: str = ""
    model_source: str = ""
    cancelled: bool = False
    notes: List[str] = field(default_factory=list)
    decided_at: str = ""

    def __post_init__(self) -> None:
        if not self.decided_at:
            self.decided_at = datetime.now().isoformat(timespec="seconds")

    @property
    def dqn_won(self) -> bool:
        return self.selected == SELECTED_DQN

    @property
    def last_check(self) -> Optional[ValidationCheck]:
        return self.checks[-1] if self.checks else None

    def headline(self) -> str:
        """One sentence a teacher can read off the screen."""
        plan = self.fixed_plan.label() if self.fixed_plan else "the configured plan"
        if self.cancelled:
            return "Hybrid was cancelled; {} was in force.".format(plan)
        if not self.checks:
            return ("Hybrid selected fixed time ({}). {}".format(
                plan, self.stopped_because or "No comparison completed."))
        check = self.checks[-1]
        if self.dqn_won:
            return ("Hybrid selected the DQN after {} check{}. Mean waiting "
                    "was {:.2f} s with fixed time and {:.2f} s with the DQN."
                    .format(len(self.checks), "" if len(self.checks) == 1 else "s",
                            check.fixed_waiting, check.dqn_waiting))
        return ("Hybrid selected tuned fixed time ({}) after {} check{}. Mean "
                "waiting was {:.2f} s with fixed time and {:.2f} s with the "
                "DQN.".format(plan, len(self.checks),
                              "" if len(self.checks) == 1 else "s",
                              check.fixed_waiting, check.dqn_waiting))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected": self.selected, "reason": self.reason,
            "workload_id": self.workload_id,
            "fixed_plan": self.fixed_plan.to_dict() if self.fixed_plan else None,
            "checks": [check.to_dict() for check in self.checks],
            "episodes_trained": self.episodes_trained,
            "elapsed_s": round(self.elapsed_s, 1),
            "stopped_because": self.stopped_because,
            "tuning_complete": self.tuning_complete,
            "validation_complete": self.validation_complete,
            "model_path": self.model_path, "model_source": self.model_source,
            "cancelled": self.cancelled, "notes": list(self.notes),
            "decided_at": self.decided_at,
            "headline": self.headline(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HybridDecision":
        plan = data.get("fixed_plan")
        return cls(
            selected=data["selected"], reason=data.get("reason", ""),
            workload_id=data.get("workload_id", ""),
            fixed_plan=FixedPlan(
                ns_green=plan["ns_green"], ew_green=plan["ew_green"],
                yellow=plan["yellow"], all_red=plan["all_red"],
                source=plan.get("source", ""),
                mean_waiting_time=plan.get("mean_waiting_time"),
                measured_on=list(plan.get("measured_on") or []),
            ) if plan else None,
            checks=[ValidationCheck(**check) for check in data.get("checks", [])],
            episodes_trained=int(data.get("episodes_trained", 0)),
            elapsed_s=float(data.get("elapsed_s", 0.0)),
            stopped_because=data.get("stopped_because", ""),
            tuning_complete=bool(data.get("tuning_complete", False)),
            validation_complete=bool(data.get("validation_complete", False)),
            model_path=data.get("model_path", ""),
            model_source=data.get("model_source", ""),
            cancelled=bool(data.get("cancelled", False)),
            notes=list(data.get("notes") or []),
            decided_at=data.get("decided_at", ""),
        )


# ===========================================================================
# Measurement - the one place a run is turned into numbers
# ===========================================================================
def _accounting(summary: Dict[str, Any], requested: int, controller: str,
                seed: int) -> RunMetrics:
    """Turn a MetricsCollector summary into the five vehicle counts.

    A congested run is only readable if these are kept apart. "200 requested,
    150 entered, 120 completed, 30 on the road, 50 still waiting to enter" is
    a sentence; "throughput 120" on its own is misleading.
    """
    entered = int(summary.get("vehicles_departed", summary.get("vehicles", 0)))
    completed = int(summary.get("throughput", 0))
    return RunMetrics(
        controller=controller,
        seed=int(seed),
        requested=int(requested),
        entered=entered,
        completed=completed,
        still_on_road=max(entered - completed, 0),
        waiting_to_enter=max(int(requested) - entered, 0),
        mean_waiting_time=float(summary.get("average_waiting_time", 0.0)),
        mean_queue_length=float(summary.get("average_queue_length", 0.0)),
        max_queue_length=float(summary.get("max_queue_length", 0.0)),
        teleports=int(summary.get("teleports", 0)),
    )


def run_fixed_plan(workload: ResolvedWorkload, plan: FixedPlan, seed: int,
                   tag: str = "_hybfix") -> RunMetrics:
    """One fixed-time run of this workload, through SUMO's own TLS program."""
    prepare_sumo_imports()
    import traci

    scoped = "{}_{}".format(tag, seed)
    files = build_workload(workload, tag=scoped, signal=plan.signal())
    collector = MetricsCollector(step_length=workload.step_length)
    steps = int(round(workload.duration_s / workload.step_length))
    label = "hybrid_fixed_{}_{}".format(workload.workload_id, seed)

    traci.start([sumo_binary(False), "-c", files["sumocfg"],
                 "--seed", str(int(seed)), "--no-warnings", "--no-step-log"],
                label=label)
    try:
        for _ in range(steps):
            traci.simulationStep()
            collector.collect_step()
    finally:
        try:
            traci.close()
        except Exception:
            pass
        cleanup_workload_inputs(workload, tag=scoped)

    return _accounting(collector.summary(workload.name),
                       workload.total_vehicles, "fixed_time", seed)


def run_dqn_policy(workload: ResolvedWorkload, agent, seed: int,
                   tag: str = "_hybdqn", profile=None) -> RunMetrics:
    """One greedy run of a phase-duration policy on this workload."""
    from rl.phase_environment import PhaseDurationEnv
    from rl.phase_state import extractor_from_contract

    env = PhaseDurationEnv(
        workload=workload, durations=agent.durations,
        profile=getattr(profile, "name", profile),
        state_extractor=extractor_from_contract(agent.observation_contract),
        tag="{}_{}".format(tag, seed))
    try:
        observation, _ = env.reset(seed=int(seed))
        collector = MetricsCollector(step_length=env.step_length,
                                     conn=env.connection)
        env.step_callback = collector.collect_step
        while True:
            action = agent.act(observation, training=False)
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        summary = collector.summary(workload.name)
    finally:
        env.close()
        env.cleanup_generated_inputs()

    return _accounting(summary, workload.total_vehicles, "dqn", seed)


# ===========================================================================
# The decision rule, on its own so it can be tested without SUMO
# ===========================================================================
def judge(dqn: Sequence[RunMetrics], fixed: Sequence[RunMetrics]
          ) -> Tuple[bool, str]:
    """Does the DQN qualify against the fixed plan on matched runs?

    Returns (qualified, reason). The reason is written for a reader, because
    it is what the Comparison page shows.
    """
    if not dqn or not fixed:
        return False, "No completed comparison, so nothing was selected on."
    if len(dqn) != len(fixed):
        return False, "The two controllers were not run on the same seeds."

    dqn_wait = statistics.fmean(m.mean_waiting_time for m in dqn)
    fixed_wait = statistics.fmean(m.mean_waiting_time for m in fixed)
    dqn_thru = statistics.fmean(m.completed for m in dqn)
    fixed_thru = statistics.fmean(m.completed for m in fixed)
    dqn_pending = statistics.fmean(m.waiting_to_enter for m in dqn)
    fixed_pending = statistics.fmean(m.waiting_to_enter for m in fixed)

    if dqn_wait >= fixed_wait:
        return False, (
            "Mean waiting was {:.2f} s with the DQN against {:.2f} s with "
            "fixed time, so fixed time keeps the signal. A tie goes to fixed "
            "time.".format(dqn_wait, fixed_wait))
    if dqn_thru < fixed_thru:
        return False, (
            "The DQN waited less ({:.2f} s against {:.2f} s) but completed "
            "fewer journeys ({:.0f} against {:.0f}), which is not an "
            "improvement.".format(dqn_wait, fixed_wait, dqn_thru, fixed_thru))
    if dqn_pending > fixed_pending:
        return False, (
            "The DQN waited less ({:.2f} s against {:.2f} s) but left more "
            "vehicles waiting to enter ({:.0f} against {:.0f}), so the gain "
            "is only on the vehicles it let in.".format(
                dqn_wait, fixed_wait, dqn_pending, fixed_pending))

    return True, (
        "The DQN waited {:.2f} s against {:.2f} s, completed {:.0f} journeys "
        "against {:.0f}, and left {:.0f} waiting to enter against {:.0f}."
        .format(dqn_wait, fixed_wait, dqn_thru, fixed_thru,
                dqn_pending, fixed_pending))


# ===========================================================================
# Tuning the fixed plan
# ===========================================================================
def tune_fixed_plan(workload: ResolvedWorkload, start: FixedPlan,
                    seed: int, deadline: Optional[float] = None,
                    should_stop: Optional[Callable[[], bool]] = None,
                    greens: Sequence[int] = TUNE_GREENS,
                    on_progress: Optional[Callable[[str], None]] = None,
                    profile=None) -> Tuple[FixedPlan, bool]:
    """Tune north-south and east-west green independently.

    Coordinate descent: hold east-west, sweep north-south, keep the best;
    then hold north-south and sweep east-west. Two passes over a dozen values
    is about twenty-four runs, which fits a five-minute budget for a
    half-hour simulation - and the demand here is asymmetric, so sweeping the
    two together would miss the optimum entirely.

    Returns the best plan ACTUALLY MEASURED and whether the sweep finished.
    """
    # Under a timing profile the sweep may only try greens that profile
    # allows, and the starting plan is pulled onto the menu - otherwise
    # Hybrid's "fallback" could be a plan the profile forbids.
    if profile is not None:
        greens = [int(d) for d in profile.durations]
        start = FixedPlan(
            ns_green=profile.nearest_duration(start.ns_green),
            ew_green=profile.nearest_duration(start.ew_green),
            yellow=start.yellow, all_red=start.all_red,
            source="configured, pulled onto the {} menu".format(profile.name))

    best = FixedPlan(ns_green=start.ns_green, ew_green=start.ew_green,
                     yellow=start.yellow, all_red=start.all_red,
                     source="tuned here", measured_on=[seed])
    best.mean_waiting_time = run_fixed_plan(workload, best, seed).mean_waiting_time
    if on_progress:
        on_progress("baseline {} -> {:.2f} s".format(best.label(),
                                                     best.mean_waiting_time))

    def out_of_time() -> bool:
        if should_stop is not None and should_stop():
            return True
        return deadline is not None and time.time() >= deadline

    complete = True
    for axis in ("ns_green", "ew_green"):
        for green in greens:
            if out_of_time():
                return best, False
            if getattr(best, axis) == green:
                continue
            trial = FixedPlan(ns_green=best.ns_green, ew_green=best.ew_green,
                              yellow=best.yellow, all_red=best.all_red,
                              source="tuned here", measured_on=[seed])
            setattr(trial, axis, float(green))
            trial.mean_waiting_time = run_fixed_plan(
                workload, trial, seed).mean_waiting_time
            if on_progress:
                on_progress("{} -> {:.2f} s".format(trial.label(),
                                                    trial.mean_waiting_time))
            if trial.mean_waiting_time < best.mean_waiting_time:
                best = trial
    return best, complete


# ===========================================================================
# Remembering a decision
# ===========================================================================
def decision_path(workload_id: str) -> str:
    return os.path.join(HYBRID_DIR, workload_id, "decision.json")


def save_decision(decision: HybridDecision) -> str:
    import json
    path = decision_path(decision.workload_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(decision.to_dict(), handle, indent=2)
    return path


def load_decision(workload_id: str) -> Optional[HybridDecision]:
    """A previous decision for this exact workload, if one was saved.

    Reused only when the workload id matches, and that id is a hash of every
    vehicle, departure time, route, speed, seed and signal setting - so an
    edited workload is never answered with the previous workload's result.
    """
    import json
    path = decision_path(workload_id)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return HybridDecision.from_dict(json.load(handle))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def configured_plan_for(workload: ResolvedWorkload) -> FixedPlan:
    """The plan traffic starts moving under, before anything is tuned."""
    signal = workload.signal
    return FixedPlan(
        ns_green=float(signal.get("ns_green", 30.0)),
        ew_green=float(signal.get("ew_green", 30.0)),
        yellow=float(signal.get("yellow", 4.0)),
        all_red=float(signal.get("all_red", 0.0)),
        source="configured")
