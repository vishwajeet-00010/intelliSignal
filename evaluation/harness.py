"""
IntelliSignal
evaluation/harness.py

One place where a controller is turned into numbers.

Every controller in a comparison - fixed-time, actuated, pressure-based or
learned - goes through this module, on the same workload, under the same
timing profile, with the same clearance, the same horizon and the same two
measurements. That is what makes the comparison fair by construction
rather than by care.

    run_fixed        a fixed plan, ported onto the profile's menu
    run_actuated     minimum green, then extend while traffic keeps coming
    run_max_pressure switch when the waiting direction is under more
                     pressure than the one being served
    run_policy       a trained duration-choosing network

Each returns a `ControllerRun` carrying:

  * the historical `MetricsCollector` summary, unchanged, so old numbers
    stay comparable;
  * the versioned delay accounting from SUMO's own trip information,
    which charges insertion delay and therefore cannot be improved by
    refusing vehicles;
  * the timing-compliance report - what greens actually ran, whether any
    red limit was breached, and how many phases were cut off by the end
    of the episode rather than completed;
  * a cross-check between the two measurements.

A note on the actuated and pressure controllers: they are here as honest
benchmarks, not as the product. Both are constrained by the same profile
as everything else, so neither is weakened to flatter the learned
controller and neither is given freedom the others do not have.
"""

from __future__ import annotations

import os
import statistics
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from custom_traffic.builder import build_workload, cleanup_workload_inputs
from custom_traffic.workload import ResolvedWorkload
from rl.phase_action import PHASE_EW, PHASE_NS
from rl.phase_environment import PhaseDurationEnv
from rl.phase_state import PhaseStateExtractor
from rl.timing_profile import TimingProfile, get_profile
from simulation.run_simulation import prepare_sumo_imports, sumo_binary
from utils.delay_accounting import (account, cross_check, enable_tripinfo,
                                    read_tripinfo)
from utils.metrics import MetricsCollector
from utils.sumo_launch import start_sumo

TRIPINFO_DIR = os.path.join(PROJECT_ROOT, "results", "tripinfo")

#: SUMO's trip-information file is read once, straight after the run that
#: wrote it, and never again - so it is deleted once it has been read.
#: Kept, they piled up to three hundred files and 39 MB of by-product. Set
#: this to True to inspect one.
KEEP_TRIPINFO = False

#: Every generated SUMO input is named after the workload and the tag, so
#: two processes evaluating the SAME workload on the SAME seed - which is
#: exactly what parallel training runs do at validation time - would write
#: over each other's files between build and SUMO reading them. The
#: process id makes every name unique without changing anything measured.
WORKER = "p{}".format(os.getpid())

#: The lanes of each direction, for the actuated and pressure controllers.
DIRECTION_LANES = {
    PHASE_NS: ["N2C_0", "N2C_1", "S2C_0", "S2C_1"],
    PHASE_EW: ["E2C_0", "E2C_1", "W2C_0", "W2C_1"],
}


@dataclass
class ControllerRun:
    """One controller, one workload, one seed - measured both ways."""

    controller: str
    profile: str
    seed: int
    summary: Dict[str, Any] = field(default_factory=dict)
    delay: Dict[str, Any] = field(default_factory=dict)
    compliance: Dict[str, Any] = field(default_factory=dict)
    crosscheck: Dict[str, Any] = field(default_factory=dict)
    greens: List[float] = field(default_factory=list)
    detail: Dict[str, Any] = field(default_factory=dict)
    #: The collector's per-second rows. Not part of `to_dict()`: it is for
    #: a caller that wants to draw the run, not for a results file.
    timeseries: List[Dict[str, Any]] = field(default_factory=list)

    # -- the four numbers a comparison leads with ----------------------
    @property
    def mean_waiting(self) -> float:
        """The historical metric, unchanged."""
        return float(self.summary.get("average_waiting_time", 0.0))

    @property
    def mean_total_delay(self) -> float:
        """Per REQUESTED vehicle, insertion delay included."""
        return float(self.delay.get("mean_total_delay_per_requested", 0.0))

    @property
    def completed(self) -> int:
        return int(self.delay.get("completed", self.summary.get("throughput", 0)))

    @property
    def waiting_to_enter(self) -> int:
        return int(self.delay.get("waiting_to_enter", 0))

    @property
    def compliant(self) -> Optional[bool]:
        """True, False, or None when nothing checked.

        None is not True. A run with no compliance report has not been
        shown to obey the timing rules, and saying it has would be the
        one kind of mistake this module exists to prevent.
        """
        if not self.compliance:
            return None
        return bool(self.compliance.get("compliant"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "controller": self.controller, "profile": self.profile,
            "seed": self.seed, "summary": self.summary, "delay": self.delay,
            "compliance": self.compliance, "crosscheck": self.crosscheck,
            "mean_green": (statistics.fmean(self.greens) if self.greens else 0.0),
            "detail": self.detail,
        }


# ---------------------------------------------------------------------------
def _tripinfo_path(workload: ResolvedWorkload, controller: str,
                   seed: int) -> str:
    os.makedirs(TRIPINFO_DIR, exist_ok=True)
    return os.path.join(TRIPINFO_DIR, "{}_{}_{}_{}.xml".format(
        workload.workload_id, controller.replace(" ", "_").replace("/", "-"),
        seed, WORKER))


def _finish(workload: ResolvedWorkload, controller: str, profile: str,
            seed: int, collector: MetricsCollector, tripinfo: str,
            greens: List[float], compliance: Dict[str, Any],
            detail: Dict[str, Any] = None) -> ControllerRun:
    summary = collector.summary(workload.name)
    trips = read_tripinfo(tripinfo)
    if not KEEP_TRIPINFO:
        try:
            os.remove(tripinfo)
        except OSError:
            pass
    still = max(int(summary.get("vehicles", 0)) - int(summary.get("throughput", 0)), 0)
    delay = account(trips,
                    scheduled_departures=[v.depart for v in workload.vehicles],
                    run_end_s=workload.duration_s, still_on_network=still)
    return ControllerRun(
        controller=controller, profile=profile, seed=int(seed),
        summary=summary, delay=delay, compliance=compliance or {},
        crosscheck=cross_check(delay, summary), greens=greens,
        detail=detail or {}, timeseries=collector.timeseries)


def _watch(collector: MetricsCollector, observer):
    """Wire a collector, and an observer if there is one, to an environment.

    Returns the two hooks an environment takes: `on_connect`, called the
    moment SUMO is up and before any second is simulated, and the
    per-step callback. Both are set BEFORE `reset`, because `reset` can
    advance the clock - the lock-and-extend environment serves its first
    minimum green there - and anything attached afterwards would miss
    those seconds in every measurement and in the recording the audit
    reads.

    An observer only reads, so a run measures the same with or without
    one.
    """
    def on_connect(connection):
        collector.conn = connection
        if observer is not None:
            observer.attach(connection)

    def step():
        collector.collect_step()
        if observer is not None:
            observer.on_step()

    return on_connect, step


# ===========================================================================
# Controllers that choose a duration once per green - run through the
# environment, so the profile is enforced on exactly the same code path
# that training and the dashboard use.
# ===========================================================================
def run_policy(workload: ResolvedWorkload, policy: Callable, profile,
               seed: int, controller: str = "policy",
               extractor: Optional[PhaseStateExtractor] = None,
               tag: str = "_h", observer=None) -> ControllerRun:
    """Run anything that picks a menu index from the observation."""
    profile = get_profile(profile)
    tripinfo = _tripinfo_path(workload, controller, seed)
    env = PhaseDurationEnv(workload=workload, profile=profile.name,
                           state_extractor=extractor or PhaseStateExtractor(),
                           tripinfo_path=tripinfo,
                           tag="{}{}_{}".format(tag, seed, WORKER))
    greens: List[float] = []
    compliance: Dict[str, Any] = {}
    collector = MetricsCollector(step_length=env.step_length)
    env.on_connect, env.step_callback = _watch(collector, observer)
    try:
        observation, _ = env.reset(seed=int(seed))
        while True:
            action = int(policy(observation, env))
            observation, _, terminated, truncated, info = env.step(action)
            # The green the end of the episode cut short was not chosen by
            # the controller, so it is left out of what is reported about
            # the controller - exactly as the compliance report does.
            if not info.get("truncated_by_episode_end"):
                greens.append(float(info["green_served"]))
            if terminated or truncated:
                compliance = info.get("timing_compliance", {})
                break
    finally:
        env.close()
        env.cleanup_generated_inputs()
    return _finish(workload, controller, profile.name, seed, collector,
                   tripinfo, greens, compliance)


def run_extend_policy(workload: ResolvedWorkload, agent, profile,
                      seed: int, extension: float = 3.0,
                      controller: str = "candidate B",
                      extractor: Optional[PhaseStateExtractor] = None,
                      tag: str = "_hx", observer=None) -> ControllerRun:
    """Run a lock-and-extend policy, measured exactly like every other one."""
    from rl.phase_extend_env import PhaseExtendEnv

    profile = get_profile(profile)
    tripinfo = _tripinfo_path(workload, controller, seed)
    env = PhaseExtendEnv(workload=workload, profile=profile.name,
                         extension=extension,
                         state_extractor=extractor or PhaseStateExtractor(),
                         tripinfo_path=tripinfo,
                         tag="{}{}_{}".format(tag, seed, WORKER))
    greens: List[float] = []
    compliance: Dict[str, Any] = {}
    collector = MetricsCollector(step_length=env.step_length)
    env.on_connect, env.step_callback = _watch(collector, observer)
    try:
        observation, _ = env.reset(seed=int(seed))
        while True:
            action = int(agent.act(observation, training=False,
                                   valid_actions=env.valid_actions()))
            observation, _, terminated, truncated, info = env.step(action)
            if info.get("phase_ended") and not info.get(
                    "truncated_by_episode_end"):
                greens.append(float(info["green_served"]))
            if terminated or truncated:
                compliance = info.get("timing_compliance", {})
                break
    finally:
        env.close()
        env.cleanup_generated_inputs()
    return _finish(workload, controller, profile.name, seed, collector,
                   tripinfo, greens, compliance, {"extension": extension})


def run_agent(workload: ResolvedWorkload, agent, profile, seed: int,
              controller: Optional[str] = None,
              tag: str = "_ha", observer=None) -> ControllerRun:
    """Run a trained model in whichever environment it was trained for.

    The one entry point for "measure this checkpoint": the action space is
    read from the checkpoint, never chosen by the caller. The run's
    `detail["decisions"]` holds one row per green with the network's own
    values at the moment it committed.
    """
    from rl.policy_rollout import (ACTION_SPACE_EXTEND, extension_of,
                                   greedy_rollout)

    profile = get_profile(profile)
    name = controller or "DQN"
    tripinfo = _tripinfo_path(workload, name, seed)
    rollout = greedy_rollout(
        workload, agent, int(seed), profile=profile,
        tag="{}{}_{}".format(tag, seed, WORKER), tripinfo_path=tripinfo,
        observer=observer)
    detail: Dict[str, Any] = {"action_space": rollout.action_space,
                              "decisions": rollout.greens,
                              "decision_log": rollout.decision_log}
    if rollout.action_space == ACTION_SPACE_EXTEND:
        detail["extension"] = extension_of(agent)
    return _finish(workload, name, profile.name, seed, rollout.collector,
                   tripinfo, rollout.completed_greens, rollout.compliance,
                   detail)


def fixed_policy(profile: TimingProfile, ns_green: float, ew_green: float):
    """A fixed plan expressed on the profile's menu."""
    ns_index = profile.durations.index(profile.nearest_duration(ns_green))
    ew_index = profile.durations.index(profile.nearest_duration(ew_green))

    def policy(observation, env):
        return ns_index if env.controller.phase == PHASE_NS else ew_index
    return policy


def max_pressure_policy(profile: TimingProfile):
    """Serve the current direction while it is the one under pressure.

    Pressure for a movement is upstream queue minus downstream queue. On
    this network every downstream edge is an exit with no signal, so the
    downstream term is zero and pressure reduces to the upstream queue.
    That simplification is why this is a benchmark rather than a claim
    about max-pressure control in general.

    The green is the shortest menu entry that is at least as long as the
    served direction needs to clear its share of the pressure, bounded by
    the profile.
    """
    def policy(observation, env):
        raw = env.state_extractor.raw(env.connection)
        phase = env.controller.phase
        served = raw["queues"][phase]
        other = raw["queues"][1 - phase]
        total = served + other
        if total <= 0:
            return 0
        share = served / total
        span = profile.max_green - profile.min_green
        want = profile.min_green + share * span
        return profile.durations.index(profile.nearest_duration(want))
    return policy


def greedy_agent_policy(agent):
    def policy(observation, env):
        return int(agent.act(observation, training=False))
    return policy


# ===========================================================================
# Actuated control, which has to decide second by second
# ===========================================================================
def run_actuated(workload: ResolvedWorkload, profile, seed: int,
                 gap_seconds: float = 3.0, tag: str = "_act",
                 observer=None) -> ControllerRun:
    """Minimum green, then extend while vehicles keep arriving.

    The classic vehicle-actuated rule, bounded by the profile: serve
    `min_green`, then extend one second at a time while traffic is still
    present on the served approaches, and gap out after `gap_seconds` with
    nothing there - or at `max_green`, whichever comes first.

    It is run here rather than through the environment because a genuine
    gap-out cannot be expressed as a duration chosen in advance. Every
    other part - the clearance, the horizon, the measurement - is
    identical.
    """
    from utils.scenario_builder import (ALL_RED, EW_GREEN, EW_YELLOW,
                                        NS_GREEN, NS_YELLOW, TLS_ID)

    profile = get_profile(profile)
    prepare_sumo_imports()
    import traci

    green_state = {PHASE_NS: NS_GREEN, PHASE_EW: EW_GREEN}
    yellow_state = {PHASE_NS: NS_YELLOW, PHASE_EW: EW_YELLOW}

    scoped = "{}{}_{}".format(tag, seed, WORKER)
    files = build_workload(workload, tag=scoped)
    tripinfo = _tripinfo_path(workload, "actuated", seed)
    enable_tripinfo(files["sumocfg"], tripinfo)

    collector = MetricsCollector(step_length=workload.step_length)
    total_steps = int(round(workload.duration_s / workload.step_length))
    yellow = float(workload.signal.get("yellow", 4.0))
    all_red = float(workload.signal.get("all_red", 0.0))

    phase = PHASE_NS
    steps_done = 0
    greens: List[float] = []
    log: List[Dict[str, Any]] = []
    last_green = {PHASE_NS: 0.0, PHASE_EW: 0.0}
    breaches = 0

    label = "actuated_{}_{}_{}".format(workload.workload_id, seed, WORKER)
    connection = start_sumo(
        [sumo_binary(False), "-c", files["sumocfg"], "--seed",
         str(int(seed)), "--no-warnings", "--no-step-log"], label=label)
    collector.conn = connection
    if observer is not None:
        observer.attach(connection)
    _, step = _watch(collector, observer)
    try:
        while steps_done < total_steps:
            # The red limit caps how long this green may run, because the
            # waiting group's red is this green plus the clearance.
            waiting = 1 - phase
            previous = last_green[waiting]
            red_cap = (profile.max_red_seconds(previous) if previous > 0
                       else profile.initial_max_red_seconds())
            ceiling = min(profile.max_green, max(profile.min_green,
                                                 red_cap - yellow - all_red))

            connection.trafficlight.setRedYellowGreenState(
                TLS_ID, green_state[phase])
            served, idle, truncated = 0.0, 0.0, False
            while steps_done < total_steps:
                connection.simulationStep()
                step()
                steps_done += 1
                served += workload.step_length
                if served >= ceiling:
                    break
                present = sum(connection.lane.getLastStepVehicleNumber(lane)
                              for lane in DIRECTION_LANES[phase])
                idle = 0.0 if present > 0 else idle + workload.step_length
                if served >= profile.min_green and idle >= gap_seconds:
                    break
            else:
                truncated = True
            if steps_done >= total_steps and served < ceiling:
                truncated = True

            red_seconds = served + yellow + all_red
            within = red_seconds <= red_cap + 1e-9
            breaches += 0 if within else 1
            if not truncated:
                greens.append(served)
            log.append({"phase": phase, "served_green": served,
                        "red_limit": red_cap, "red_within_limit": within,
                        "truncated_by_episode_end": truncated})
            last_green[phase] = served

            connection.trafficlight.setRedYellowGreenState(
                TLS_ID, yellow_state[phase])
            for _ in range(int(round(yellow / workload.step_length))):
                if steps_done >= total_steps:
                    break
                connection.simulationStep()
                step()
                steps_done += 1
            if all_red > 0:
                connection.trafficlight.setRedYellowGreenState(TLS_ID, ALL_RED)
                for _ in range(int(round(all_red / workload.step_length))):
                    if steps_done >= total_steps:
                        break
                    connection.simulationStep()
                    step()
                    steps_done += 1
            phase = 1 - phase
    finally:
        try:
            connection.close()
        except Exception:
            pass
        cleanup_workload_inputs(workload, tag=scoped)

    completed = [e for e in log if not e["truncated_by_episode_end"]]
    served_greens = [e["served_green"] for e in completed]
    compliance = {
        "profile": profile.name,
        "decisions": len(log),
        "completed_phases": len(completed),
        "truncated_phases": len(log) - len(completed),
        "min_green_seen": min(served_greens) if served_greens else None,
        "max_green_seen": max(served_greens) if served_greens else None,
        "greens_below_min": sum(1 for g in served_greens
                                if g < profile.min_green - 1e-9),
        "greens_above_max": sum(1 for g in served_greens
                                if g > profile.max_green + 1e-9),
        "red_limit_breaches": breaches,
        "interventions": 0,
        "compliant": (breaches == 0
                      and all(profile.min_green - 1e-9 <= g <=
                              profile.max_green + 1e-9 for g in served_greens)),
    }
    return _finish(workload, "actuated (gap {:g}s)".format(gap_seconds),
                   profile.name, seed, collector, tripinfo, greens,
                   compliance, {"gap_seconds": gap_seconds})


# ===========================================================================
def aggregate(runs: Sequence[ControllerRun]) -> Dict[str, Any]:
    """Mean and spread across seeds, for one controller."""
    if not runs:
        return {}
    waiting = [run.mean_waiting for run in runs]
    delay = [run.mean_total_delay for run in runs]
    completed = [run.completed for run in runs]
    pending = [run.waiting_to_enter for run in runs]
    queues = [float(run.summary.get("average_queue_length", 0.0)) for run in runs]
    greens = [g for run in runs for g in run.greens]

    def spread(values):
        return statistics.pstdev(values) if len(values) > 1 else 0.0

    return {
        "controller": runs[0].controller,
        "profile": runs[0].profile,
        "seeds": [run.seed for run in runs],
        "n": len(runs),
        "mean_waiting": round(statistics.fmean(waiting), 3),
        "sd_waiting": round(spread(waiting), 3),
        "mean_total_delay": round(statistics.fmean(delay), 3),
        "sd_total_delay": round(spread(delay), 3),
        "mean_completed": round(statistics.fmean(completed), 1),
        "mean_waiting_to_enter": round(statistics.fmean(pending), 1),
        "mean_queue": round(statistics.fmean(queues), 3),
        "mean_green": round(statistics.fmean(greens), 2) if greens else 0.0,
        "all_compliant": all(run.compliant is True for run in runs),
        "unchecked_runs": sum(1 for run in runs if run.compliant is None),
        "red_breaches": sum(int(run.compliance.get("red_limit_breaches", 0))
                            for run in runs),
        "truncated_phases": sum(int(run.compliance.get("truncated_phases", 0))
                                for run in runs),
    }
