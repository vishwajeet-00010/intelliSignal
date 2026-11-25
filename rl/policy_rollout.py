"""
IntelliSignal
rl/policy_rollout.py

Running a trained policy in the environment its action space belongs to.

There are two kinds of trained policy in this project, and they are not
interchangeable:

    duration menu   one decision per green: pick the whole green from a
                    menu of durations (`rl/phase_environment.py`)
    lock-and-extend serve the minimum green, then decide every few seconds
                    whether to extend it or end it (`rl/phase_extend_env.py`)

Both are stored as a `PhaseDQNAgent`, so nothing in the type stops a
lock-and-extend network being handed to the duration-menu environment. It
runs without an error there, too: its two outputs are read as "menu entry
0" and "menu entry 1", and the signal alternates between the two shortest
greens on the menu whatever the traffic is doing. That is exactly what the
dashboard and the Hybrid controller did with the promoted standard-profile
model until this module existed, while the evaluation harness - which chose
the environment by hand - reported the real numbers of that model.

So the choice is made here, once, from what the checkpoint says it is, and
every caller that wants "this model, on this traffic" goes through
`greedy_rollout`.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl.phase_state import extractor_from_contract
from rl.timing_profile import DEFAULT_PROFILE, PROFILES, get_profile
from utils.metrics import MetricsCollector

ACTION_SPACE_MENU = "duration_menu"
ACTION_SPACE_EXTEND = "lock_extend"

PHASE_LABELS = {0: "North-South", 1: "East-West"}


class PolicyMismatchError(RuntimeError):
    """A policy was asked to run somewhere its action space does not fit."""


@dataclass
class Rollout:
    """One greedy episode: what was measured and what the policy chose."""

    action_space: str
    profile: Optional[str]
    seed: int
    summary: Dict[str, Any] = field(default_factory=dict)
    timeseries: List[Dict[str, Any]] = field(default_factory=list)
    greens: List[Dict[str, Any]] = field(default_factory=list)
    compliance: Dict[str, Any] = field(default_factory=dict)
    decision_log: List[Dict[str, Any]] = field(default_factory=list)
    #: The live collector, for callers that derive more than the summary.
    collector: Any = None

    @property
    def served_greens(self) -> List[float]:
        """Every green, including one the end of the run cut short."""
        return [float(row["green_seconds"]) for row in self.greens]

    @property
    def completed_greens(self) -> List[float]:
        """Only the greens the controller itself chose to end.

        The last green of a run is usually stopped by the clock rather
        than by the controller, so its length says nothing about what
        the controller would have done.
        """
        return [float(row["green_seconds"]) for row in self.greens
                if row.get("completed", True)]


# ---------------------------------------------------------------------------
def action_space_of(agent) -> str:
    """Which of the two action spaces this policy was trained in."""
    declared = str(getattr(agent, "action_space", "") or "")
    if declared in (ACTION_SPACE_MENU, ACTION_SPACE_EXTEND):
        return declared
    extra = getattr(agent, "extra", None) or {}
    if str(extra.get("candidate", "")).upper() == "B":
        return ACTION_SPACE_EXTEND
    if "candidate:B" in str(getattr(agent, "scenario", "")):
        return ACTION_SPACE_EXTEND
    return ACTION_SPACE_MENU


def extension_of(agent, default: float = 3.0) -> float:
    """Seconds one EXTEND adds, for a lock-and-extend policy."""
    declared = float(getattr(agent, "extension", 0.0) or 0.0)
    if declared > 0:
        return declared
    extra = getattr(agent, "extra", None) or {}
    try:
        value = float(extra.get("extension", 0.0) or 0.0)
    except (TypeError, ValueError):
        value = 0.0
    if value > 0:
        return value
    # Not inferred from the checkpoint's durations: for a lock-and-extend
    # model those are the profile's menu, whose spacing has nothing to do
    # with the extension.
    return float(default)


def ladder_of(agent, profile=None) -> List[float]:
    """Every green a lock-and-extend policy can actually produce."""
    from rl.phase_extend_env import extension_ladder

    chosen = profile_of(agent, profile)
    return [float(g) for g in extension_ladder(chosen, extension_of(agent))]


def profile_of(agent, profile=None):
    """The timing profile to run this policy under.

    An explicit profile wins. Without one, a lock-and-extend policy is run
    under the profile it was trained for - it cannot be run without one,
    because the lock IS the minimum green of the profile. The trainer
    records the profile beside the weights; an older checkpoint is matched
    by the limits of the menu it stores.
    """
    if profile is not None:
        return get_profile(getattr(profile, "name", profile))
    if action_space_of(agent) != ACTION_SPACE_EXTEND:
        return None
    extra = getattr(agent, "extra", None) or {}
    recorded = (extra.get("profile") or {}).get("name") \
        if isinstance(extra.get("profile"), dict) else extra.get("profile")
    if recorded in PROFILES:
        return PROFILES[recorded]
    greens = [float(d) for d in getattr(agent, "durations", [])]
    if greens:
        for candidate in PROFILES.values():
            if (abs(candidate.min_green - min(greens)) < 1e-6
                    and abs(candidate.max_green - max(greens)) < 1e-6):
                return candidate
    return get_profile(DEFAULT_PROFILE)


def describe(agent) -> Dict[str, Any]:
    """What kind of controller this checkpoint is, for a reader."""
    kind = action_space_of(agent)
    greens = [float(d) for d in getattr(agent, "durations", [])]
    if kind == ACTION_SPACE_EXTEND:
        step = extension_of(agent)
        greens = ladder_of(agent)
        text = ("Lock-and-extend: every green runs for at least {:g} s, then "
                "the network decides every {:g} s whether to extend it or end "
                "it, up to {:g} s.".format(greens[0], step, greens[-1]))
    else:
        step = 0.0
        menu = (", ".join("{:g}".format(g) for g in greens) + " s"
                if greens else "its menu")
        text = ("Duration menu: at the start of every green the network picks "
                "its whole length from {}.".format(menu))
    return {"action_space": kind, "extension": step, "greens": greens,
            "summary": text}


# ---------------------------------------------------------------------------
def make_env(workload, agent, profile=None, tag: str = "",
             tripinfo_path: Optional[str] = None, extractor=None):
    """The environment this policy's actions mean something in."""
    kind = action_space_of(agent)
    chosen = profile_of(agent, profile)
    extractor = extractor or extractor_from_contract(
        getattr(agent, "observation_contract", None))

    if kind == ACTION_SPACE_EXTEND:
        from rl.phase_extend_env import PhaseExtendEnv, extension_ladder

        extension = extension_of(agent)
        ladder = extension_ladder(chosen, extension)
        # The checkpoint stores the menu of the profile it was trained
        # under. Its limits must be this profile's limits: the lock is the
        # minimum green, so a model trained with a 4 s lock is a different
        # controller under a 15 s one, not a more cautious version of it.
        trained = [float(d) for d in agent.durations]
        if trained and (abs(min(trained) - ladder[0]) > 1e-6
                        or abs(max(trained) - ladder[-1]) > 1e-6):
            raise PolicyMismatchError(
                "This lock-and-extend model was trained for greens of {:g}-"
                "{:g} s, but the {!r} profile allows {:g}-{:g} s.".format(
                    min(trained), max(trained), chosen.name,
                    ladder[0], ladder[-1]))
        if int(agent.num_actions) != 2:
            raise PolicyMismatchError(
                "A lock-and-extend model has two outputs (extend, end); "
                "this checkpoint has {}.".format(agent.num_actions))
        return PhaseExtendEnv(
            workload=workload, profile=chosen.name, extension=extension,
            state_extractor=extractor, tripinfo_path=tripinfo_path, tag=tag)

    from rl.phase_environment import PhaseDurationEnv

    if int(agent.num_actions) != len(agent.durations):
        raise PolicyMismatchError(
            "A duration-menu model needs one output per menu entry; this "
            "checkpoint has {} outputs for {} durations.".format(
                agent.num_actions, len(agent.durations)))
    if chosen is not None:
        # Under a profile the environment uses the PROFILE's menu, so a
        # checkpoint with any other menu would have its outputs read as
        # different greens from the ones it learned.
        menu = tuple(float(d) for d in chosen.durations)
        trained = tuple(float(d) for d in agent.durations)
        if menu != trained:
            raise PolicyMismatchError(
                "This model chooses from greens of {} s, but the {!r} "
                "profile's menu is {} s.".format(
                    list(trained), chosen.name, list(menu)))
    return PhaseDurationEnv(
        workload=workload, durations=agent.durations,
        profile=chosen.name if chosen is not None else None,
        state_extractor=extractor, tripinfo_path=tripinfo_path, tag=tag)


# ---------------------------------------------------------------------------
def greedy_rollout(workload, agent, seed: int, profile=None,
                   tag: str = "_roll", tripinfo_path: Optional[str] = None,
                   extractor=None, observer=None) -> Rollout:
    """Run the policy greedily for one episode and measure it.

    Returns the `MetricsCollector` summary and per-second series, the
    timing-compliance report, and one row per green the policy served -
    with the network's own values at the moment it committed, so the
    dashboard can show why as well as what.

    `observer`, if given, is attached to the connection once SUMO is up
    and told about every simulation step (`attach(connection)`,
    `on_step()`). It only watches; it cannot change the run.
    """
    kind = action_space_of(agent)
    env = make_env(workload, agent, profile=profile, tag=tag,
                   tripinfo_path=tripinfo_path, extractor=extractor)
    chosen = getattr(env, "profile", None)
    greens: List[Dict[str, Any]] = []
    compliance: Dict[str, Any] = {}
    decision_log: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}
    timeseries: List[Dict[str, Any]] = []
    # Attached BEFORE reset, because reset advances the clock: the
    # lock-and-extend environment serves its first minimum green there
    # so that the first question is asked at a real decision point.
    # Attaching afterwards silently lost those fifteen seconds from
    # every measurement and from the recording the audit reads.
    collector = MetricsCollector(step_length=env.step_length)

    def attach(connection):
        collector.conn = connection
        if observer is not None:
            observer.attach(connection)

    def step_callback():
        collector.collect_step()
        if observer is not None:
            observer.on_step()

    env.on_connect = attach
    env.step_callback = step_callback
    try:
        observation, _ = env.reset(seed=int(seed))
        decisions_in_green = 0
        while True:
            phase = env.controller.phase
            values = [float(v) for v in agent.q_values(observation)]
            if kind == ACTION_SPACE_EXTEND:
                action = int(agent.act(observation, training=False,
                                       valid_actions=env.valid_actions()))
            else:
                action = int(agent.act(observation, training=False))
            observation, _, terminated, truncated, info = env.step(action)
            decisions_in_green += 1

            if kind == ACTION_SPACE_EXTEND:
                if info.get("phase_ended"):
                    cut_off = bool(info.get("truncated_by_episode_end"))
                    greens.append({
                        "time": float(info["simulation_time"]),
                        "phase": PHASE_LABELS[phase],
                        "green_seconds": float(info["green_served"]),
                        "decisions": decisions_in_green,
                        "extensions": max(decisions_in_green - 1, 0),
                        "forced_end": bool(info.get("forced_end_at_max_green")),
                        "completed": not cut_off,
                        "q_values": {"extend": round(values[0], 4),
                                     "end": round(values[1], 4)},
                    })
                    decisions_in_green = 0
            else:
                greens.append({
                    "time": float(info["simulation_time"]),
                    "phase": PHASE_LABELS[phase],
                    "green_seconds": float(info["green_served"]),
                    "requested_seconds": float(info.get(
                        "requested_green", info["green_served"])),
                    "intervention": bool(info.get("intervention")),
                    "completed": not bool(info.get(
                        "truncated_by_episode_end")),
                    "q_values": {"{:g}s".format(d): round(v, 4)
                                 for d, v in zip(agent.durations, values)},
                })
                decisions_in_green = 0

            if terminated or truncated:
                compliance = dict(info.get("timing_compliance") or {})
                decision_log = list(info.get("decision_log") or [])
                break
        summary = collector.summary(workload.name)
        timeseries = collector.timeseries
    finally:
        env.close()
        env.cleanup_generated_inputs()

    summary["timing_compliance"] = compliance
    profile_name = getattr(chosen, "name", chosen) if chosen is not None else None
    return Rollout(
        action_space=kind, profile=profile_name, seed=int(seed),
        summary=summary, timeseries=timeseries, greens=greens,
        compliance=compliance, decision_log=decision_log,
        collector=collector)
