"""
IntelliSignal
simulation/engine.py

One engine behind the whole product.

Everything the dashboard runs - a fixed plan, the actuated benchmark,
max-pressure, the trained model, and the runs Hybrid makes while it
decides - comes through this module, and this module goes through
`evaluation/harness.py`. That is the point: the numbers on the screen and
the numbers in the report are produced by the same code on the same
traffic under the same timing rules, so they cannot drift apart.

Every run carries five things back:

    metrics     the measurements, both ways: the live collector's and
                SUMO's own trip information
    compliance  what the controller says it did
    audit       what the SIGNAL actually showed, checked independently
                (`simulation/recorder.py`), because a controller with a
                bug would report its own bug as compliant
    decisions   one row per green: how long, and what the network thought
    replay      where every vehicle was, second by second

Traffic is always an explicit list of vehicles - this class of vehicle,
in this lane, departing at this second - so two controllers meet exactly
the same traffic and a comparison is a measurement rather than an
estimate. The presets below are Indian arterial demand at four levels;
their seeds are disjoint from the training, validation and test splits,
so demonstrating the product never touches the evidence.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from custom_traffic.fleet import get_fleet, mean_pcu, mean_startup_delay
from custom_traffic.workload import ResolvedWorkload
from rl.timing_profile import DEFAULT_PROFILE, get_profile
from simulation.recorder import ReplayRecorder, audit_signal, green_timeline, \
    signal_segments

RUN_DIR = os.path.join(PROJECT_ROOT, "results", "runs")
RUN_FORMAT = 2

#: The one profile the product operates under. `responsive` still exists in
#: `rl/timing_profile.py` so that historical 4-20 s models can be loaded and
#: explained, but nothing here will run traffic under it.
OPERATING_PROFILE = DEFAULT_PROFILE


# ===========================================================================
# Traffic presets
# ===========================================================================
@dataclass(frozen=True)
class Preset:
    """One ready-made traffic pattern, described in plain words."""

    key: str
    label: str
    shape: str
    intensity: str
    seed: int
    headline: str
    detail: str


#: Seeds are 50000+, disjoint from train (11000), validation (22000) and
#: test (33000), so a demonstration can never be run on the traffic a model
#: was selected or reported on.
PRESETS: Dict[str, Preset] = {
    p.key: p for p in (
        Preset("offpeak", "Off-peak", "balanced", "light", 50001,
               "Light, even traffic on all four arms",
               "Mid-morning on the arterial: about 280 vehicles an hour on "
               "each approach, evenly spread. Any sensible plan copes; the "
               "question is how much time it wastes."),
        Preset("typical", "Typical day", "balanced", "moderate", 50002,
               "Steady traffic, busy but not saturated",
               "The ordinary working day: about 480 vehicles an hour on each "
               "approach. The plan starts to matter."),
        Preset("peak", "Evening peak", "directional", "heavy", 50003,
               "One axis carries three times the other",
               "The evening rush: north-south runs heavy while east-west "
               "stays light. A symmetric plan spends half its green on an "
               "almost empty road."),
        Preset("lane_rush", "Kerb-lane rush", "lane_skew", "heavy", 50004,
               "One lane of each approach carries most of the traffic",
               "Left-turning and parking traffic piles into the kerb lane. "
               "The per-direction totals look fine while one lane blocks."),
        Preset("burst", "Sudden surge", "burst", "moderate", 50005,
               "Quiet, then a heavy pulse through the middle",
               "A procession, a school, a cricket match ending: east-west "
               "demand triples for a third of the run, then stops."),
    )
}
PRESET_ORDER = ["offpeak", "typical", "peak", "lane_rush", "burst"]


def preset_workload(key: str, duration_s: float = 900.0,
                    seed: Optional[int] = None) -> ResolvedWorkload:
    """Build a preset's traffic. Same key and duration give same traffic."""
    from training.workload_library import build

    preset = PRESETS[str(key)]
    return build(preset.shape, preset.intensity, float(duration_s),
                 int(seed if seed is not None else preset.seed))


# ===========================================================================
# Controllers
# ===========================================================================
@dataclass(frozen=True)
class Controller:
    """One way of working the signal, as the dashboard offers it."""

    key: str
    label: str
    kind: str                     # fixed | actuated | pressure | model
    headline: str
    detail: str


CONTROLLERS: Dict[str, Controller] = {
    c.key: c for c in (
        Controller("fixed_time", "Fixed-time", "fixed",
                   "The same plan every cycle",
                   "A fixed green for north-south and another for east-west, "
                   "repeated whatever the traffic does. This is what most "
                   "junctions run, and it is the baseline everything else "
                   "has to beat."),
        Controller("actuated", "Actuated", "actuated",
                   "Extends the green while vehicles keep arriving",
                   "The classic vehicle-actuated rule: serve the minimum "
                   "green, then hold it while traffic is still coming, and "
                   "end it after a three-second gap or at the maximum."),
        Controller("max_pressure", "Max-pressure", "pressure",
                   "Serves whichever direction is under more pressure",
                   "A well-known research baseline. On this junction the "
                   "downstream roads are exits, so pressure reduces to the "
                   "queue waiting on each approach."),
        Controller("dqn", "DQN (trained)", "model",
                   "A trained network decides how long each green runs",
                   "The promoted model for the operating timing rules. It "
                   "sees the queues, the approaching traffic and how long "
                   "each direction has waited, and chooses."),
    )
}
CONTROLLER_ORDER = ["dqn", "fixed_time", "actuated", "max_pressure"]

#: What the Comparison page runs when the user asks for 'everything'.
BENCHMARKS = ["fixed_time", "actuated", "max_pressure"]


# ===========================================================================
# A run
# ===========================================================================
@dataclass
class RunResult:
    """One controller, one traffic pattern, one seed - and all the evidence."""

    controller: str
    label: str
    workload_id: str
    workload_name: str
    seed: int
    profile: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    compliance: Dict[str, Any] = field(default_factory=dict)
    audit: Dict[str, Any] = field(default_factory=dict)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    greens: List[Dict[str, Any]] = field(default_factory=list)
    series: List[Dict[str, Any]] = field(default_factory=list)
    replay: Dict[str, Any] = field(default_factory=dict)
    detail: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now().isoformat(timespec="seconds")

    # -- the numbers the pages lead with -------------------------------
    @property
    def delay(self) -> float:
        return float(self.metrics.get("delay_per_requested", 0.0))

    @property
    def waiting(self) -> float:
        return float(self.metrics.get("waiting_per_vehicle", 0.0))

    @property
    def completed(self) -> int:
        return int(self.metrics.get("completed", 0))

    def matches_model(self, status: Dict[str, Any]) -> Optional[bool]:
        """Was this run produced by the model that is in force now?

        None when the question does not apply: not a model run, or a run
        saved before the fingerprint was recorded.
        """
        if self.controller != "dqn":
            return None
        mine = (self.detail.get("model") or {}).get("members")
        if not mine:
            return None
        record = status.get("record") or {}
        theirs = record.get("members") or [
            {"path": record.get("path", ""),
             "sha256": record.get("sha256", "")}]
        return ([str(m.get("sha256", "")).lower() for m in mine]
                == [str(t.get("sha256", "")).lower() for t in theirs])

    @property
    def stale_metrics(self) -> bool:
        """Measured by an older version of the delay accounting."""
        from utils.delay_accounting import METRICS_VERSION

        recorded = self.detail.get("metrics_version")
        return recorded is not None and int(recorded) != int(METRICS_VERSION)

    @property
    def trustworthy(self) -> bool:
        """Every check that could be made was made, and all of them passed.

        A run nobody audited is not trustworthy, it is unexamined, so the
        absence of an audit counts against it rather than for it.
        """
        return bool(self.audit.get("compliant") is True
                    and self.metrics.get("measurements_agree", True)
                    and not self.metrics.get("collisions", 0))

    def to_dict(self, with_replay: bool = True) -> Dict[str, Any]:
        data = {
            "format": RUN_FORMAT, "controller": self.controller,
            "label": self.label, "workload_id": self.workload_id,
            "workload_name": self.workload_name, "seed": self.seed,
            "profile": self.profile, "metrics": self.metrics,
            "compliance": self.compliance, "audit": self.audit,
            "decisions": self.decisions, "greens": self.greens,
            "series": self.series, "detail": self.detail,
            "created_at": self.created_at,
        }
        if with_replay:
            data["replay"] = self.replay
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunResult":
        return cls(
            controller=data.get("controller", ""),
            label=data.get("label", ""),
            workload_id=data.get("workload_id", ""),
            workload_name=data.get("workload_name", ""),
            seed=int(data.get("seed", 0)),
            profile=data.get("profile", OPERATING_PROFILE),
            metrics=dict(data.get("metrics") or {}),
            compliance=dict(data.get("compliance") or {}),
            audit=dict(data.get("audit") or {}),
            decisions=list(data.get("decisions") or []),
            greens=list(data.get("greens") or []),
            series=list(data.get("series") or []),
            replay=dict(data.get("replay") or {}),
            detail=dict(data.get("detail") or {}),
            created_at=data.get("created_at", ""))


# ---------------------------------------------------------------------------
def _metrics(run, recorder: Optional[ReplayRecorder]) -> Dict[str, Any]:
    """The measurements of one harness run, under one set of names.

    Both measurements are kept and both are reported. `delay_per_requested`
    leads because it charges the time vehicles spend unable to get in, so
    it cannot be improved by refusing traffic; `waiting_per_vehicle` is the
    historical headline and is kept so older numbers stay comparable.
    """
    summary, delay = run.summary, run.delay
    requested = int(delay.get("requested", summary.get("vehicles", 0)))
    entered = int(delay.get("entered", summary.get("vehicles_departed", 0)))
    completed = int(delay.get("completed", summary.get("throughput", 0)))
    return {
        "delay_per_requested": float(delay.get(
            "mean_total_delay_per_requested", 0.0)),
        "time_lost_per_entered": float(delay.get(
            "mean_time_loss_per_entered", 0.0)),
        "waiting_per_vehicle": float(summary.get("average_waiting_time", 0.0)),
        "queue_mean": float(summary.get("average_queue_length", 0.0)),
        "queue_max": float(summary.get("max_queue_length", 0.0)),
        "requested": requested,
        "entered": entered,
        "completed": completed,
        "still_on_road": max(entered - completed, 0),
        "waiting_to_enter": int(delay.get("waiting_to_enter",
                                          max(requested - entered, 0))),
        "completion_rate": (completed / requested) if requested else 0.0,
        "teleports": int(summary.get("teleports", 0)),
        "collisions": int(recorder.collisions) if recorder else 0,
        "mean_green": (sum(run.greens) / len(run.greens)) if run.greens else 0.0,
        "switches": len(run.greens),
        "simulation_seconds": float(summary.get("simulation_time", 0.0)),
        "measurements_agree": bool(run.crosscheck.get("completed_agree", True)),
        "crosscheck": run.crosscheck,
    }


def _series(run, recorder: Optional[ReplayRecorder]) -> List[Dict[str, Any]]:
    """Per-second rows: the collector's view and the per-approach queues."""
    rows = [dict(row) for row in (run.timeseries or [])]
    if recorder is None:
        return rows
    extra = recorder.series()
    for index, row in enumerate(rows):
        if index < len(extra):
            row.update({key: value for key, value in extra[index].items()
                        if key != "time"})
    return rows


# ---------------------------------------------------------------------------
def run_controller(workload: ResolvedWorkload, controller: str,
                   seed: Optional[int] = None, profile: str = OPERATING_PROFILE,
                   plan: Optional[Dict[str, float]] = None,
                   agent=None, record: bool = True,
                   label: Optional[str] = None) -> RunResult:
    """Run one controller on one traffic pattern and bring back everything.

    `plan` is for `fixed_time`: {'ns': seconds, 'ew': seconds}. Without
    one the workload's own configured plan is used, pulled onto the
    profile's menu. `agent` overrides the registered model for `dqn`.
    """
    from evaluation.harness import (fixed_policy, max_pressure_policy,
                                    run_actuated, run_agent, run_policy)

    chosen = get_profile(profile)
    seed = int(seed if seed is not None else workload.seed)
    spec = CONTROLLERS.get(controller)
    if spec is None:
        raise ValueError("Unknown controller {!r}. Use one of {}.".format(
            controller, ", ".join(CONTROLLER_ORDER)))
    recorder = ReplayRecorder(workload) if record else None
    detail: Dict[str, Any] = {}
    name = label or spec.label

    if controller == "fixed_time":
        plan = plan or configured_plan(workload, chosen)
        ns = chosen.nearest_duration(float(plan["ns"]))
        ew = chosen.nearest_duration(float(plan["ew"]))
        detail = {"ns_green": ns, "ew_green": ew,
                  "source": plan.get("source", "configured"),
                  "cycle": ns + ew + 2 * (float(workload.signal.get("yellow", 3.0))
                                          + float(workload.signal.get("all_red", 2.0)))}
        name = label or "Fixed-time ns {:g} s / ew {:g} s".format(ns, ew)
        run = run_policy(workload, fixed_policy(chosen, ns, ew), chosen, seed,
                         controller=name, tag="_ui", observer=recorder)

    elif controller == "actuated":
        run = run_actuated(workload, chosen, seed, gap_seconds=3.0,
                           tag="_ui", observer=recorder)
        detail = {"gap_seconds": 3.0}

    elif controller == "max_pressure":
        run = run_policy(workload, max_pressure_policy(chosen), chosen, seed,
                         controller=name, tag="_ui", observer=recorder)

    else:                                                   # dqn
        from agents.phase_dqn_agent import load_registered_profile_model
        from rl.policy_rollout import describe

        agent = agent if agent is not None else \
            load_registered_profile_model(chosen.name)
        run = run_agent(workload, agent, chosen, seed, controller=name,
                        tag="_ui", observer=recorder)
        detail = dict(run.detail)
        detail.update(describe(agent))
        # Which network produced this, by fingerprint. Without it a run
        # reloaded weeks later says "DQN" and means nothing: a registry
        # entry is a pointer, and pointers move.
        detail["model"] = model_identity(agent)
        detail["model_path"] = ", ".join(detail["model"]["paths"])
        detail.pop("decision_log", None)

    decisions = list(run.detail.get("decisions") or [])
    audit: Dict[str, Any] = {}
    replay: Dict[str, Any] = {}
    greens: List[Dict[str, Any]] = []
    if recorder is not None:
        audit = audit_signal(recorder.signal, chosen,
                             float(workload.signal.get("yellow", 3.0)),
                             float(workload.signal.get("all_red", 2.0)),
                             workload.step_length)
        replay = recorder.replay()
        greens = green_timeline(signal_segments(recorder.signal,
                                                recorder.step_length))

    from utils.delay_accounting import METRICS_VERSION

    detail["metrics_version"] = METRICS_VERSION
    detail["traffic_model"] = workload.traffic_model
    return RunResult(
        controller=controller, label=name,
        workload_id=workload.workload_id, workload_name=workload.name,
        seed=seed, profile=chosen.name,
        metrics=_metrics(run, recorder), compliance=dict(run.compliance),
        audit=audit, decisions=decisions, greens=greens,
        series=_series(run, recorder), replay=replay, detail=detail)


def configured_plan(workload: ResolvedWorkload, profile=None
                    ) -> Dict[str, Any]:
    """The plan the traffic arrives with, pulled onto the profile's menu."""
    chosen = get_profile(profile or OPERATING_PROFILE)
    signal = workload.signal or {}
    return {"ns": chosen.nearest_duration(float(signal.get("ns_green", 30.0))),
            "ew": chosen.nearest_duration(float(signal.get("ew_green", 25.0))),
            "source": "configured for this junction"}


def reference_plan(profile=None) -> Optional[Dict[str, Any]]:
    """The best fixed plan found when the model was selected, if recorded.

    Using it saves the user a sixty-four-run sweep for the usual case,
    and it is the plan the published comparison was made against. It is
    only offered, never silently substituted.
    """
    record = _selection_record()
    if not record:
        return None
    best = record.get("best_fixed") or {}
    if "ns" not in best or "ew" not in best:
        return None
    chosen = get_profile(profile or OPERATING_PROFILE)
    return {"ns": chosen.nearest_duration(float(best["ns"])),
            "ew": chosen.nearest_duration(float(best["ew"])),
            "source": "best of every plan on the menu, measured on the "
                      "validation traffic"}


# ===========================================================================
# What the model is, and whether it can be trusted today
# ===========================================================================
def model_identity(agent) -> Dict[str, Any]:
    """Which checkpoints a policy is, so a saved run can be attributed.

    A registry entry is a pointer and pointers move. Recording the
    fingerprint of the files that actually ran is what lets a result
    from last month be matched to the model that produced it, or shown
    as belonging to a model that is no longer the one in force.
    """
    from agents.phase_dqn_agent import sha256_of

    paths = [p for p in str(getattr(agent, "model_path", "") or "").split(";")
             if p]
    members = []
    for path in paths:
        full = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT,
                                                             path)
        try:
            members.append({"path": _relative(full),
                            "sha256": sha256_of(full)})
        except OSError:
            members.append({"path": _relative(path), "sha256": ""})
    return {
        "paths": [item["path"] for item in members],
        "members": members,
        "action_space": str(getattr(agent, "action_space", "") or ""),
        "networks": len(getattr(agent, "members", []) or [1]),
    }


def _relative(path: str) -> str:
    if not path:
        return ""
    try:
        return os.path.relpath(path, PROJECT_ROOT).replace("\\", "/")
    except ValueError:
        return path


def selection_report(profile: str = OPERATING_PROFILE) -> Dict[str, Any]:
    """The selection run the active model was promoted from, if recorded.

    `evaluation/select_profile_model.py` writes it: every trained run
    re-scored on the validation traffic, every fixed plan on the menu
    measured, and the test split read once at the end. The Model page
    shows it, so the claim on screen and the evidence behind it are the
    same file.
    """
    from agents.phase_dqn_agent import registered_profile_model

    record = registered_profile_model(profile) or {}
    path = record.get("selection_report") or record.get("report")
    if not path:
        return {}
    full = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    try:
        with open(full, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def _selection_record() -> Dict[str, Any]:
    return selection_report(OPERATING_PROFILE)


def model_status(profile: str = OPERATING_PROFILE) -> Dict[str, Any]:
    """Everything that decides whether the DQN may take the signal.

    Each check is reported with its answer, so the dashboard can show why
    the model is or is not in charge instead of just that it is not.
    """
    from agents.phase_dqn_agent import (PhaseModelError,
                                        load_registered_profile_model,
                                        registered_profile_model)

    chosen = get_profile(profile)
    record = registered_profile_model(chosen.name)
    status: Dict[str, Any] = {
        "profile": chosen.name, "registered": bool(record),
        "record": dict(record or {}), "usable": False, "checks": [],
        "reason": "", "agent": None, "describe": {},
    }

    def check(name: str, ok: bool, detail: str) -> None:
        status["checks"].append({"check": name, "ok": bool(ok),
                                 "detail": detail})

    if not record:
        status["reason"] = ("No model has been promoted for the {} timing "
                            "rules, so Hybrid decides instead.".format(
                                chosen.name))
        check("a model is registered", False, status["reason"])
        return status
    check("a model is registered", True,
          "{} ({})".format(record.get("label", "model"),
                           record.get("path", "")))

    try:
        agent = load_registered_profile_model(chosen.name)
    except PhaseModelError as exc:
        status["reason"] = str(exc)
        check("the file matches its recorded fingerprint and the timing "
              "rules", False, str(exc))
        return status
    check("the file matches its recorded fingerprint", True,
          "sha256 as recorded when it was promoted")

    from rl.policy_rollout import describe

    story = describe(agent)
    status["agent"] = agent
    status["describe"] = story
    # The loader refuses a checkpoint whose greens fall outside the
    # profile, so reaching this line IS the check passing - but saying so
    # is better than printing a tick that was written by hand.
    greens = story["greens"]
    inside = bool(greens) and all(chosen.contains_green(g) for g in greens)
    check("its greens are inside {:g}-{:g} s".format(
        chosen.min_green, chosen.max_green), inside,
        "{} It can produce {}.".format(
            story["summary"],
            ", ".join("{:g}".format(g) for g in greens) + " s"))
    if not inside:
        status["reason"] = ("The registered model can produce greens the "
                            "timing rules do not allow.")
        return status

    measurement = record.get("measurement") or {}
    mine = measurement.get("mean_total_delay")
    baseline = measurement.get("baseline_total_delay")
    if mine is None or baseline is None:
        check("it was measured against a tuned fixed plan", False,
              "Its promotion record does not say what it was measured "
              "against, so it is not trusted to take the signal.")
        status["reason"] = ("The promotion record of the registered model "
                            "does not show it beating a tuned fixed plan.")
        return status
    beat = float(mine) < float(baseline)
    check("it beat the tuned fixed plan when promoted", beat,
          "{:.2f} s of delay per requested vehicle against the plan's "
          "{:.2f} s, on the {} traffic it had never been selected on."
          .format(float(mine), float(baseline),
                  measurement.get("split", "test")))
    if not beat:
        status["reason"] = ("The registered model did not beat the tuned "
                            "fixed plan it was measured against.")
        return status
    if str(record.get("selected_on", "")) == "test":
        check("it was selected on traffic it was not reported on", False,
              "The record says it was selected on the test split.")
        status["reason"] = ("The registered model was selected on the split "
                            "it is reported on.")
        return status
    check("it was selected on traffic it was not reported on", True,
          "selected on the {} traffic, reported on the {} traffic".format(
              record.get("selected_on", "validation"),
              measurement.get("split", "test")))

    status["usable"] = True
    status["members"] = len(getattr(agent, "members", []) or [1])
    status["reason"] = "The registered model is valid for these timing rules."
    return status


def environment_status() -> Dict[str, Any]:
    """Is everything the product needs actually installed and present?

    Shown rather than assumed, because the first thing that goes wrong on
    a machine that is not this one is SUMO not being found, and a blank
    page is a worse answer than a sentence saying so.
    """
    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    try:
        from simulation.run_simulation import prepare_sumo_imports
        prepare_sumo_imports()
        import sumolib
        binary = sumolib.checkBinary("sumo")
        check("SUMO is installed", True, binary)
    except BaseException as exc:                       # noqa: BLE001
        check("SUMO is installed", False,
              "{}. Set SUMO_HOME, or pip install eclipse-sumo.".format(exc))

    try:
        import torch
        check("PyTorch is installed", True,
              "version {}".format(torch.__version__))
    except Exception as exc:                           # noqa: BLE001
        check("PyTorch is installed", False, str(exc))

    network = os.path.join(PROJECT_ROOT, "sumo", "network",
                           "intersection.net.xml")
    check("the junction is built", os.path.isfile(network),
          network if os.path.isfile(network)
          else "run python sumo/network/build_network.py")

    return {"ready": all(c["ok"] for c in checks), "checks": checks}


def fleet_summary(traffic_model: str = "indian_urban") -> Dict[str, Any]:
    """The fleet on the road, as rows a table can render."""
    fleet = get_fleet(traffic_model)
    return {
        "traffic_model": traffic_model,
        "mean_pcu": round(mean_pcu(fleet), 3),
        "mean_startup_delay": round(mean_startup_delay(fleet), 3),
        "rows": [{
            "vehicle": v.name, "share": v.share, "length_m": v.length,
            "startup_delay_s": v.startup_delay, "pcu": v.pcu,
            "max_speed_kmh": v.max_speed_kmh,
        } for v in fleet],
    }


def green_value(green_seconds: float, traffic_model: str = "indian_urban"
                ) -> Dict[str, float]:
    """How much of a green of this length actually moves traffic.

    The queue's reaction cascades back from the stop line, so the loss is
    the mean per-vehicle startup delay plus about two seconds for the
    wave to travel - which is why a short green spends nearly all of
    itself starting the queue and almost none of it clearing it.
    """
    fleet = get_fleet(traffic_model)
    lost = 2.0 + mean_startup_delay(fleet)
    usable = max(float(green_seconds) - lost, 0.0)
    return {"green": float(green_seconds), "lost": round(lost, 2),
            "usable": round(usable, 2),
            "share": round(usable / green_seconds if green_seconds else 0.0, 4)}


# ===========================================================================
# Saving a run so it can be looked at again
# ===========================================================================
def run_path(workload_id: str, controller: str, seed: int,
             directory: str = RUN_DIR) -> str:
    return os.path.join(directory, "{}_{}_{}.json".format(
        workload_id, controller, seed))


def save_run(result: RunResult, directory: str = RUN_DIR,
             with_replay: bool = True) -> str:
    os.makedirs(directory, exist_ok=True)
    path = run_path(result.workload_id, result.controller, result.seed,
                    directory)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result.to_dict(with_replay=with_replay), handle,
                  separators=(",", ":"), default=float)
    return path


def load_run(workload_id: str, controller: str, seed: int,
             directory: str = RUN_DIR) -> Optional[RunResult]:
    path = run_path(workload_id, controller, seed, directory)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return RunResult.from_dict(json.load(handle))
    except (OSError, ValueError):
        return None


def saved_runs(directory: str = RUN_DIR) -> List[Dict[str, Any]]:
    """A short description of every saved run, newest first."""
    if not os.path.isdir(directory):
        return []
    rows = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(directory, name), "r",
                      encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            continue
        rows.append({
            "file": name, "controller": data.get("controller", ""),
            "label": data.get("label", ""),
            "workload_id": data.get("workload_id", ""),
            "workload_name": data.get("workload_name", ""),
            "seed": int(data.get("seed", 0)),
            "created_at": data.get("created_at", ""),
            "delay": float((data.get("metrics") or {}).get(
                "delay_per_requested", 0.0)),
            "waiting": float((data.get("metrics") or {}).get(
                "waiting_per_vehicle", 0.0)),
        })
    return sorted(rows, key=lambda row: row["created_at"], reverse=True)


# ===========================================================================
# Comparing runs
# ===========================================================================
def compare(results: Sequence[RunResult]) -> Dict[str, Any]:
    """A table of runs on the SAME traffic, with the baseline called out.

    Refuses to put two different traffic patterns in one table: that is
    the mistake the table exists to prevent.
    """
    results = [r for r in results if r is not None]
    if not results:
        return {"rows": [], "baseline": None, "workload_id": ""}
    ids = {r.workload_id for r in results}
    if len(ids) > 1:
        raise ValueError(
            "These runs are on different traffic ({}), so they cannot be "
            "compared in one table.".format(", ".join(sorted(ids))))
    seeds = {r.seed for r in results}
    if len(seeds) > 1:
        raise ValueError(
            "These runs used different SUMO seeds ({}), so the table would "
            "compare different runs of the same traffic.".format(
                ", ".join(str(s) for s in sorted(seeds))))

    baseline = next((r for r in results if r.controller == "fixed_time"), None)
    rows = []
    for result in sorted(results, key=lambda r: r.delay):
        row = {
            "controller": result.controller, "label": result.label,
            "delay": result.delay, "waiting": result.waiting,
            "completed": result.completed,
            "waiting_to_enter": result.metrics.get("waiting_to_enter", 0),
            "queue_mean": result.metrics.get("queue_mean", 0.0),
            "mean_green": result.metrics.get("mean_green", 0.0),
            "compliant": result.audit.get("compliant"),
            "trustworthy": result.trustworthy,
            "seed": result.seed,
        }
        if baseline is not None and baseline.delay > 0:
            row["change_percent"] = round(
                100.0 * (result.delay - baseline.delay) / baseline.delay, 1)
            row["change_seconds"] = round(result.delay - baseline.delay, 2)
        rows.append(row)
    return {
        "rows": rows,
        "baseline": baseline.controller if baseline else None,
        "workload_id": results[0].workload_id,
        "workload_name": results[0].workload_name,
        "seed": results[0].seed,
        "profile": results[0].profile,
    }
