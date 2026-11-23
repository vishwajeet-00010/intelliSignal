"""
IntelliSignal
simulation/custom_runs.py

Running a custom workload under one controller, and saving it so the
dashboard can replay it later.

This is the one place a custom run is produced, for every control mode, so
the three modes cannot drift into measuring different things. Each run
returns the same five vehicle counts, because a congested run is only
readable when they are kept apart:

    requested        vehicles the workload asked for
    entered          vehicles SUMO managed to put on the network
    completed        vehicles that finished their route
    still on road    entered but not yet finished at the end
    waiting to enter requested but never inserted, because the lane was full

"Throughput 120" on its own hides the difference between a signal that is
working and a lane that could not accept the demand in the first place.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from custom_traffic.builder import build_workload, cleanup_workload_inputs
from custom_traffic.workload import ResolvedWorkload
from simulation.hybrid import FixedPlan, configured_plan_for
from simulation.run_simulation import prepare_sumo_imports, sumo_binary
from utils.metrics import MetricsCollector

CUSTOM_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "custom_runs")


# ---------------------------------------------------------------------------
def _accounting(summary: Dict[str, Any], requested: int) -> Dict[str, Any]:
    entered = int(summary.get("vehicles_departed", summary.get("vehicles", 0)))
    completed = int(summary.get("throughput", 0))
    summary["vehicles_requested"] = int(requested)
    summary["vehicles_entered"] = entered
    summary["vehicles_completed"] = completed
    summary["vehicles_still_on_road"] = max(entered - completed, 0)
    summary["vehicles_waiting_to_enter"] = max(int(requested) - entered, 0)
    return summary


# ---------------------------------------------------------------------------
def run_fixed(workload: ResolvedWorkload, plan: Optional[FixedPlan] = None,
              seed: Optional[int] = None, use_gui: bool = False,
              tag: str = "_run"):
    """One fixed-time run of the workload, through SUMO's own TLS program."""
    prepare_sumo_imports()
    import traci

    plan = plan or configured_plan_for(workload)
    seed = int(seed if seed is not None else workload.seed)
    files = build_workload(workload, tag=tag, signal=plan.signal())
    collector = MetricsCollector(step_length=workload.step_length)
    steps = int(round(workload.duration_s / workload.step_length))
    label = "custom_fixed_{}_{}".format(workload.workload_id, seed)

    command = [sumo_binary(use_gui), "-c", files["sumocfg"],
               "--seed", str(seed), "--no-warnings", "--no-step-log"]
    if use_gui:
        command += ["--start", "--quit-on-end"]

    traci.start(command, label=label)
    try:
        for _ in range(steps):
            traci.simulationStep()
            collector.collect_step()
    finally:
        try:
            traci.close()
        except Exception:
            pass
        cleanup_workload_inputs(workload, tag=tag)

    summary = _accounting(collector.summary(workload.name),
                          workload.total_vehicles)
    summary.update({"controller": "fixed_time", "seed": seed,
                    "ns_green": plan.ns_green, "ew_green": plan.ew_green,
                    "plan_source": plan.source})
    return summary, collector.timeseries


# ---------------------------------------------------------------------------
def run_dqn(workload: ResolvedWorkload, agent, seed: Optional[int] = None,
            tag: str = "_rundqn"):
    """One greedy run of a phase-duration policy on the workload."""
    from rl.phase_environment import PhaseDurationEnv
    from rl.phase_state import extractor_from_contract

    seed = int(seed if seed is not None else workload.seed)
    env = PhaseDurationEnv(
        workload=workload, durations=agent.durations,
        state_extractor=extractor_from_contract(agent.observation_contract),
        tag=tag)
    greens: List[Dict[str, Any]] = []
    try:
        observation, _ = env.reset(seed=seed)
        collector = MetricsCollector(step_length=env.step_length,
                                     conn=env.connection)
        env.step_callback = collector.collect_step
        while True:
            phase = env.controller.phase
            action = agent.act(observation, training=False)
            values = agent.q_values(observation)
            observation, _, terminated, truncated, info = env.step(action)
            greens.append({
                "time": float(info["simulation_time"]),
                "phase": "North-South" if phase == 0 else "East-West",
                "green_seconds": float(info["green_served"]),
                "q_values": {"{:g}s".format(d): round(float(v), 4)
                             for d, v in zip(agent.durations, values)},
            })
            if terminated or truncated:
                break
        summary = collector.summary(workload.name)
        timeseries = collector.timeseries
    finally:
        env.close()
        env.cleanup_generated_inputs()

    summary = _accounting(summary, workload.total_vehicles)
    summary.update({"controller": "dqn", "seed": seed,
                    "model": getattr(agent, "model_path", "") or "",
                    "decisions": len(greens)})
    return summary, timeseries, greens


# ---------------------------------------------------------------------------
def save_custom_run(workload: ResolvedWorkload, summary: Dict[str, Any],
                    timeseries, decisions=None,
                    directory: str = CUSTOM_RESULTS_DIR) -> Dict[str, str]:
    """Write the run so the dashboard can reload and replay it."""
    import pandas as pd

    os.makedirs(directory, exist_ok=True)
    stem = "{}_{}".format(workload.workload_id, summary.get("controller", "run"))

    results_path = os.path.join(directory, "{}_results.csv".format(stem))
    pd.DataFrame([summary]).to_csv(results_path, index=False)

    series_path = ""
    if timeseries:
        series_path = os.path.join(directory, "{}_timeseries.csv".format(stem))
        pd.DataFrame(timeseries).to_csv(series_path, index=False)

    decisions_path = ""
    if decisions:
        decisions_path = os.path.join(directory, "{}_decisions.json".format(stem))
        with open(decisions_path, "w", encoding="utf-8") as handle:
            json.dump(decisions, handle, indent=2, default=float)

    meta_path = os.path.join(directory, "{}_run.json".format(stem))
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump({
            "workload_id": workload.workload_id,
            "workload_name": workload.name,
            "controller": summary.get("controller"),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "summary": summary,
            "results_csv": os.path.relpath(results_path, PROJECT_ROOT),
            "timeseries_csv": (os.path.relpath(series_path, PROJECT_ROOT)
                               if series_path else ""),
            "decisions_json": (os.path.relpath(decisions_path, PROJECT_ROOT)
                               if decisions_path else ""),
        }, handle, indent=2, default=float)

    return {"results": results_path, "timeseries": series_path,
            "decisions": decisions_path, "meta": meta_path}


# ---------------------------------------------------------------------------
def saved_custom_runs(directory: str = CUSTOM_RESULTS_DIR
                      ) -> List[Dict[str, Any]]:
    """Every saved custom run, newest first, for the replay list."""
    if not os.path.isdir(directory):
        return []
    rows = []
    for name in os.listdir(directory):
        if not name.endswith("_run.json"):
            continue
        try:
            with open(os.path.join(directory, name), "r",
                      encoding="utf-8") as handle:
                rows.append(json.load(handle))
        except (OSError, ValueError):
            continue
    return sorted(rows, key=lambda row: row.get("saved_at", ""), reverse=True)


def load_custom_run(workload_id: str, controller: str,
                    directory: str = CUSTOM_RESULTS_DIR
                    ) -> Optional[Dict[str, Any]]:
    path = os.path.join(directory, "{}_{}_run.json".format(workload_id,
                                                           controller))
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None
