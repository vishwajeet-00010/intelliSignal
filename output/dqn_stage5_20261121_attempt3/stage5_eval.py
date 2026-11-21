"""Validation and sealed confirmatory evaluation for Stage 5."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))

from agents.base_agent import (ACTION_EW_GREEN, ACTION_NS_GREEN, PHASE_EW,
                               PHASE_NS, TrafficAgent, TrafficState)
from agents.dqn_agent import DQNAgent, set_seeds
from agents.fixed_time_agent import FixedTimeAgent
from rl.environment import IntelliSignalEnv
from simulation.agent_simulation import run_agent_episode, run_fixed_time_episode

TRAIN_SEEDS = (6207, 6208, 6209, 6210, 6211)
VALIDATION_SEEDS = (6301, 6302, 6303, 6304, 6305)
CALIBRATION_SEEDS = tuple(range(6401, 6411))
CONFIRMATORY_SEEDS = tuple(range(6501, 6521))
# Keep the frozen control explicit: changing the general environment default
# must not change either Stage 5 arm during evaluation.
ARMS = {"control": 20.0, "treatment": 60.0}
SCENARIOS = ("moderate", "heavy")
APPROACHES = ("north", "east", "south", "west")


class PressureActuatedAgent(TrafficAgent):
    name = "Pressure-actuated"
    description = "Switches when opposing queue/wait pressure exceeds current pressure."
    is_learned = False

    def __init__(self, switch_margin: float, max_green: float, min_green: float = 5.0):
        self.switch_margin = float(switch_margin)
        self.max_green = float(max_green)
        self.min_green = float(min_green)

    @staticmethod
    def pressure(queue: float, waiting: float) -> float:
        return float(queue) + 0.02 * float(waiting)

    def select_action(self, state):
        current = state.current_phase
        if state.time_since_phase_change < self.min_green:
            return ACTION_NS_GREEN if current == PHASE_NS else ACTION_EW_GREEN
        if state.time_since_phase_change >= self.max_green:
            return ACTION_EW_GREEN if current == PHASE_NS else ACTION_NS_GREEN
        if current == PHASE_NS:
            here = self.pressure(state.ns_queue, state.ns_waiting)
            other = self.pressure(state.ew_queue, state.ew_waiting)
            other_action = ACTION_EW_GREEN
        else:
            here = self.pressure(state.ew_queue, state.ew_waiting)
            other = self.pressure(state.ns_queue, state.ns_waiting)
            other_action = ACTION_NS_GREEN
        if other - here >= self.switch_margin:
            return other_action
        return ACTION_NS_GREEN if current == PHASE_NS else ACTION_EW_GREEN


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def mean(rows: list[dict], metric: str) -> float:
    return statistics.fmean(float(row[metric]) for row in rows)


def load_dqn(path: Path, scenario: str) -> DQNAgent:
    agent = DQNAgent.load(str(path), scenario=scenario)
    agent.set_epsilon(0.0)
    return agent


def run_policy(scenario: str, label: str, policy, seeds, phase_scale=None):
    rows = []
    for seed in seeds:
        set_seeds(seed)
        if label == "fixed_time_original":
            row, _ = run_fixed_time_episode(scenario, seed=seed, duration=1800)
        else:
            if isinstance(policy, DQNAgent):
                policy.set_epsilon(0.0)
            row, _ = run_agent_episode(
                scenario, policy, seed=seed, duration=1800, controller=label,
                phase_elapsed_scale=phase_scale)
        row["controller"] = label
        row["scenario"] = scenario
        row["phase_elapsed_scale"] = phase_scale
        rows.append(row)
        print(f"{scenario} {label} seed={seed} wait="
              f"{row['average_waiting_time']:.3f} queue="
              f"{row['average_queue_length']:.3f} throughput="
              f"{row['throughput']}", flush=True)
    return rows


def collect_observation_stats(scenario: str, model_path: Path, arm: str) -> dict:
    """Collect raw, normalized, and clipped feature distributions."""
    import numpy as np
    agent = load_dqn(model_path, scenario)
    scale = ARMS[arm]
    values = {"queues": {a: [] for a in APPROACHES},
              "waiting": {a: [] for a in APPROACHES}, "phase_elapsed": []}
    clipped = {"queues": {a: 0 for a in APPROACHES},
               "waiting": {a: 0 for a in APPROACHES}, "phase_elapsed": 0}
    normalised_phase = []
    phase_scale = None
    for seed in CALIBRATION_SEEDS:
        set_seeds(seed)
        env = IntelliSignalEnv(scenario=scenario, seed=seed,
                               phase_elapsed_scale=scale)
        phase_scale = env.state_extractor.phase_elapsed_scale
        def sample_observation():
            raw = env.state_extractor.raw(env.connection)
            for kind in ("queues", "waiting"):
                divisor = (env.state_extractor.max_queue if kind == "queues"
                           else env.state_extractor.max_waiting)
                for approach, value in raw[kind].items():
                    values[kind][approach].append(float(value))
                    clipped[kind][approach] += int(float(value) > divisor)
            elapsed = float(env.controller.green_elapsed)
            values["phase_elapsed"].append(elapsed)
            normalised_phase.append(min(1.0, elapsed / phase_scale))
            clipped["phase_elapsed"] += int(elapsed > phase_scale)
        try:
            observation, info = env.reset(seed=seed)
            while True:
                sample_observation()
                action = agent.env_action(TrafficState.from_env_info(info, observation))
                observation, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
        finally:
            env.close()

    summary = {}
    for kind, groups in values.items():
        if kind in ("queues", "waiting"):
            summary[kind] = {}
            for approach, samples in groups.items():
                array = np.asarray(samples, dtype=float)
                summary[kind][approach] = {
                    "p50": float(np.percentile(array, 50)),
                    "p95": float(np.percentile(array, 95)),
                    "p99": float(np.percentile(array, 99)),
                    "max": float(np.max(array)),
                    "fraction_clipped": clipped[kind][approach] / len(array),
                }
        else:
            array = np.asarray(groups, dtype=float)
            normalized = np.asarray(normalised_phase, dtype=float)
            summary[kind] = {
                "p50": float(np.percentile(array, 50)),
                "p95": float(np.percentile(array, 95)),
                "p99": float(np.percentile(array, 99)),
                "max": float(np.max(array)),
                "fraction_clipped": clipped[kind] / len(array),
                "normalization_scale_seconds": phase_scale,
                "normalized_p50": float(np.percentile(normalized, 50)),
                "normalized_p95": float(np.percentile(normalized, 95)),
                "normalized_p99": float(np.percentile(normalized, 99)),
            }
    return {"scenario": scenario, "arm": arm, "model": str(model_path),
            "calibration_seeds": list(CALIBRATION_SEEDS),
            "raw_and_normalized_percentiles": summary}


def tune_actuated(scenario: str, incumbent_rows: list[dict], seeds):
    base_queue, base_throughput = (mean(incumbent_rows, "average_queue_length"),
                                   mean(incumbent_rows, "throughput"))
    results, grid_rows = [], []
    for margin in (0.0, 2.0, 4.0):
        for max_green in (30.0, 40.0, 50.0):
            label = f"actuated_m{margin:g}_g{max_green:g}"
            batch = run_policy(scenario, label, PressureActuatedAgent(margin, max_green), seeds)
            grid_rows += batch
            passed = (mean(batch, "average_queue_length") <= base_queue * 1.02 and
                      mean(batch, "throughput") >= base_throughput * 0.99 and
                      sum(int(r.get("blocked_switches") or 0) for r in batch) == 0 and
                      sum(int(r.get("teleports") or 0) for r in batch) == 0)
            results.append({"switch_margin": margin, "max_green": max_green,
                            "mean_waiting_time": mean(batch, "average_waiting_time"),
                            "mean_average_queue": mean(batch, "average_queue_length"),
                            "mean_throughput": mean(batch, "throughput"),
                            "blocked_switches": sum(int(r.get("blocked_switches") or 0) for r in batch),
                            "teleports": sum(int(r.get("teleports") or 0) for r in batch),
                            "passes_validation_guard": passed})
    eligible = [r for r in results if r["passes_validation_guard"]]
    selected = min(eligible or results, key=lambda row: row["mean_waiting_time"])
    selected["selected_within_guardrails"] = bool(eligible)
    policy = PressureActuatedAgent(selected["switch_margin"], selected["max_green"])
    tuned_rows = run_policy(scenario, "actuated_tuned", policy, seeds)
    return grid_rows + tuned_rows, {"grid": results, "selected": selected}


def validation(scenario: str) -> None:
    folder = OUT / scenario
    rows = []
    incumbent_path = ROOT / "output" / "dqn_retraining_20261120" / "models" / f"dqn_{scenario}" / "best.pt"
    starting_path = ROOT / "models" / f"dqn_{scenario}" / "best.pt"
    incumbent = run_policy(scenario, "incumbent_dqn", load_dqn(incumbent_path, scenario), VALIDATION_SEEDS)
    rows += incumbent
    rows += run_policy(scenario, "starting_dqn", load_dqn(starting_path, scenario), VALIDATION_SEEDS)
    rows += run_policy(scenario, "fixed_time_original", None, VALIDATION_SEEDS)
    green = 5.0 if scenario == "moderate" else 7.0
    rows += run_policy(scenario, "fixed_time_tuned",
                       FixedTimeAgent(ns_green=green, ew_green=green, yellow=3.0),
                       VALIDATION_SEEDS)

    per_run, arm_results, candidate_rows = {}, {}, {}
    for arm, scale in ARMS.items():
        for seed in TRAIN_SEEDS:
            run_dir = folder / arm / f"seed{seed}"
            checkpoint = run_dir / "models" / "best.pt"
            summary_path = run_dir / "training_summary.json"
            if not checkpoint.is_file() or not summary_path.is_file():
                per_run[f"{arm}_seed{seed}"] = {"status": "missing_training_artifact"}
                continue
            result = run_policy(scenario, f"candidate_{arm}_seed{seed}",
                                load_dqn(checkpoint, scenario), VALIDATION_SEEDS,
                                phase_scale=scale)
            rows += result
            candidate_rows[(arm, seed)] = result
            wait, queue, throughput = (mean(result, "average_waiting_time"),
                                       mean(result, "average_queue_length"),
                                       mean(result, "throughput"))
            passed = (wait <= mean(incumbent, "average_waiting_time") * 1.03 and
                      queue <= mean(incumbent, "average_queue_length") * 1.02 and
                      throughput >= mean(incumbent, "throughput") * 0.99 and
                      sum(int(r.get("blocked_switches") or 0) for r in result) == 0 and
                      sum(int(r.get("teleports") or 0) for r in result) == 0)
            per_run[f"{arm}_seed{seed}"] = {
                "status": "evaluated", "mean_waiting_time": wait,
                "mean_average_queue": queue, "mean_max_queue": mean(result, "max_queue_length"),
                "mean_throughput": throughput,
                "mean_vehicles_left": mean(result, "vehicles_still_in_network"),
                "blocked_switches": sum(int(r.get("blocked_switches") or 0) for r in result),
                "teleports": sum(int(r.get("teleports") or 0) for r in result),
                "passes_validation_gate": passed,
                "checkpoint": str(checkpoint), "phase_elapsed_scale": scale,
            }
        arm_rows = [v for key, v in per_run.items()
                    if key.startswith(arm + "_") and v.get("status") == "evaluated"]
        passes = sum(bool(v["passes_validation_gate"]) for v in arm_rows)
        arm_results[arm] = {
            "runs_evaluated": len(arm_rows), "runs_passed": passes,
            "validation_eligible": len(arm_rows) == 5 and passes >= 4,
            "mean_waiting_time_across_runs": statistics.fmean(
                v["mean_waiting_time"] for v in arm_rows) if arm_rows else None,
        }

    calibration = {}
    for arm in ARMS:
        arm_evaluated = [(seed, entry) for key, entry in per_run.items()
                   if key.startswith(arm + "_seed") and entry.get("status") == "evaluated"
                   for seed in [int(key.rsplit("seed", 1)[1])]]
        if arm_evaluated:
            seed, best = min(arm_evaluated, key=lambda pair: pair[1]["mean_waiting_time"])
            stats = collect_observation_stats(scenario, Path(best["checkpoint"]), arm)
            cal_path = folder / "calibration" / f"observation_stats_{arm}.json"
            cal_path.parent.mkdir(parents=True, exist_ok=True)
            stats["selected_training_seed_for_diagnostics"] = seed
            stats["path"] = str(cal_path)
            cal_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
            calibration[arm] = stats

    actuator_rows, actuator_result = tune_actuated(scenario, incumbent, VALIDATION_SEEDS)
    rows += actuator_rows
    qualified = []
    for (arm, seed), result in candidate_rows.items():
        if arm_results[arm]["validation_eligible"] and per_run[f"{arm}_seed{seed}"]["passes_validation_gate"]:
            qualified.append({"arm": arm, "seed": seed,
                               "phase_elapsed_scale": ARMS[arm],
                               "checkpoint": per_run[f"{arm}_seed{seed}"]["checkpoint"],
                               "mean_waiting_time": mean(result, "average_waiting_time"),
                               "mean_average_queue": mean(result, "average_queue_length"),
                               "mean_max_queue": mean(result, "max_queue_length"),
                               "mean_throughput": mean(result, "throughput"),
                               "blocked_switches": sum(int(r.get("blocked_switches") or 0) for r in result),
                               "teleports": sum(int(r.get("teleports") or 0) for r in result)})
    selected = min(qualified, key=lambda row: row["mean_waiting_time"]) if qualified else None
    result = {"scenario": scenario, "training_seeds": list(TRAIN_SEEDS),
              "validation_seeds": list(VALIDATION_SEEDS),
              "calibration_seeds": list(CALIBRATION_SEEDS),
              "duration_seconds": 1800, "candidate_checks": per_run,
              "configuration_checks": arm_results,
              "calibration": calibration,
              "actuated_tuning": actuator_result,
              "eligible_checkpoints": qualified,
              "selected_candidate": selected,
              "selected_candidate_allowed_for_confirmatory": selected is not None}
    vdir = folder / "validation"
    write_rows(vdir / "validation_rows.csv", rows)
    (vdir / "validation_selection.json").write_text(json.dumps(result, indent=2),
                                                    encoding="utf-8")
    manifest_path = OUT / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("validation", {})[scenario] = result
    manifest.setdefault("actuated_baseline", {})[scenario] = actuator_result
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def confirmatory(scenario: str) -> None:
    folder = OUT / scenario
    selection = json.loads((folder / "validation" / "validation_selection.json").read_text(encoding="utf-8"))
    rows = []
    rows += run_policy(scenario, "fixed_time_original", None, CONFIRMATORY_SEEDS)
    green = 5.0 if scenario == "moderate" else 7.0
    rows += run_policy(scenario, "fixed_time_tuned",
                       FixedTimeAgent(ns_green=green, ew_green=green, yellow=3.0),
                       CONFIRMATORY_SEEDS)
    selected_act = selection["actuated_tuning"]["selected"]
    rows += run_policy(scenario, "actuated_tuned",
                       PressureActuatedAgent(selected_act["switch_margin"],
                                             selected_act["max_green"]), CONFIRMATORY_SEEDS)
    rows += run_policy(scenario, "incumbent_dqn",
                       load_dqn(ROOT / "output" / "dqn_retraining_20261120" / "models" / f"dqn_{scenario}" / "best.pt", scenario),
                       CONFIRMATORY_SEEDS)
    rows += run_policy(scenario, "starting_dqn",
                       load_dqn(ROOT / "models" / f"dqn_{scenario}" / "best.pt", scenario),
                       CONFIRMATORY_SEEDS)
    candidate = selection.get("selected_candidate")
    if candidate:
        arm = candidate["arm"]
        rows += run_policy(scenario, "stage5_candidate",
                           load_dqn(Path(candidate["checkpoint"]), scenario),
                           CONFIRMATORY_SEEDS, phase_scale=ARMS[arm])
    out = folder / "confirmatory"
    write_rows(out / "confirmatory_rows.csv", rows)
    record = {"scenario": scenario, "seeds": list(CONFIRMATORY_SEEDS),
              "duration_seconds": 1800, "candidate_included": candidate is not None,
              "selected_candidate": candidate,
              "rows_csv": str(out / "confirmatory_rows.csv")}
    (out / "confirmatory_run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    manifest_path = OUT / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("confirmatory_runs", {})[scenario] = record
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("validation", "confirmatory"))
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    args = parser.parse_args()
    if args.phase == "validation":
        validation(args.scenario)
    else:
        confirmatory(args.scenario)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
