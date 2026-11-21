"""Isolated Stage 4 validation and sealed confirmatory evaluation runner."""
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
from simulation.agent_simulation import run_agent_episode, run_fixed_time_episode

VALIDATION_SEEDS = [5101, 5102, 5103, 5104, 5105]
CALIBRATION_SEEDS = list(range(5201, 5211))
CONFIRMATORY_SEEDS = list(range(6101, 6121))
SCENARIOS = ("moderate", "heavy")
METRICS = ("average_waiting_time", "average_queue_length", "max_queue_length",
           "throughput", "vehicles", "vehicles_still_in_network", "teleports",
           "switch_count", "blocked_switches", "decisions")


class PressureActuatedAgent(TrafficAgent):
    """Queue/wait pressure controller, sampled at the environment decision step."""
    name = "Pressure-actuated"
    description = "Switches when opposing demand pressure exceeds current pressure."
    is_learned = False

    def __init__(self, switch_margin: float, max_green: float, min_green: float = 5.0):
        self.switch_margin = float(switch_margin)
        self.max_green = float(max_green)
        self.min_green = float(min_green)

    @staticmethod
    def _pressure(queue: float, waiting: float) -> float:
        return float(queue) + 0.02 * float(waiting)

    def select_action(self, state: TrafficState) -> int:
        current = state.current_phase
        if state.time_since_phase_change < self.min_green:
            return ACTION_NS_GREEN if current == PHASE_NS else ACTION_EW_GREEN
        if state.time_since_phase_change >= self.max_green:
            return ACTION_EW_GREEN if current == PHASE_NS else ACTION_NS_GREEN
        if current == PHASE_NS:
            current_pressure = self._pressure(state.ns_queue, state.ns_waiting)
            other_pressure = self._pressure(state.ew_queue, state.ew_waiting)
            other_action = ACTION_EW_GREEN
        else:
            current_pressure = self._pressure(state.ew_queue, state.ew_waiting)
            other_pressure = self._pressure(state.ns_queue, state.ns_waiting)
            other_action = ACTION_NS_GREEN
        if other_pressure - current_pressure >= self.switch_margin:
            return other_action
        return ACTION_NS_GREEN if current == PHASE_NS else ACTION_EW_GREEN


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run_policy(scenario: str, label: str, policy, seeds: list[int]) -> list[dict]:
    rows = []
    for seed in seeds:
        set_seeds(seed)
        if label == "fixed_time_original":
            summary, _ = run_fixed_time_episode(scenario, seed=seed, duration=1800)
        else:
            if isinstance(policy, DQNAgent):
                policy.set_epsilon(0.0)
            summary, _ = run_agent_episode(scenario, policy, seed=seed,
                                           duration=1800, controller=label)
        summary["controller"] = label
        summary["scenario"] = scenario
        rows.append(summary)
        print(f"{scenario} {label} seed={seed} wait="
              f"{summary['average_waiting_time']:.3f} queue="
              f"{summary['average_queue_length']:.3f} throughput="
              f"{summary['throughput']}", flush=True)
    return rows


def mean(rows: list[dict], metric: str) -> float:
    return statistics.fmean(float(row[metric]) for row in rows)


def load_dqn(path: Path, scenario: str) -> DQNAgent:
    agent = DQNAgent.load(str(path), scenario=scenario)
    agent.set_epsilon(0.0)
    return agent


def validation(scenario: str) -> None:
    folder = OUT / scenario
    run_rows = []
    policies: dict[str, object] = {}
    incumbent_path = (OUT.parent / "dqn_retraining_20261120" / "models" /
                      f"dqn_{scenario}" / "best.pt")
    start_path = ROOT / "models" / f"dqn_{scenario}" / "best.pt"
    policies["incumbent_dqn"] = load_dqn(incumbent_path, scenario)
    policies["starting_dqn"] = load_dqn(start_path, scenario)
    candidate_paths = {}
    for seed in (55, 56):
        model = folder / f"seed{seed}" / "models" / "best.pt"
        summary_file = folder / f"seed{seed}" / "training_summary.json"
        if model.exists() and summary_file.exists():
            policies[f"candidate_seed{seed}"] = load_dqn(model, scenario)
            candidate_paths[seed] = model
    if len(candidate_paths) != 2:
        raise RuntimeError(f"Both independent candidate runs are required: {scenario}")

    for label, policy in policies.items():
        run_rows += run_policy(scenario, label, policy, VALIDATION_SEEDS)
    # Existing scenario plan and the tuned fixed-time timings from Stage 3.
    run_rows += run_policy(scenario, "fixed_time_original", None, VALIDATION_SEEDS)
    green = 5.0 if scenario == "moderate" else 7.0
    run_rows += run_policy(scenario, "fixed_time_tuned",
                           FixedTimeAgent(ns_green=green, ew_green=green,
                                          yellow=3.0), VALIDATION_SEEDS)

    incumbent_rows = [r for r in run_rows if r["controller"] == "incumbent_dqn"]
    incumbent_wait = mean(incumbent_rows, "average_waiting_time")
    incumbent_queue = mean(incumbent_rows, "average_queue_length")
    incumbent_throughput = mean(incumbent_rows, "throughput")
    # Tune the actuated pressure margin/max-green only on the declared validation set.
    actuator_results = []
    for margin in (0.0, 2.0, 4.0):
        for max_green in (30.0, 40.0, 50.0):
            label = f"actuated_m{margin:g}_g{max_green:g}"
            policy = PressureActuatedAgent(margin, max_green)
            result = run_policy(scenario, label, policy, VALIDATION_SEEDS)
            run_rows += result
            passed = (
                mean(result, "average_queue_length") <= incumbent_queue * 1.02 and
                mean(result, "throughput") >= incumbent_throughput * 0.99 and
                sum(int(row["blocked_switches"]) for row in result) == 0 and
                sum(int(row["teleports"]) for row in result) == 0)
            actuator_results.append({
                "switch_margin": margin, "max_green": max_green,
                "mean_waiting_time": mean(result, "average_waiting_time"),
                "mean_average_queue": mean(result, "average_queue_length"),
                "mean_max_queue": mean(result, "max_queue_length"),
                "mean_throughput": mean(result, "throughput"),
                "mean_vehicles_left": mean(result, "vehicles_still_in_network"),
                "blocked_switches": sum(int(row["blocked_switches"]) for row in result),
                "teleports": sum(int(row["teleports"]) for row in result),
                "passes_validation_safety_and_tradeoff_guard": passed,
            })
    eligible_actuators = [row for row in actuator_results
                          if row["passes_validation_safety_and_tradeoff_guard"]]
    selected_actuator = min(eligible_actuators or actuator_results,
                            key=lambda row: row["mean_waiting_time"])
    selected_actuator["selected_within_declared_guardrails"] = bool(eligible_actuators)
    if selected_actuator:
        selected_label = (f"actuated_m{selected_actuator['switch_margin']:g}_"
                          f"g{selected_actuator['max_green']:g}")
        selected = PressureActuatedAgent(selected_actuator["switch_margin"],
                                         selected_actuator["max_green"])
        run_rows += run_policy(scenario, "actuated_tuned", selected, VALIDATION_SEEDS)
        selected_actuator["selected_label"] = selected_label

    candidate_checks = {}
    observation_stats = {}
    for seed in (55, 56):
        rows = [r for r in run_rows if r["controller"] == f"candidate_seed{seed}"]
        wait = mean(rows, "average_waiting_time")
        queue = mean(rows, "average_queue_length")
        throughput = mean(rows, "throughput")
        valid = (
            wait <= incumbent_wait * 1.03 and
            queue <= incumbent_queue * 1.02 and
            throughput >= incumbent_throughput * 0.99 and
            sum(int(r["blocked_switches"]) for r in rows) == 0 and
            sum(int(r["teleports"]) for r in rows) == 0)
        candidate_checks[str(seed)] = {
            "checkpoint": str(candidate_paths[seed]),
            "mean_waiting_time": wait, "mean_average_queue": queue,
            "mean_max_queue": mean(rows, "max_queue_length"),
            "mean_throughput": throughput,
            "mean_vehicles_left": mean(rows, "vehicles_still_in_network"),
            "blocked_switches": sum(int(r["blocked_switches"]) for r in rows),
            "teleports": sum(int(r["teleports"]) for r in rows),
            "passes_validation_gate": valid,
        }
        # Diagnostics only: a distinct calibration split does not retune scales.
        from evaluation.collect_observation_stats import collect
        stats = collect(scenario, str(candidate_paths[seed]), CALIBRATION_SEEDS)
        stats_path = folder / "validation" / f"observation_stats_seed{seed}.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        observation_stats[str(seed)] = {"path": str(stats_path),
                                        "percentiles": stats["percentiles"]}
    validation_passed = all(row["passes_validation_gate"]
                            for row in candidate_checks.values())
    selected_candidate = None
    if validation_passed:
        selected_candidate = min(candidate_checks.items(),
                                 key=lambda item: item[1]["mean_waiting_time"])[0]
    result = {
        "scenario": scenario, "validation_seeds": VALIDATION_SEEDS,
        "calibration_seeds": CALIBRATION_SEEDS,
        "duration_seconds": 1800,
        "candidate_checks": candidate_checks,
        "observation_stats": observation_stats,
        "all_independent_candidates_pass": validation_passed,
        "selected_candidate_seed": int(selected_candidate) if selected_candidate else None,
        "actuated_tuning_grid": actuator_results,
        "selected_actuated": selected_actuator,
        "confirmatory_candidate_allowed": validation_passed,
        "rows_csv": str(folder / "validation" / "validation_rows.csv"),
    }
    val_dir = folder / "validation"
    write_rows(val_dir / "validation_rows.csv", run_rows)
    (val_dir / "validation_selection.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def confirmatory(scenario: str) -> None:
    folder = OUT / scenario
    selection_path = folder / "validation" / "validation_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    rows = []
    incumbent_path = (OUT.parent / "dqn_retraining_20261120" / "models" /
                      f"dqn_{scenario}" / "best.pt")
    start_path = ROOT / "models" / f"dqn_{scenario}" / "best.pt"
    policies = {
        "incumbent_dqn": load_dqn(incumbent_path, scenario),
        "starting_dqn": load_dqn(start_path, scenario),
    }
    selected_seed = selection.get("selected_candidate_seed")
    if selected_seed is not None:
        cp = folder / f"seed{selected_seed}" / "models" / "best.pt"
        policies["stage4_candidate"] = load_dqn(cp, scenario)
    rows += run_policy(scenario, "fixed_time_original", None, CONFIRMATORY_SEEDS)
    green = 5.0 if scenario == "moderate" else 7.0
    rows += run_policy(scenario, "fixed_time_tuned",
                       FixedTimeAgent(ns_green=green, ew_green=green, yellow=3.0),
                       CONFIRMATORY_SEEDS)
    selected_actuator = selection.get("selected_actuated")
    if selected_actuator:
        rows += run_policy(
            scenario, "actuated_tuned",
            PressureActuatedAgent(selected_actuator["switch_margin"],
                                  selected_actuator["max_green"]),
            CONFIRMATORY_SEEDS)
    for label, policy in policies.items():
        rows += run_policy(scenario, label, policy, CONFIRMATORY_SEEDS)
    out = folder / "confirmatory"
    write_rows(out / "confirmatory_rows.csv", rows)
    result = {"scenario": scenario, "seeds": CONFIRMATORY_SEEDS,
              "duration_seconds": 1800,
              "candidate_included": "stage4_candidate" in policies,
              "rows_csv": str(out / "confirmatory_rows.csv")}
    (out / "confirmatory_run.json").write_text(json.dumps(result, indent=2),
                                               encoding="utf-8")


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
