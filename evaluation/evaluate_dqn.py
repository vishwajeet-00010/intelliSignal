"""
IntelliSignal - Stage 3
evaluation/evaluate_dqn.py

EVALUATES a trained DQN on the SUMO intersection. No learning happens here.

Usage
-----
    python evaluation/evaluate_dqn.py --scenario moderate --seeds 42 43 44
    python evaluation/evaluate_dqn.py --scenario heavy \
        --model models/dqn_heavy/best.pt --seeds 42 43 44

    # the checkpoint is optional: without --model the scenario's best.pt is
    # used, and the run is refused outright if no model has been trained
    python evaluation/evaluate_dqn.py --scenario moderate --model models/dqn_moderate/final.pt

What "evaluation" means here
----------------------------
    epsilon = 0            every action is argmax Q(state); no exploration
    no remember()          nothing is written to the replay buffer
    no learn()             no gradient step, so the weights cannot change
    same network           the same intersection, demand and duration as the
                           Stage 1 fixed-time baseline

The script asserts afterwards that the agent took no gradient step, so "we
did not train during evaluation" is checked rather than promised.

Output
------
    results/evaluation/dqn_<scenario>_evaluation.csv          one row per seed
    results/evaluation/dqn_<scenario>_seed<seed>_timeseries.csv
    results/evaluation/dqn_<scenario>_evaluation_summary.csv   mean / std

results/csv/ (the recorded Stage 1 baseline) is never written.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.dqn_agent import (DQNAgent, DQNModelNotFoundError,  # noqa: E402
                              default_model_path, model_directory, set_seeds)
from simulation.agent_simulation import (DQN, RESULT_COLUMNS,    # noqa: E402
                                         run_agent_episode)

EVALUATION_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "evaluation")

#: The metrics every Stage 3 table reports, in the order they are reported.
METRICS = [
    "average_waiting_time",
    "average_queue_length",
    "max_queue_length",
    "throughput",
    "vehicles_still_in_network",
    "teleports",
]

METRIC_LABELS = {
    "average_waiting_time": "Average waiting time (s/vehicle)",
    "average_queue_length": "Average queue length (vehicles)",
    "max_queue_length": "Maximum queue length (vehicles)",
    "throughput": "Throughput (vehicles completed)",
    "vehicles_still_in_network": "Vehicles still in network",
    "teleports": "Teleports",
}


# ===========================================================================
def resolve_model(scenario: str, model: Optional[str] = None) -> str:
    """The checkpoint to evaluate, or a clear refusal.

    Never falls back to an untrained network and never falls back to the
    fixed-time controller: a missing model is an error, not a silent
    substitution.
    """
    if model:
        if not os.path.isfile(model):
            raise DQNModelNotFoundError(
                "Checkpoint not found: {}".format(model))
        return model
    path = default_model_path(scenario)
    if path is None:
        raise DQNModelNotFoundError(
            "No trained model for scenario '{}' in {}.\n\n"
            "Train one first:\n"
            "    python training/train_dqn.py --scenario {} --episodes 20 --seed 42"
            .format(scenario,
                    os.path.relpath(model_directory(scenario), PROJECT_ROOT),
                    scenario))
    return path


# ===========================================================================
def evaluate(scenario: str = "moderate", model: Optional[str] = None,
             seeds: List[int] = None, duration: Optional[float] = None,
             results_dir: Optional[str] = None, save: bool = True,
             quiet: bool = False, record_decisions: bool = False
             ) -> Dict[str, Any]:
    """Run the trained policy greedily on one scenario, once per seed."""
    seeds = [int(s) for s in (seeds or [42])]
    results_dir = results_dir or EVALUATION_RESULTS_DIR
    model_path = resolve_model(scenario, model)

    agent = DQNAgent.load(model_path, scenario=scenario)
    agent.set_epsilon(0.0)                 # greedy, stated explicitly
    gradient_steps_before = agent.learn_steps
    buffer_before = len(agent.memory)

    if not quiet:
        _print_header(scenario, model_path, agent, seeds, duration)

    rows: List[Dict[str, Any]] = []
    timeseries_paths: List[str] = []

    for seed in seeds:
        set_seeds(seed)                    # the agent is greedy, but be exact
        summary, timeseries = run_agent_episode(
            scenario, agent, seed=seed, duration=duration, controller=DQN,
            record_decisions=record_decisions)
        summary["model"] = os.path.relpath(model_path, PROJECT_ROOT)
        rows.append(summary)
        if not quiet:
            _print_row(summary)

        if save:
            from simulation.agent_simulation import save_run
            paths = save_run(summary, timeseries, results_dir,
                             "dqn_{}_seed{}".format(scenario, seed),
                             write_timeseries=True)
            if paths["timeseries"]:
                timeseries_paths.append(paths["timeseries"])

    # --- evaluation must not have trained ---------------------------------
    if agent.learn_steps != gradient_steps_before:
        raise RuntimeError(
            "The agent took {} gradient step(s) during evaluation. Evaluation "
            "must not train.".format(agent.learn_steps - gradient_steps_before))
    if len(agent.memory) != buffer_before:
        raise RuntimeError(
            "The replay buffer changed during evaluation. Evaluation must not "
            "collect experience.")

    result = {
        "scenario": scenario,
        "model": model_path,
        "seeds": seeds,
        "rows": rows,
        "aggregate": aggregate(rows, METRICS),
        "epsilon": agent.epsilon,
        "architecture": agent.architecture(),
        "timeseries": timeseries_paths,
    }

    if save:
        result["raw_csv"] = write_raw(rows, results_dir,
                                      "dqn_{}_evaluation".format(scenario))
        result["summary_csv"] = write_aggregate(
            result["aggregate"], results_dir,
            "dqn_{}_evaluation_summary".format(scenario), scenario, DQN)

    if not quiet:
        _print_aggregate(result)
    return result


# ===========================================================================
# aggregation helpers - shared with compare_controllers.py
# ===========================================================================
def aggregate(rows: List[Dict[str, Any]], metrics: List[str] = None
              ) -> Dict[str, Dict[str, Optional[float]]]:
    """mean, sample standard deviation, min, max and n for each metric.

    The standard deviation is the SAMPLE standard deviation (n-1), which is
    the right one for "we ran three seeds out of all possible seeds". With a
    single seed it is undefined and reported as None rather than as 0.0,
    because 0.0 would suggest a precision that one run cannot give.
    """
    import statistics

    metrics = metrics or METRICS
    output: Dict[str, Dict[str, Optional[float]]] = {}
    for metric in metrics:
        values = [float(r[metric]) for r in rows
                  if r.get(metric) is not None]
        if not values:
            output[metric] = {"mean": None, "std": None, "min": None,
                              "max": None, "n": 0}
            continue
        output[metric] = {
            "mean": float(statistics.fmean(values)),
            "std": (float(statistics.stdev(values)) if len(values) > 1 else None),
            "min": min(values),
            "max": max(values),
            "n": len(values),
        }
    return output


def write_raw(rows: List[Dict[str, Any]], directory: str, stem: str) -> str:
    """One row per (controller, seed) run - the raw results, never aggregated."""
    import pandas as pd

    os.makedirs(directory, exist_ok=True)
    columns = list(RESULT_COLUMNS)
    for extra in ("episode_return", "switch_count", "decisions", "model"):
        if extra not in columns:
            columns.append(extra)
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    path = os.path.join(directory, "{}.csv".format(stem))
    frame[columns].to_csv(path, index=False)
    return path


def write_aggregate(aggregated: Dict[str, Dict[str, Optional[float]]],
                    directory: str, stem: str, scenario: str,
                    controller: str) -> str:
    """Long-format mean/std table: one row per metric, kept apart from the raw."""
    import pandas as pd

    os.makedirs(directory, exist_ok=True)
    rows = []
    for metric, stats in aggregated.items():
        rows.append({
            "scenario": scenario,
            "controller": controller,
            "metric": metric,
            "mean": stats["mean"],
            "std": stats["std"],
            "min": stats["min"],
            "max": stats["max"],
            "seeds": stats["n"],
        })
    path = os.path.join(directory, "{}.csv".format(stem))
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def format_mean_std(stats: Dict[str, Optional[float]], digits: int = 2) -> str:
    """'10.04 ± 0.31', or '10.04 (1 seed)' when the deviation is undefined."""
    if stats is None or stats.get("mean") is None:
        return "n/a"
    mean = stats["mean"]
    if stats.get("std") is None:
        return "{:.{d}f} (1 seed)".format(mean, d=digits)
    return "{:.{d}f} ± {:.{d}f}".format(mean, stats["std"], d=digits)


# ===========================================================================
# console output
# ===========================================================================
def _print_header(scenario, model_path, agent, seeds, duration) -> None:
    print("=" * 74)
    print("  IntelliSignal - Stage 3  |  DQN evaluation (greedy, no learning)")
    print("=" * 74)
    print("  Scenario      : {}".format(scenario))
    print("  Model         : {}".format(os.path.relpath(model_path, PROJECT_ROOT)))
    print("  Architecture  : {}".format(agent.architecture()))
    print("  Trained for   : {} episodes".format(agent.episode))
    print("  Epsilon       : {:.2f}  (0 = no exploration)".format(agent.epsilon))
    print("  Seeds         : {}".format(", ".join(str(s) for s in seeds)))
    print("  Duration      : {}".format(
        "{:.0f} s".format(duration) if duration else "scenario default (1800 s)"))
    print("-" * 74)
    print("  seed   wait(s)   queue   maxQ   thru   left  tele  switches")


def _print_row(summary) -> None:
    print("  {seed:>4}  {wait:>8.2f}  {queue:>6.2f}  {maxq:>4}  {thru:>5}  "
          "{left:>5}  {tele:>4}  {sw:>8}".format(
              seed=summary["seed"], wait=summary["average_waiting_time"],
              queue=summary["average_queue_length"],
              maxq=summary["max_queue_length"], thru=summary["throughput"],
              left=summary["vehicles_still_in_network"],
              tele=summary["teleports"], sw=summary.get("switch_count", 0)))


def _print_aggregate(result) -> None:
    print("-" * 74)
    print("  MEAN ± SAMPLE STANDARD DEVIATION over {} seed(s)".format(
        len(result["seeds"])))
    print("-" * 74)
    for metric in METRICS:
        digits = 0 if metric in ("max_queue_length", "throughput",
                                 "vehicles_still_in_network", "teleports") else 2
        print("  {:<38} {}".format(
            METRIC_LABELS[metric],
            format_mean_std(result["aggregate"][metric], digits)))
    print("-" * 74)
    if "raw_csv" in result:
        print("  Raw results   : {}".format(
            os.path.relpath(result["raw_csv"], PROJECT_ROOT)))
        print("  Aggregated    : {}".format(
            os.path.relpath(result["summary_csv"], PROJECT_ROOT)))
    print("  These are DQN numbers only. They say nothing about whether the")
    print("  DQN beat the fixed-time plan - run compare_controllers.py for that.")
    print("=" * 74)


# ===========================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IntelliSignal Stage 3 - evaluate a trained DQN greedily.")
    parser.add_argument("--scenario", "-s", default="moderate",
                        help="Scenario to evaluate on (moderate or heavy).")
    parser.add_argument("--model", "-m", default=None,
                        help="Checkpoint to load "
                             "(default: models/dqn_<scenario>/best.pt).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="One evaluation run per seed, e.g. --seeds 42 43 44.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Shorthand for a single --seeds value.")
    parser.add_argument("--duration", type=float, default=None,
                        help="Simulated seconds per run "
                             "(default: the scenario's 1800 s).")
    parser.add_argument("--results-dir", default=None, dest="results_dir",
                        help="Where to write the CSVs "
                             "(default: results/evaluation).")
    parser.add_argument("--no-save", action="store_true",
                        help="Print the results without writing any file.")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    seeds = [args.seed] if args.seed is not None else args.seeds
    try:
        evaluate(scenario=args.scenario, model=args.model, seeds=seeds,
                 duration=args.duration, results_dir=args.results_dir,
                 save=not args.no_save, quiet=args.quiet)
    except DQNModelNotFoundError as error:
        print("\n{}\n".format(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
