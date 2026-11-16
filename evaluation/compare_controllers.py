"""
IntelliSignal - Stage 3
evaluation/compare_controllers.py

FIXED-TIME vs DQN, on identical traffic.

Usage
-----
    python evaluation/compare_controllers.py --scenario moderate --seeds 42 43 44
    python evaluation/compare_controllers.py --scenario heavy --seeds 42 43 44

    # add an untrained network as a control (see "Why the control" below)
    python evaluation/compare_controllers.py --scenario moderate --seeds 42 43 44 \
        --include-untrained

What makes the comparison fair
-------------------------------
For each seed, ONE set of SUMO input files is built and both controllers run
on it:

    same road network      sumo/network/intersection.net.xml (unchanged)
    same traffic demand    the same generated route file, same seed
    same duration          the same number of simulated seconds
    same measurements      utils.metrics.MetricsCollector, the class that
                           produced the Stage 1 baseline

The only difference is who decides when the signal changes: SUMO's fixed-time
program, or the trained Q-network. Both runs go through
simulation/agent_simulation.py, so there is one code path and not two.

Why the control
---------------
The DQN acts every 10 s and may switch as often as min_green allows, while
the fixed-time plan holds 30 s of green per direction. Part of any difference
could therefore come from the shorter decision interval rather than from
anything the network learned. `--include-untrained` runs the SAME
architecture with RANDOM weights on the same traffic. If the trained network
is no better than the untrained one, the learning did not do the work, and
that is worth knowing before claiming it did.

Output
------
    results/evaluation/comparison_<scenario>_raw.csv
        one row per (controller, seed) - every individual run, nothing hidden
    results/evaluation/comparison_<scenario>_aggregated.csv
        one row per (controller, metric): mean, std, min, max
    results/evaluation/comparison_<scenario>_improvement.csv
        one row per metric: fixed-time mean, DQN mean, improvement %
    results/evaluation/comparison_<scenario>_<controller>_seed<seed>_timeseries.csv
        per-second queue length, for the queue-over-time chart

The recorded Stage 1 baseline in results/csv/ is read for reference and is
never written.
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
                              set_seeds)
from evaluation.evaluate_dqn import (EVALUATION_RESULTS_DIR,     # noqa: E402
                                     METRIC_LABELS, METRICS, aggregate,
                                     format_mean_std, resolve_model,
                                     write_raw)
from simulation.agent_simulation import (DQN, FIXED_TIME,        # noqa: E402
                                         LOWER_IS_BETTER, improvement,
                                         run_agent_episode,
                                         run_fixed_time_episode, save_run)

#: An untrained network, used only as a control. Never presented as a result.
UNTRAINED = "dqn_untrained"

CONTROLLER_LABELS = {
    FIXED_TIME: "Fixed-Time",
    DQN: "DQN",
    UNTRAINED: "DQN (untrained control)",
}

BASELINE_CSV_DIR = os.path.join(PROJECT_ROOT, "results", "csv")


# ===========================================================================
def recorded_baseline(scenario: str) -> Optional[Dict[str, Any]]:
    """The Stage 1 baseline row from results/csv/, for reference only."""
    import pandas as pd

    path = os.path.join(BASELINE_CSV_DIR, "{}_results.csv".format(scenario))
    if not os.path.isfile(path):
        return None
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    if frame.empty:
        return None
    row = frame.iloc[0].to_dict()
    row["_path"] = path
    return row


# ===========================================================================
def compare(scenario: str = "moderate", model: Optional[str] = None,
            seeds: List[int] = None, duration: Optional[float] = None,
            results_dir: Optional[str] = None,
            include_untrained: bool = False,
            save: bool = True, quiet: bool = False) -> Dict[str, Any]:
    """Run both controllers on every seed and return the whole comparison."""
    seeds = [int(s) for s in (seeds or [42, 43, 44])]
    results_dir = results_dir or EVALUATION_RESULTS_DIR
    model_path = resolve_model(scenario, model)

    agent = DQNAgent.load(model_path, scenario=scenario)
    agent.set_epsilon(0.0)

    control = None
    if include_untrained:
        # Same architecture, random weights, same seed for its initialisation.
        set_seeds(0)
        control = DQNAgent(observation_size=agent.observation_size,
                           num_actions=agent.num_actions,
                           config=agent.config, trainable=True, seed=0)
        control.set_epsilon(0.0)

    if not quiet:
        _print_header(scenario, model_path, agent, seeds, duration,
                      include_untrained)

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        # --- fixed-time, on the files this seed builds ------------------
        fixed_summary, fixed_ts = run_fixed_time_episode(
            scenario, seed=seed, duration=duration)
        rows.append(fixed_summary)
        if save:
            save_run(fixed_summary, fixed_ts, results_dir,
                     "comparison_{}_fixed_time_seed{}".format(scenario, seed))

        # --- the trained DQN, on the SAME files -------------------------
        set_seeds(seed)
        dqn_summary, dqn_ts = run_agent_episode(
            scenario, agent, seed=seed, duration=duration, controller=DQN)
        dqn_summary["model"] = os.path.relpath(model_path, PROJECT_ROOT)
        rows.append(dqn_summary)
        if save:
            save_run(dqn_summary, dqn_ts, results_dir,
                     "comparison_{}_dqn_seed{}".format(scenario, seed))

        # --- the untrained control, if asked for ------------------------
        if control is not None:
            set_seeds(seed)
            control_summary, control_ts = run_agent_episode(
                scenario, control, seed=seed, duration=duration,
                controller=UNTRAINED)
            control_summary["model"] = "random weights (control)"
            rows.append(control_summary)
            if save:
                save_run(control_summary, control_ts, results_dir,
                         "comparison_{}_untrained_seed{}".format(scenario, seed))

        if not quiet:
            _print_seed(seed, rows)

    controllers = [FIXED_TIME, DQN] + ([UNTRAINED] if control is not None else [])
    aggregates = {
        controller: aggregate([r for r in rows if r["controller"] == controller],
                              METRICS)
        for controller in controllers
    }
    improvements = improvement_table(aggregates[FIXED_TIME], aggregates[DQN])

    result = {
        "scenario": scenario,
        "model": model_path,
        "seeds": seeds,
        "controllers": controllers,
        "rows": rows,
        "aggregates": aggregates,
        "improvements": improvements,
        "recorded_baseline": recorded_baseline(scenario),
        "duration": duration,
    }

    if save:
        result["raw_csv"] = write_raw(
            rows, results_dir, "comparison_{}_raw".format(scenario))
        result["aggregated_csv"] = _write_aggregated(
            aggregates, results_dir, "comparison_{}_aggregated".format(scenario),
            scenario)
        result["improvement_csv"] = _write_improvement(
            improvements, results_dir,
            "comparison_{}_improvement".format(scenario), scenario)
        try:
            from evaluation.plot_comparison import plot_comparison
            result["plots"] = plot_comparison(scenario,
                                              results_dir=results_dir)
        except Exception as exc:                    # pragma: no cover
            result["plots"] = []
            if not quiet:
                print("  (comparison charts were not generated: {})".format(exc))

    if not quiet:
        _print_comparison(result)
    return result


# ===========================================================================
def improvement_table(fixed: Dict[str, Dict[str, Optional[float]]],
                      dqn: Dict[str, Dict[str, Optional[float]]]
                      ) -> List[Dict[str, Any]]:
    """Per-metric improvement of the DQN over fixed-time, on the means.

    A positive percentage always means the DQN is better, for every metric.
    A negative percentage means it is worse and is reported as such - the
    sign is never flipped and a bad row is never dropped.
    """
    table = []
    for metric in METRICS:
        fixed_mean = fixed.get(metric, {}).get("mean")
        dqn_mean = dqn.get(metric, {}).get("mean")
        percent = improvement(fixed_mean, dqn_mean, metric)
        table.append({
            "metric": metric,
            "label": METRIC_LABELS[metric],
            "lower_is_better": bool(LOWER_IS_BETTER.get(metric, True)),
            "fixed_time_mean": fixed_mean,
            "fixed_time_std": fixed.get(metric, {}).get("std"),
            "dqn_mean": dqn_mean,
            "dqn_std": dqn.get(metric, {}).get("std"),
            "difference": (None if fixed_mean is None or dqn_mean is None
                           else float(dqn_mean) - float(fixed_mean)),
            "improvement_percent": percent,
            "dqn_better": (None if percent is None else bool(percent > 0)),
        })
    return table


# ===========================================================================
def _write_aggregated(aggregates, directory: str, stem: str,
                      scenario: str) -> str:
    import pandas as pd

    os.makedirs(directory, exist_ok=True)
    rows = []
    for controller, metrics in aggregates.items():
        for metric, stats in metrics.items():
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


def _write_improvement(table, directory: str, stem: str, scenario: str) -> str:
    import pandas as pd

    os.makedirs(directory, exist_ok=True)
    frame = pd.DataFrame(table)
    frame.insert(0, "scenario", scenario)
    path = os.path.join(directory, "{}.csv".format(stem))
    frame.to_csv(path, index=False)
    return path


# ===========================================================================
# console output
# ===========================================================================
def _print_header(scenario, model_path, agent, seeds, duration,
                  include_untrained) -> None:
    print("=" * 78)
    print("  IntelliSignal - Stage 3  |  Fixed-Time vs DQN")
    print("=" * 78)
    print("  Scenario   : {}".format(scenario))
    print("  Model      : {}  ({} episodes trained)".format(
        os.path.relpath(model_path, PROJECT_ROOT), agent.episode))
    print("  Seeds      : {}".format(", ".join(str(s) for s in seeds)))
    print("  Duration   : {}".format(
        "{:.0f} s".format(duration) if duration else "1800 s (scenario default)"))
    print("  Controllers: fixed-time, DQN{}".format(
        ", untrained control" if include_untrained else ""))
    print("  Both controllers run on the SAME generated network, demand and")
    print("  seed, and are measured by the same Stage 1 metrics collector.")
    print("-" * 78)


def _print_seed(seed, rows) -> None:
    print("  seed {}".format(seed))
    print("    {:<26} {:>9} {:>8} {:>6} {:>7} {:>6}".format(
        "controller", "wait(s)", "queue", "maxQ", "thru", "tele"))
    for row in [r for r in rows if r["seed"] == seed]:
        print("    {:<26} {:>9.2f} {:>8.2f} {:>6} {:>7} {:>6}".format(
            CONTROLLER_LABELS.get(row["controller"], row["controller"]),
            row["average_waiting_time"], row["average_queue_length"],
            row["max_queue_length"], row["throughput"], row["teleports"]))


def _print_comparison(result) -> None:
    print("-" * 78)
    print("  RESULTS - mean ± sample standard deviation over {} seeds".format(
        len(result["seeds"])))
    print("-" * 78)
    header = "  {:<32}".format("metric")
    for controller in result["controllers"]:
        header += " {:>20}".format(CONTROLLER_LABELS[controller])
    header += " {:>13}".format("improvement")
    print(header)

    for row in result["improvements"]:
        metric = row["metric"]
        digits = 0 if metric in ("max_queue_length", "throughput",
                                 "vehicles_still_in_network", "teleports") else 2
        line = "  {:<32}".format(METRIC_LABELS[metric])
        for controller in result["controllers"]:
            line += " {:>20}".format(
                format_mean_std(result["aggregates"][controller][metric], digits))
        percent = row["improvement_percent"]
        line += " {:>13}".format(
            "n/a" if percent is None else "{:+.1f} %".format(percent))
        print(line)

    print("-" * 78)
    print("  A POSITIVE improvement means the DQN did better; a NEGATIVE one")
    print("  means it did worse. Nothing is hidden and no sign is flipped.")

    baseline = result.get("recorded_baseline")
    if baseline:
        print("-" * 78)
        print("  Reference - the recorded Stage 1 baseline (results/csv/, "
              "seed 42, untouched):")
        print("    waiting {:.2f} s | queue {:.2f} | max queue {} | throughput {}"
              .format(baseline.get("average_waiting_time", float("nan")),
                      baseline.get("average_queue_length", float("nan")),
                      baseline.get("max_queue_length", "?"),
                      baseline.get("throughput", "?")))

    print("-" * 78)
    for key, label in (("raw_csv", "Raw results  "),
                       ("aggregated_csv", "Aggregated   "),
                       ("improvement_csv", "Improvement  ")):
        if key in result:
            print("  {}: {}".format(label,
                                    os.path.relpath(result[key], PROJECT_ROOT)))
    for plot in result.get("plots", []):
        print("  Chart        : {}".format(os.path.relpath(plot, PROJECT_ROOT)))
    print("=" * 78)


# ===========================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IntelliSignal Stage 3 - compare the fixed-time baseline "
                    "against a trained DQN on identical traffic.")
    parser.add_argument("--scenario", "-s", default="moderate",
                        help="Scenario to compare on (moderate or heavy).")
    parser.add_argument("--model", "-m", default=None,
                        help="Checkpoint to use "
                             "(default: models/dqn_<scenario>/best.pt).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                        help="Seeds to run both controllers on.")
    parser.add_argument("--duration", type=float, default=None,
                        help="Simulated seconds per run "
                             "(default: the scenario's 1800 s).")
    parser.add_argument("--results-dir", default=None, dest="results_dir",
                        help="Where to write the CSVs "
                             "(default: results/evaluation).")
    parser.add_argument("--include-untrained", action="store_true",
                        dest="include_untrained",
                        help="Also run an untrained network as a control.")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        compare(scenario=args.scenario, model=args.model, seeds=args.seeds,
                duration=args.duration, results_dir=args.results_dir,
                include_untrained=args.include_untrained,
                save=not args.no_save, quiet=args.quiet)
    except DQNModelNotFoundError as error:
        print("\n{}\n".format(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
