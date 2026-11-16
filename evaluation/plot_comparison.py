"""
IntelliSignal - Stage 3
evaluation/plot_comparison.py

Draws the Fixed-Time vs DQN comparison charts.

    python evaluation/plot_comparison.py --scenario moderate

Reads only the CSV files that `evaluation/compare_controllers.py` wrote, so
every bar and every line is a measurement from a real SUMO run. Nothing is
typed in, and no value is drawn that is not in a CSV.

Produces, in results/plots/comparison/:

    <scenario>_waiting_time.png     average waiting time, mean over seeds
    <scenario>_queue_length.png     average queue length
    <scenario>_throughput.png       throughput
    <scenario>_max_queue.png        maximum queue length
    <scenario>_queue_over_time.png  per-second queue length, both controllers
    <scenario>_summary.png          the four bar charts on one sheet

Error bars are the sample standard deviation across the evaluation seeds.
With a single seed there is no deviation to draw and none is drawn - an
invisible error bar is honest, a fabricated one is not.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib                                                   # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

from evaluation.evaluate_dqn import EVALUATION_RESULTS_DIR          # noqa: E402

PLOTS_DIR = os.path.join(PROJECT_ROOT, "results", "plots", "comparison")

COLOURS = {
    "fixed_time": "#4c78a8",
    "dqn": "#e45756",
    "dqn_untrained": "#b0b4b9",
}
LABELS = {
    "fixed_time": "Fixed-Time",
    "dqn": "DQN",
    "dqn_untrained": "DQN (untrained control)",
}

#: (metric, chart title, y label, file suffix)
BAR_CHARTS = [
    ("average_waiting_time", "Average waiting time",
     "Seconds per vehicle", "waiting_time"),
    ("average_queue_length", "Average queue length",
     "Vehicles", "queue_length"),
    ("throughput", "Throughput", "Vehicles completed", "throughput"),
    ("max_queue_length", "Maximum queue length", "Vehicles", "max_queue"),
]


# ===========================================================================
def _load(scenario: str, results_dir: str):
    import pandas as pd

    raw_path = os.path.join(results_dir,
                            "comparison_{}_raw.csv".format(scenario))
    aggregated_path = os.path.join(
        results_dir, "comparison_{}_aggregated.csv".format(scenario))
    if not os.path.isfile(aggregated_path):
        raise FileNotFoundError(
            "No comparison results for '{}'. Run:\n"
            "    python evaluation/compare_controllers.py --scenario {} "
            "--seeds 42 43 44".format(scenario, scenario))
    raw = pd.read_csv(raw_path) if os.path.isfile(raw_path) else None
    return raw, pd.read_csv(aggregated_path)


def _controllers(frame) -> List[str]:
    """Controllers in a stable, meaningful order: baseline first."""
    present = list(dict.fromkeys(frame["controller"].tolist()))
    order = ["fixed_time", "dqn", "dqn_untrained"]
    return [c for c in order if c in present] + [c for c in present
                                                 if c not in order]


def _bar(axis, aggregated, metric: str, title: str, ylabel: str,
         seeds: Optional[int] = None) -> bool:
    rows = aggregated[aggregated["metric"] == metric]
    if rows.empty:
        axis.set_axis_off()
        return False

    controllers = _controllers(rows)
    means, errors, colours, labels = [], [], [], []
    for controller in controllers:
        row = rows[rows["controller"] == controller].iloc[0]
        means.append(float(row["mean"]))
        std = row["std"]
        errors.append(0.0 if std != std or std is None else float(std))
        colours.append(COLOURS.get(controller, "#888888"))
        labels.append(LABELS.get(controller, controller))

    positions = range(len(means))
    bars = axis.bar(positions, means,
                    yerr=errors if any(errors) else None,
                    capsize=6, color=colours, width=0.55,
                    error_kw={"ecolor": "#333", "elinewidth": 1.2})
    axis.set_xticks(list(positions))
    axis.set_xticklabels(labels, fontsize=9)
    axis.set_title(title, fontsize=11)
    axis.set_ylabel(ylabel)
    axis.grid(True, axis="y", alpha=0.3)
    axis.set_axisbelow(True)

    span = max(means) - min(means + [0.0])
    for bar, mean in zip(bars, means):
        axis.text(bar.get_x() + bar.get_width() / 2,
                  bar.get_height() + max(span, max(means)) * 0.02,
                  "{:.2f}".format(mean) if mean < 100 else "{:.0f}".format(mean),
                  ha="center", va="bottom", fontsize=9)
    if seeds:
        axis.set_xlabel("error bars: sample std over {} seeds".format(seeds),
                        fontsize=8, color="#666")
    return True


# ===========================================================================
def _queue_over_time(axis, scenario: str, results_dir: str, seed: int) -> bool:
    """Per-second queue length for both controllers, same seed, same traffic."""
    import pandas as pd

    drawn = False
    for controller in ("fixed_time", "dqn"):
        path = os.path.join(results_dir,
                            "comparison_{}_{}_seed{}_timeseries.csv".format(
                                scenario, controller, seed))
        if not os.path.isfile(path):
            continue
        frame = pd.read_csv(path)
        if frame.empty or "queue_length" not in frame.columns:
            continue
        axis.plot(frame["time"], frame["queue_length"],
                  color=COLOURS[controller], linewidth=0.9,
                  label=LABELS[controller], alpha=0.85)
        drawn = True

    if not drawn:
        axis.set_axis_off()
        return False

    axis.set_title("Queue length over time - {} traffic, seed {}".format(
        scenario, seed), fontsize=11)
    axis.set_xlabel("Simulation time (s)")
    axis.set_ylabel("Halting vehicles (8 incoming lanes)")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=9)
    return True


# ===========================================================================
def plot_comparison(scenario: str = "moderate", results_dir: str = None,
                    output_dir: str = None) -> List[str]:
    """Draw every comparison chart for one scenario. Returns the files written."""
    results_dir = results_dir or EVALUATION_RESULTS_DIR
    output_dir = output_dir or PLOTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    raw, aggregated = _load(scenario, results_dir)
    seeds = None
    if raw is not None and "seed" in raw.columns:
        seeds = int(raw["seed"].nunique())
    first_seed = int(raw["seed"].min()) if raw is not None else 42

    written: List[str] = []

    for metric, title, ylabel, suffix in BAR_CHARTS:
        figure, axis = plt.subplots(figsize=(6.0, 4.4))
        if _bar(axis, aggregated, metric,
                "{} - {} traffic".format(title, scenario), ylabel, seeds):
            figure.tight_layout()
            path = os.path.join(output_dir, "{}_{}.png".format(scenario, suffix))
            figure.savefig(path, dpi=140)
            written.append(path)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 4.4))
    if _queue_over_time(axis, scenario, results_dir, first_seed):
        figure.tight_layout()
        path = os.path.join(output_dir,
                            "{}_queue_over_time.png".format(scenario))
        figure.savefig(path, dpi=140)
        written.append(path)
    plt.close(figure)

    # --- one summary sheet --------------------------------------------
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    for axis, (metric, title, ylabel, _) in zip(axes.flatten(), BAR_CHARTS):
        _bar(axis, aggregated, metric, title, ylabel, seeds)
    figure.suptitle(
        "IntelliSignal Stage 3 - Fixed-Time vs DQN, {} traffic "
        "({} seed{}, 1800 s each)".format(
            scenario, seeds or 1, "s" if (seeds or 1) != 1 else ""),
        fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(output_dir, "{}_summary.png".format(scenario))
    figure.savefig(path, dpi=130)
    plt.close(figure)
    written.append(path)

    return written


# ===========================================================================
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Draw the Fixed-Time vs DQN comparison charts from the "
                    "evaluation CSVs.")
    parser.add_argument("--scenario", "-s", default="moderate")
    parser.add_argument("--results-dir", default=None, dest="results_dir")
    args = parser.parse_args(argv)

    for path in plot_comparison(args.scenario, results_dir=args.results_dir):
        print("  Saved: {}".format(os.path.relpath(path, PROJECT_ROOT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
