"""
IntelliSignal - Stage 3
training/plot_training.py

Draws the training curves from a training history CSV.

    python training/plot_training.py --scenario moderate --seed 42

Produces, in results/plots/training/:

    dqn_<scenario>_seed<seed>_reward.png     return per episode
    dqn_<scenario>_seed<seed>_loss.png       mean TD loss per episode
    dqn_<scenario>_seed<seed>_epsilon.png    exploration rate per episode
    dqn_<scenario>_seed<seed>_queue.png      average queue per episode
    dqn_<scenario>_seed<seed>_waiting.png    average waiting time per episode
    dqn_<scenario>_seed<seed>_overview.png   all five on one sheet

Honesty rules this file follows
-------------------------------
  * Every point is a value read from the CSV that the training loop wrote.
    Nothing is smoothed into the data, interpolated or invented.
  * A moving average is drawn ON TOP of the raw curve, never instead of it,
    and the legend says how long the window is. Traffic reinforcement
    learning is noisy and the noise is part of the result.
  * No trend line is fitted and no axis is cropped to make a curve look
    better than it is.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib                                                  # noqa: E402

matplotlib.use("Agg")                   # headless: write files, open no window
import matplotlib.pyplot as plt                                    # noqa: E402

PLOTS_DIR = os.path.join(PROJECT_ROOT, "results", "plots", "training")

RAW_COLOUR = "#9aa3ab"
LINE_COLOUR = "#1f77b4"
ACCENT = "#d62728"

#: (column, title, y label, colour)
CURVES = [
    ("total_reward", "DQN return per episode", "Total reward", LINE_COLOUR),
    ("loss", "DQN training loss per episode", "Mean TD loss", ACCENT),
    ("epsilon", "Exploration rate per episode", "Epsilon", "#2ca02c"),
    ("average_queue_length", "Average queue length per episode",
     "Vehicles", "#ff7f0e"),
    ("average_waiting_time", "Average waiting time per episode",
     "Seconds per vehicle", "#9467bd"),
]


def _moving_average(values, window: int):
    """Simple trailing moving average, ignoring missing values."""
    output = []
    buffer: List[float] = []
    for value in values:
        if value is None or value != value:          # NaN
            output.append(None)
            continue
        buffer.append(float(value))
        if len(buffer) > window:
            buffer.pop(0)
        output.append(sum(buffer) / len(buffer))
    return output


def _draw(axis, episodes, values, title, ylabel, colour, window: int) -> bool:
    """One curve: raw values plus a clearly labelled moving average."""
    pairs = [(e, v) for e, v in zip(episodes, values)
             if v is not None and v == v]
    if not pairs:
        axis.set_axis_off()
        return False

    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    axis.plot(x, y, color=RAW_COLOUR, linewidth=1.0, marker="o",
              markersize=2.5, label="per episode (raw)")
    if len(y) >= window > 1:
        axis.plot(x, _moving_average(y, window), color=colour, linewidth=2.0,
                  label="{}-episode moving average".format(window))
    axis.set_title(title, fontsize=11)
    axis.set_xlabel("Episode")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8, loc="best")
    return True


def plot_training_history(csv_path: str, output_dir: str = None,
                          window: int = 10) -> List[str]:
    """Draw every curve for one training run. Returns the files written."""
    import pandas as pd

    if not os.path.isfile(csv_path):
        raise FileNotFoundError("Training history not found: {}".format(csv_path))

    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError("Training history is empty: {}".format(csv_path))

    output_dir = output_dir or PLOTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.basename(csv_path).replace("_training.csv", "")
    episodes = frame["episode"].tolist()
    window = max(1, min(int(window), max(len(episodes) // 3, 1)))

    written: List[str] = []

    # --- one file per curve ------------------------------------------
    short = {"total_reward": "reward", "loss": "loss", "epsilon": "epsilon",
             "average_queue_length": "queue", "average_waiting_time": "waiting"}
    for column, title, ylabel, colour in CURVES:
        if column not in frame.columns:
            continue
        figure, axis = plt.subplots(figsize=(7.2, 4.0))
        drawn = _draw(axis, episodes, frame[column].tolist(), title, ylabel,
                      colour, window)
        if drawn:
            figure.tight_layout()
            path = os.path.join(output_dir, "{}_{}.png".format(
                stem, short.get(column, column)))
            figure.savefig(path, dpi=140)
            written.append(path)
        plt.close(figure)

    # --- one overview sheet -------------------------------------------
    figure, axes = plt.subplots(3, 2, figsize=(13, 11))
    flat = axes.flatten()
    for index, (column, title, ylabel, colour) in enumerate(CURVES):
        if column in frame.columns:
            _draw(flat[index], episodes, frame[column].tolist(), title,
                  ylabel, colour, window)
        else:
            flat[index].set_axis_off()
    flat[5].set_axis_off()
    flat[5].text(0.5, 0.5,
                 "Raw values as recorded.\nNo smoothing is applied to the\n"
                 "underlying data; the coloured line is a\n"
                 "{}-episode moving average drawn on top.".format(window),
                 ha="center", va="center", fontsize=10, color="#555")
    figure.suptitle("IntelliSignal Stage 3 - {}".format(stem), fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    path = os.path.join(output_dir, "{}_overview.png".format(stem))
    figure.savefig(path, dpi=130)
    plt.close(figure)
    written.append(path)

    return written


# ===========================================================================
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Draw the DQN training curves from a training history CSV.")
    parser.add_argument("--csv", default=None,
                        help="Path to a training history CSV.")
    parser.add_argument("--scenario", "-s", default="moderate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window", type=int, default=10,
                        help="Moving-average window, in episodes.")
    args = parser.parse_args(argv)

    csv_path = args.csv
    if csv_path is None:
        from training.train_dqn import history_path
        csv_path = history_path(args.scenario, args.seed)

    for path in plot_training_history(csv_path, window=args.window):
        print("  Saved: {}".format(os.path.relpath(path, PROJECT_ROOT)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
