"""
IntelliSignal - Stage 1
plot_results.py

Creates the Stage 1 result graphs.

Every number plotted here is read from the CSV files in results/csv/, which
are written by run_simulation.py directly from the SUMO simulation. No value
is typed into this file by hand - if a CSV is missing, the corresponding
graph is simply not produced.

Usage
-----
    python simulation/plot_results.py
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")           # render to files, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd              # noqa: E402

CSV_DIR = os.path.join(PROJECT_ROOT, "results", "csv")
PLOT_DIR = os.path.join(PROJECT_ROOT, "results", "plots")

# Scenarios are plotted in this order, each with its own fixed colour.
SCENARIO_ORDER = ["moderate", "heavy"]
SCENARIO_LABEL = {"moderate": "Moderate traffic", "heavy": "Heavy traffic"}
SCENARIO_COLOR = {"moderate": "#3b6fb6", "heavy": "#d98324"}

SURFACE = "#fcfcfb"
INK = "#1f2326"
INK_MUTED = "#6b7075"
GRID = "#dcdedf"


# --------------------------------------------------------------------------
def load_summary() -> pd.DataFrame:
    """Read every <scenario>_results.csv that exists into one DataFrame."""
    frames = []
    for scenario in SCENARIO_ORDER:
        path = os.path.join(CSV_DIR, "{}_results.csv".format(scenario))
        if os.path.isfile(path):
            frames.append(pd.read_csv(path))
        else:
            print("  (skipped) no results yet for '{}': {}".format(
                scenario, os.path.relpath(path, PROJECT_ROOT)))
    if not frames:
        raise SystemExit(
            "No result CSV files found in results/csv/.\n"
            "Run the scenarios first:\n"
            "    python simulation/run_simulation.py --scenario moderate\n"
            "    python simulation/run_simulation.py --scenario heavy")
    return pd.concat(frames, ignore_index=True)


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, length=0, labelsize=10)


def bar_chart(df: pd.DataFrame, column: str, title: str, ylabel: str,
              value_format: str, filename: str) -> str:
    """One metric, one bar per scenario, value printed on top of the bar."""
    labels = [SCENARIO_LABEL.get(s, s) for s in df["scenario"]]
    colors = [SCENARIO_COLOR.get(s, "#7a7f85") for s in df["scenario"]]
    values = df[column].astype(float).tolist()

    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)

    bars = ax.bar(labels, values, width=0.5, color=colors)
    for rect, value in zip(bars, values):
        ax.annotate(value_format.format(value),
                    xy=(rect.get_x() + rect.get_width() / 2, value),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=11, color=INK)

    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=14)
    ax.set_ylabel(ylabel, fontsize=10, color=INK_MUTED)
    ax.set_ylim(0, max(values) * 1.18 if values else 1)

    fig.text(0.01, 0.01,
             "IntelliSignal Stage 1 - fixed-time signal - measured in SUMO via TraCI",
             fontsize=7.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def queue_over_time() -> str:
    """Queue length second by second, taken from the time-series CSV files."""
    series = {}
    for scenario in SCENARIO_ORDER:
        path = os.path.join(CSV_DIR, "{}_timeseries.csv".format(scenario))
        if os.path.isfile(path):
            series[scenario] = pd.read_csv(path)
    if not series:
        return ""

    fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)

    for scenario, frame in series.items():
        ax.plot(frame["time"], frame["queue_length"],
                linewidth=1.4, color=SCENARIO_COLOR.get(scenario, "#7a7f85"),
                label=SCENARIO_LABEL.get(scenario, scenario))

    ax.set_title("Queue length at the intersection over time",
                 loc="left", fontsize=13, color=INK, pad=14)
    ax.set_xlabel("Simulation time (s)", fontsize=10, color=INK_MUTED)
    ax.set_ylabel("Halting vehicles on the 8 incoming lanes",
                  fontsize=10, color=INK_MUTED)
    legend = ax.legend(frameon=False, fontsize=10, loc="upper left")
    for text in legend.get_texts():
        text.set_color(INK)

    fig.text(0.01, 0.01,
             "IntelliSignal Stage 1 - measured in SUMO via TraCI, one sample per simulation second",
             fontsize=7.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    path = os.path.join(PLOT_DIR, "04_queue_length_over_time.png")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def summary_panel(df: pd.DataFrame) -> str:
    """The three Stage 1 metrics side by side - one figure for a slide."""
    labels = [SCENARIO_LABEL.get(s, s).replace(" traffic", "") for s in df["scenario"]]
    colors = [SCENARIO_COLOR.get(s, "#7a7f85") for s in df["scenario"]]

    panels = [
        ("average_waiting_time", "Average waiting time", "seconds / vehicle", "{:.1f}"),
        ("average_queue_length", "Average queue length", "vehicles", "{:.1f}"),
        ("throughput", "Throughput", "vehicles completed", "{:.0f}"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.0), dpi=150)
    fig.patch.set_facecolor(SURFACE)

    for ax, (column, title, ylabel, fmt) in zip(axes, panels):
        _style_axes(ax)
        values = df[column].astype(float).tolist()
        bars = ax.bar(labels, values, width=0.5, color=colors)
        for rect, value in zip(bars, values):
            ax.annotate(fmt.format(value),
                        xy=(rect.get_x() + rect.get_width() / 2, value),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, color=INK)
        ax.set_title(title, loc="left", fontsize=11.5, color=INK, pad=10)
        ax.set_ylabel(ylabel, fontsize=9, color=INK_MUTED)
        ax.set_ylim(0, max(values) * 1.20 if values else 1)

    fig.suptitle("Stage 1 baseline results - moderate vs heavy traffic",
                 x=0.012, ha="left", fontsize=14, color=INK)
    fig.text(0.012, 0.015,
             "IntelliSignal Stage 1 - fixed-time signal - measured in SUMO via TraCI",
             fontsize=7.5, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))

    path = os.path.join(PLOT_DIR, "05_summary_comparison.png")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
def generate_plots() -> list:
    """Create every Stage 1 graph from the CSV results. Returns the paths."""
    os.makedirs(PLOT_DIR, exist_ok=True)
    df = load_summary()

    print("Generating graphs from:", os.path.relpath(CSV_DIR, PROJECT_ROOT))
    created = [
        bar_chart(df, "average_waiting_time",
                  "Average waiting time per vehicle",
                  "seconds / vehicle", "{:.2f} s",
                  "01_average_waiting_time.png"),
        bar_chart(df, "average_queue_length",
                  "Average queue length at the intersection",
                  "vehicles", "{:.2f}",
                  "02_average_queue_length.png"),
        bar_chart(df, "throughput",
                  "Throughput - vehicles that passed through",
                  "vehicles", "{:.0f}",
                  "03_throughput.png"),
    ]
    ts = queue_over_time()
    if ts:
        created.append(ts)
    created.append(summary_panel(df))

    for path in created:
        print("  saved:", os.path.relpath(path, PROJECT_ROOT))
    return created


if __name__ == "__main__":
    generate_plots()
