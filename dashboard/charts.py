"""
IntelliSignal - Dashboard
dashboard/charts.py

Every chart the dashboard draws.

All of them are built from the CSV files the simulation produced. No chart
function invents a number, and none of them recalculates a metric that
`utils/metrics.py` already measured - they read and draw.

Design rules followed here:
  * one colour per scenario, assigned by identity and never by rank, so a
    scenario keeps its colour on every chart and every page
  * a value is printed on every bar, so the chart never depends on colour
    alone and is readable in print or with colour-vision deficiency
  * thin marks, recessive grid lines, no chart junk
  * a tooltip on every mark, because a dashboard chart is meant to be probed
"""

from __future__ import annotations

from typing import List, Optional

import altair as alt
import pandas as pd

from dashboard.theme import (BORDER, INK, INK_MUTED, SCENARIO_COLORS,
                             SCENARIO_LABELS, scenario_label)

# Fixed order, so the colour scale never re-assigns colours when a scenario
# is missing from the data.
SCENARIO_DOMAIN: List[str] = ["Moderate traffic", "Heavy traffic"]
SCENARIO_RANGE: List[str] = [SCENARIO_COLORS["moderate"], SCENARIO_COLORS["heavy"]]

_AXIS = dict(labelColor=INK_MUTED, titleColor=INK_MUTED, labelFontSize=11,
             titleFontSize=11, tickColor=BORDER, domainColor=BORDER,
             gridColor="#eceef0")


def _base(frame: pd.DataFrame, height: int = 260) -> alt.Chart:
    return alt.Chart(frame).properties(height=height)


def _color_scale() -> alt.Scale:
    return alt.Scale(domain=SCENARIO_DOMAIN, range=SCENARIO_RANGE)


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the display label used as the categorical key."""
    out = frame.copy()
    out["Scenario"] = out["scenario"].map(scenario_label)
    return out


# ==========================================================================
# Bar charts - one metric, one bar per scenario
# ==========================================================================
def metric_bar(frame: pd.DataFrame, column: str, title: str,
               axis_title: str, value_format: str = ".2f",
               height: int = 270) -> Optional[alt.LayerChart]:
    """A bar per scenario for one metric, with the value printed on the bar."""
    if frame is None or frame.empty or column not in frame.columns:
        return None
    data = _prepare(frame.dropna(subset=[column]))
    if data.empty:
        return None

    bars = _base(data, height).mark_bar(
        size=52, cornerRadiusTopLeft=4, cornerRadiusTopRight=4
    ).encode(
        x=alt.X("Scenario:N", title=None, axis=alt.Axis(labelAngle=0, **_AXIS),
                sort=SCENARIO_DOMAIN),
        y=alt.Y("{}:Q".format(column), title=axis_title,
                axis=alt.Axis(**_AXIS)),
        color=alt.Color("Scenario:N", scale=_color_scale(), legend=None),
        tooltip=[
            alt.Tooltip("Scenario:N", title="Scenario"),
            alt.Tooltip("{}:Q".format(column), title=axis_title, format=value_format),
            alt.Tooltip("source:N", title="Source"),
        ],
    )
    labels = _base(data, height).mark_text(
        dy=-9, color=INK, fontSize=13
    ).encode(
        x=alt.X("Scenario:N", sort=SCENARIO_DOMAIN),
        y=alt.Y("{}:Q".format(column)),
        text=alt.Text("{}:Q".format(column), format=value_format),
    )
    return (bars + labels).properties(title=alt.TitleParams(
        text=title, anchor="start", color=INK, fontSize=13, fontWeight=600,
        offset=10))


def waiting_time_chart(frame):
    return metric_bar(frame, "average_waiting_time",
                      "Average waiting time per vehicle",
                      "seconds / vehicle", ".2f")


def queue_length_chart(frame):
    return metric_bar(frame, "average_queue_length",
                      "Average queue length at the intersection",
                      "vehicles", ".2f")


def throughput_chart(frame):
    return metric_bar(frame, "throughput",
                      "Throughput — vehicles that passed through",
                      "vehicles", ".0f")


# ==========================================================================
# Line chart - queue length over time
# ==========================================================================
def queue_over_time_chart(series: dict, height: int = 320) -> Optional[alt.Chart]:
    """Recorded queue length second by second, one line per scenario.

    `series` maps a scenario name to its timeseries DataFrame.
    """
    frames = []
    for scenario, frame in (series or {}).items():
        if frame is None or frame.empty or "queue_length" not in frame.columns:
            continue
        part = frame[["time", "queue_length"]].copy()
        part["Scenario"] = scenario_label(scenario)
        frames.append(part)
    if not frames:
        return None

    data = pd.concat(frames, ignore_index=True)
    hover = alt.selection_point(fields=["time"], nearest=True, on="mouseover",
                                empty=False)

    lines = alt.Chart(data).mark_line(strokeWidth=1.4).encode(
        x=alt.X("time:Q", title="Simulation time (s)", axis=alt.Axis(**_AXIS)),
        y=alt.Y("queue_length:Q", title="Halting vehicles at the intersection",
                axis=alt.Axis(**_AXIS)),
        color=alt.Color("Scenario:N", scale=_color_scale(),
                        legend=alt.Legend(title=None, orient="top-left",
                                          labelColor=INK, labelFontSize=11,
                                          symbolStrokeWidth=3)),
    )
    points = alt.Chart(data).mark_circle(size=52, opacity=0).encode(
        x="time:Q", y="queue_length:Q",
        color=alt.Color("Scenario:N", scale=_color_scale(), legend=None),
        tooltip=[
            alt.Tooltip("Scenario:N", title="Scenario"),
            alt.Tooltip("time:Q", title="Time (s)", format=".0f"),
            alt.Tooltip("queue_length:Q", title="Queue", format=".0f"),
        ],
    ).add_params(hover)

    return (lines + points).properties(
        height=height,
        title=alt.TitleParams(text="Queue length over time",
                              anchor="start", color=INK, fontSize=13,
                              fontWeight=600, offset=10),
    ).interactive(bind_y=False)


# ==========================================================================
# Vehicles in the network over time
# ==========================================================================
def vehicles_over_time_chart(series: dict, height: int = 260) -> Optional[alt.Chart]:
    frames = []
    for scenario, frame in (series or {}).items():
        if frame is None or frame.empty or "running_vehicles" not in frame.columns:
            continue
        part = frame[["time", "running_vehicles"]].copy()
        part["Scenario"] = scenario_label(scenario)
        frames.append(part)
    if not frames:
        return None
    data = pd.concat(frames, ignore_index=True)
    return alt.Chart(data).mark_line(strokeWidth=1.4).encode(
        x=alt.X("time:Q", title="Simulation time (s)", axis=alt.Axis(**_AXIS)),
        y=alt.Y("running_vehicles:Q", title="Vehicles in the network",
                axis=alt.Axis(**_AXIS)),
        color=alt.Color("Scenario:N", scale=_color_scale(),
                        legend=alt.Legend(title=None, orient="top-left",
                                          labelColor=INK, labelFontSize=11,
                                          symbolStrokeWidth=3)),
        tooltip=[
            alt.Tooltip("Scenario:N", title="Scenario"),
            alt.Tooltip("time:Q", title="Time (s)", format=".0f"),
            alt.Tooltip("running_vehicles:Q", title="Vehicles", format=".0f"),
        ],
    ).properties(
        height=height,
        title=alt.TitleParams(text="Vehicles in the network over time",
                              anchor="start", color=INK, fontSize=13,
                              fontWeight=600, offset=10),
    )


# ==========================================================================
# Signal timing plan
# ==========================================================================
def signal_plan_chart(segments, height: int = 110) -> Optional[alt.Chart]:
    """The configured fixed-time cycle drawn as a timeline.

    `segments` is the list of (label, seconds) from FixedTimeAgent.timeline().
    """
    if not segments:
        return None
    from dashboard.theme import SIGNAL_COLORS

    rows, start = [], 0.0
    for label, seconds in segments:
        rows.append({"Phase": label, "start": start, "end": start + seconds,
                     "seconds": seconds})
        start += seconds
    data = pd.DataFrame(rows)

    color_for = {
        "NS green": SIGNAL_COLORS["green"],
        "EW green": SIGNAL_COLORS["green"],
        "Yellow": SIGNAL_COLORS["yellow"],
        "All red": SIGNAL_COLORS["red"],
    }
    domain = list(dict.fromkeys(data["Phase"].tolist()))
    scale = alt.Scale(domain=domain,
                      range=[color_for.get(p, "#7a7f85") for p in domain])

    bars = alt.Chart(data).mark_bar(height=34, cornerRadius=3, stroke="#ffffff",
                                    strokeWidth=2).encode(
        x=alt.X("start:Q", title="Seconds into the cycle", axis=alt.Axis(**_AXIS)),
        x2="end:Q",
        color=alt.Color("Phase:N", scale=scale,
                        legend=alt.Legend(title=None, orient="bottom",
                                          labelColor=INK, labelFontSize=11)),
        tooltip=[alt.Tooltip("Phase:N"),
                 alt.Tooltip("seconds:Q", title="Duration (s)", format=".0f")],
    )
    labels = alt.Chart(data).mark_text(color="#ffffff", fontSize=11,
                                       fontWeight=600).encode(
        x=alt.X("start:Q"), x2="end:Q",
        text=alt.Text("seconds:Q", format=".0f"),
    )
    return (bars + labels).properties(
        height=height,
        title=alt.TitleParams(text="Configured fixed-time cycle",
                              anchor="start", color=INK, fontSize=13,
                              fontWeight=600, offset=10))


# ==========================================================================
# Stage 3: controller comparison and training curves
# ==========================================================================
#: One colour per CONTROLLER, assigned by identity like the scenario colours,
#: so Fixed-Time and DQN keep their colour on every chart and every page.
CONTROLLER_COLORS = {
    "fixed_time": "#4c78a8",
    "dqn": "#e45756",
    "dqn_untrained": "#b0b4b9",
}
CONTROLLER_LABELS = {
    "fixed_time": "Fixed-Time",
    "dqn": "DQN",
    "dqn_untrained": "DQN (untrained control)",
}
CONTROLLER_DOMAIN: List[str] = ["Fixed-Time", "DQN", "DQN (untrained control)"]
CONTROLLER_RANGE: List[str] = [CONTROLLER_COLORS["fixed_time"],
                               CONTROLLER_COLORS["dqn"],
                               CONTROLLER_COLORS["dqn_untrained"]]


def _controller_scale() -> alt.Scale:
    return alt.Scale(domain=CONTROLLER_DOMAIN, range=CONTROLLER_RANGE)


def controller_metric_chart(aggregated: pd.DataFrame, metric: str,
                            title: str, unit: str = "",
                            height: int = 260) -> Optional[alt.Chart]:
    """One metric, one bar per controller, with the std as an error bar.

    `aggregated` is the long-format mean/std table written by
    evaluation/compare_controllers.py. The error bar is the sample standard
    deviation across the evaluation seeds; when only one seed was run there
    is no deviation and none is drawn.
    """
    if aggregated is None or aggregated.empty:
        return None
    rows = aggregated[aggregated["metric"] == metric].copy()
    if rows.empty:
        return None

    rows["Controller"] = rows["controller"].map(
        lambda c: CONTROLLER_LABELS.get(c, str(c)))
    rows["std_filled"] = rows["std"].fillna(0.0)
    rows["low"] = rows["mean"] - rows["std_filled"]
    rows["high"] = rows["mean"] + rows["std_filled"]

    tooltip = [alt.Tooltip("Controller:N"),
               alt.Tooltip("mean:Q", title="Mean", format=".2f"),
               alt.Tooltip("std:Q", title="Std deviation", format=".2f"),
               alt.Tooltip("seeds:Q", title="Seeds", format=".0f")]

    bars = alt.Chart(rows).mark_bar(cornerRadiusTopLeft=3,
                                    cornerRadiusTopRight=3, size=46).encode(
        x=alt.X("Controller:N", title=None, sort=CONTROLLER_DOMAIN,
                axis=alt.Axis(labelAngle=0, **_AXIS)),
        y=alt.Y("mean:Q", title=unit or None, axis=alt.Axis(**_AXIS)),
        color=alt.Color("Controller:N", scale=_controller_scale(), legend=None),
        tooltip=tooltip,
    )
    errors = alt.Chart(rows).mark_rule(strokeWidth=1.4, color=INK_MUTED).encode(
        x=alt.X("Controller:N", sort=CONTROLLER_DOMAIN),
        y="low:Q", y2="high:Q",
    )
    labels = alt.Chart(rows).mark_text(dy=-8, color=INK, fontSize=11,
                                       fontWeight=600).encode(
        x=alt.X("Controller:N", sort=CONTROLLER_DOMAIN),
        y=alt.Y("high:Q"),
        text=alt.Text("mean:Q", format=".2f"),
    )
    return (bars + errors + labels).properties(
        height=height,
        title=alt.TitleParams(text=title, anchor="start", color=INK,
                              fontSize=13, fontWeight=600, offset=10))


def queue_by_controller_chart(series: dict, height: int = 320
                              ) -> Optional[alt.Chart]:
    """Per-second queue length for each controller on the same traffic.

    `series` maps a controller key to its recorded time-series DataFrame.
    """
    if not series:
        return None
    frames = []
    for controller, frame in series.items():
        if frame is None or frame.empty or "queue_length" not in frame.columns:
            continue
        part = frame[["time", "queue_length"]].copy()
        part["Controller"] = CONTROLLER_LABELS.get(controller, str(controller))
        frames.append(part)
    if not frames:
        return None
    data = pd.concat(frames, ignore_index=True)

    return alt.Chart(data).mark_line(strokeWidth=1.1, opacity=0.9).encode(
        x=alt.X("time:Q", title="Simulation time (s)", axis=alt.Axis(**_AXIS)),
        y=alt.Y("queue_length:Q", title="Halting vehicles",
                axis=alt.Axis(**_AXIS)),
        color=alt.Color("Controller:N", scale=_controller_scale(),
                        legend=alt.Legend(title=None, orient="top",
                                          labelColor=INK, labelFontSize=11)),
        tooltip=[alt.Tooltip("time:Q", title="t (s)", format=".0f"),
                 alt.Tooltip("queue_length:Q", title="Queue", format=".0f"),
                 alt.Tooltip("Controller:N")],
    ).properties(
        height=height,
        title=alt.TitleParams(
            text="Queue length over time — same network, same demand, same seed",
            anchor="start", color=INK, fontSize=13, fontWeight=600, offset=10))


def training_curve_chart(history: pd.DataFrame, column: str, title: str,
                         unit: str = "", height: int = 260
                         ) -> Optional[alt.Chart]:
    """One training curve per seed, drawn from the recorded history CSV.

    The raw per-episode values are drawn. Nothing is smoothed: a noisy
    learning curve is the result, and hiding the noise would misrepresent it.
    """
    if history is None or history.empty or column not in history.columns:
        return None
    data = history[["episode", column] + (
        ["seed"] if "seed" in history.columns else [])].copy()
    data = data.dropna(subset=[column])
    if data.empty:
        return None
    if "seed" not in data.columns:
        data["seed"] = "42"
    data["Seed"] = data["seed"].astype(str)

    return alt.Chart(data).mark_line(strokeWidth=1.6, point=False).encode(
        x=alt.X("episode:Q", title="Episode", axis=alt.Axis(**_AXIS)),
        y=alt.Y("{}:Q".format(column), title=unit or None,
                axis=alt.Axis(**_AXIS), scale=alt.Scale(zero=False)),
        color=alt.Color("Seed:N", legend=alt.Legend(
            title="Training seed", orient="top", labelColor=INK,
            labelFontSize=11, titleFontSize=11, titleColor=INK_MUTED)),
        tooltip=[alt.Tooltip("episode:Q", title="Episode", format=".0f"),
                 alt.Tooltip("{}:Q".format(column), title=title, format=".4f"),
                 alt.Tooltip("Seed:N")],
    ).properties(
        height=height,
        title=alt.TitleParams(text=title, anchor="start", color=INK,
                              fontSize=13, fontWeight=600, offset=10))
