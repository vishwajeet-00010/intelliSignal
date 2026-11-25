"""
IntelliSignal
dashboard/charts.py

The charts, all of them, in one place.

Each one answers a question a viewer actually asks, and nothing is
plotted that cannot be read off the simulation that produced it:

    queue_over_time      when did traffic pile up, and on which arm
    signal_timeline      what the signal showed, second by second
    green_lengths        how long the greens were, and how they varied
    delay_bars           which controller lost the least time
    paired_bars          and whether it won on every pattern or on average
    training_curve       how a model's validation score moved while it learned
    fixed_grid           the whole grid of fixed plans that were measured

Colours come from `dashboard/ui.py`, so north-south is the same blue
here, in the junction drawing and in the tables.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import altair as alt
import pandas as pd

from dashboard.ui import (BAD, BRAND, CONTROLLER_COLORS, EW_COLOR, GOOD,
                          INK, LINE, MUTED, NS_COLOR)

FONT = "Segoe UI, system-ui, sans-serif"


def _base(chart: alt.Chart, height: int = 220) -> alt.Chart:
    return (chart.properties(height=height)
            .configure_view(strokeWidth=0)
            .configure_axis(labelFont=FONT, titleFont=FONT, labelColor=MUTED,
                            titleColor=MUTED, labelFontSize=11,
                            titleFontSize=11, gridColor=LINE, domainColor=LINE,
                            tickColor=LINE)
            .configure_legend(labelFont=FONT, titleFont=FONT,
                              labelColor=INK, titleColor=MUTED,
                              labelFontSize=11, titleFontSize=11)
            .configure_title(font=FONT, fontSize=12, color=MUTED,
                             anchor="start", fontWeight=600))


# ---------------------------------------------------------------------------
def queue_over_time(series: Sequence[Dict[str, Any]], height: int = 210
                    ) -> Optional[alt.Chart]:
    """Vehicles halted on each axis, through the run."""
    if not series:
        return None
    frame = pd.DataFrame(list(series))
    if "queue_NS" not in frame.columns:
        if "queue_length" not in frame.columns:
            return None
        frame = frame.rename(columns={"queue_length": "All approaches"})
        long = frame.melt("time", ["All approaches"], "axis", "queue")
    else:
        frame = frame.rename(columns={"queue_NS": "North-South",
                                      "queue_EW": "East-West"})
        long = frame.melt("time", ["North-South", "East-West"],
                          "axis", "queue")
    chart = alt.Chart(long).mark_area(opacity=0.55, interpolate="step-after")
    return _base(
        chart.encode(
            x=alt.X("time:Q", title="seconds"),
            y=alt.Y("queue:Q", title="vehicles halted", stack=True),
            color=alt.Color("axis:N", title=None,
                            scale=alt.Scale(
                                domain=["North-South", "East-West",
                                        "All approaches"],
                                range=[NS_COLOR, EW_COLOR, BRAND])),
            tooltip=[alt.Tooltip("time:Q", title="second"),
                     alt.Tooltip("axis:N", title=""),
                     alt.Tooltip("queue:Q", title="halted")]),
        height)


def queue_compare(runs: Dict[str, Sequence[Dict[str, Any]]],
                  height: int = 210) -> Optional[alt.Chart]:
    """Total queue through the run, one line per controller."""
    rows: List[Dict[str, Any]] = []
    for label, series in runs.items():
        for row in series:
            value = row.get("queue_total", row.get("queue_length"))
            if value is None:
                continue
            rows.append({"time": row.get("time", 0), "queue": value,
                         "controller": label})
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    chart = alt.Chart(frame).mark_line(strokeWidth=1.6, opacity=0.9)
    return _base(
        chart.encode(
            x=alt.X("time:Q", title="seconds"),
            y=alt.Y("queue:Q", title="vehicles halted"),
            color=alt.Color("controller:N", title=None),
            tooltip=["controller:N", alt.Tooltip("time:Q", title="second"),
                     alt.Tooltip("queue:Q", title="halted")]),
        height)


# ---------------------------------------------------------------------------
def signal_timeline(greens: Sequence[Dict[str, Any]], height: int = 110
                    ) -> Optional[alt.Chart]:
    """Every green the signal showed, as a bar on a time axis."""
    if not greens:
        return None
    frame = pd.DataFrame([{
        "start": row["start"], "end": row["end"],
        "phase": row["phase"], "seconds": row["green_seconds"],
        "completed": "served" if row.get("completed", True) else "cut off by "
                     "the end of the run",
    } for row in greens])
    chart = alt.Chart(frame).mark_bar(height=22, cornerRadius=3)
    return _base(
        chart.encode(
            x=alt.X("start:Q", title="seconds"),
            x2="end:Q",
            y=alt.Y("phase:N", title=None, sort=["North-South", "East-West"]),
            color=alt.Color("phase:N", legend=None,
                            scale=alt.Scale(
                                domain=["North-South", "East-West"],
                                range=[NS_COLOR, EW_COLOR])),
            opacity=alt.Opacity("completed:N", legend=None,
                                scale=alt.Scale(domain=["served",
                                                        "cut off by the end "
                                                        "of the run"],
                                                range=[0.92, 0.4])),
            tooltip=[alt.Tooltip("phase:N", title="direction"),
                     alt.Tooltip("start:Q", title="from"),
                     alt.Tooltip("seconds:Q", title="green (s)"),
                     alt.Tooltip("completed:N", title="")]),
        height)


def green_lengths(greens: Sequence[Dict[str, Any]], min_green: float,
                  max_green: float, height: int = 190) -> Optional[alt.Chart]:
    """How long the greens were - the shape of what the controller chose."""
    served = [row for row in greens if row.get("completed", True)]
    if not served:
        return None
    frame = pd.DataFrame([{"phase": row["phase"],
                           "seconds": row["green_seconds"]}
                          for row in served])
    bars = (alt.Chart(frame).mark_bar(opacity=0.85)
            .encode(
                x=alt.X("seconds:Q", bin=alt.Bin(step=1),
                        title="green length (s)",
                        scale=alt.Scale(domain=[min_green - 1, max_green + 1])),
                y=alt.Y("count()", title="greens"),
                color=alt.Color("phase:N", title=None,
                                scale=alt.Scale(
                                    domain=["North-South", "East-West"],
                                    range=[NS_COLOR, EW_COLOR])),
                tooltip=[alt.Tooltip("count()", title="greens"),
                         alt.Tooltip("phase:N", title="direction")]))
    rules = (alt.Chart(pd.DataFrame({"x": [min_green, max_green]}))
             .mark_rule(color=MUTED, strokeDash=[4, 3], opacity=0.7)
             .encode(x="x:Q"))
    return _base(bars + rules, height)


# ---------------------------------------------------------------------------
def delay_bars(rows: Sequence[Dict[str, Any]], height: int = 210,
               field: str = "delay",
               title: str = "delay per requested vehicle (s)"
               ) -> Optional[alt.Chart]:
    """One bar per controller. Shorter is better, and it says so."""
    if not rows:
        return None
    frame = pd.DataFrame([{"controller": row["label"],
                           "key": row.get("controller", ""),
                           "value": row.get(field, 0.0)} for row in rows])
    chart = alt.Chart(frame).mark_bar(cornerRadiusEnd=4, height=26)
    return _base(
        chart.encode(
            x=alt.X("value:Q", title=title),
            y=alt.Y("controller:N", title=None, sort="x"),
            color=alt.Color("key:N", legend=None,
                            scale=alt.Scale(
                                domain=list(CONTROLLER_COLORS),
                                range=list(CONTROLLER_COLORS.values()))),
            tooltip=[alt.Tooltip("controller:N", title=""),
                     alt.Tooltip("value:Q", title=title, format=".2f")]),
        height)


def paired_bars(rows: Sequence[Dict[str, Any]], height: int = 210
                ) -> Optional[alt.Chart]:
    """Change against the fixed plan, per traffic pattern.

    Below zero is better. The colour says so without the reader having
    to remember which way round the metric runs.
    """
    if not rows:
        return None
    frame = pd.DataFrame(list(rows))
    chart = alt.Chart(frame).mark_bar(cornerRadius=2)
    return _base(
        chart.encode(
            x=alt.X("difference:Q",
                    title="change against the fixed plan (s per vehicle)"),
            y=alt.Y("workload:N", title=None, sort="-x"),
            color=alt.condition(alt.datum.difference < 0,
                                alt.value(GOOD), alt.value(BAD)),
            tooltip=[alt.Tooltip("workload:N", title="traffic"),
                     alt.Tooltip("difference:Q", title="change (s)",
                                 format="+.2f")]),
        height)


# ---------------------------------------------------------------------------
def training_curve(history: Sequence[Dict[str, Any]], height: int = 220
                   ) -> Optional[alt.Chart]:
    """Validation delay against episode, with the selected point marked."""
    rows = [{"episode": row.get("episode"),
             "delay": row.get("mean_total_delay")}
            for row in history
            if row.get("episode") is not None
            and row.get("mean_total_delay") is not None]
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    best = frame.loc[frame["delay"].idxmin()]
    line = (alt.Chart(frame).mark_line(color=BRAND, strokeWidth=2,
                                       point=alt.OverlayMarkDef(size=28))
            .encode(x=alt.X("episode:Q", title="episode"),
                    y=alt.Y("delay:Q", title="validation delay (s)",
                            scale=alt.Scale(zero=False)),
                    tooltip=[alt.Tooltip("episode:Q"),
                             alt.Tooltip("delay:Q", format=".2f")]))
    mark = (alt.Chart(pd.DataFrame([best])).mark_point(
        color=GOOD, size=160, filled=True, shape="diamond")
        .encode(x="episode:Q", y="delay:Q"))
    return _base(line + mark, height)


def seed_scores(runs: Sequence[Dict[str, Any]], height: int = 260
                ) -> Optional[alt.Chart]:
    """Every trained run's validation score, so the spread is visible."""
    rows = [{"run": run.get("name", ""),
             "candidate": "A - duration menu" if run.get("candidate") == "A"
                          else "B - lock and extend",
             "delay": (run.get("validation") or {}).get("delay")}
            for run in runs if (run.get("validation") or {}).get("delay")]
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    chart = alt.Chart(frame).mark_circle(size=110, opacity=0.85)
    return _base(
        chart.encode(
            x=alt.X("delay:Q", title="validation delay (s)",
                    scale=alt.Scale(zero=False)),
            y=alt.Y("candidate:N", title=None),
            color=alt.Color("candidate:N", legend=None,
                            scale=alt.Scale(range=[BRAND, EW_COLOR])),
            tooltip=[alt.Tooltip("run:N", title="run"),
                     alt.Tooltip("delay:Q", format=".2f")]),
        height)


def fixed_grid(grid: Sequence[Dict[str, Any]], height: int = 300
               ) -> Optional[alt.Chart]:
    """Every fixed plan that was measured, as a heat map."""
    rows = [{"ns": row["ns"], "ew": row["ew"], "delay": row["delay"]}
            for row in grid if "ns" in row and "delay" in row]
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    chart = alt.Chart(frame).mark_rect()
    return _base(
        chart.encode(
            x=alt.X("ew:O", title="east-west green (s)"),
            y=alt.Y("ns:O", title="north-south green (s)", sort="descending"),
            color=alt.Color("delay:Q", title="delay (s)",
                            scale=alt.Scale(scheme="blueorange",
                                            reverse=True)),
            tooltip=[alt.Tooltip("ns:O", title="ns green"),
                     alt.Tooltip("ew:O", title="ew green"),
                     alt.Tooltip("delay:Q", title="delay", format=".2f")]),
        height)


# ---------------------------------------------------------------------------
def arrivals(workload, height: int = 170) -> Optional[alt.Chart]:
    """When the traffic arrives, by approach - the shape of the demand."""
    vehicles = getattr(workload, "vehicles", None)
    if not vehicles:
        return None
    frame = pd.DataFrame([{"depart": v.depart,
                           "approach": v.approach.title()}
                          for v in vehicles])
    chart = alt.Chart(frame).mark_bar(opacity=0.85)
    return _base(
        chart.encode(
            x=alt.X("depart:Q", bin=alt.Bin(maxbins=60),
                    title="arrival time (s)"),
            y=alt.Y("count()", title="vehicles", stack=True),
            color=alt.Color("approach:N", title=None,
                            scale=alt.Scale(
                                domain=["North", "South", "East", "West"],
                                range=[NS_COLOR, "#5a92cd", EW_COLOR,
                                       "#d9a06a"])),
            tooltip=[alt.Tooltip("count()", title="vehicles"),
                     alt.Tooltip("approach:N", title="from")]),
        height)


def fleet_donut(mix: Dict[str, int], height: int = 170) -> Optional[alt.Chart]:
    """What is actually on the road, by count."""
    rows = [{"vehicle": name, "count": count} for name, count in mix.items()
            if count]
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    from dashboard.ui import VEHICLE_COLORS

    chart = alt.Chart(frame).mark_arc(innerRadius=42, cornerRadius=2)
    return _base(
        chart.encode(
            theta=alt.Theta("count:Q", stack=True),
            color=alt.Color("vehicle:N", title=None,
                            scale=alt.Scale(
                                domain=list(VEHICLE_COLORS),
                                range=list(VEHICLE_COLORS.values()))),
            tooltip=["vehicle:N", alt.Tooltip("count:Q", title="vehicles")]),
        height)
