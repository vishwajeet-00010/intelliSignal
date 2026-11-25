"""
IntelliSignal
dashboard/views/common.py

The pieces more than one page shows: a run's headline numbers, the
audit of its signal, its decisions, and the traffic it was measured on.

They live here rather than in each page so that the same run is
described the same way wherever it appears - and so that the audit is
never quietly left off a page that reports a number.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from dashboard import charts, junction, ui
from simulation.engine import green_value
from rl.timing_profile import get_profile


# ---------------------------------------------------------------------------
def run_tiles(result, baseline=None) -> None:
    """The five numbers a run is judged on."""
    metrics = result.metrics
    delay_note = "lower is better"
    tone = "lead"
    if baseline is not None and baseline is not result and baseline.delay:
        change = 100.0 * (result.delay - baseline.delay) / baseline.delay
        delay_note = "{:+.1f} % against the fixed plan".format(change)
        tone = "good" if change < 0 else "bad"
    pending = int(metrics.get("waiting_to_enter", 0))
    ui.tiles([
        {"label": "Delay per vehicle", "value": result.delay, "unit": "s",
         "note_text": delay_note, "tone": tone},
        {"label": "Waiting per vehicle", "value": result.waiting, "unit": "s",
         "note_text": "time spent stopped"},
        {"label": "Completed", "value": result.completed, "unit": "",
         "digits": 0,
         "note_text": "of {:,} requested".format(
             int(metrics.get("requested", 0)))},
        {"label": "Turned away", "value": pending, "unit": "", "digits": 0,
         "note_text": "never got onto the road",
         "tone": "bad" if pending else "good"},
        {"label": "Average green", "value": metrics.get("mean_green", 0.0),
         "unit": "s", "note_text": "{} greens served".format(
             int(metrics.get("switches", 0)))},
    ])


def metric_glossary() -> None:
    with st.expander("What these numbers mean"):
        for key in ("delay_per_requested", "waiting_per_vehicle", "completed",
                    "waiting_to_enter", "queue_mean"):
            st.markdown("**{}** — {}".format(
                {"delay_per_requested": "Delay per vehicle",
                 "waiting_per_vehicle": "Waiting per vehicle",
                 "completed": "Completed",
                 "waiting_to_enter": "Turned away",
                 "queue_mean": "Average queue"}[key],
                ui.METRIC_HELP[key]))


# ---------------------------------------------------------------------------
def audit_panel(result, expanded: bool = False) -> None:
    """What was checked about this run, and what the answer was.

    The timing check reads the signal SUMO displayed, not the
    controller's own account of it, so a controller that breaks its own
    rules is caught here rather than believed.
    """
    audit = result.audit or {}
    metrics = result.metrics
    violations = list(audit.get("violations") or [])
    profile = get_profile(result.profile)

    rows: List[Dict[str, Any]] = []
    if audit:
        rows.append({
            "check": "Every green obeyed the {:g}-{:g} s rule".format(
                profile.min_green, profile.max_green),
            "ok": not any(v["rule"] in ("minimum green", "maximum green")
                          for v in violations),
            "detail": "{} greens were served, from {} s to {} s. Read from "
                      "the signal itself, not from the controller.".format(
                          audit.get("greens_completed", 0),
                          ui.fmt(audit.get("shortest_green"), 0),
                          ui.fmt(audit.get("longest_green"), 0))})
        rows.append({
            "check": "Amber and all-red ran their full length",
            "ok": not any(v["rule"] in ("amber length", "all-red length",
                                        "amber after green",
                                        "all-red after amber", "alternation")
                          for v in violations),
            "detail": "{:g} s amber then {:g} s all-red after every green, "
                      "and the other direction next.".format(
                          audit.get("yellow", 3.0), audit.get("all_red", 2.0))})
        longest = audit.get("longest_red") or {}
        rows.append({
            "check": "No direction was held on red too long",
            "ok": not any(v["rule"] == "maximum red" for v in violations),
            "detail": "Longest red: north-south {} s, east-west {} s. The "
                      "limit is {:g} times the preceding green.".format(
                          ui.fmt(longest.get("NS"), 0),
                          ui.fmt(longest.get("EW"), 0),
                          profile.max_red_factor)})
    else:
        rows.append({"check": "Signal timing", "ok": None,
                     "detail": "This run was measured without recording the "
                               "signal, so the timing was not audited."})

    rows.append({
        "check": "The two measurements agree",
        "ok": bool(metrics.get("measurements_agree", True)),
        "detail": "SUMO's own trip records and the live counter agree on how "
                  "many vehicles finished."})
    rows.append({
        "check": "No collisions and no teleports",
        "ok": not (metrics.get("collisions", 0) or metrics.get("teleports", 0)),
        "detail": "{} collisions, {} teleports. A teleport is SUMO removing a "
                  "vehicle that was stuck, and it would flatter the "
                  "numbers.".format(int(metrics.get("collisions", 0)),
                                    int(metrics.get("teleports", 0)))})

    failed = [row for row in rows if row["ok"] is False]
    title = ("Checks on this run — all passed" if not failed
             else "Checks on this run — {} FAILED".format(len(failed)))
    with st.expander(title, expanded=expanded or bool(failed)):
        ui.checks(rows)
        if violations:
            st.markdown("**What the signal actually did wrong**")
            st.dataframe(
                pd.DataFrame([{"at (s)": v.get("at"), "rule": v["rule"],
                               "what happened": v["detail"]}
                              for v in violations]),
                hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
def replay_block(result, height: int = 640) -> None:
    junction.replay_view(
        result.replay, height=height,
        title="{} — {}".format(result.label, result.workload_name))


def timeline_block(result) -> None:
    chart = charts.signal_timeline(result.greens)
    if chart is not None:
        st.markdown("**When each direction had the green**")
        st.altair_chart(chart, width="stretch")
    queue = charts.queue_over_time(result.series)
    if queue is not None:
        st.markdown("**Vehicles halted, second by second**")
        st.altair_chart(queue, width="stretch")


def green_block(result) -> None:
    profile = get_profile(result.profile)
    chart = charts.green_lengths(result.greens, profile.min_green,
                                 profile.max_green)
    if chart is not None:
        st.altair_chart(chart, width="stretch")
        value = green_value(result.metrics.get("mean_green", 0.0))
        ui.footnote(
            "The dashed lines are the {:g} s minimum and {:g} s maximum. An "
            "average green of {:g} s loses about {:g} s to vehicles starting "
            "off, so roughly {:.0f} % of it actually moves traffic.".format(
                profile.min_green, profile.max_green, value["green"],
                value["lost"], 100 * value["share"]))


def decisions_block(result) -> None:
    """One row per green: what the controller did, and what it expected."""
    decisions = result.decisions or []
    if not decisions:
        return
    kind = (result.detail or {}).get("action_space", "")
    rows = []
    for row in decisions:
        entry = {"at (s)": row.get("time"), "direction": row.get("phase"),
                 "green (s)": row.get("green_seconds"),
                 "ended by": ("the controller" if row.get("completed", True)
                              else "the end of the run")}
        if kind == "lock_extend":
            entry["extensions"] = row.get("extensions", 0)
            entry["ended at the maximum"] = bool(row.get("forced_end"))
            q = row.get("q_values") or {}
            entry["Q extend"] = q.get("extend")
            entry["Q end"] = q.get("end")
        else:
            q = row.get("q_values") or {}
            if q:
                best = max(q, key=lambda k: q[k])
                entry["best option"] = best
                entry["margin"] = round(
                    q[best] - max(v for k, v in q.items() if k != best), 3) \
                    if len(q) > 1 else None
        rows.append(entry)
    with st.expander("Every decision the controller made ({})".format(
            len(rows))):
        if kind == "lock_extend":
            st.caption(
                "This controller holds the minimum green, then every few "
                "seconds chooses between extending and ending. Q is what it "
                "expects each choice to cost from here on - less negative is "
                "better - so the two columns show how close the call was.")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     height=min(38 * len(rows) + 40, 420))


# ---------------------------------------------------------------------------
def traffic_card(workload, compact: bool = False) -> None:
    """What is on the road: how much, which vehicles, arriving when."""
    if workload is None:
        ui.note("No traffic has been chosen yet. Open the Traffic page.",
                "warn")
        return
    mix = workload.fleet_mix()
    pcu = workload.total_pcu()
    per_hour = workload.total_vehicles * 3600.0 / max(workload.duration_s, 1)
    ui.tiles([
        {"label": "Vehicles", "value": workload.total_vehicles, "digits": 0,
         "note_text": "{:,.0f} an hour".format(per_hour)},
        {"label": "Road space", "value": pcu, "unit": "PCU", "digits": 0,
         "note_text": "{:.2f} PCU per vehicle".format(
             pcu / max(workload.total_vehicles, 1))},
        {"label": "Run length", "value": workload.duration_s / 60.0,
         "unit": "min", "digits": 0,
         "note_text": "{:,.0f} seconds simulated".format(workload.duration_s)},
        {"label": "Two-wheelers", "value": 100.0 * mix.get("motorcycle", 0)
         / max(workload.total_vehicles, 1), "unit": "%", "digits": 0,
         "note_text": "of every vehicle on the road"},
    ])
    if compact:
        return
    left, right = st.columns([3, 2])
    with left:
        chart = charts.arrivals(workload)
        if chart is not None:
            st.markdown("**When the traffic arrives**")
            st.altair_chart(chart, width="stretch")
    with right:
        donut = charts.fleet_donut(mix)
        if donut is not None:
            st.markdown("**What is on the road**")
            st.altair_chart(donut, width="stretch")


def comparison_table(table: Dict[str, Any]) -> None:
    """The one table that puts controllers side by side."""
    rows = table.get("rows") or []
    if not rows:
        return
    frame = pd.DataFrame([{
        "Controller": row["label"],
        "Delay per vehicle (s)": round(row["delay"], 2),
        "Against fixed": ("—" if row.get("change_percent") is None
                          else "{:+.1f} %".format(row["change_percent"])),
        "Waiting (s)": round(row["waiting"], 2),
        "Completed": row["completed"],
        "Turned away": row["waiting_to_enter"],
        "Average green (s)": round(row["mean_green"], 1),
        "Checks": "passed" if row["trustworthy"] else "SEE CHECKS",
    } for row in rows])
    st.dataframe(frame, hide_index=True, width="stretch")
    ui.footnote(
        "Every row is the same traffic, the same vehicles, the same seed and "
        "the same timing rules — the only difference is the controller.")
