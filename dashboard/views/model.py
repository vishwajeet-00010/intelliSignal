"""
IntelliSignal
dashboard/views/model.py

Which model is in charge, why it is allowed to be, and what it cost to
find out.

The page is deliberately in that order. A trained controller is only
worth anything if someone can say how it was chosen and what it was
measured against, so the checks come before the results and the results
come before the training curves.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import pandas as pd
import streamlit as st

from dashboard import charts, ui
from simulation.engine import OPERATING_PROFILE, selection_report
from rl.timing_profile import get_profile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def _registry_card(status: Dict[str, Any]) -> None:
    record = status.get("record") or {}
    story = status.get("describe") or {}
    tone, headline = (
        ("ok", "The trained model is in charge")
        if status.get("usable")
        else ("warn", "The trained model is not in charge"))
    st.markdown(ui.badge(headline, tone), unsafe_allow_html=True)
    st.write("")
    if story.get("summary"):
        ui.note(story["summary"], "info", "What this controller does")
    if not status.get("usable"):
        ui.note(status.get("reason", ""), "warn",
                "Why Hybrid is deciding instead")

    members = record.get("members") or []
    ui.tiles([
        {"label": "Registered as", "value": record.get("label", "nothing"),
         "note_text": "models/active_models.json"},
        {"label": "Networks", "value": max(len(members), 1), "digits": 0,
         "note_text": ("averaged together" if len(members) > 1
                       else "a single network")},
        {"label": "Greens allowed", "value": "{:g}-{:g} s".format(
            get_profile(status.get("profile", OPERATING_PROFILE)).min_green,
            get_profile(status.get("profile", OPERATING_PROFILE)).max_green),
         "note_text": "enforced on every run"},
        {"label": "Promoted", "value": record.get("promoted_on", "--"),
         "note_text": "selected on {} traffic".format(
             record.get("selected_on", "validation"))},
    ])
    st.write("")
    st.markdown("**Before this model is allowed to take the signal**")
    ui.checks(status.get("checks") or [])
    if record.get("note"):
        ui.footnote(record["note"])


def _measurement(record: Dict[str, Any]) -> None:
    measurement = record.get("measurement") or {}
    if not measurement.get("mean_total_delay"):
        return
    paired = measurement.get("paired") or {}
    st.markdown("### What it was measured at")
    st.caption(
        "On the test traffic, which is not the traffic it was trained on "
        "and not the traffic it was chosen on. Measured once.")
    ui.tiles([
        {"label": "Model", "value": measurement["mean_total_delay"],
         "unit": "s", "tone": "good",
         "note_text": "delay per requested vehicle"},
        {"label": "Tuned fixed plan",
         "value": measurement.get("baseline_total_delay"), "unit": "s",
         "note_text": "the baseline it had to beat"},
        {"label": "Improvement",
         "value": (100.0 * (measurement["mean_total_delay"]
                            - measurement["baseline_total_delay"])
                   / measurement["baseline_total_delay"])
         if measurement.get("baseline_total_delay") else None,
         "unit": "%", "digits": 1, "tone": "good",
         "note_text": "lower delay is better"},
        {"label": "Patterns won",
         "value": "{} / {}".format(paired.get("wins", "--"),
                                   paired.get("workloads", "--")),
         "note_text": "worst case {:+.2f} s".format(
             paired.get("worst_delta", 0.0)) if paired else ""},
    ])


def _selection(report: Dict[str, Any]) -> None:
    if not report:
        ui.note(
            "No selection report is recorded for the active model, so this "
            "page can show what it was measured at but not what it was "
            "chosen from. Run evaluation/select_profile_model.py to produce "
            "one.", "warn")
        return

    st.markdown("### How it was chosen")
    st.caption(report.get("rule", ""))
    runs = report.get("runs") or []
    ensembles = report.get("ensembles") or []
    best_fixed = report.get("best_fixed") or {}
    grid = report.get("fixed_grid") or []

    ui.tiles([
        {"label": "Runs trained", "value": len(runs), "digits": 0,
         "note_text": "each a different random seed"},
        {"label": "Ensembles tried", "value": len(ensembles), "digits": 0,
         "note_text": "the leading runs averaged"},
        {"label": "Fixed plans measured", "value": len(grid), "digits": 0,
         "note_text": "every pair on the menu"},
        {"label": "Seeds per pattern",
         "value": report.get("seeds_per_workload", 1), "digits": 0,
         "note_text": "so a result is not one lucky run"},
    ])

    tabs = st.tabs(["Every trained run", "Every fixed plan",
                    "The test result"])
    with tabs[0]:
        chart = charts.seed_scores(runs)
        if chart is not None:
            st.altair_chart(chart, width="stretch")
        table = pd.DataFrame([{
            "Run": run.get("name"),
            "Action space": ("duration menu" if run.get("candidate") == "A"
                             else "lock and extend"),
            "Seed": run.get("seed"),
            "Validation delay (s)": round(
                (run.get("validation") or {}).get("delay", 0.0), 3),
            "Waiting (s)": round(
                (run.get("validation") or {}).get("waiting", 0.0), 2),
            "Obeyed the timing rules":
                "yes" if (run.get("validation") or {}).get("all_compliant")
                else "NO",
        } for run in runs])
        st.dataframe(table, hide_index=True, width="stretch")
        if ensembles:
            st.markdown("**Averaged ensembles**")
            st.dataframe(pd.DataFrame([{
                "Ensemble": e.get("name"),
                "Members": ", ".join(e.get("members") or []),
                "Validation delay (s)": round(
                    (e.get("validation") or {}).get("delay", 0.0), 3),
            } for e in ensembles]), hide_index=True, width="stretch")
            ui.footnote(
                "Averaging the Q-values of several independently trained "
                "networks keeps what they agree on and cancels much of what "
                "is peculiar to any one of them.")

    with tabs[1]:
        chart = charts.fixed_grid(grid)
        if chart is not None:
            st.altair_chart(chart, width="stretch")
        st.caption(
            "Every north-south and east-west pair on the menu, measured on "
            "the validation traffic. The best of them, {:g} s / {:g} s, is "
            "the baseline the model had to beat — not a plan that was left "
            "deliberately weak.".format(best_fixed.get("ns", 0),
                                        best_fixed.get("ew", 0)))

    with tabs[2]:
        test = report.get("test") or []
        if test:
            st.dataframe(pd.DataFrame([{
                "Controller": row["controller"],
                "Delay per vehicle (s)": round(row["delay"], 2),
                "Against the best fixed plan": (
                    "—" if not row.get("against_best_fixed")
                    else "{:+.1f} %".format(
                        row["against_best_fixed"]["relative_change_percent"])),
                "Won on": (
                    "—" if not row.get("against_best_fixed")
                    else "{} / {}".format(row["against_best_fixed"]["wins"],
                                          row["against_best_fixed"]["of"])),
                "Waiting (s)": round(row["waiting"], 2),
                "Completed": row["completed"],
                "Turned away": row["pending"],
            } for row in test]), hide_index=True, width="stretch")
            ui.footnote(
                "Test traffic, {} seeds per pattern, measured once after the "
                "choice was made.".format(report.get("seeds_per_workload", 1)))


def _training(record: Dict[str, Any]) -> None:
    """The validation curve of the promoted run, if its history is on disk."""
    path = record.get("path", "")
    if not path:
        return
    history_path = os.path.join(
        PROJECT_ROOT, os.path.dirname(os.path.dirname(path)),
        "validation_history.json")
    if not os.path.isfile(history_path):
        return
    try:
        with open(history_path, "r", encoding="utf-8") as handle:
            history = json.load(handle)
    except (OSError, ValueError):
        return
    chart = charts.training_curve(history)
    if chart is None:
        return
    st.markdown("### While it was learning")
    st.altair_chart(chart, width="stretch")
    ui.footnote(
        "Validation delay after every twentieth episode. The diamond is the "
        "checkpoint that was kept — chosen on this curve, never on the test "
        "traffic.")


# ---------------------------------------------------------------------------
def render(ctx: Dict[str, Any]) -> None:
    status = ctx["model_status"]
    record = status.get("record") or {}
    st.markdown("## The model")
    _registry_card(status)
    st.divider()
    _measurement(record)
    st.divider()
    _selection(selection_report(status.get("profile", OPERATING_PROFILE)))
    st.divider()
    _training(record)

    with st.expander("Older models kept as evidence, not in use"):
        st.write(
            "The project ran three earlier stages before this one, and their "
            "checkpoints are still in the repository so their published "
            "numbers can be reproduced. They are not offered here because "
            "they were trained on a single generic car with no startup "
            "delay, under different timing rules, so their numbers are not "
            "comparable with anything on this dashboard.")
        from agents.phase_dqn_agent import (REGISTRY_PATH,
                                            registered_phase_model)
        rows = []
        for scenario in ("moderate", "heavy"):
            old = registered_phase_model(scenario)
            if old:
                rows.append({
                    "Scenario": scenario,
                    "Model": old.get("label", ""),
                    "Greens": "{:g}-{:g} s".format(
                        min(old.get("durations") or [0]),
                        max(old.get("durations") or [0])),
                    "File": old.get("path", ""),
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        ui.footnote("Registry: {}".format(
            os.path.relpath(REGISTRY_PATH, PROJECT_ROOT).replace("\\", "/")))
