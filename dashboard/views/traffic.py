"""
IntelliSignal
dashboard/views/traffic.py

What is on the road.

Two ways in: a ready-made pattern, or traffic described lane by lane.
Both end in the same thing - an explicit list of vehicles, each with its
class, its lane, its turn and the second it arrives - which is what makes
a comparison between two controllers a measurement rather than an
estimate. Nothing here is a rate that SUMO will sample differently next
time.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import streamlit as st

from dashboard import custom_form, state, ui
from dashboard.views import common
from simulation.engine import (PRESET_ORDER, PRESETS, fleet_summary,
                               green_value, preset_workload)


def _preset_cards() -> None:
    chosen = state.get("preset")
    st.markdown("#### Ready-made traffic")
    st.caption(
        "Five patterns taken from how this arterial actually behaves. Their "
        "random seeds are deliberately different from the traffic the model "
        "was trained, selected and reported on, so a demonstration can never "
        "accidentally be run on the evidence.")
    columns = st.columns(len(PRESET_ORDER))
    for column, key in zip(columns, PRESET_ORDER):
        preset = PRESETS[key]
        with column:
            with st.container(border=True):
                st.markdown("**{}**".format(preset.label))
                st.caption(preset.headline)
                if key == chosen:
                    st.markdown(ui.badge("in use", "ok"),
                                unsafe_allow_html=True)
                elif st.button("Use this", key="preset_" + key,
                               width="stretch"):
                    state.put("preset", key)
                    state.put("traffic_kind", "preset")
                    state.reset_traffic()
                    st.rerun()
    st.caption(PRESETS[chosen].detail if chosen in PRESETS else "")


def _fleet_panel() -> None:
    fleet = fleet_summary("indian_urban")
    st.markdown("#### The fleet on the road")
    st.caption(
        "Every run uses this mix. It is the reason the greens are not short: "
        "a stopped vehicle does not move the instant the light changes, and "
        "the delay cascades back down the queue.")
    table = pd.DataFrame([{
        "Vehicle": row["vehicle"],
        "Share": "{:.0f} %".format(100 * row["share"]),
        "Length": "{:.1f} m".format(row["length_m"]),
        "Moves off after": "{:.1f} s".format(row["startup_delay_s"]),
        "Road space (PCU)": "{:.2f}".format(row["pcu"]),
    } for row in fleet["rows"]])
    left, right = st.columns([3, 2])
    with left:
        st.dataframe(table, hide_index=True, width="stretch")
        ui.footnote(
            "Mean {:.2f} PCU and {:.2f} s of startup delay per vehicle. "
            "Clearance between phases is a 3 s amber and a 2 s all-red, as "
            "Indian practice uses.".format(fleet["mean_pcu"],
                                           fleet["mean_startup_delay"]))
    with right:
        rows = [green_value(g) for g in (4, 8, 15, 22, 30)]
        st.dataframe(pd.DataFrame([{
            "Green": "{:.0f} s".format(row["green"]),
            "Lost starting off": "{:.1f} s".format(row["lost"]),
            "Actually moving": "{:.1f} s".format(row["usable"]),
            "Useful": "{:.0f} %".format(100 * row["share"]),
        } for row in rows]), hide_index=True, width="stretch")
        ui.footnote("Which is why a four-second green is the worst plan on "
                    "this junction, not the most responsive one.")


# ---------------------------------------------------------------------------
def render(ctx: Dict[str, Any]) -> None:
    st.markdown("## Traffic")

    kind = state.get("traffic_kind")
    st.caption(
        "Traffic comes from a ready-made pattern or from what you type "
        "here; the switch is in the sidebar, under **Traffic**. Either "
        "way it ends as an explicit list of vehicles, which is what lets "
        "two controllers meet the same traffic.")

    st.divider()
    if kind == "preset":
        _preset_cards()
        st.write("")
        columns = st.columns([1, 1, 3])
        with columns[0]:
            duration = st.select_slider(
                "Run length", [300.0, 600.0, 900.0, 1200.0, 1800.0],
                format_func=lambda s: "{:.0f} min".format(s / 60),
                key="duration_s", on_change=state.reset_traffic)
        with columns[1]:
            st.number_input(
                "Seed", min_value=0, max_value=999999, step=1, key="seed",
                on_change=state.reset_traffic,
                help="Fixes SUMO's own randomness so a run can be repeated "
                     "exactly. The traffic itself is the same whatever this "
                     "is.")
        workload = preset_workload(state.get("preset"), float(duration))
    else:
        spec = custom_form.render_form()
        custom_form.render_actions(spec)
        data = state.get("custom_workload")
        workload = ctx["workload"] if data else None
        if workload is None:
            ui.note("Fill in at least one lane and press **Preview traffic**. "
                    "Nothing runs until the traffic exists.", "info")
            st.divider()
            _fleet_panel()
            return
        st.divider()
        custom_form.render_preview(workload)

    st.divider()
    st.markdown("#### This is what will run")
    common.traffic_card(workload)
    ui.footnote("Workload {} — the identifier is a hash of every vehicle, "
                "its lane, its turn and its departure second, so two runs "
                "with the same identifier met exactly the same traffic."
                .format(workload.workload_id[:12]))

    st.divider()
    _fleet_panel()
