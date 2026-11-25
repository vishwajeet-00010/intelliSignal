"""
IntelliSignal - Dashboard
dashboard/custom_form.py

The Custom Traffic page: describe the traffic, see exactly what will run.

The page has two halves and they are always in this order:

    1. the form      - only the lanes the user selected, nothing else
    2. the preview   - the resolved workload, with generated values marked

The preview is not a summary of the form. It is the object that gets
simulated: `custom_traffic/builder.py` writes one `<vehicle>` element per row
behind it, so what the preview promises and what SUMO runs are the same
thing. That is why Run is disabled until a preview exists, and why
navigating away and coming back does not quietly re-resolve: the workload is
held in session state until the user edits something or presses Regenerate.
"""

from __future__ import annotations

from typing import Dict, Optional

import streamlit as st

from custom_traffic import (APPROACHES, LANE_SPEED_KMH, LANES_PER_APPROACH,
                            TURNS_AS_COUNTS, TURNS_AS_PERCENT, CustomSpec,
                            Resolution, example_spec, lane_label,
                            movements_for_lane, resolve)
from custom_traffic.workload import FROM_AUTO, FROM_DERIVED, FROM_USER
from dashboard import state

#: Shown next to a value the user did not type.
SOURCE_MARK = {FROM_USER: "", FROM_AUTO: " ·auto", FROM_DERIVED: " ·derived"}


# ---------------------------------------------------------------------------
def _spec() -> CustomSpec:
    raw = state.get("custom_spec")
    if isinstance(raw, dict):
        return CustomSpec.from_dict(raw)
    return CustomSpec.empty()


def _store(spec: CustomSpec) -> None:
    state.put("custom_spec", spec.to_dict())


def _invalidate() -> None:
    """The form changed, so the preview no longer describes it."""
    state.put("custom_workload", None)
    state.put("custom_issues", [])


def _number(label: str, key: str, value, *, minimum=0.0, maximum=None,
            step=1.0, integer: bool = True, help_text: str = ""):
    """A number box that can be left empty, because empty means something."""
    text = "" if value is None else ("{:g}".format(value))
    raw = st.text_input(label, value=text, key=key, help=help_text,
                        placeholder="auto")
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        number = float(raw)
    except ValueError:
        st.caption(":red[Enter a number, or leave it blank for automatic.]")
        return None
    if minimum is not None:
        number = max(number, minimum)
    if maximum is not None:
        number = min(number, maximum)
    return int(round(number)) if integer else number


# ---------------------------------------------------------------------------
#: The junction the product models. The form no longer offers the older
#: single-car scenarios: they exist as historical evidence, and traffic
#: built here is always the Indian fleet with real startup lost time.
BASE_SCENARIO = "indian_arterial"


def render_form() -> CustomSpec:
    """Draw the lane form and return the spec it describes."""
    spec = _spec()
    spec.base_scenario = BASE_SCENARIO

    st.markdown("#### 1 · Describe the traffic")
    st.caption(
        "Tick the lanes that carry traffic and fill in what you know. "
        "Anything left blank is filled in for you from the base scenario and "
        "marked in the preview. Lane 0 is the kerb-side lane (right turn and "
        "straight ahead); lane 1 is the centre lane (straight ahead and left "
        "turn).")

    top = st.columns([2, 2, 2, 2])
    base = BASE_SCENARIO
    with top[0]:
        st.text_input("Junction", value="Indian urban arterial",
                      disabled=True,
                      help="The signal plan, clearance times and vehicle "
                           "mix come from config/indian_arterial.yaml. Only "
                           "the demand is replaced by what you enter here.")
    with top[1]:
        duration = st.number_input(
            "Run length (s)", min_value=60, max_value=7200, step=60,
            value=int(spec.duration_s or 1800),
            help="Simulated seconds, not real seconds.")
    with top[2]:
        seed = st.number_input(
            "Seed", min_value=0, max_value=999999, step=1, value=int(spec.seed),
            help="Fixes both the generated values and SUMO's own randomness, "
                 "so the same form always produces the same run.")
    with top[3]:
        st.write("")
        st.write("")
        if st.button("Generate example", width="stretch",
                     help="Fill the form with the worked example: 200 cars on "
                          "one lane, 300-350 on another."):
            new = example_spec(base_scenario=base, seed=int(seed))
            new.duration_s = float(duration)
            _store(new)
            _invalidate()
            st.rerun()

    spec.base_scenario = base
    spec.duration_s = float(duration)
    spec.seed = int(seed)

    st.write("")
    for approach in APPROACHES:
        header = st.columns([1, 6])
        with header[0]:
            st.markdown("**{}**".format(approach.capitalize()))
        with header[1]:
            enabled_now = [
                spec.lane(approach, index).enabled
                for index in range(LANES_PER_APPROACH)]
            st.caption("lane 0 right/straight · lane 1 straight/left"
                       if any(enabled_now) else "no traffic on this approach")

        for index in range(LANES_PER_APPROACH):
            lane = spec.lane(approach, index)
            key = lane.key
            on = st.checkbox(lane_label(approach, index), value=lane.enabled,
                             key="use_{}".format(key))
            lane.enabled = bool(on)
            if not on:
                continue

            with st.container(border=True):
                row = st.columns([1.2, 1.2, 1.2, 1.4, 1.4, 1.2])
                with row[0]:
                    lane.count = _number(
                        "Vehicles", "cnt_{}".format(key), lane.count,
                        help_text="Exact number of vehicles. Leave blank to "
                                  "use a range, or blank both for automatic.")
                with row[1]:
                    lane.count_min = _number(
                        "or min", "cmin_{}".format(key), lane.count_min)
                with row[2]:
                    lane.count_max = _number(
                        "max", "cmax_{}".format(key), lane.count_max)
                with row[3]:
                    lane.window_start_s = _number(
                        "Arrive from (s)", "ws_{}".format(key),
                        lane.window_start_s, integer=False,
                        help_text="Start of the arrival window, in simulated "
                                  "seconds from the beginning of the run.")
                with row[4]:
                    lane.window_end_s = _number(
                        "to (s)", "we_{}".format(key), lane.window_end_s,
                        integer=False,
                        help_text="End of the arrival window. 200 cars over "
                                  "1800 s and 200 cars over 300 s are "
                                  "different runs.")
                with row[5]:
                    lane.speed_kmh = _number(
                        "Speed (km/h)", "sp_{}".format(key), lane.speed_kmh,
                        integer=False, maximum=130.0,
                        help_text="Desired speed. The lane limit is {:.0f} "
                                  "km/h and SUMO sets the actual speed from "
                                  "the traffic ahead.".format(LANE_SPEED_KMH))

                allowed = movements_for_lane(index)
                turn_cols = st.columns([1.4] + [1.2] * len(allowed))
                with turn_cols[0]:
                    mode = st.radio(
                        "Turns as", [TURNS_AS_COUNTS, TURNS_AS_PERCENT],
                        index=0 if lane.turns_mode == TURNS_AS_COUNTS else 1,
                        format_func=lambda m: ("vehicle counts"
                                               if m == TURNS_AS_COUNTS
                                               else "percentages"),
                        key="tm_{}".format(key), horizontal=False)
                    lane.turns_mode = mode
                turns: Dict[str, Optional[float]] = {}
                for position, movement in enumerate(allowed, start=1):
                    with turn_cols[position]:
                        turns[movement] = _number(
                            movement.capitalize(),
                            "tn_{}_{}".format(key, movement),
                            lane.turns.get(movement),
                            integer=(mode == TURNS_AS_COUNTS),
                            maximum=100.0 if mode == TURNS_AS_PERCENT else None,
                            help_text="Leave blank and it is worked out from "
                                      "the others.")
                lane.turns = turns

    _store(spec)
    return spec


# ---------------------------------------------------------------------------
def render_actions(spec: CustomSpec) -> None:
    """Resolve / Regenerate, and the issue list."""
    st.write("")
    buttons = st.columns([1.3, 1.3, 1.3, 3])
    with buttons[0]:
        if st.button("Preview traffic", type="primary", width="stretch"):
            _resolve_into_state(spec, bump=False)
    with buttons[1]:
        if st.button("Regenerate", width="stretch",
                     help="Draw new values for the blanks. Anything you "
                          "typed is kept exactly as it is."):
            _resolve_into_state(spec, bump=True)
    with buttons[2]:
        if st.button("Clear form", width="stretch"):
            _store(CustomSpec.empty(base_scenario=spec.base_scenario,
                                    seed=spec.seed))
            _invalidate()
            st.rerun()

    for issue in state.get("custom_issues") or []:
        if issue.get("severity") == "error":
            st.error("**{}** — {}".format(
                issue.get("where") or "Form", issue["message"]))
        else:
            st.warning("**{}** — {}".format(
                issue.get("where") or "Note", issue["message"]))


def _resolve_into_state(spec: CustomSpec, bump: bool) -> None:
    token = int(state.get("custom_regen_token") or 0)
    if bump:
        token += 1
        state.put("custom_regen_token", token)
    resolution: Resolution = resolve(spec, regenerate_token=token)
    state.put("custom_issues", [
        {"where": issue.lane_key or "Form", "message": issue.message,
         "severity": issue.severity} for issue in resolution.issues])
    state.put(
        "custom_workload",
        resolution.workload.to_dict() if resolution.workload else None)
    st.rerun()


# ---------------------------------------------------------------------------
def render_preview(workload) -> None:
    """The resolved workload, exactly as it will be simulated."""
    import pandas as pd

    st.markdown("#### 2 · This is what will run")
    head = st.columns([1.3, 1.3, 1.3, 1.3])
    head[0].metric("Vehicles", "{:,}".format(workload.total_vehicles))
    head[1].metric("Run length", "{:.0f} s".format(workload.duration_s))
    head[2].metric("Lanes used", str(len(workload.active_lanes)))
    head[3].metric("Workload id", workload.workload_id[:8])

    rows = workload.preview_rows()
    if not rows:
        st.info("No lane carries traffic yet.")
        return

    table = []
    for row in rows:
        table.append({
            "Lane": row["Lane"],
            "Vehicles": "{}{}".format(row["Vehicles"],
                                      SOURCE_MARK.get(row["count_source"], "")),
            "Arrive between": "{}{}".format(
                row["Arrive between"], SOURCE_MARK.get(row["window_source"], "")),
            "Rate": row["Rate"],
            "Desired speed": "{}{}".format(
                row["Desired speed"], SOURCE_MARK.get(row["speed_source"], "")),
            "Turns": "{}{}".format(row["Turns"],
                                   SOURCE_MARK.get(row["turns_source"], "")),
        })
    st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)
    st.caption(
        "`·auto` was generated for you, `·derived` was worked out from your "
        "other entries, and everything else is exactly what you typed. "
        "Press Regenerate to redraw only the generated values.")

    with st.expander("The first vehicles, as SUMO will receive them"):
        preview = [{
            "Vehicle": vehicle.vehicle_id,
            "Departs at": "{:.0f} s".format(vehicle.depart),
            "Lane": "{} {}".format(vehicle.approach, vehicle.lane),
            "Movement": vehicle.movement,
            "Desired speed": "{:.0f} km/h".format(vehicle.speed_kmh),
        } for vehicle in workload.vehicles[:25]]
        st.dataframe(pd.DataFrame(preview), width="stretch", hide_index=True)
        st.caption(
            "{} vehicles in total. Each one is written into the route file "
            "individually, so both controllers in a comparison receive "
            "identical traffic.".format(workload.total_vehicles))
