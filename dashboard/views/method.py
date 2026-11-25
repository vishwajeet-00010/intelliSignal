"""
IntelliSignal
dashboard/views/method.py

How the junction is modelled, how a controller is measured, and what
this project does not claim.

The limitations are on the page rather than in a footnote because a
result whose limits are hidden is worth less than one whose limits are
stated, and because every one of them has at some point been the thing
that made a number wrong.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import streamlit as st

from dashboard import ui
from rl.timing_profile import get_profile
from simulation.engine import OPERATING_PROFILE, fleet_summary, green_value


def _timing() -> None:
    profile = get_profile(OPERATING_PROFILE)
    st.markdown("### The timing rules every controller obeys")
    ui.tiles([
        {"label": "Minimum green", "value": profile.min_green, "unit": "s",
         "digits": 0, "note_text": "long enough to cross on foot"},
        {"label": "Maximum green", "value": profile.max_green, "unit": "s",
         "digits": 0, "note_text": "nobody waits through a whole cycle"},
        {"label": "Clearance", "value": 5, "unit": "s", "digits": 0,
         "note_text": "3 s amber then 2 s all-red"},
        {"label": "Maximum red", "value": "{:g}x".format(
            profile.max_red_factor),
         "note_text": "of the preceding green"},
    ])
    st.caption(
        "These are enforced in the environment, not requested of the "
        "controller: a green outside them cannot be served, whatever the "
        "controller asks for. Every run is then audited a second time from "
        "the signal SUMO actually displayed, which is what the checks panel "
        "on each run reports.")
    analysis = profile.red_limit_analysis(3.0, 2.0)
    ui.footnote(analysis["explanation"])


def _why_long_greens() -> None:
    st.markdown("### Why the greens are not short")
    st.write(
        "When a signal turns green the first driver takes about a second to "
        "react, the driver behind reacts to them, and the wave takes several "
        "seconds to reach the back of the queue. Those seconds are spent "
        "whatever the green length, so a short green spends almost all of "
        "itself starting the queue and hardly any of it clearing one.")
    left, right = st.columns([2, 3])
    with left:
        st.dataframe(pd.DataFrame([{
            "Green": "{:.0f} s".format(row["green"]),
            "Lost starting off": "{:.1f} s".format(row["lost"]),
            "Actually moving": "{:.1f} s".format(row["usable"]),
            "Useful": "{:.0f} %".format(100 * row["share"]),
        } for row in (green_value(g) for g in (4, 8, 12, 15, 22, 30))]),
            hide_index=True, width="stretch")
    with right:
        st.markdown(
            "This is also the correction at the centre of the project. An "
            "earlier version of the simulation had no startup delay at all: "
            "every vehicle was one generic car that moved the instant the "
            "light changed. In that model a four-second green really did "
            "discharge four seconds of traffic, so the learned controller "
            "quite correctly chose four-second greens — and no traffic "
            "engineer would have deployed them.")
        st.markdown(
            "With the Indian fleet and real startup delay, measured on this "
            "junction, a 4 s green gives **61.3 s** of mean waiting and a "
            "12 s green gives **8.2 s**. The model was wrong, not the "
            "method.")


def _measurement() -> None:
    st.markdown("### How a controller is measured")
    st.write(
        "Every controller — fixed-time, actuated, max-pressure and the "
        "trained model — is run by the same code on the same vehicles with "
        "the same clearance times and the same horizon, and measured two "
        "independent ways.")
    st.markdown(
        "**Delay per requested vehicle** is the headline. It counts the "
        "time lost on the road *plus* the time a vehicle spent unable to "
        "get on at all, divided by every vehicle the traffic asked for. It "
        "comes from SUMO's own trip records. A controller cannot improve it "
        "by blocking an approach, which the older headline could be.")
    st.markdown(
        "**Mean waiting time** is kept beside it, because it is what the "
        "earlier stages of this project reported and dropping it would make "
        "the older numbers unreadable.")
    st.markdown(
        "The two are cross-checked against each other after every run. If "
        "they disagree about how many vehicles finished, the run says so "
        "instead of picking one.")
    st.caption(
        "Traffic is written out one `<vehicle>` element at a time, so two "
        "controllers meet the same vehicles in the same lanes at the same "
        "seconds. Nothing is resampled between runs.")


def _splits() -> None:
    st.markdown("### Training, choosing and reporting are kept apart")
    st.write(
        "The model learns on one set of traffic patterns, the checkpoint "
        "and the run to keep are chosen on a second - the **validation** "
        "patterns - and the number that gets reported comes from a third, "
        "the **test** patterns, which are read once after the choice has "
        "been made. The traffic this dashboard demonstrates on is a fourth "
        "set again, so a demonstration cannot accidentally be run on the "
        "evidence.")
    st.dataframe(pd.DataFrame([
        {"Traffic": "Training (18 patterns)",
         "Used for": "learning",
         "Seen by the model": "yes, every episode"},
        {"Traffic": "Validation (6 patterns)",
         "Used for": "choosing which checkpoint and which run to keep",
         "Seen by the model": "no"},
        {"Traffic": "Test (6 patterns)",
         "Used for": "the number that gets reported",
         "Seen by the model": "no, and read once"},
        {"Traffic": "Demonstration (5 presets)",
         "Used for": "this dashboard",
         "Seen by the model": "no, and separate from all three"},
    ]), hide_index=True, width="stretch")
    ui.footnote(
        "The four use different traffic seeds and different SUMO seeds, so "
        "a model cannot be chosen on the traffic it was fitted to, reported "
        "on the traffic it was chosen on, or demonstrated on the traffic it "
        "was reported on.")


def _limits() -> None:
    st.markdown("### What this does not claim")
    for text in (
        "**The fleet is representative, not surveyed.** 62 % two-wheelers, "
        "22 % cars, 11 % auto-rickshaws and 5 % buses is typical of an "
        "Indian urban arterial, but it is not a count taken at a particular "
        "junction. A real deployment would calibrate it from local data.",
        "**Saturation headway was not validated against field data.** It "
        "emerges from the headway, minimum gap and startup delay given to "
        "each vehicle class, and lands in a plausible range — which is not "
        "the same as being measured.",
        "**Demand is deterministic.** Vehicles are written out "
        "individually, which is what makes two controllers comparable, but "
        "it is not the same as stochastic arrivals. A controller is "
        "therefore not tested against the variability a real junction sees.",
        "**One junction, one layout.** Four arms, two lanes each way, no "
        "pedestrians modelled, no bus stops, no parking, and the roads "
        "leaving the junction are free — so a queue can never back up into "
        "it from downstream.",
        "**The older scenarios are not comparable.** `moderate` and "
        "`heavy` use a single generic car with no startup delay. Their "
        "numbers are kept so the earlier stages reproduce, and nothing "
        "puts them in a table with anything measured on the Indian fleet.",
    ):
        st.markdown("- " + text)


# ---------------------------------------------------------------------------
def render(ctx: Dict[str, Any]) -> None:
    st.markdown("## Method")
    st.caption(
        "IntelliSignal is a traffic-signal controller for a four-arm Indian "
        "urban junction, trained with reinforcement learning and measured "
        "against the alternatives a traffic engineer would actually "
        "consider.")
    _timing()
    st.divider()
    _why_long_greens()
    st.divider()
    _measurement()
    st.divider()
    _splits()
    st.divider()

    fleet = fleet_summary("indian_urban")
    st.markdown("### The junction and the fleet")
    columns = st.columns(2, gap="medium")
    with columns[0]:
        st.dataframe(pd.DataFrame([{
            "Vehicle": row["vehicle"],
            "Share": "{:.0f} %".format(100 * row["share"]),
            "Length": "{:.1f} m".format(row["length_m"]),
            "Moves off after": "{:.1f} s".format(row["startup_delay_s"]),
            "PCU": "{:.2f}".format(row["pcu"]),
        } for row in fleet["rows"]]), hide_index=True, width="stretch")
    with columns[1]:
        st.dataframe(pd.DataFrame([
            {"Property": "Arms", "Value": "4 (north, south, east, west)"},
            {"Property": "Lanes", "Value": "2 in and 2 out on each arm"},
            {"Property": "Approach length", "Value": "190 m"},
            {"Property": "Speed limit", "Value": "50 km/h"},
            {"Property": "Lane 0", "Value": "right turn and straight ahead"},
            {"Property": "Lane 1", "Value": "straight ahead and left turn"},
            {"Property": "Simulator", "Value": "SUMO, 1 second steps"},
        ]), hide_index=True, width="stretch")
    st.divider()
    _limits()
