"""
IntelliSignal - Dashboard
dashboard/traffic_panel.py

The pieces that make the numbers mean something to a person watching.

Three things a viewer needs to be told, and none of them are obvious from
a waiting-time figure:

1. **What is on the road.** "628 vehicles" hides that most of them are
   two-wheelers. The same count of buses would not fit through the
   junction at all, which is why the demand is also shown in PCU.

2. **Why a short green is wasteful.** When the signal turns green a
   driver takes about a second to react, the driver behind reacts to
   them, and the wave takes several seconds to reach the back of the
   queue. A four-second green spends nearly all of itself on that. This
   is the single most useful sentence on the screen, because it is the
   answer to "why not just make the greens short?"

3. **What the signal plan actually is**, in seconds, as a cycle a traffic
   engineer would recognise.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import streamlit as st

from custom_traffic.fleet import (GENERIC, INDIAN_URBAN, describe,
                                  effective_green_note, get_fleet, mean_pcu,
                                  mean_startup_delay)

FLEET_LABEL = {
    INDIAN_URBAN: "Indian urban mix",
    GENERIC: "single generic car",
}


# ---------------------------------------------------------------------------
def fleet_summary(traffic_model: str, expanded: bool = False) -> None:
    """What is actually on the road."""
    fleet = get_fleet(traffic_model)
    label = FLEET_LABEL.get(traffic_model, traffic_model)

    if traffic_model == GENERIC:
        st.caption(
            "Fleet: **{}** — one vehicle type that accelerates away the "
            "instant the light turns green. Kept for the original Moderate "
            "and Heavy scenarios so their recorded numbers still "
            "reproduce.".format(label))
        return

    st.caption(
        "Fleet: **{}** — {:.0f} % two-wheelers, and every vehicle waits "
        "{:.1f} s on average before it moves off. Mean {:.2f} PCU per "
        "vehicle.".format(label, 100 * fleet[0].share,
                          mean_startup_delay(fleet), mean_pcu(fleet)))

    with st.expander("What is on the road", expanded=expanded):
        import pandas as pd
        st.dataframe(pd.DataFrame(describe(fleet)), width="stretch",
                     hide_index=True)
        st.caption(
            "PCU is passenger car units — how much road space a vehicle "
            "takes compared with a car. Indian signal design works in PCU "
            "because sixty motorcycles are not sixty cars.")


# ---------------------------------------------------------------------------
def green_explainer(green_seconds: float, traffic_model: str) -> None:
    """Why a short green wastes itself. The key sentence of the demo."""
    fleet = get_fleet(traffic_model)
    if traffic_model == GENERIC:
        return
    st.info("**{}**".format(effective_green_note(green_seconds, fleet)))


def green_comparison(traffic_model: str) -> None:
    """The same sentence across several greens, as a small table."""
    fleet = get_fleet(traffic_model)
    if traffic_model == GENERIC:
        return
    import pandas as pd

    lost = 2.0 + mean_startup_delay(fleet)
    rows = []
    for green in (4, 8, 15, 20, 30):
        usable = max(green - lost, 0.0)
        rows.append({
            "Green": "{} s".format(green),
            "Lost to starting off": "{:.1f} s".format(min(lost, green)),
            "Actually moving traffic": "{:.1f} s".format(usable),
            "Useful share": "{:.0f} %".format(100 * usable / green),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(
        "This is why a four-second green is not a plan: almost all of it "
        "is spent waiting for the queue to start moving. It is also why "
        "the minimum green is 15 s rather than something shorter.")


# ---------------------------------------------------------------------------
def signal_plan(ns_green: float, ew_green: float, yellow: float,
                all_red: float, source: str = "") -> None:
    """The plan as a traffic engineer would read it."""
    cycle = ns_green + ew_green + 2 * (yellow + all_red)
    columns = st.columns(4)
    columns[0].metric("North–South green", "{:.0f} s".format(ns_green))
    columns[1].metric("East–West green", "{:.0f} s".format(ew_green))
    columns[2].metric("Clearance", "{:.0f} s".format(yellow + all_red),
                      help="{:.0f} s amber plus {:.0f} s all-red, twice per "
                           "cycle".format(yellow, all_red))
    columns[3].metric("Cycle", "{:.0f} s".format(cycle))
    if source:
        st.caption("Plan: {}.".format(source))


# ---------------------------------------------------------------------------
def demand_summary(workload) -> None:
    """Vehicles and PCU, with the fleet mix."""
    columns = st.columns(4)
    columns[0].metric("Vehicles", "{:,}".format(workload.total_vehicles))
    try:
        columns[1].metric("Demand in PCU", "{:,.0f}".format(workload.total_pcu()))
    except Exception:
        columns[1].metric("Demand in PCU", "—")
    columns[2].metric("Over", "{:.0f} s".format(workload.duration_s))
    columns[3].metric("Lanes used", str(len(workload.active_lanes)))

    mix = workload.fleet_mix() if hasattr(workload, "fleet_mix") else {}
    if mix and len(mix) > 1:
        st.caption("Mix: " + " · ".join(
            "{} {}".format(count, name) for name, count in mix.items()))


# ---------------------------------------------------------------------------
def auto_hybrid_notice(reason: Optional[str]) -> None:
    """Say plainly that the DQN handed over, and why."""
    if not reason:
        return
    st.warning(
        "**The DQN did not take the signal — Hybrid is deciding instead.**\n\n"
        "{}\n\nHybrid tunes the fixed plan on this traffic, tests whatever "
        "DQN exists against it, and uses whichever actually wins. A tie "
        "goes to fixed time.".format(reason))


# ---------------------------------------------------------------------------
def practical_note(profile) -> None:
    """One line tying the timing limits to real practice."""
    st.caption(
        "Greens are held between **{:.0f} s and {:.0f} s**. The lower bound "
        "is what a pedestrian needs to cross and what a queue needs to get "
        "moving; the upper bound stops one direction being held at red for "
        "an unreasonable time.".format(profile.min_green, profile.max_green))
