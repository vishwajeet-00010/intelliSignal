"""
IntelliSignal
dashboard.py

The product: a traffic-signal controller for an Indian urban junction,
with the evidence for everything it claims.

    streamlit run dashboard.py

Five pages, in the order a visitor needs them:

    Control room   choose a controller, run the junction, watch it
    Traffic        what is on the road, ready-made or described by hand
    Compare        every controller on the same vehicles, and Hybrid
    The model      which model is in charge, and how it was chosen
    Method         how this is modelled and measured, and what it does
                   not claim

Everything any page shows comes from `simulation/engine.py`, which runs
every controller through `evaluation/harness.py`. There is no second
code path for the picture, and nothing on screen is pre-computed except
runs the user has already asked for.
"""

from __future__ import annotations

import os
import sys

import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(
    page_title="IntelliSignal",
    page_icon="|",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dashboard import state, ui                                 # noqa: E402
from dashboard.views import (compare, control_room, method,     # noqa: E402
                             model, traffic)
from rl.timing_profile import get_profile                       # noqa: E402
from simulation.engine import (OPERATING_PROFILE,              # noqa: E402
                               environment_status, model_status)

PAGES = {
    "Control room": control_room,
    "Traffic": traffic,
    "Compare": compare,
    "The model": model,
    "Method": method,
}


@st.cache_resource(show_spinner=False)
def _status_cache_token() -> dict:
    """A place to hang the model status so it is read once per server."""
    return {}


def current_status() -> dict:
    """Which model is in charge. Re-read when the registry file changes."""
    from agents.phase_dqn_agent import REGISTRY_PATH

    cache = _status_cache_token()
    try:
        stamp = os.path.getmtime(REGISTRY_PATH)
    except OSError:
        stamp = 0.0
    if cache.get("stamp") != stamp:
        cache["stamp"] = stamp
        cache["status"] = model_status(OPERATING_PROFILE)
    return cache["status"]


# ---------------------------------------------------------------------------
def sidebar(status: dict):
    """Where the traffic and the page are chosen, and nothing else."""
    from simulation.engine import PRESET_ORDER, PRESETS

    with st.sidebar:
        st.markdown("### IntelliSignal")
        st.caption("Adaptive traffic signal control")
        st.write("")

        page = st.radio("Page", list(PAGES), key="page",
                        label_visibility="collapsed")

        st.divider()
        st.markdown("**Traffic**")
        # These two widgets are keyed on the state they set, so the
        # widget and the value are one thing. Giving them separate keys
        # and writing the state by hand meant a second control for the
        # same choice could disagree with them, and Streamlit resolves
        # that by keeping the widget - which showed as a page that reset
        # itself as soon as it was touched.
        kind = st.radio(
            "Source", ["preset", "custom"],
            format_func=lambda k: ("Ready-made" if k == "preset"
                                   else "My own"),
            horizontal=True, key="traffic_kind",
            label_visibility="collapsed", on_change=state.reset_traffic)

        if kind == "preset":
            chosen = st.selectbox(
                "Pattern", list(PRESET_ORDER),
                format_func=lambda k: PRESETS[k].label, key="preset",
                on_change=state.reset_traffic)
            st.caption(PRESETS[chosen].headline)
        else:
            data = state.get("custom_workload")
            if data:
                st.caption("{:,} vehicles over {:.0f} minutes.".format(
                    len(data.get("vehicles") or []),
                    float(data.get("duration_s", 0)) / 60))
            else:
                st.caption("Not defined yet — open the Traffic page.")

        st.divider()
        st.markdown("**Signal**")
        profile = get_profile(state.get("profile"))
        st.caption("Greens of {:g}-{:g} s, 3 s amber, 2 s all-red. Enforced "
                   "on every controller.".format(profile.min_green,
                                                 profile.max_green))
        st.markdown(
            ui.badge("model in charge", "ok") if status.get("usable")
            else ui.badge("Hybrid decides", "warn"),
            unsafe_allow_html=True)
        st.caption(status.get("reason", ""))

        st.divider()
        environment = environment_status()
        if environment["ready"]:
            st.markdown(ui.badge("environment ready", "ok"),
                        unsafe_allow_html=True)
        else:
            st.markdown(ui.badge("environment incomplete", "bad"),
                        unsafe_allow_html=True)
            for item in environment["checks"]:
                if not item["ok"]:
                    st.caption("{}: {}".format(item["check"], item["detail"]))
        with st.expander("What is installed"):
            ui.checks(environment["checks"])
        ui.footnote("One junction, Indian urban fleet, SUMO.")
    return page


# ---------------------------------------------------------------------------
def _page_from_url() -> None:
    """`?page=Compare` opens that page.

    It is how the pages are screenshotted for the report, and it means a
    link to one page of the dashboard goes where it says it does.
    """
    wanted = st.query_params.get("page")
    if wanted in PAGES and st.session_state.get("page") != wanted:
        st.session_state["page"] = wanted


def main() -> None:
    state.init()
    _page_from_url()
    ui.inject_css()
    status = current_status()
    page = sidebar(status)

    workload = None
    error = ""
    try:
        workload = state.current_workload()
    except Exception as exc:                        # noqa: BLE001
        error = "{}: {}".format(type(exc).__name__, exc)

    profile = get_profile(state.get("profile"))
    ui.masthead(
        "IntelliSignal",
        "Adaptive signal control for an Indian urban junction",
        ui.badge(state.traffic_label(), "info"))
    ui.rules_strip([
        ("Junction", "4 arms, 2 lanes each way"),
        ("Greens", "{:g}-{:g} s, 3 s amber, 2 s all-red".format(
            profile.min_green, profile.max_green)),
        ("Fleet", "62 % two-wheelers, with real startup delay"),
        ("Traffic", state.traffic_label()),
        ("Signal", "trained model" if status.get("usable")
         else "Hybrid decides"),
    ])

    if error:
        ui.note(error, "bad", "The traffic could not be built")

    context = {"workload": workload, "model_status": status,
               "profile": state.get("profile")}
    PAGES[page].render(context)


if __name__ == "__main__":
    main()
