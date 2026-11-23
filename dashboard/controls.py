"""
IntelliSignal - Dashboard
dashboard/controls.py

The sidebar: navigation, what to run, and Run.

It is deliberately short. A teacher watching this on a laptop should be able
to read the whole sidebar without scrolling, so only the four things that
change what happens live at the top level:

    where you are · which traffic · which controller · Run

Everything occasional - run length, seed, the DQN checkpoint, the Hybrid
limits, the SUMO window - is behind **Advanced**. None of it is removed; it
is one click away and it remembers what you set.
"""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from dashboard import state
from dashboard.theme import SCENARIO_LABELS

#: The third option in the scenario list. It has no YAML file - the demand
#: comes from the Custom Traffic page instead.
CUSTOM = "custom"

LABELS = dict(SCENARIO_LABELS)
LABELS[CUSTOM] = "Custom traffic"


def _label(scenario: str) -> str:
    return LABELS.get(scenario, scenario.title())


# ---------------------------------------------------------------------------
def sidebar(controller) -> Dict[str, Any]:
    """Draw the sidebar and return the current selections."""
    with st.sidebar:
        st.markdown('<div class="is-brand">INTELLISIGNAL</div>'
                    '<div class="is-brand-sub">Adaptive traffic signal '
                    'management<br>SUMO · TraCI · Deep RL</div>',
                    unsafe_allow_html=True)

        page = st.radio("Navigation", state.PAGES,
                        index=state.PAGES.index(state.get("page")),
                        label_visibility="collapsed", key="nav_radio")
        state.set_value("page", page)

        st.divider()

        scenarios = list(controller.available_scenarios())
        if not scenarios:
            st.error("No scenarios found in config/.")
            return _empty(page)
        options = scenarios + [CUSTOM]

        st.markdown('<div class="is-side-label">Traffic</div>',
                    unsafe_allow_html=True)
        current = state.get("scenario")
        if current not in options:
            current = options[0]
        scenario = st.selectbox(
            "Traffic", options, index=options.index(current),
            format_func=_label, label_visibility="collapsed",
            key="scenario_select")
        state.set_value("scenario", scenario)

        workload = state.get("custom_workload")
        if scenario == CUSTOM:
            if workload:
                st.caption("{} vehicles · {:.0f} s · id {}".format(
                    len(workload.get("vehicles") or []),
                    workload.get("duration_s", 0),
                    str(workload.get("workload_id", ""))[:8]))
            else:
                st.caption("No traffic defined yet — open **Custom Traffic**.")

        # --- controller ------------------------------------------------
        st.markdown('<div class="is-side-label">Controller</div>',
                    unsafe_allow_html=True)
        mode_scenario = (workload.get("base_scenario", "moderate")
                         if scenario == CUSTOM and workload else scenario)
        modes = controller.control_modes(
            mode_scenario if mode_scenario in scenarios else None)
        keys = [mode.key for mode in modes]
        current_mode = state.get("control_mode")
        if current_mode not in keys:
            current_mode = keys[0]
        mode_key = st.radio(
            "Controller", keys, index=keys.index(current_mode),
            format_func=lambda k: next(
                "{}{}".format(m.label, "" if m.available else "  ·  unavailable")
                for m in modes if m.key == k),
            label_visibility="collapsed", key="mode_radio")
        state.set_value("control_mode", mode_key)
        selected = next(mode for mode in modes if mode.key == mode_key)
        if not selected.available:
            st.warning(selected.unavailable_reason)

        # --- run --------------------------------------------------------
        st.write("")
        ready = selected.available and (scenario != CUSTOM or bool(workload))
        run_clicked = st.button(
            "Run", type="primary", width="stretch", key="run_button",
            disabled=not ready,
            help=None if ready else
            "Define traffic on the Custom Traffic page first.")

        # --- everything occasional -------------------------------------
        with st.expander("Advanced"):
            summary = None
            if scenario in scenarios:
                try:
                    summary = controller.scenario_summary(scenario)
                except Exception:
                    summary = None
            default_duration = int(summary["duration"]) if summary else 1800

            duration = st.number_input(
                "Run length (simulated s)", min_value=60, max_value=7200,
                value=int(state.get("duration") or default_duration), step=60,
                key="duration_input",
                help="Simulated seconds. Custom traffic takes its length "
                     "from the Custom Traffic page instead.")
            state.set_value("duration", int(duration))

            seed = st.number_input(
                "Seed", min_value=0, max_value=100000,
                value=int(state.get("seed") or 42), step=1, key="seed_input",
                help="SUMO's random seed. The same seed gives the same "
                     "traffic, which is what makes two controllers "
                     "comparable.")
            state.set_value("seed", int(seed))

            model_name = None
            if mode_key == "dqn_adaptive" and selected.available \
                    and scenario in scenarios:
                checkpoints = controller.available_models(scenario)
                names = [entry["name"] for entry in checkpoints]
                if names:
                    chosen = state.get("model_name")
                    if chosen not in names:
                        chosen = names[0]
                    model_name = st.selectbox(
                        "Checkpoint", names, index=names.index(chosen),
                        key="model_select")
                    state.set_value("model_name", model_name)

            if mode_key == "hybrid":
                st.markdown("**Hybrid limits**")
                state.set_value("hybrid_wall_clock_s", int(st.number_input(
                    "Time limit (wall-clock s)", min_value=30, max_value=3600,
                    value=int(state.get("hybrid_wall_clock_s") or 300),
                    step=30, key="hyb_clock",
                    help="Real seconds from pressing Run, including tuning, "
                         "training and validation. Not simulated seconds.")))
                state.set_value("hybrid_max_episodes", int(st.number_input(
                    "Extra training episodes", min_value=0, max_value=500,
                    value=int(state.get("hybrid_max_episodes") or 30), step=5,
                    key="hyb_eps")))
                state.set_value("hybrid_patience", int(st.number_input(
                    "Patience (failed checks)", min_value=1, max_value=20,
                    value=int(state.get("hybrid_patience") or 3), step=1,
                    key="hyb_pat",
                    help="Consecutive completed validation checks the "
                         "candidate may lose before the search gives up.")))

            gui = st.checkbox(
                "Open the SUMO window", value=False, key="gui_checkbox",
                help="Runs in sumo-gui instead of headless. Needs a display "
                     "on the machine running the dashboard.")

        st.caption("Baseline results were measured at 1800 s with seed 42 "
                   "and live in results/csv/. Dashboard runs are written "
                   "elsewhere and never overwrite them.")

    return {
        "page": page,
        "scenario": scenario,
        "mode": mode_key,
        "mode_object": selected,
        "duration": int(duration),
        "seed": int(seed),
        "model_name": model_name,
        "run_clicked": bool(run_clicked),
        "gui": bool(gui),
        "is_custom": scenario == CUSTOM,
    }


def _empty(page: str) -> Dict[str, Any]:
    return {"page": page, "scenario": None, "mode": None, "mode_object": None,
            "duration": None, "seed": 42, "model_name": None,
            "run_clicked": False, "gui": False, "is_custom": False}
