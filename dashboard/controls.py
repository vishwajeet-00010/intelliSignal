"""
IntelliSignal - Dashboard
dashboard/controls.py

The sidebar: navigation, scenario, control mode, duration and the Run button.

The control-mode selector is the important part architecturally. It is built
from `SimulationController.control_modes(scenario)`, so an unavailable mode is
shown, labelled and explained rather than hidden.

Stage 3 note: the mode list is asked for PER SCENARIO, because whether the
DQN can run depends on whether a trained checkpoint exists for that scenario.
Selecting "heavy" with no heavy model therefore disables the DQN mode and
explains why, without any code in this file knowing what a model is. The
model and seed pickers appear only when the DQN mode is selected, so the
fixed-time sidebar is exactly what it was.
"""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from dashboard import state
from dashboard.theme import SCENARIO_LABELS


def sidebar(controller) -> Dict[str, Any]:
    """Draw the sidebar and return the current selections."""
    with st.sidebar:
        st.markdown('<div class="is-brand">INTELLISIGNAL</div>'
                    '<div class="is-brand-sub">Adaptive Traffic Signal Management<br>'
                    'SUMO · TraCI · Deep Reinforcement Learning</div>',
                    unsafe_allow_html=True)

        # --- navigation ------------------------------------------------
        st.markdown('<div class="is-side-label">Navigation</div>',
                    unsafe_allow_html=True)
        page = st.radio("Navigation", state.PAGES,
                        index=state.PAGES.index(state.get("page")),
                        label_visibility="collapsed", key="nav_radio")
        state.set_value("page", page)

        st.divider()

        # --- scenario --------------------------------------------------
        scenarios = controller.available_scenarios()
        if not scenarios:
            st.error("No scenarios found in config/.")
            return {"page": page, "scenario": None, "mode": None,
                    "mode_object": None, "duration": None, "seed": 42,
                    "model_name": None, "run_clicked": False, "gui": False}

        st.markdown('<div class="is-side-label">Scenario</div>',
                    unsafe_allow_html=True)
        current = state.get("scenario")
        if current not in scenarios:
            current = scenarios[0]
        scenario = st.selectbox(
            "Scenario", scenarios, index=scenarios.index(current),
            format_func=lambda s: SCENARIO_LABELS.get(s, s.title()),
            label_visibility="collapsed", key="scenario_select")
        state.set_value("scenario", scenario)

        # --- control mode ----------------------------------------------
        # Asked per scenario: the DQN mode is available only when a trained
        # checkpoint exists for THIS scenario.
        st.markdown('<div class="is-side-label">Control Mode</div>',
                    unsafe_allow_html=True)
        modes = controller.control_modes(scenario)
        labels = {m.key: "{}{}".format(
            m.label, "" if m.available else "  ·  no trained model")
            for m in modes}
        mode_keys = [m.key for m in modes]
        current_mode = state.get("control_mode")
        if current_mode not in mode_keys:
            current_mode = mode_keys[0]
        mode_key = st.radio(
            "Control mode", mode_keys, index=mode_keys.index(current_mode),
            format_func=lambda k: labels[k],
            label_visibility="collapsed", key="mode_radio")
        state.set_value("control_mode", mode_key)

        selected_mode = next(m for m in modes if m.key == mode_key)
        if not selected_mode.available:
            st.warning(selected_mode.unavailable_reason)

        # --- the trained model, only when the DQN mode is selected ------
        model_name = None
        if selected_mode.key == "dqn_adaptive" and selected_mode.available:
            checkpoints = controller.available_models(scenario)
            names = [m["name"] for m in checkpoints]
            st.markdown('<div class="is-side-label">Trained Model</div>',
                        unsafe_allow_html=True)
            chosen = state.get("model_name")
            if chosen not in names:
                chosen = names[0]
            model_name = st.selectbox(
                "Model", names, index=names.index(chosen),
                label_visibility="collapsed", key="model_select",
                help="Checkpoints found in models/dqn_{}/. best.pt is the one "
                     "with the best mean return during training; final.pt is "
                     "the last episode.".format(scenario))
            state.set_value("model_name", model_name)
            st.caption("models/dqn_{}/{}".format(scenario, model_name))

        # --- duration ---------------------------------------------------
        summary = None
        try:
            summary = controller.scenario_summary(scenario)
        except Exception:
            pass
        default_duration = int(summary["duration"]) if summary else 1800

        st.markdown('<div class="is-side-label">Simulation Duration</div>',
                    unsafe_allow_html=True)
        duration = st.number_input(
            "Simulation duration (s)", min_value=60, max_value=7200,
            value=int(state.get("duration") or default_duration), step=60,
            label_visibility="collapsed", key="duration_input",
            help="Simulated seconds. The Stage 1 results were measured at "
                 "1800 s; a shorter run is useful for a quick demonstration.")
        state.set_value("duration", int(duration))

        # --- seed, only for a DQN run -----------------------------------
        # The fixed-time command line takes its seed from the scenario YAML
        # (42), so offering a seed box for it would be offering a control
        # that does nothing. A DQN run sets the SUMO seed directly.
        seed = int(state.get("seed") or 42)
        if selected_mode.key == "dqn_adaptive" and selected_mode.available:
            st.markdown('<div class="is-side-label">Random Seed</div>',
                        unsafe_allow_html=True)
            seed = st.number_input(
                "Seed", min_value=0, max_value=100000, value=seed, step=1,
                label_visibility="collapsed", key="seed_input",
                help="SUMO's random seed. The same seed gives the same "
                     "traffic, which is how a DQN run and a fixed-time run "
                     "can be compared fairly.")
            state.set_value("seed", int(seed))

        # --- run --------------------------------------------------------
        st.write("")
        run_clicked = st.button(
            "Run Simulation", type="primary", width="stretch",
            disabled=not selected_mode.available,
            key="run_button")

        gui = st.checkbox(
            "Open SUMO visualization window", value=False, key="gui_checkbox",
            help="Runs the simulation in sumo-gui instead of headless. Works "
                 "when the dashboard runs on your own machine with a display.")

        st.divider()
        st.caption("Stage 1 baseline results were measured at 1800 s with "
                   "seed 42 and are stored in results/csv/. Dashboard runs "
                   "are written to results/data/ and never overwrite them.")

    return {
        "page": page,
        "scenario": scenario,
        "mode": mode_key,
        "mode_object": selected_mode,
        "duration": int(duration),
        "seed": int(seed),
        "model_name": model_name,
        "run_clicked": bool(run_clicked),
        "gui": bool(gui),
    }
