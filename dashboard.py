"""
IntelliSignal - Streamlit dashboard
dashboard.py

Run it with:

    streamlit run dashboard.py

The demonstration this is built for
-----------------------------------
A user enters traffic for one intersection, sees exactly what will be
simulated, presses Run, and gets a measured answer saying which controller
worked better and why. Everything on the screen serves that path:

    Dashboard        where you are: the junction, the traffic, the controller
    Custom Traffic   describe traffic lane by lane, then see the preview
    Simulation       watch one run, second by second
    Comparison       the outcome, in one sentence and one table
    Results          the measured numbers, with the detail folded away
    DQN Model        what the learned controller is
    About            what is real here and what is not

Architecture
------------
                USER
                 v
          STREAMLIT DASHBOARD          this file + dashboard/
                 v
        SIMULATION CONTROLLER          simulation/controller.py
                 v
     Fixed-Time | DQN | Hybrid         agents/, simulation/hybrid*.py
                 v
             TraCI -> SUMO
                 v
             MetricsCollector          utils/metrics.py

This file never imports SUMO, TraCI or an agent directly. It asks the
controller and renders the answer.

What is real here
-----------------
Every number shown comes from a simulation that actually ran. The signal
colours are decoded from the signal states TraCI recorded. The intersection
drawing is a schematic and is labelled as one. When something has not been
measured, the page says so rather than showing the nearest number it has.
"""

from __future__ import annotations

import os
import sys

import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(
    page_title="IntelliSignal — Adaptive Traffic Signal Management",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dashboard import (charts, components, controls, custom_form,  # noqa: E402
                       hybrid_panel, state)
from dashboard.theme import CSS, fmt, scenario_label          # noqa: E402
from rl.timing_profile import PROFILES, compatible, get_profile  # noqa: E402
from simulation.controller import (ControlModeUnavailableError,  # noqa: E402
                                   MODE_DQN, MODE_FIXED_TIME, MODE_HYBRID,
                                   SimulationController, SimulationError)

CUSTOM = controls.CUSTOM


# ==========================================================================
@st.cache_resource
def get_controller() -> SimulationController:
    """One controller for the whole session."""
    return SimulationController()


@st.cache_data(ttl=30, show_spinner=False)
def cached_environment_status(_controller) -> dict:
    status = _controller.environment_status()
    return {
        "indicators": status.indicators(),
        "ready": status.simulation_ready,
        "sumo_version": status.sumo_version,
        "sumo_home": status.sumo_home,
        "sumo_binary": status.sumo_binary,
        "traci_version": status.traci_version,
        "python_version": status.python_version,
        "gymnasium": status.gymnasium_available,
        "torch": status.torch_available,
        "torch_version": status.torch_version,
        "trained_models": status.trained_models,
        "errors": status.errors,
        "scenarios": status.scenarios,
    }


# ==========================================================================
# Shared helpers
# ==========================================================================
def current_workload():
    """The resolved custom workload held in session state, if any."""
    from custom_traffic.workload import ResolvedWorkload
    raw = state.get("custom_workload")
    return ResolvedWorkload.from_dict(raw) if raw else None


def traffic_label(selection) -> str:
    if selection.get("is_custom"):
        workload = current_workload()
        return ("Custom traffic · {} vehicles".format(workload.total_vehicles)
                if workload else "Custom traffic · not defined yet")
    return scenario_label(selection["scenario"])


def active_profile(selection):
    return get_profile(selection.get("profile") or "responsive")


def profile_banner(selection) -> None:
    """Say which timing rules are in force, on every page that shows a run."""
    profile = active_profile(selection)
    st.caption(
        "Timing profile **{}** — green {:g}-{:g} s, red at most {:g}x the "
        "preceding green. {}".format(
            profile.name, profile.min_green, profile.max_green,
            profile.max_red_factor, profile.description))


def controller_label(mode: str) -> str:
    return {MODE_FIXED_TIME: "Fixed-Time", MODE_DQN: "DQN Adaptive",
            MODE_HYBRID: "Hybrid"}.get(mode, mode)


def hybrid_limits():
    from simulation.hybrid import HybridLimits
    return HybridLimits(
        wall_clock_s=float(state.get("hybrid_wall_clock_s") or 300),
        max_episodes=int(state.get("hybrid_max_episodes") or 30),
        patience=int(state.get("hybrid_patience") or 3))


def vehicle_accounting(run) -> None:
    """Requested / entered / completed / on road / waiting to enter."""
    requested = run.metric("vehicles_requested")
    if requested is None:
        return
    columns = st.columns(5)
    for column, (label, key, hint) in zip(columns, [
            ("Requested", "vehicles_requested", "asked for by the workload"),
            ("Entered", "vehicles_entered", "actually put on the network"),
            ("Completed", "vehicles_completed", "finished their route"),
            ("Still on road", "vehicles_still_on_road", "at the end of the run"),
            ("Waiting to enter", "vehicles_waiting_to_enter",
             "never inserted - the lane was full")]):
        column.metric(label, "{:,}".format(int(run.metric(key) or 0)),
                      help=hint)


# ==========================================================================
# Running
# ==========================================================================
def do_run(controller, selection) -> None:
    """Handle the Run button for every combination of traffic and mode."""
    mode = selection["mode"]
    scenario = selection["scenario"]
    is_custom = selection["is_custom"]

    try:
        if mode == MODE_HYBRID:
            _run_hybrid(controller, selection)
            return

        if is_custom:
            workload = current_workload()
            if workload is None:
                state.record_error(
                    "Define traffic on the Custom Traffic page first.")
                return
            profile = active_profile(selection)
            agent = None
            if mode == MODE_DQN:
                agent = controller.load_phase_agent(workload.base_scenario,
                                                    profile=profile.name)
                if not compatible(profile, agent.durations):
                    state.record_error(
                        "The registered model's greens are {} s, which the "
                        "{} profile ({:g}-{:g} s) does not allow. It is kept "
                        "as historical evidence, not activated here. Train a "
                        "model for this profile, or switch the profile back."
                        .format("/".join("{:g}".format(d)
                                         for d in agent.durations),
                                profile.name, profile.min_green,
                                profile.max_green))
                    return
            run = controller.run_custom(workload, mode=mode, agent=agent,
                                        seed=selection["seed"],
                                        use_gui=selection["gui"],
                                        profile=profile.name)
        else:
            run = controller.run(
                scenario, mode=mode, duration=selection["duration"],
                use_gui=selection["gui"], seed=selection["seed"],
                model_path=controller.model_path(scenario,
                                                 selection["model_name"])
                if selection["model_name"] else None)
        state.record_run(run)
        st.success(
            "Run finished. Average waiting time {} s, {:,} vehicles "
            "completed.".format(fmt(run.metric("average_waiting_time")),
                                int(run.metric("throughput") or 0)))
    except (SimulationError, ControlModeUnavailableError) as exc:
        state.record_error(str(exc))
    except Exception as exc:                      # pragma: no cover
        state.record_error("{}: {}".format(type(exc).__name__, exc))


def _run_hybrid(controller, selection) -> None:
    """Start the Hybrid search, or reconnect to the one already running."""
    from custom_traffic.resolver import workload_from_scenario
    from simulation.hybrid import configured_plan_for
    from simulation.hybrid_job import start_job

    if selection["is_custom"]:
        workload = current_workload()
        if workload is None:
            state.record_error(
                "Define traffic on the Custom Traffic page first.")
            return
    else:
        workload = workload_from_scenario(
            selection["scenario"], duration_s=float(selection["duration"]),
            seed=int(selection["seed"]))

    state.set_value("hybrid_workload_id", workload.workload_id)
    state.set_value("hybrid_workload", workload.to_dict())
    start_job(workload, hybrid_limits(), configured_plan_for(workload),
              profile=active_profile(selection).name)
    state.set_value("page", "Comparison")


# ==========================================================================
# Pages
# ==========================================================================
def page_dashboard(controller, selection, env) -> None:
    components.project_header()
    components.status_lights(env["indicators"])

    run = state.current_run(controller, selection["scenario"],
                            selection["mode"]) if not selection["is_custom"] \
        else state.get("last_run")

    left, right = st.columns([1.15, 1])
    with left:
        signal = {"NS": "green", "EW": "red"}
        components.intersection_panel(signal)
    with right:
        st.markdown("### {}".format(traffic_label(selection)))
        st.caption("Controller: **{}**".format(
            controller_label(selection["mode"])))
        profile_banner(selection)

        if run is not None and getattr(run, "metrics", None):
            columns = st.columns(3)
            columns[0].metric("Average waiting time", "{} s".format(
                fmt(run.metric("average_waiting_time"))))
            columns[1].metric("Vehicles completed",
                              "{:,}".format(int(run.metric("throughput") or 0)))
            columns[2].metric("Average queue length", "{} veh".format(
                fmt(run.metric("average_queue_length"))))
            st.caption("From {}.".format(run.source))
        else:
            st.info("Nothing measured yet. Choose the traffic and the "
                    "controller in the sidebar, then press **Run**.")

        if selection["is_custom"]:
            workload = current_workload()
            if workload is None:
                st.warning("Custom traffic is selected but no traffic has "
                           "been defined. Open **Custom Traffic**.")
            else:
                st.caption("Workload `{}` · {} vehicles over {:.0f} s".format(
                    workload.workload_id[:8], workload.total_vehicles,
                    workload.duration_s))

    if state.get("last_error"):
        st.error(state.get("last_error"))
        state.set_value("last_error", None)


# --------------------------------------------------------------------------
def page_custom(controller, selection, env) -> None:
    st.markdown("## Custom traffic")
    st.caption(
        "Describe the traffic for this intersection, then check the preview. "
        "The preview is what runs — every vehicle in it is written into the "
        "SUMO route file individually, so both controllers receive identical "
        "traffic.")

    spec = custom_form.render_form(controller)
    custom_form.render_actions(spec)

    workload = current_workload()
    st.divider()
    if workload is None:
        st.info("Press **Preview traffic** to resolve the form. Run stays "
                "disabled until there is a preview to run.")
        return
    custom_form.render_preview(workload)

    st.divider()
    st.caption("Select **Custom traffic** in the sidebar and press **Run** to "
               "simulate exactly this.")


# --------------------------------------------------------------------------
def page_simulation(controller, selection, env) -> None:
    st.markdown("## Simulation")

    run = (state.get("last_run") if selection["is_custom"]
           else state.current_run(controller, selection["scenario"],
                                  selection["mode"]))
    if run is None:
        st.info("No run to show yet. Press **Run** in the sidebar.")
        return

    series = _timeseries_for(controller, run, selection)
    if series is None or not len(series):
        st.warning("This run has no second-by-second record to replay.")
        _headline_metrics(run)
        return

    duration = float(series["time"].max())
    moment = st.slider("Time in the run (simulated seconds)", 0.0, duration,
                       float(min(state.get("signal_time") or 0, duration)),
                       step=1.0, key="signal_slider")
    state.set_value("signal_time", moment)

    row = series.iloc[(series["time"] - moment).abs().argsort().iloc[0]]
    directions = components.direction_states(row.get("signal_state"))

    left, right = st.columns([1.15, 1])
    with left:
        components.intersection_panel(directions)
    with right:
        components.signal_readout(directions)
        st.markdown("#### {}".format(_one_sentence(directions, row)))
        st.caption("At {:.0f} s: {:.0f} vehicles on the network, "
                   "{:.0f} halting.".format(
                       float(row.get("time", 0)),
                       float(row.get("running_vehicles", 0)),
                       float(row.get("queue_length", 0))))
        _headline_metrics(run)

    with st.expander("Decision details"):
        _decision_details(controller, run, selection, moment)


def _one_sentence(directions, row) -> str:
    green = [name for name, colour in directions.items() if colour == "green"]
    if not green:
        return "Both directions are stopped (clearance)."
    which = "North–South" if green[0] == "NS" else "East–West"
    return "{} has green.".format(which)


def _headline_metrics(run) -> None:
    columns = st.columns(3)
    columns[0].metric("Average waiting", "{} s".format(
        fmt(run.metric("average_waiting_time"))))
    columns[1].metric("Completed",
                      "{:,}".format(int(run.metric("throughput") or 0)))
    columns[2].metric("Average queue", "{} veh".format(
        fmt(run.metric("average_queue_length"))))


def _timeseries_for(controller, run, selection):
    import pandas as pd
    path = getattr(run, "timeseries_path", "")
    if path and os.path.isfile(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    if not selection["is_custom"]:
        try:
            return controller.load_timeseries(selection["scenario"],
                                              selection["mode"])
        except Exception:
            return None
    return None


def _decision_details(controller, run, selection, moment) -> None:
    st.caption(
        "The controller's own output, recorded while the run happened. "
        "Q-values are what the network produced at that moment, not a "
        "re-evaluation afterwards.")
    if run.mode != MODE_DQN:
        st.write("This run used a fixed-time plan, so there are no Q-values. "
                 "The signal follows the configured cycle: "
                 "north–south green, yellow, east–west green, yellow.")
        return

    import json
    meta_path = str(getattr(run, "csv_path", "")).replace(
        "_results.csv", "_run.json")
    decisions = None
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            path = os.path.join(PROJECT_ROOT, payload.get("decisions_json", ""))
            if payload.get("decisions_json") and os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as handle:
                    decisions = json.load(handle)
        except Exception:
            decisions = None

    if not decisions:
        st.write("No decision log was recorded for this run.")
        return

    import pandas as pd
    nearest = min(decisions, key=lambda entry: abs(entry["time"] - moment))
    st.write("At {:.0f} s the controller gave **{}** a **{:.0f} s** green."
             .format(nearest["time"], nearest["phase"],
                     nearest["green_seconds"]))
    st.dataframe(pd.DataFrame([nearest["q_values"]]), width="stretch",
                 hide_index=True)
    st.caption("One Q-value per green duration on the menu. The largest is "
               "the one the controller chose.")


# --------------------------------------------------------------------------
def page_comparison(controller, selection, env) -> None:
    st.markdown("## Comparison")

    workload_id = state.get("hybrid_workload_id")
    if workload_id:
        _hybrid_section(workload_id)
        st.divider()

    if selection["is_custom"]:
        _custom_comparison()
        return

    st.markdown("#### Fixed-Time against the DQN, on the recorded runs")
    try:
        result = controller.controller_comparison(selection["scenario"])
    except Exception as exc:
        components.show_error(exc)
        return
    _render_scenario_comparison(result)


def _hybrid_section(workload_id: str) -> None:
    from simulation.hybrid import load_decision
    from simulation.hybrid_job import get_job

    st.markdown("#### Hybrid")
    job = get_job(workload_id)
    if job is not None and job.alive:
        status = job.status()
        hybrid_panel.progress(status)
        columns = st.columns([1.2, 1.2, 3])
        with columns[0]:
            if st.button("Cancel", width="stretch"):
                job.cancel()
                st.rerun()
        with columns[1]:
            if st.button("Refresh", width="stretch"):
                st.rerun()
        st.caption("The search runs in the background. Refreshing the page "
                   "reconnects to it; pressing Run again does not start a "
                   "second one.")
        return

    payload = None
    if job is not None and job.decision is not None:
        payload = job.decision.to_dict()
    else:
        saved = load_decision(workload_id)
        payload = saved.to_dict() if saved else None
    if job is not None and job.state == "failed":
        st.error(job.error)
    hybrid_panel.decision(payload)


def _custom_comparison() -> None:
    from simulation.custom_runs import load_custom_run
    workload = current_workload()
    if workload is None:
        st.info("No custom traffic defined.")
        return

    fixed = load_custom_run(workload.workload_id, "fixed_time")
    learned = load_custom_run(workload.workload_id, "dqn")
    if not fixed and not learned:
        st.info("Run this traffic under Fixed-Time and under the DQN to "
                "compare them. Both runs use the same vehicles.")
        return

    import pandas as pd
    rows = []
    for label, payload in (("Fixed-Time", fixed), ("DQN", learned)):
        if not payload:
            continue
        summary = payload["summary"]
        rows.append({
            "Controller": label,
            "Average waiting (s)": round(
                float(summary.get("average_waiting_time", 0)), 2),
            "Completed": int(summary.get("vehicles_completed", 0)),
            "Average queue (veh)": round(
                float(summary.get("average_queue_length", 0)), 2),
            "Waiting to enter": int(
                summary.get("vehicles_waiting_to_enter", 0)),
        })

    if len(rows) == 2:
        better = min(rows, key=lambda row: row["Average waiting (s)"])
        other = max(rows, key=lambda row: row["Average waiting (s)"])
        st.success(
            "**{} waited less on this traffic: {:.2f} s against {:.2f} s.**"
            .format(better["Controller"], better["Average waiting (s)"],
                    other["Average waiting (s)"]))
    else:
        st.info("Only one controller has been run on this traffic so far.")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("Both controllers were given the identical vehicle list of "
               "workload `{}`.".format(workload.workload_id[:8]))


def _render_scenario_comparison(result) -> None:
    import pandas as pd
    rows = result.get("rows") or []
    if not rows:
        st.info("No comparison has been recorded for this scenario yet.")
        return
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    with st.expander("Per-seed results and how they were measured"):
        st.json(result, expanded=False)


# --------------------------------------------------------------------------
def page_results(controller, selection, env) -> None:
    st.markdown("## Results")
    run = (state.get("last_run") if selection["is_custom"]
           else state.current_run(controller, selection["scenario"],
                                  selection["mode"]))
    if run is None:
        st.info("No results yet for this combination. Press **Run**.")
        return

    components.controller_badge(run.mode,
                                str(run.metric("model", "") or ""))
    profile_banner(selection)
    _headline_metrics(run)
    vehicle_accounting(run)
    st.caption("From {}.".format(run.source))
    _timing_compliance(run)

    with st.expander("Every measured value from this run"):
        st.json(run.metrics, expanded=False)
    with st.expander("Saved runs on disk"):
        _saved_runs_table()


def _timing_compliance(run) -> None:
    """Did the run obey the profile? Measured from the recorded phases."""
    report = run.metric("timing_compliance")
    if not report:
        return
    ok = report.get("compliant", True)
    (st.success if ok else st.error)(
        "Timing {}: greens ran {:g}-{:g} s, {} below the minimum, {} above "
        "the maximum, {} red-limit breach(es).".format(
            "compliant" if ok else "NOT compliant",
            report.get("min_green_seen") or 0,
            report.get("max_green_seen") or 0,
            report.get("greens_below_min", 0),
            report.get("greens_above_max", 0),
            report.get("red_limit_breaches", 0)))
    truncated = report.get("truncated_phases", 0)
    if truncated:
        st.caption("{} phase(s) were cut short by the end of the run. They "
                   "are counted as truncated, not as short greens."
                   .format(truncated))


def _saved_runs_table() -> None:
    from simulation.custom_runs import saved_custom_runs
    rows = saved_custom_runs()
    if not rows:
        st.write("No saved custom runs yet.")
        return
    import pandas as pd
    st.dataframe(pd.DataFrame([{
        "Workload": row.get("workload_id", "")[:8],
        "Name": row.get("workload_name", ""),
        "Controller": row.get("controller", ""),
        "Saved": row.get("saved_at", ""),
        "Average waiting (s)": round(float(
            (row.get("summary") or {}).get("average_waiting_time", 0)), 2),
    } for row in rows]), width="stretch", hide_index=True)
    st.caption("Saved under results/custom_runs/. A run can be reloaded by "
               "its workload id.")


# --------------------------------------------------------------------------
def page_model(controller, selection, env) -> None:
    st.markdown("## DQN model")
    st.write(
        "The learned controller chooses **how long each green lasts**, once "
        "per green, from a menu bounded by the minimum and maximum green. "
        "It cannot ask for an unsafe green, every change passes through the "
        "yellow clearance, and every decision hands the road to the other "
        "direction next.")

    from agents.phase_dqn_agent import registered_phase_model
    base = (current_workload().base_scenario
            if selection["is_custom"] and current_workload()
            else selection["scenario"])
    record = registered_phase_model(base) if base else None

    if not record:
        st.info("No duration-choosing model is registered for '{}'.".format(base))
    else:
        measurement = record.get("measurement") or {}
        columns = st.columns(3)
        columns[0].metric("Measured waiting", "{} s".format(
            fmt(measurement.get("mean_waiting_time"))))
        columns[1].metric("Best fixed plan", "{} s".format(
            fmt(measurement.get("baseline_mean_waiting_time"))))
        learned = measurement.get("mean_waiting_time")
        baseline = measurement.get("baseline_mean_waiting_time")
        columns[2].metric(
            "Difference",
            "{:+.1f} %".format(100.0 * (learned - baseline) / baseline)
            if learned and baseline else "—")
        st.caption(record.get("note", ""))

        with st.expander("Checkpoint details"):
            st.json({k: v for k, v in record.items()
                     if k != "measurement"}, expanded=False)
        with st.expander("The measurement behind the promotion"):
            st.json(measurement, expanded=False)

    with st.expander("Training curves"):
        try:
            history = controller.training_history(base)
            if history is None or not len(history):
                st.write("No training history recorded for this scenario.")
            else:
                st.altair_chart(
                    charts.training_curve_chart(
                        history, "average_waiting_time",
                        "Validation waiting time"),
                    width="stretch")
        except Exception:
            st.write("No training history recorded for this scenario.")


# --------------------------------------------------------------------------
def page_about(controller, selection, env) -> None:
    st.markdown("## About this project")
    st.write(
        "IntelliSignal compares a conventional fixed-time traffic signal "
        "with a reinforcement-learning controller on one four-way "
        "intersection, simulated in SUMO. Every number in this dashboard "
        "comes from a simulation that actually ran.")

    columns = st.columns(3)
    columns[0].metric("Scenarios", str(len(env.get("scenarios") or [])))
    columns[1].metric("SUMO", env.get("sumo_version") or "not found")
    columns[2].metric("PyTorch", env.get("torch_version") or "not installed")

    with st.expander("The three controllers"):
        st.markdown(
            "- **Fixed-Time** — the conventional plan from the scenario "
            "file. Each direction gets a fixed green, every cycle.\n"
            "- **DQN Adaptive** — chooses each green's duration from the "
            "traffic state.\n"
            "- **Hybrid** — starts on a valid fixed plan, then spends a "
            "fixed amount of wall-clock time tuning that plan and testing "
            "the DQN against it. The DQN takes over only if it measurably "
            "wins; a tie goes to fixed time.")

    with st.expander("What is real, and what is not"):
        st.markdown(
            "- The intersection drawing is a **schematic**, not a SUMO "
            "rendering. Real SUMO screenshots are on the Results page.\n"
            "- Signal colours are decoded from the states TraCI recorded.\n"
            "- Demand for Moderate and Heavy is generated without "
            "`random=\"true\"`, so arrivals are on a fixed schedule. Custom "
            "traffic writes every vehicle out individually.\n"
            "- When something has not been measured, the page says so.")

    with st.expander("Environment"):
        st.json({k: env.get(k) for k in
                 ("python_version", "sumo_version", "sumo_home",
                  "sumo_binary", "traci_version", "torch_version",
                  "gymnasium", "trained_models")}, expanded=False)
        if env.get("errors"):
            st.warning("\n".join(env["errors"]))


# ==========================================================================
PAGE_FUNCTIONS = {
    "Dashboard": page_dashboard,
    "Custom Traffic": page_custom,
    "Simulation": page_simulation,
    "Comparison": page_comparison,
    "Results": page_results,
    "DQN Model": page_model,
    "About": page_about,
}


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    state.init_state()
    controller = get_controller()
    env = cached_environment_status(controller)

    selection = controls.sidebar(controller)
    if selection.get("run_clicked"):
        with st.spinner("Running…"):
            do_run(controller, selection)

    page = selection.get("page") or "Dashboard"
    PAGE_FUNCTIONS.get(page, page_dashboard)(controller, selection, env)


if __name__ == "__main__":
    main()
