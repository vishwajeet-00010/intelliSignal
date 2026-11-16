"""
IntelliSignal - Streamlit dashboard
dashboard.py

Run it with:

    streamlit run dashboard.py

Architecture
------------
                USER
                 |
                 v
          STREAMLIT DASHBOARD          <-- this file + dashboard/
                 |
                 v
        SIMULATION CONTROLLER          simulation/controller.py
                 |
        +--------+--------+
        |                 |
        v                 v
 Fixed-Time           DQN Agent       agents/
 Controller          (Stage 3)
        |                 |
        +--------+--------+
                 v
                TraCI
                 v
                SUMO
                 v
             Metrics                   utils/metrics.py
                 v
              Results                  results/
                 v
          Streamlit Charts             dashboard/charts.py

This file never imports SUMO, TraCI or an agent. It asks the controller for
everything and renders the answer, which is what made the DQN stage a
back-end change rather than a rewrite of the interface.

What is real here
-----------------
Every number shown comes from a CSV that a SUMO simulation wrote. The status
lights are detected from the running environment. The signal colours on the
Simulation page are decoded from the signal states TraCI recorded during the
run. Nothing is hard-coded.

Stage 3 honesty rules
---------------------
  * The DQN mode is offered only when PyTorch is installed AND a trained
    checkpoint exists for the selected scenario. The controller checks for
    the file every time the sidebar is drawn.
  * When no model exists the mode is disabled and says "DQN unavailable —
    train a model first", with the command that trains one.
  * A DQN run is a real SUMO simulation driven by the loaded network. DQN
    results live in their own files, and the dashboard never falls back to
    fixed-time numbers under a DQN heading - it reports that there is no DQN
    run yet.
  * Every result on screen is labelled with the controller that produced it.
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

from dashboard import charts, components, controls, state  # noqa: E402
from dashboard.theme import CSS, fmt, scenario_label  # noqa: E402
from simulation.controller import (ControlModeUnavailableError,  # noqa: E402
                                   SimulationController, SimulationError)


# ==========================================================================
@st.cache_resource
def get_controller() -> SimulationController:
    """One controller for the whole session."""
    return SimulationController()


@st.cache_data(ttl=30, show_spinner=False)
def cached_environment_status(_controller) -> dict:
    """Detected environment health.

    Cached briefly so that probing SUMO does not run on every widget click,
    but not so long that a fixed installation keeps showing as broken.
    """
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
        "sumo_rl": status.sumo_rl_available,
        "torch": status.torch_available,
        "torch_version": status.torch_version,
        "trained_models": status.trained_models,
        "errors": status.errors,
        "scenarios": status.scenarios,
    }


def load_comparison(controller):
    try:
        return controller.comparison_table()
    except Exception:
        import pandas as pd
        return pd.DataFrame()


# ==========================================================================
# Pages
# ==========================================================================
def page_dashboard(controller, selection, env) -> None:
    scenario = selection["scenario"]
    mode = selection["mode"]
    run = state.current_run(controller, scenario, mode)

    components.section("Current Results",
                       "Headline metrics for the selected scenario and control "
                       "mode, read from the CSV a SUMO run produced.")
    components.controller_badge(mode, selection.get("model_name") or "")
    st.write("")
    if run is None and mode == "dqn_adaptive":
        st.info("No DQN run for this scenario yet. Press **Run Simulation** "
                "in the sidebar to run one. The dashboard will not show "
                "fixed-time numbers here — switch the control mode to "
                "Fixed-Time to see those.")
    components.kpi_row(run)

    if run is not None:
        st.write("")
        components.secondary_metrics(run)
        st.markdown('<div class="is-caption">Source: <code>{}</code></div>'.format(
            os.path.relpath(run.csv_path, PROJECT_ROOT)), unsafe_allow_html=True)

    st.markdown('<hr class="is-rule">', unsafe_allow_html=True)

    left, right = st.columns([1, 1.25], gap="large")

    with left:
        components.section("Intersection")
        # Show the configured start of the cycle rather than dead grey lamps.
        # This is the plan, not a live reading - the caption says so, and the
        # Simulation page shows the signal states actually recorded by TraCI.
        start_signal, start_caption = {"NS": "off", "EW": "off"}, ""
        try:
            agent = controller.agent_for_mode("fixed_time", scenario)
            from agents import PHASE_NS
            if agent.phase_at(0) == PHASE_NS:
                start_signal = {"NS": "green", "EW": "red"}
            else:
                start_signal = {"NS": "red", "EW": "green"}
            start_caption = ("Signal colours show the configured start of the "
                             "fixed-time cycle, not a live reading.")
        except Exception:
            pass
        components.intersection_panel(start_signal, caption=start_caption)

    with right:
        components.section("Scenario Configuration",
                           "Read from the YAML file, not typed into the dashboard.")
        try:
            summary = controller.scenario_summary(scenario)
        except SimulationError as error:
            components.show_error(error)
            summary = None

        if summary:
            columns = st.columns(2)
            with columns[0]:
                st.markdown(
                    "**Traffic demand**\n\n"
                    "- Total: **{total:.0f}** veh/h\n"
                    "- North: {north} · South: {south} veh/h\n"
                    "- East: {east} · West: {west} veh/h\n"
                    "- Turns: {right:.0%} right · {straight:.0%} straight · "
                    "{left:.0%} left".format(
                        total=summary["total_demand"],
                        **summary["demand_per_approach"],
                        **summary["turn_ratios"]))
            with columns[1]:
                st.markdown(
                    "**Fixed-time signal plan**\n\n"
                    "- NS green: **{ns:.0f} s**\n"
                    "- EW green: **{ew:.0f} s**\n"
                    "- Yellow: **{y:.0f} s**\n"
                    "- Cycle: **{c:.0f} s**".format(
                        ns=summary["ns_green"], ew=summary["ew_green"],
                        y=summary["yellow"], c=summary["cycle_time"]))
            st.caption("Duration {:.0f} s · step {:.0f} s · seed {}".format(
                summary["duration"], summary["step_length"], summary["seed"]))

    st.markdown('<hr class="is-rule">', unsafe_allow_html=True)

    components.section("Scenario Comparison",
                       "Same network, same signal plan — only the traffic "
                       "demand differs.")
    render_comparison(controller)


def page_simulation(controller, selection, env) -> None:
    scenario = selection["scenario"]
    mode = selection["mode_object"]

    components.section("Traffic Simulation",
                       "Runs the existing project pipeline: "
                       "SUMO → TraCI → metrics → CSV.")

    if not env["ready"]:
        st.error("SUMO is not available.\n\n"
                 "Please verify that SUMO is installed and SUMO_HOME is "
                 "configured correctly.")
        for problem in env["errors"]:
            st.caption("· " + problem)
        return

    columns = st.columns([1.1, 1])
    with columns[0]:
        st.markdown(
            "**Scenario:** {}  \n"
            "**Control mode:** {} {}  \n"
            "**Duration:** {} simulated seconds".format(
                scenario_label(scenario), mode.label,
                "(implemented)" if mode.available else "(not implemented)",
                selection["duration"]))
        st.caption("Press **Run Simulation** in the sidebar. The simulation "
                   "only runs when that button is pressed — never on a page "
                   "refresh.")
    with columns[1]:
        if controller.sumo_gui_available():
            st.markdown("**Open SUMO Visualization**")
            st.caption("Run this in a terminal on the machine with the display "
                       "to watch the intersection in sumo-gui:")
            st.code(controller.gui_command(scenario, selection["duration"]),
                    language="bash")
        else:
            st.caption("sumo-gui was not found, so the visualization command "
                       "is not offered.")

    if state.get("last_error"):
        st.error(state.get("last_error"))

    run = state.current_run(controller, scenario, selection["mode"])
    if run is None:
        st.info("No {} results for this scenario yet. Press **Run "
                "Simulation**.".format(mode.label))
        return

    st.markdown('<hr class="is-rule">', unsafe_allow_html=True)
    components.section("Measured Result", "From the run recorded in {}".format(
        os.path.relpath(run.csv_path, PROJECT_ROOT)))
    components.controller_badge(run.mode, str(run.metric("model", "") or ""))
    st.write("")
    components.kpi_row(run)
    st.write("")
    components.secondary_metrics(run)

    if run.mode == "dqn_adaptive":
        st.caption(
            "Produced by the trained network in `{model}` ({episodes} training "
            "episodes, architecture {arch}), driving the Stage 2 environment "
            "through the same safety controller as every other run. Episode "
            "return {ret}, signal switches {sw}.".format(
                model=run.metric("model", "?"),
                episodes=run.metric("episodes_trained", "?"),
                arch=run.metric("architecture", "?"),
                ret=fmt(run.metric("episode_return"), 2),
                sw=fmt(run.metric("switch_count"), 0)))

    # --- recorded signal state ------------------------------------------
    st.markdown('<hr class="is-rule">', unsafe_allow_html=True)
    timeseries = controller.load_timeseries(scenario, selection["mode"])
    components.section(
        "Signal State During the Run",
        "Recorded from TraCI at every simulated second and saved to the "
        "time-series CSV. Move the slider to step through the run.")

    if timeseries is None or "signal_state" not in timeseries.columns:
        st.info("No recorded signal data for this scenario. The time-series "
                "CSV is written whenever a simulation is run.")
        return

    max_time = int(timeseries["time"].max())
    chosen = st.slider("Simulation time (s)", 0, max_time,
                       min(state.get("signal_time", 0), max_time), step=1,
                       key="signal_slider")
    state.set_value("signal_time", chosen)

    row = timeseries.iloc[(timeseries["time"] - chosen).abs().argmin()]
    signal_state = str(row.get("signal_state", ""))
    directions = components.direction_states(signal_state)

    left, right = st.columns([1, 1.2], gap="large")
    with left:
        components.intersection_panel(
            directions,
            caption="Signal colours are decoded from the SUMO state recorded "
                    "at t = {:.0f} s.".format(row["time"]))
    with right:
        st.markdown("**Current Signal** · t = {:.0f} s".format(row["time"]))
        components.signal_readout(
            directions,
            "Raw SUMO state: <code>{}</code> — 16 controlled links, "
            "4 per approach.".format(signal_state))
        st.write("")
        metric_columns = st.columns(2)
        with metric_columns[0]:
            components.kpi_card("Queue at this second",
                                fmt(row.get("queue_length"), 0), " vehicles")
        with metric_columns[1]:
            components.kpi_card("Vehicles in network",
                                fmt(row.get("running_vehicles"), 0), "")

        st.write("")
        if run.mode == "dqn_adaptive":
            render_dqn_decision(controller, scenario, float(row["time"]))
        else:
            try:
                summary = controller.scenario_summary(scenario)
                agent = controller.agent_for_mode("fixed_time", scenario)
                from agents import PHASE_EW, PHASE_NS, TrafficState
                phase = (PHASE_NS if directions["NS"] in ("green", "yellow")
                         else PHASE_EW)
                traffic_state = TrafficState(
                    current_phase=phase,
                    time_since_phase_change=float(row["time"]) % summary["cycle_time"],
                    simulation_time=float(row["time"]))
                components.agent_decision_panel(agent.explain(traffic_state))
            except Exception:
                components.agent_decision_panel(
                    None, "The controller decision could not be shown for "
                          "this step.")


def render_dqn_decision(controller, scenario: str, moment: float) -> None:
    """The network's RECORDED decision nearest a moment in the run.

    The Q-values shown here were produced by the network during the run and
    written to the decision log. They are not recomputed from a guessed state:
    the 10-number observation cannot be rebuilt from the saved time series, so
    rather than invent one, the dashboard shows what was recorded or says it
    has nothing to show.
    """
    decisions = controller.load_decisions(scenario)
    if decisions is None or "q_keep" not in decisions.columns:
        components.agent_decision_panel(
            None, "No decision log for this run. Run the DQN mode from the "
                  "sidebar and the Q-values the network produces will be "
                  "recorded here.")
        return

    row = decisions.iloc[(decisions["simulation_time"] - moment).abs().argmin()]
    keep = float(row["q_keep"])
    switch = float(row["q_switch"])
    chose_switch = int(row.get("env_action", 0)) == 1
    phase = int(row.get("phase", 0))
    phase_name = "North-South green" if phase == 0 else "East-West green"
    components.agent_decision_panel({
        "agent": "DQN Adaptive Controller",
        "action_name": ("East-West green" if (phase == 0) == chose_switch
                        else "North-South green"),
        "is_learned": True,
        "uses_traffic_state": True,
        "env_action_name": "SWITCH" if chose_switch else "KEEP",
        "q_values": {"keep": keep, "switch": switch},
        "model_path": str(controller.model_path(scenario) or ""),
        "reason": ("Recorded at t = {:.0f} s: Q(keep) = {:+.4f}, "
                   "Q(switch) = {:+.4f}, so the network {} the signal."
                   .format(float(row["simulation_time"]), keep, switch,
                           "changed" if chose_switch else "held")),
        "state": {
            "current phase": phase_name,
            "total queue": row.get("total_queue"),
            "total waiting time": row.get("total_waiting_time"),
            "decision number": row.get("decision"),
        },
    })


def page_results(controller, selection, env) -> None:
    components.section("Results",
                       "Charts built from the CSV files the simulations wrote.")

    comparison = load_comparison(controller)
    if comparison.empty:
        st.info("No results found yet. Run a scenario from the **Simulation** "
                "page, or run `python simulation/run_scenario.py all` in a "
                "terminal.")
        return

    columns = st.columns(3, gap="medium")
    for column, chart in zip(columns, (charts.waiting_time_chart(comparison),
                                       charts.queue_length_chart(comparison),
                                       charts.throughput_chart(comparison))):
        with column:
            if chart is not None:
                st.altair_chart(chart, width="stretch")

    st.markdown('<hr class="is-rule">', unsafe_allow_html=True)

    series = {}
    for scenario in controller.available_scenarios():
        frame = controller.load_timeseries(scenario)
        if frame is not None:
            series[scenario] = frame

    chart = charts.queue_over_time_chart(series)
    if chart is not None:
        st.altair_chart(chart, width="stretch")
        st.caption("One sample per simulated second, recorded through TraCI. "
                   "The saw-tooth pattern is the signal cycle: queues build "
                   "during red and discharge during green.")
    else:
        st.info("No time-series data available yet.")

    chart = charts.vehicles_over_time_chart(series)
    if chart is not None:
        st.altair_chart(chart, width="stretch")

    st.markdown('<hr class="is-rule">', unsafe_allow_html=True)
    components.section("Scenario Comparison")
    render_comparison(controller)

    st.markdown('<hr class="is-rule">', unsafe_allow_html=True)
    components.section("SUMO Screenshots",
                       "Captured from the running simulation by sumo-gui.")
    screenshots = controller.screenshots()
    if not screenshots:
        st.info("No screenshots found. Generate them with "
                "`python presentation/capture_screenshots.py`.")
    else:
        for start in range(0, len(screenshots), 3):
            row = screenshots[start:start + 3]
            for column, shot in zip(st.columns(len(row)), row):
                with column:
                    st.image(shot["path"], caption=shot["caption"],
                             width="stretch")


def page_model(controller, selection, env) -> None:
    """The trained model: its status, where it came from, and its curves."""
    scenario = selection["scenario"]

    components.section(
        "DQN Model",
        "What has actually been trained for this scenario. Every value is "
        "read from the checkpoint and from the training history CSV.")

    status = controller.model_status(scenario)

    if not status["available"]:
        st.warning(status["reason"] or "No trained DQN model.")
        st.markdown("**Train one with:**")
        st.code("python training/train_dqn.py --scenario {} --episodes 20 "
                "--seed 42".format(scenario), language="bash")
        st.caption("A 20-episode run is a quick demonstration; 100 episodes "
                   "is the default training run. Neither guarantees a good "
                   "policy — the Comparison page is what decides that.")
        if not status["torch_available"]:
            st.code("pip install -r requirements.txt", language="bash")
        return

    summary = status["summary"]
    metadata = status["metadata"] or {}

    columns = st.columns(4)
    for column, (label, value, unit) in zip(columns, [
            ("Model status", "Trained", ""),
            ("Episodes trained", fmt(summary.get("episodes_trained"), 0), ""),
            ("Parameters", fmt(summary.get("parameters"), 0), ""),
            ("Discount γ", fmt(summary.get("gamma"), 2), "")]):
        with column:
            components.kpi_card(label, value, unit)

    st.write("")
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown(
            "**Model**\n\n"
            "- Path: `{path}`\n"
            "- Scenario: **{scenario}**\n"
            "- Architecture: `{arch}`\n"
            "- Observation: **{obs}** values → **{actions}** Q-values\n"
            "- Seed: `{seed}`\n"
            "- Trained on: {date}".format(
                path=status["model_path"],
                scenario=summary.get("scenario") or scenario,
                arch=summary.get("architecture", "?"),
                obs=summary.get("observation_size", "?"),
                actions=summary.get("num_actions", "?"),
                seed=summary.get("seed", "?"),
                date=metadata.get("training_date", "not recorded")))
    with right:
        epsilon = metadata.get("epsilon", {})
        st.markdown(
            "**Hyperparameters used**\n\n"
            "- Learning rate: `{lr}`\n"
            "- Batch size: `{batch}`\n"
            "- Replay capacity: `{replay}`\n"
            "- Target update: every `{target}` gradient steps\n"
            "- Exploration: `{eps}`\n"
            "- Epsilon now: `{now}` (0 = greedy, as the dashboard runs it)".format(
                lr=summary.get("learning_rate"), batch=summary.get("batch_size"),
                replay=summary.get("replay_capacity"),
                target=summary.get("target_update_frequency"),
                eps=(epsilon.get("schedule") if isinstance(epsilon, dict)
                     else epsilon) or "not recorded",
                now=fmt(summary.get("epsilon"), 2)))

    st.caption("Checkpoints found: " + ", ".join(
        "{} ({:.0f} kB)".format(c["name"], c["size_kb"])
        for c in status["checkpoints"]))

    # --- the architecture, as a picture a panel can follow -------------
    st.markdown('<hr class="is-rule">', unsafe_allow_html=True)
    components.section("How a decision is made")
    st.code(
        "  traffic state in SUMO\n"
        "          |\n"
        "  {obs}-value observation      queues, waiting times, phase, phase age\n"
        "          |\n"
        "     Q-network  {arch}\n"
        "          |\n"
        "  Q(keep)   Q(switch)           expected future congestion cost\n"
        "          |\n"
        "   pick the larger Q-value      (greedy: no exploration here)\n"
        "          |\n"
        "  safety controller             yellow clearance, min/max green\n"
        "          |\n"
        "   traffic signal in SUMO".format(
            obs=summary.get("observation_size", 10),
            arch=summary.get("architecture", "")),
        language="text")

    # --- training curves ------------------------------------------------
    st.markdown('<hr class="is-rule">', unsafe_allow_html=True)
    components.section(
        "Training History",
        "Read from results/training/. Raw per-episode values — noisy "
        "learning curves are the real result and are not smoothed.")

    history = controller.training_history(scenario)
    if history is None:
        st.info("No training history CSV found for this scenario in "
                "results/training/.")
        return

    pairs = [("total_reward", "Return per episode", "Total reward"),
             ("loss", "Training loss per episode", "Mean TD loss"),
             ("epsilon", "Exploration rate", "Epsilon"),
             ("average_queue_length", "Average queue per episode", "Vehicles"),
             ("average_waiting_time", "Average waiting time per episode",
              "Seconds per vehicle"),
             ("throughput", "Throughput per episode", "Vehicles")]
    for start in range(0, len(pairs), 2):
        row_columns = st.columns(2, gap="medium")
        for column, (field, title, unit) in zip(row_columns, pairs[start:start + 2]):
            with column:
                chart = charts.training_curve_chart(history, field, title, unit)
                if chart is not None:
                    st.altair_chart(chart, width="stretch")

    episodes = int(history["episode"].max())
    st.caption("{} episodes recorded across {} training seed(s). The reward "
               "is the sum of the per-decision rewards, which are negative "
               "congestion costs — closer to zero is better.".format(
                   episodes, history["seed"].nunique()
                   if "seed" in history.columns else 1))


def page_comparison(controller, selection, env) -> None:
    """Fixed-Time vs DQN: the multi-seed experiment and the live runs."""
    scenario = selection["scenario"]

    components.section(
        "Fixed-Time vs DQN",
        "Both controllers on the same network, the same demand, the same "
        "duration and the same seed, measured with the same collector.")

    evaluation = controller.evaluation_results(scenario)

    if evaluation["available"]:
        seeds = evaluation["seeds"]
        st.markdown(
            "**Multi-seed experiment** — {n} seed(s): {seeds}. "
            "Values are mean ± sample standard deviation.".format(
                n=len(seeds), seeds=", ".join(str(s) for s in seeds)))

        improvement = evaluation["improvement"]
        if improvement is not None:
            rows = improvement.to_dict("records")
            components.comparison_table(
                [{"metric": r["metric"],
                  "lower_is_better": bool(r.get("lower_is_better", True)),
                  "fixed_time": r.get("fixed_time_mean"),
                  "dqn": r.get("dqn_mean"),
                  "improvement_percent": (
                      None if r.get("improvement_percent") != r.get("improvement_percent")
                      else r.get("improvement_percent"))}
                 for r in rows])

            better = [r for r in rows
                      if r.get("dqn_better") is True or r.get("dqn_better") == "True"]
            worse = [r for r in rows
                     if r.get("dqn_better") is False or r.get("dqn_better") == "False"]
            st.caption(
                "The DQN is better on {b} of {t} metrics and worse on {w}. "
                "A negative percentage is shown exactly as measured — nothing "
                "is hidden.".format(b=len(better), t=len(rows), w=len(worse)))

        aggregated = evaluation["aggregated"]
        st.write("")
        columns = st.columns(2, gap="medium")
        specs = [("average_waiting_time", "Average waiting time",
                  "Seconds per vehicle"),
                 ("average_queue_length", "Average queue length", "Vehicles"),
                 ("throughput", "Throughput", "Vehicles completed"),
                 ("max_queue_length", "Maximum queue length", "Vehicles")]
        for index, (metric, title, unit) in enumerate(specs):
            with columns[index % 2]:
                chart = charts.controller_metric_chart(aggregated, metric,
                                                       title, unit)
                if chart is not None:
                    st.altair_chart(chart, width="stretch")

        series = controller.evaluation_timeseries(scenario)
        if series:
            st.write("")
            chart = charts.queue_by_controller_chart(series)
            if chart is not None:
                st.altair_chart(chart, width="stretch")
                st.caption("One sample per simulated second, recorded through "
                           "TraCI during the two runs of the same seed.")

        st.write("")
        with st.expander("Every individual run (no seed is hidden)"):
            raw = evaluation["raw"]
            if raw is not None:
                st.dataframe(raw, hide_index=True, width="stretch")
            st.caption("Files: " + ", ".join(
                "`{}`".format(p) for p in evaluation.get("paths", {}).values()))
    else:
        st.info(evaluation["reason"] + "\n\nRun the experiment on the command "
                "line — it runs both controllers on three seeds and writes "
                "the CSVs this page reads.")
        st.code(evaluation["command"], language="bash")

    # --- the live pair, from runs started in this dashboard -------------
    st.markdown('<hr class="is-rule">', unsafe_allow_html=True)
    components.section(
        "Runs started from this dashboard",
        "The last fixed-time run and the last DQN run for this scenario, "
        "from results/data/.")

    live = controller.controller_comparison(scenario)
    if not live["available"]:
        st.info(live["reason"])
        return
    if not live["comparable"]:
        st.warning(live["reason"])

    columns = st.columns(2, gap="large")
    for column, (run, label) in zip(columns, [(live["fixed"], "Fixed-Time"),
                                              (live["dqn"], "DQN")]):
        with column:
            st.markdown("**{}**".format(label))
            components.controller_badge(run.mode,
                                        str(run.metric("model", "") or ""))
            st.caption("{} · {:.0f} simulated seconds · {}".format(
                run.source, float(run.metric("simulation_time") or 0),
                os.path.relpath(run.csv_path, PROJECT_ROOT)))
    st.write("")
    components.comparison_table(live["rows"])


def page_about(controller, selection, env) -> None:
    components.section("About IntelliSignal")
    st.markdown(
        "**IntelliSignal** studies whether a traffic signal that decides from "
        "live traffic conditions can outperform a conventional pre-timed "
        "signal. The project is built in stages, and this dashboard is the "
        "presentation layer for what has actually been built.")

    st.markdown("### Implementation status")
    st.markdown(
        "| Stage | Component | Status |\n"
        "|---|---|---|\n"
        "| 1 | SUMO four-way intersection, fixed-time signal, two scenarios, "
        "TraCI metrics, CSV results, graphs | **Implemented** |\n"
        "| 2 | Gymnasium reinforcement-learning environment — state, action, "
        "reward, `reset()` / `step()` | **Implemented** |\n"
        "| — | This Streamlit dashboard | **Implemented** |\n"
        "| 3 | DQN agent, replay buffer, target network, training loop, "
        "evaluation, Fixed-Time vs DQN comparison | **Implemented** |\n"
        "| — | CCTV/YOLO detection, real hardware, multiple intersections | "
        "**Future scope** |\n")

    trained = env.get("trained_models") or []
    if trained:
        st.info(
            "A DQN is implemented and trained for: **{}**. Every DQN number "
            "in this dashboard comes from a SUMO run driven by a trained "
            "network — results are kept in their own files and are never "
            "mixed with the fixed-time baseline.".format(
                ", ".join(scenario_label(s) for s in trained)))
    else:
        st.warning(
            "The DQN is implemented, but no trained model is present in "
            "`models/`, so the DQN control mode is disabled. Train one with "
            "`python training/train_dqn.py --scenario moderate --episodes 20 "
            "--seed 42`. The dashboard will not produce DQN numbers without "
            "a trained network.")

    st.markdown("### Architecture")
    st.code(
        "                USER\n"
        "                 |\n"
        "                 v\n"
        "          STREAMLIT DASHBOARD          dashboard/\n"
        "                 |\n"
        "                 v\n"
        "        SIMULATION CONTROLLER          simulation/controller.py\n"
        "                 |\n"
        "        +--------+--------+\n"
        "        |                 |\n"
        "        v                 v\n"
        " Fixed-Time           DQN Agent        agents/\n"
        " Controller          (Stage 3)\n"
        "        |                 |\n"
        "        +--------+--------+\n"
        "                 v\n"
        "                TraCI\n"
        "                 v\n"
        "                SUMO                   sumo/\n"
        "                 v\n"
        "         Traffic Simulation\n"
        "                 v\n"
        "             Metrics                   utils/metrics.py\n"
        "                 v\n"
        "              Results                  results/\n"
        "                 v\n"
        "          Streamlit Charts             dashboard/charts.py",
        language="text")

    st.markdown("### Fixed-Time Baseline")
    scenario = selection["scenario"]
    try:
        summary = controller.scenario_summary(scenario)
        agent = controller.agent_for_mode("fixed_time", scenario)
    except SimulationError as error:
        components.show_error(error)
        return

    st.markdown(
        "Signal timing is predetermined and never reacts to traffic:\n\n"
        "`NS Green → Yellow → EW Green → Yellow`\n")
    columns = st.columns(4)
    for column, (label, value, unit) in zip(columns, [
            ("NS Green", summary["ns_green"], "sec"),
            ("EW Green", summary["ew_green"], "sec"),
            ("Yellow", summary["yellow"], "sec"),
            ("Cycle", summary["cycle_time"], "sec")]):
        with column:
            components.kpi_card(label, fmt(value, 0), " " + unit)

    chart = charts.signal_plan_chart(agent.timeline())
    if chart is not None:
        st.altair_chart(chart, width="stretch")
    st.caption("Values read from {} — the dashboard does not hard-code any "
               "signal timing.".format(
                   os.path.relpath(summary["config_path"], PROJECT_ROOT)))

    st.markdown("### DQN Adaptive Controller")
    st.markdown(
        "A Deep Q-Network attached to the reinforcement-learning environment "
        "in `rl/`. It reads the 10-value observation — the four approach "
        "queues, the four approach waiting times, the current phase and how "
        "long that phase has been green — and produces two Q-values, one for "
        "holding the green and one for switching. It takes the larger, and "
        "the existing safety controller turns that into a legal signal "
        "change with the Stage 1 yellow clearance.\n\n"
        "It learns from the same congestion reward Stage 2 defined, using a "
        "replay buffer and a separate target network. The **DQN Model** page "
        "shows what has been trained; the **Comparison** page shows how it "
        "measures against the fixed-time plan above.")

    st.markdown("### Environment")
    columns = st.columns(2)
    with columns[0]:
        st.markdown(
            "- Python: `{}`\n- SUMO: `{}`\n- TraCI: `{}`".format(
                env["python_version"], env["sumo_version"] or "not found",
                env["traci_version"] or "not found"))
    with columns[1]:
        st.markdown(
            "- Gymnasium: `{}`\n- SUMO-RL: `{}`\n- PyTorch: `{}`\n"
            "- SUMO_HOME: `{}`".format(
                "available" if env["gymnasium"] else "not installed",
                "available" if env["sumo_rl"] else "not installed",
                env.get("torch_version") or ("available" if env.get("torch")
                                             else "not installed"),
                env["sumo_home"] or "not set"))

    st.markdown("### Run it yourself")
    st.code("streamlit run dashboard.py\n"
            "python simulation/run_scenario.py all\n"
            "python rl/run_environment.py --scenario moderate\n"
            "python training/train_dqn.py --scenario moderate --episodes 20 --seed 42\n"
            "python evaluation/evaluate_dqn.py --scenario moderate --seeds 42 43 44\n"
            "python evaluation/compare_controllers.py --scenario moderate --seeds 42 43 44\n"
            "python -m pytest -q", language="bash")


# ==========================================================================
def render_comparison(controller) -> None:
    comparison = load_comparison(controller)
    if comparison.empty:
        st.info("No results to compare yet.")
        return

    # A comparison is only fair when both runs cover the same simulated time.
    durations = comparison["simulation_time"].dropna().unique()
    if len(durations) > 1:
        st.warning(
            "These scenarios were run for different lengths of time ({}). "
            "Re-run them with the same duration before comparing the "
            "numbers.".format(
                ", ".join("{:.0f} s".format(d) for d in sorted(durations))))

    display = comparison.copy()
    display["Scenario"] = display["scenario"].map(scenario_label)
    table = display[["Scenario", "vehicles", "average_waiting_time",
                     "average_queue_length", "throughput", "simulation_time",
                     "source"]].rename(columns={
        "vehicles": "Vehicles",
        "average_waiting_time": "Avg waiting time (s)",
        "average_queue_length": "Avg queue (vehicles)",
        "throughput": "Throughput",
        "simulation_time": "Sim time (s)",
        "source": "Source"})
    st.dataframe(table, hide_index=True, width="stretch")

    columns = st.columns(3, gap="medium")
    for column, chart in zip(columns, (charts.waiting_time_chart(comparison),
                                       charts.queue_length_chart(comparison),
                                       charts.throughput_chart(comparison))):
        with column:
            if chart is not None:
                st.altair_chart(chart, width="stretch")


# ==========================================================================
def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    state.init_state()
    controller = get_controller()

    try:
        env = cached_environment_status(controller)
    except Exception as error:          # never crash on a detection failure
        env = {"indicators": [], "ready": False, "errors": [str(error)],
               "sumo_version": "", "sumo_home": "", "sumo_binary": "",
               "traci_version": "", "python_version": "", "gymnasium": False,
               "sumo_rl": False, "torch": False, "torch_version": "",
               "trained_models": [], "scenarios": []}

    selection = controls.sidebar(controller)

    components.project_header()
    components.status_lights(env["indicators"])

    # --- the Run button is the ONLY thing that starts a simulation -----
    if selection["run_clicked"] and selection["scenario"]:
        try:
            label = (selection["mode_object"].label
                     if selection.get("mode_object") else selection["mode"])
            with st.spinner("Running {} traffic in SUMO for {} simulated "
                            "seconds under {} control…".format(
                                selection["scenario"], selection["duration"],
                                label)):
                run = controller.run(selection["scenario"],
                                     mode=selection["mode"],
                                     duration=float(selection["duration"]),
                                     use_gui=selection["gui"],
                                     seed=selection.get("seed"),
                                     model_path=controller.model_path(
                                         selection["scenario"],
                                         selection.get("model_name"))
                                     if selection["mode"] == "dqn_adaptive"
                                     else None)
            state.record_run(run)
            st.success(
                "{label} finished in {wall:.1f} s of real time — average "
                "waiting time {wait} s, average queue {queue} vehicles, "
                "throughput {thru} vehicles.".format(
                    label=label, wall=run.duration_seconds or 0.0,
                    wait=fmt(run.metric("average_waiting_time"), 2),
                    queue=fmt(run.metric("average_queue_length"), 2),
                    thru=fmt(run.metric("throughput"), 0)))
        except ControlModeUnavailableError as error:
            state.record_error(str(error))
            components.show_error(error)
        except SimulationError as error:
            state.record_error(str(error))
            components.show_error(error)
        except Exception as error:      # last line of defence: no traceback
            message = "Unexpected problem while running the simulation: {}".format(error)
            state.record_error(message)
            st.error(message)
            st.caption("The full error is in the terminal running Streamlit.")

    page = selection["page"]
    if not selection["scenario"]:
        st.error("No scenarios were found in config/. The dashboard needs at "
                 "least one scenario YAML file to work.")
        return

    if page == "Dashboard":
        page_dashboard(controller, selection, env)
    elif page == "Simulation":
        page_simulation(controller, selection, env)
    elif page == "Results":
        page_results(controller, selection, env)
    elif page == "DQN Model":
        page_model(controller, selection, env)
    elif page == "Comparison":
        page_comparison(controller, selection, env)
    else:
        page_about(controller, selection, env)


if __name__ == "__main__":
    main()
