"""
IntelliSignal
dashboard/views/control_room.py

The page the demonstration starts on: choose a controller, run it, and
watch the junction it produced.

Everything on this page comes from one run of `simulation/engine.py`, so
the animation, the numbers, the decisions and the audit are all the same
simulation. There is no separate "live view" that re-creates the traffic
for the picture.
"""

from __future__ import annotations

import time
from typing import Any, Dict

import streamlit as st

from dashboard import jobs, junction, state, ui
from dashboard.views import common
from simulation.engine import (CONTROLLER_ORDER, CONTROLLERS, load_run,
                               reference_plan)


def _controller_choice(status: Dict[str, Any]) -> str:
    """Pick the controller, and say what each one is while choosing."""
    options = list(CONTROLLER_ORDER)
    labels = {key: CONTROLLERS[key].label for key in options}
    if not status.get("usable"):
        labels["dqn"] = CONTROLLERS["dqn"].label + " — unavailable"

    current = state.get("controller")
    if current not in options:
        current = "fixed_time"
    choice = st.radio(
        "Controller", options, index=options.index(current),
        format_func=lambda key: labels[key], horizontal=True,
        key="controller_choice",
        help="Each one works the same junction on the same traffic.")
    state.put("controller", choice)
    st.caption(CONTROLLERS[choice].detail)
    return choice


def _plan_controls(workload) -> None:
    """The two greens, when a fixed plan is what is being run."""
    from simulation.engine import configured_plan

    profile_plan = configured_plan(workload)
    best = reference_plan()
    menu = [float(d) for d in __import__(
        "rl.timing_profile", fromlist=["get_profile"]
    ).get_profile(state.get("profile")).durations]

    ns = state.get("plan_ns") or profile_plan["ns"]
    ew = state.get("plan_ew") or profile_plan["ew"]
    columns = st.columns([1, 1, 2])
    with columns[0]:
        ns = st.select_slider("North-south green", menu,
                              value=min(menu, key=lambda d: abs(d - ns)),
                              key="plan_ns_slider")
    with columns[1]:
        ew = st.select_slider("East-west green", menu,
                              value=min(menu, key=lambda d: abs(d - ew)),
                              key="plan_ew_slider")
    state.put("plan_ns", ns)
    state.put("plan_ew", ew)
    with columns[2]:
        cycle = ns + ew + 2 * (float(workload.signal.get("yellow", 3.0))
                               + float(workload.signal.get("all_red", 2.0)))
        st.markdown("&nbsp;", unsafe_allow_html=True)
        ui.kv("Cycle:", "{:.0f} s  ({:g} + {:g} green, 2 x 5 s clearance)"
              .format(cycle, ns, ew))
        if best and (best["ns"], best["ew"]) != (ns, ew):
            if st.button("Use the best plan found when the model was "
                         "selected ({:g} / {:g})".format(best["ns"],
                                                         best["ew"]),
                         key="use_reference_plan"):
                state.put("plan_ns", best["ns"])
                state.put("plan_ew", best["ew"])
                st.rerun()


# ---------------------------------------------------------------------------
def render(ctx: Dict[str, Any]) -> None:
    workload = ctx["workload"]
    status = ctx["model_status"]

    st.markdown("## Control room")
    if workload is None:
        ui.note("Choose a traffic pattern in the sidebar, or build your own "
                "on the Traffic page, and the junction will run.", "warn")
        junction.still(height=430, label="Waiting for traffic")
        return

    left, right = st.columns([3, 2], gap="medium")
    with left:
        controller = _controller_choice(status)
    with right:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        run_clicked = st.button(
            "Run the junction", type="primary", width="stretch",
            key="run_button",
            help="Runs {:.0f} minutes of this traffic through SUMO and "
                 "measures it.".format(workload.duration_s / 60))

    if controller == "fixed_time":
        _plan_controls(workload)

    if controller == "dqn" and not status.get("usable"):
        ui.note(status.get("reason", "No validated model is available."),
                "warn", "The trained model cannot take the signal")
        st.caption("The Model page shows which check failed. Hybrid, on the "
                   "Compare page, will tune a fixed plan and test any model "
                   "against it rather than assume either.")

    # --- start, or pick up, the job -----------------------------------
    key = jobs.job_key(workload.workload_id, [controller], state.get("seed"))
    if run_clicked:
        if controller == "dqn" and not status.get("usable"):
            st.stop()
        jobs.forget_job(key)
        jobs.start_job(workload, [controller], seed=int(state.get("seed")),
                       profile=state.get("profile"),
                       plan=state.plan_override())
        state.put("job_key", key)

    job = jobs.get_job(key)
    result = None
    if job is not None and job.alive:
        snapshot = job.status()
        st.progress(
            min(0.08 + 0.9 * snapshot["progress"], 0.98),
            text="Running {} through SUMO — {:.0f} s so far".format(
                snapshot["current_label"] or "the junction",
                snapshot["elapsed_s"]))
        st.caption("Nine hundred seconds of traffic take about twenty "
                   "seconds to simulate and measure.")
        time.sleep(0.8)
        st.rerun()
    elif job is not None and job.state == "failed":
        ui.note(job.error, "bad", "The run did not finish")
    elif job is not None and controller in job.results:
        result = job.results[controller]
    else:
        result = load_run(workload.workload_id, controller,
                          int(state.get("seed")))
        if result is not None:
            st.caption("Showing the last run of this controller on this "
                       "traffic. Press **Run the junction** to run it again.")

    if result is None:
        ui.note("Press **Run the junction** to simulate this traffic under "
                "the controller above. Nothing on this page is pre-computed.",
                "info")
        junction.still(height=430, label="{} — not run yet".format(
            CONTROLLERS[controller].label))
        return

    # --- the result ----------------------------------------------------
    baseline = load_run(workload.workload_id, "fixed_time",
                        int(state.get("seed")))
    st.markdown("### {} {}".format(result.label, ""), unsafe_allow_html=False)
    st.markdown(ui.verdict_badge(result), unsafe_allow_html=True)
    common.run_tiles(result, baseline if controller != "fixed_time" else None)
    st.write("")
    common.replay_block(result, height=620)
    st.write("")
    columns = st.columns([3, 2], gap="medium")
    with columns[0]:
        common.timeline_block(result)
    with columns[1]:
        st.markdown("**How long the greens were**")
        common.green_block(result)
        common.audit_panel(result)
    common.decisions_block(result)
    common.metric_glossary()
    _saved_runs(workload)


def _saved_runs(workload) -> None:
    """Every run this dashboard has measured and kept, newest first."""
    import pandas as pd
    from simulation.engine import CONTROLLERS, saved_runs

    rows = saved_runs()
    if not rows:
        return
    with st.expander("Runs this dashboard has measured ({})".format(
            len(rows))):
        st.dataframe(pd.DataFrame([{
            "When": row["created_at"].replace("T", " "),
            "Traffic": row["workload_name"],
            "Controller": (CONTROLLERS[row["controller"]].label
                           if row["controller"] in CONTROLLERS
                           else row["controller"]),
            "Delay per vehicle (s)": round(row["delay"], 2),
            "Waiting (s)": round(row["waiting"], 2),
            "Seed": row["seed"],
            "This traffic": ("yes" if row["workload_id"]
                             == workload.workload_id else ""),
        } for row in rows]), hide_index=True, width="stretch")
        ui.footnote(
            "A run is reloaded automatically when the traffic, controller "
            "and seed all match, which is why nothing here can be shown "
            "beside traffic it was not measured on.")
