"""
IntelliSignal
dashboard/views/compare.py

Several controllers, one set of vehicles, one table.

The comparison is fair by construction rather than by care: every
controller on this page is run by `simulation/engine.py` on the same
workload, the same seed, the same clearance times and the same timing
rules, and measured by the same two measurements. The only difference
between the rows is the thing being compared.

Hybrid has its own half of the page because it is not another
controller - it is the rule that decides between a tuned fixed plan and
the model, with a stopwatch running.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import streamlit as st

from dashboard import charts, jobs, state, ui
from dashboard.views import common
from simulation.engine import (CONTROLLER_ORDER, CONTROLLERS, compare,
                               load_run)


def _controller_picker(status: Dict[str, Any]) -> List[str]:
    options = list(CONTROLLER_ORDER)
    default = [key for key in state.get("compare_with") if key in options]
    if "fixed_time" not in default:
        default.append("fixed_time")
    if status.get("usable") and "dqn" not in default:
        default.insert(0, "dqn")
    chosen = st.multiselect(
        "Controllers to run", options, default=default,
        format_func=lambda key: CONTROLLERS[key].label,
        key="compare_pick",
        help="Fixed-time is the baseline every other row is measured "
             "against, so it is always included.")
    if "fixed_time" not in chosen:
        chosen = ["fixed_time"] + list(chosen)
    if "dqn" in chosen and not status.get("usable"):
        chosen = [key for key in chosen if key != "dqn"]
        ui.note(status.get("reason", ""), "warn",
                "The trained model is not available, so it is not in the "
                "table")
    state.put("compare_with", [key for key in chosen if key != "fixed_time"])
    return chosen


def _hybrid_section(workload, ctx) -> None:
    """The time-limited search: tune a plan, test the model, keep the winner."""
    from simulation.hybrid import (HybridLimits, configured_plan_for,
                                   load_decision)
    from simulation.hybrid_job import cancel_job, get_job, start_job
    from dashboard import hybrid_panel

    st.markdown("### Hybrid — let the junction decide for itself")
    st.caption(
        "Hybrid does not assume either controller is better. It tunes a "
        "fixed plan on this exact traffic, tests whatever model exists "
        "against it, and keeps whichever wins — a tie goes to fixed time. "
        "It is the setting that cannot produce a flattering result, which "
        "is why it is what a cautious engineer would deploy.")

    job = get_job(workload.workload_id)
    columns = st.columns([1.2, 1.2, 1, 3])
    with columns[0]:
        limit_min = st.number_input("Time limit (minutes)", 1, 30, 5, 1,
                                    key="hybrid_minutes")
    with columns[1]:
        episodes = st.number_input("Training episodes at most", 0, 200, 30, 5,
                                   key="hybrid_episodes")
    with columns[2]:
        patience = st.number_input("Patience", 1, 10, 3, 1,
                                   key="hybrid_patience",
                                   help="Stop after this many checks in a "
                                        "row that do not qualify.")
    with columns[3]:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        if job is not None and job.alive:
            if st.button("Stop the search", key="hybrid_cancel"):
                cancel_job(workload.workload_id)
                st.rerun()
        elif st.button("Let Hybrid decide", type="primary",
                       key="hybrid_start"):
            start_job(workload,
                      HybridLimits(wall_clock_s=float(limit_min) * 60,
                                   max_episodes=int(episodes),
                                   patience=int(patience)).validate(),
                      configured_plan=configured_plan_for(workload),
                      profile=ctx["profile"])
            state.put("hybrid_workload_id", workload.workload_id)
            st.rerun()

    limits = HybridLimits(wall_clock_s=float(limit_min) * 60,
                          max_episodes=int(episodes), patience=int(patience))
    if job is not None:
        hybrid_panel.progress(job.status())
        if job.alive:
            time.sleep(1.0)
            st.rerun()
        if job.decision is not None:
            hybrid_panel.decision(job.decision.to_dict())
            return

    # A stored decision answers the question it was asked: this traffic,
    # these timing rules, this budget. Change any of them and it is not
    # an answer any more, so it is looked up by all three.
    saved = load_decision(workload.workload_id, profile=ctx["profile"],
                          limits=limits)
    if saved is not None:
        ui.note("Restored from a search that finished on {}. Nothing has "
                "run just now.".format(
                    (saved.decided_at or "an earlier date").replace("T", " ")),
                "info")
        hybrid_panel.decision(saved.to_dict())
    elif job is None:
        hybrid_panel.not_started({"wall_clock_s": limits.wall_clock_s,
                                  "max_episodes": limits.max_episodes,
                                  "patience": limits.patience})


# ---------------------------------------------------------------------------
def render(ctx: Dict[str, Any]) -> None:
    workload = ctx["workload"]
    st.markdown("## Compare")
    if workload is None:
        ui.note("Choose a traffic pattern first — the whole point of this "
                "page is that every controller meets the same vehicles.",
                "warn")
        return

    common.traffic_card(workload, compact=True)
    st.write("")
    chosen = _controller_picker(ctx["model_status"])

    key = jobs.job_key(workload.workload_id, chosen, state.get("seed"))
    columns = st.columns([1.4, 1.4, 4])
    with columns[0]:
        if st.button("Run them all", type="primary", width="stretch",
                     key="compare_run"):
            jobs.forget_job(key)
            jobs.start_job(workload, chosen, seed=int(state.get("seed")),
                           profile=ctx["profile"], plan=state.plan_override())
            state.put("compare_job_key", key)
            st.rerun()
    with columns[1]:
        st.markdown("&nbsp;", unsafe_allow_html=True)
    with columns[2]:
        ui.footnote(
            "{} runs of {:.0f} simulated minutes each, about {:.0f} seconds "
            "of real time in total.".format(
                len(chosen), workload.duration_s / 60,
                len(chosen) * workload.duration_s / 45))

    job = jobs.get_job(key)
    results = {}
    if job is not None:
        results = dict(job.results)
        if job.alive:
            snapshot = job.status()
            st.progress(
                min(0.05 + 0.92 * snapshot["progress"], 0.98),
                text="{} of {} done — running {} ({:.0f} s elapsed)".format(
                    len(snapshot["finished"]), snapshot["total"],
                    snapshot["current_label"], snapshot["elapsed_s"]))
            time.sleep(1.0)
            st.rerun()
        elif job.state == "failed":
            ui.note(job.error, "bad", "The comparison did not finish")

    for controller in chosen:
        if controller not in results:
            saved = load_run(workload.workload_id, controller,
                             int(state.get("seed")))
            if saved is not None:
                results[controller] = saved

    ready = [results[key_] for key_ in chosen if key_ in results]
    if len(ready) < 2:
        ui.note("Press **Run them all** to measure these controllers on this "
                "traffic.", "info")
        st.divider()
        _hybrid_section(workload, ctx)
        return

    table = compare(ready)
    st.markdown("### Results on this traffic")
    common.comparison_table(table)

    columns = st.columns(2, gap="medium")
    with columns[0]:
        chart = charts.delay_bars(table["rows"])
        if chart is not None:
            st.markdown("**Delay per requested vehicle**")
            st.altair_chart(chart, width="stretch")
    with columns[1]:
        chart = charts.queue_compare(
            {result.label: result.series for result in ready})
        if chart is not None:
            st.markdown("**Vehicles halted, through the run**")
            st.altair_chart(chart, width="stretch")

    worst = [row for row in table["rows"] if not row["trustworthy"]]
    if worst:
        ui.note("{} of these runs did not pass every check. Open the run "
                "below before quoting its number.".format(len(worst)), "bad")

    st.divider()
    tabs = st.tabs([CONTROLLERS[r.controller].label for r in ready])
    for tab, result in zip(tabs, ready):
        with tab:
            common.run_tiles(result, results.get("fixed_time"))
            common.audit_panel(result)
            common.timeline_block(result)
            common.decisions_block(result)

    st.divider()
    _hybrid_section(workload, ctx)
