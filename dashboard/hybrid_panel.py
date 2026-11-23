"""
IntelliSignal - Dashboard
dashboard/hybrid_panel.py

Showing what Hybrid is doing, and what it decided.

Two rules this panel exists to keep:

  * **Say which plan is actually in force.** While the search is running the
    signal is on the configured plan; afterwards it is on the tuned plan, or
    on the DQN. Showing a decision before one exists would be the dashboard
    inventing a result.
  * **Report a measured loss only when a completed comparison supports it.**
    If tuning or validation did not finish, the panel says so instead of
    quoting a number from a comparison that never happened.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st


# ---------------------------------------------------------------------------
def progress(status: Dict[str, Any]) -> None:
    """The live state of a running search."""
    limit = max(float(status.get("limit_s") or 1.0), 1.0)
    elapsed = float(status.get("elapsed_s") or 0.0)
    remaining = float(status.get("remaining_s") or 0.0)

    st.progress(min(elapsed / limit, 1.0),
                text="{}  ·  {:.0f} s of {:.0f} s used".format(
                    status.get("phase", "working"), elapsed, limit))
    columns = st.columns([1.2, 1.2, 2.6])
    columns[0].metric("Time left", "{:.0f} s".format(remaining))
    columns[1].metric("State", str(status.get("state", "")).title())
    with columns[2]:
        st.caption("Wall-clock time, not simulated time. A 1800 s "
                   "simulation runs in about 15 s.")

    messages = status.get("messages") or []
    if messages:
        with st.expander("What the search is doing", expanded=False):
            st.code("\n".join(messages), language=None)


# ---------------------------------------------------------------------------
def decision(payload: Optional[Dict[str, Any]]) -> None:
    """The outcome, led by the one sentence that answers the question."""
    if not payload:
        st.info("No Hybrid decision yet for this traffic.")
        return

    selected_dqn = payload.get("selected") == "dqn"
    headline = payload.get("headline") or ""
    (st.success if selected_dqn else st.info)("**{}**".format(headline))

    plan = payload.get("fixed_plan") or {}
    columns = st.columns([1.3, 1.3, 1.3, 1.3])
    columns[0].metric("In force",
                      "DQN" if selected_dqn else "Fixed-time")
    columns[1].metric("Tuned plan", "{:.0f} / {:.0f} s".format(
        plan.get("ns_green", 0), plan.get("ew_green", 0)),
        help="North-south green / east-west green, measured on this traffic.")
    columns[2].metric("Checks completed", str(len(payload.get("checks") or [])))
    columns[3].metric("Search time", "{:.0f} s".format(
        payload.get("elapsed_s") or 0))

    st.caption("Stopped because {}.".format(
        payload.get("stopped_because") or "the search finished"))

    if not payload.get("tuning_complete", True):
        st.warning("Tuning did not finish inside the time limit. The plan "
                   "above is the best one that was actually measured, not "
                   "the best possible plan.")
    if not payload.get("validation_complete", True):
        st.warning("No validation check completed, so no comparison between "
                   "the two controllers is being reported. Fixed time kept "
                   "the signal by default.")

    st.markdown("**Why**")
    st.write(payload.get("reason") or "")

    checks = payload.get("checks") or []
    if checks:
        with st.expander("Validation checks ({})".format(len(checks))):
            import pandas as pd
            st.dataframe(pd.DataFrame([{
                "Check": check["index"],
                "After episode": check["episode"],
                "DQN waiting (s)": round(check["dqn_waiting"], 2),
                "Fixed waiting (s)": round(check["fixed_waiting"], 2),
                "DQN completed": round(check["dqn_throughput"], 1),
                "Fixed completed": round(check["fixed_throughput"], 1),
                "DQN waiting to enter": round(check["dqn_waiting_to_enter"], 1),
                "Fixed waiting to enter": round(
                    check["fixed_waiting_to_enter"], 1),
                "Result": "qualifies" if check["qualified"] else "no",
            } for check in checks]), width="stretch", hide_index=True)
            st.caption(
                "Each check froze the candidate and ran both controllers on "
                "the same vehicles and the same seeds ({}). The DQN is "
                "selected only with lower waiting, at least equal "
                "throughput and no increase in vehicles waiting to enter."
                .format(", ".join(str(s) for s in checks[0].get("seeds", []))))

    notes = payload.get("notes") or []
    if notes:
        with st.expander("Notes"):
            for note in notes:
                st.write("· {}".format(note))

    if payload.get("model_path"):
        st.caption("Qualifying checkpoint: `{}` ({})".format(
            payload["model_path"], payload.get("model_source", "")))


# ---------------------------------------------------------------------------
def not_started(limits: Dict[str, Any]) -> None:
    """What Hybrid will do, before Run is pressed."""
    st.info(
        "**Hybrid** starts the traffic on a valid fixed-time plan straight "
        "away, then spends up to **{:.0f} s of wall-clock time** tuning that "
        "plan and testing a DQN against it. The DQN takes over only if it "
        "measurably wins; a tie goes to fixed time."
        .format(limits.get("wall_clock_s", 300)))
    st.caption(
        "Limits: {:.0f} s, at most {} extra training episodes, and patience "
        "of {} consecutive failed checks. All three are editable under "
        "Advanced.".format(limits.get("wall_clock_s", 300),
                           limits.get("max_episodes", 30),
                           limits.get("patience", 3)))
