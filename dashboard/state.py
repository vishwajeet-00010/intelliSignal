"""
IntelliSignal - Dashboard
dashboard/state.py

Streamlit session state.

Streamlit re-runs the whole script on every interaction, so anything that must
survive a click lives here. The rule that matters for this project: **a
simulation runs only when the Run button is pressed**, never on a page
refresh. The last run is kept in session state and every page reads it from
there.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

PAGES: List[str] = ["Dashboard", "Custom Traffic", "Simulation",
                    "Comparison", "Results", "DQN Model", "About"]

DEFAULTS: Dict[str, Any] = {
    "page": "Dashboard",
    "scenario": "indian_arterial",
    "control_mode": "hybrid",
    "duration": 1800,
    "seed": 42,                # SUMO seed for a DQN run
    "model_name": None,        # which checkpoint the DQN run should use
    "last_run": None,          # SimulationRun produced by the Run button
    "last_error": None,        # str, shown once then cleared
    "run_history": [],         # list of dicts, newest first
    "signal_time": 0,          # position of the signal time slider

    # --- custom traffic ------------------------------------------------
    # The spec is what was typed; the workload is the resolved preview.
    # They are kept apart so that navigating away and back does not
    # re-resolve and quietly change the generated values.
    "custom_spec": None,          # CustomSpec.to_dict()
    "custom_workload": None,      # ResolvedWorkload.to_dict()
    "custom_issues": [],          # [{where, message, severity}]
    "custom_regen_token": 0,      # bumped only by "Regenerate"
    "auto_hybrid_reason": None,   # why the DQN mode handed over to Hybrid

    # --- hybrid ----------------------------------------------------------
    "hybrid_wall_clock_s": 300,
    "hybrid_max_episodes": 30,
    "hybrid_patience": 3,
    "hybrid_workload_id": None,   # the job the page is watching

    # --- operational timing profile --------------------------------------
    # Which green limits every controller is held to. It changes what is
    # reachable, so it is recorded with every run and shown on the pages
    # rather than left implicit.
    "timing_profile": "standard",
}


def init_state() -> None:
    """Put the defaults in place. Safe to call on every re-run."""
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = (list(value) if isinstance(value, list)
                                     else value)


def get(key: str, default: Any = None) -> Any:
    return st.session_state.get(key, DEFAULTS.get(key, default))


def set_value(key: str, value: Any) -> None:
    st.session_state[key] = value


# --------------------------------------------------------------------------
def record_run(run) -> None:
    """Remember a finished simulation."""
    st.session_state["last_run"] = run
    st.session_state["last_error"] = None
    history = st.session_state.get("run_history", [])
    history.insert(0, {
        "scenario": run.scenario,
        "mode": run.mode,
        "controller": run.metrics.get("controller", run.mode),
        "finished_at": run.finished_at,
        "wall_clock": run.duration_seconds,
        "average_waiting_time": run.metric("average_waiting_time"),
        "average_queue_length": run.metric("average_queue_length"),
        "throughput": run.metric("throughput"),
        "simulation_time": run.metric("simulation_time"),
    })
    st.session_state["run_history"] = history[:10]


def record_error(message: str) -> None:
    st.session_state["last_error"] = message


def current_run(controller, scenario: Optional[str] = None,
                mode: Optional[str] = None):
    """The run to display for a scenario and control mode.

    Prefers a simulation started from the dashboard in this session; otherwise
    falls back to whatever results are on disk, so the pages have real numbers
    to show before anything has been run.

    The MODE is part of the identity of a run. A fixed-time run is never
    returned when DQN results were asked for: `load_results` has no fallback
    for the DQN, so the caller gets None and says "no DQN run yet" rather
    than showing baseline numbers under a DQN heading.
    """
    scenario = scenario or get("scenario")
    mode = mode or get("control_mode")
    last = st.session_state.get("last_run")
    if (last is not None and last.scenario == scenario
            and getattr(last, "mode", None) == mode):
        return last
    try:
        return controller.load_results(scenario, mode)
    except Exception:
        return None
