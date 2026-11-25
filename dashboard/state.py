"""
IntelliSignal
dashboard/state.py

What the dashboard remembers between clicks.

Streamlit re-runs the whole script on every interaction, so anything that
must survive a click lives in `st.session_state`. Keeping the keys and
their defaults in one place means a page cannot invent a key that another
page then fails to find, and `reset_traffic` can clear everything that
belonged to the previous traffic in one call - which is what stops a run
measured on one pattern being shown beside the name of another.

The workload itself is stored as its dictionary rather than the object,
because Streamlit pickles session state between reruns and a plain dict
reloads identically every time.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st

from simulation.engine import OPERATING_PROFILE

#: Every key the dashboard uses, with the value it starts at.
DEFAULTS: Dict[str, Any] = {
    "page": "Control room",
    # --- the traffic -------------------------------------------------
    "traffic_kind": "preset",          # preset | custom
    "preset": "typical",
    "duration_s": 900.0,
    "seed": 6101,
    "custom_workload": None,           # the resolved workload, as a dict
    "custom_spec": None,               # the form's own state, as a dict
    "custom_issues": [],               # what the resolver said about it
    "custom_regen_token": 0,           # bumped to redraw the generated values
    # --- the controller ----------------------------------------------
    "controller": "dqn",
    "plan_ns": None,                   # None = use the configured plan
    "plan_ew": None,
    "compare_with": ["fixed_time", "actuated"],
    # --- what is running or has run ----------------------------------
    "job_key": "",
    "compare_job_key": "",
    "hybrid_workload_id": "",
    "last_error": "",
    "profile": OPERATING_PROFILE,
}


def init() -> None:
    """Fill in anything missing. Safe to call on every rerun."""
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = (list(value) if isinstance(value, list)
                                     else value)


def get(key: str, default: Any = None) -> Any:
    return st.session_state.get(key, DEFAULTS.get(key, default))


def put(key: str, value: Any) -> None:
    st.session_state[key] = value


def reset_traffic() -> None:
    """Forget everything that belonged to the previous traffic.

    A run is only ever shown next to the traffic it was measured on, so
    changing the traffic must clear the jobs as well as the workload.
    """
    for key in ("job_key", "compare_job_key", "hybrid_workload_id",
                "last_error"):
        st.session_state[key] = DEFAULTS[key]


# ---------------------------------------------------------------------------
def current_workload():
    """The traffic the whole dashboard is working on right now.

    One function, used by every page, so two pages can never disagree
    about what is on the road.
    """
    from custom_traffic.workload import ResolvedWorkload
    from simulation.engine import preset_workload

    if get("traffic_kind") == "custom":
        data = get("custom_workload")
        if not data:
            return None
        try:
            return ResolvedWorkload.from_dict(data)
        except (KeyError, TypeError, ValueError):
            return None
    return preset_workload(get("preset"), float(get("duration_s")))


def set_custom_workload(workload) -> None:
    put("custom_workload", workload.to_dict() if workload is not None else None)
    reset_traffic()


def traffic_label() -> str:
    """How the current traffic is named on screen."""
    from simulation.engine import PRESETS

    if get("traffic_kind") == "custom":
        data = get("custom_workload")
        return "Custom traffic" if data else "Custom traffic (not defined yet)"
    preset = PRESETS.get(get("preset"))
    return preset.label if preset else str(get("preset"))


def plan_override() -> Optional[Dict[str, float]]:
    """The fixed plan the user chose, or None to use the configured one."""
    ns, ew = get("plan_ns"), get("plan_ew")
    if ns is None or ew is None:
        return None
    return {"ns": float(ns), "ew": float(ew), "source": "chosen here"}
