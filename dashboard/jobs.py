"""
IntelliSignal
dashboard/jobs.py

Running simulations without freezing the screen.

A nine-hundred-second run of this junction takes the better part of
half a minute, and a comparison runs several. Doing that inside the
Streamlit callback leaves the page blank and the user guessing, so every
run goes onto a worker thread and the page polls it. The job owns the
list of controllers to run, which one it is on, and the results it has
finished; the page only ever reads a snapshot.

Jobs are keyed by traffic and by the set of controllers, so going to
another page and coming back finds the same job still running rather
than starting a second one.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from simulation.engine import CONTROLLERS, RunResult, run_controller, save_run

#: Every live job in this server process, by key.
_JOBS: Dict[str, "RunJob"] = {}
_LOCK = threading.Lock()


@dataclass
class RunJob:
    """One or more controllers, run on one traffic pattern, on a thread."""

    key: str
    workload: Any
    controllers: List[str]
    seed: int
    profile: str
    plan: Optional[Dict[str, float]] = None
    record: bool = True
    save: bool = True

    state: str = "waiting"             # waiting | running | done | failed
    current: str = ""
    finished: List[str] = field(default_factory=list)
    results: Dict[str, RunResult] = field(default_factory=dict)
    error: str = ""
    started_at: float = 0.0
    ended_at: float = 0.0
    _thread: Optional[threading.Thread] = None
    _cancel: threading.Event = field(default_factory=threading.Event)

    # ------------------------------------------------------------------
    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def elapsed_s(self) -> float:
        if not self.started_at:
            return 0.0
        return (self.ended_at or time.time()) - self.started_at

    @property
    def progress(self) -> float:
        if not self.controllers:
            return 1.0
        return len(self.finished) / len(self.controllers)

    def start(self) -> "RunJob":
        self.started_at = time.time()
        self.state = "running"
        self._thread = threading.Thread(
            target=self._run, daemon=True,
            name="run-{}".format(self.key[:16]))
        self._thread.start()
        return self

    def cancel(self) -> None:
        """Stop before the next controller. A run in flight finishes."""
        self._cancel.set()

    # ------------------------------------------------------------------
    def _run(self) -> None:
        try:
            for controller in self.controllers:
                if self._cancel.is_set():
                    self.state = "done"
                    self.current = ""
                    break
                self.current = controller
                result = run_controller(
                    self.workload, controller, seed=self.seed,
                    profile=self.profile,
                    plan=self.plan if controller == "fixed_time" else None,
                    record=self.record)
                self.results[controller] = result
                self.finished.append(controller)
                if self.save:
                    try:
                        save_run(result)
                    except OSError:
                        pass          # a saved copy is a convenience
            else:
                self.state = "done"
            self.current = ""
        except BaseException as exc:                  # noqa: BLE001
            self.state = "failed"
            self.current = ""
            self.error = "{}: {}".format(type(exc).__name__, exc)
            self.detail = traceback.format_exc()
        finally:
            self.ended_at = time.time()

    # ------------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        """A snapshot the page can render without blocking."""
        done = list(self.finished)
        return {
            "key": self.key, "state": self.state, "alive": self.alive,
            "current": self.current,
            "current_label": (CONTROLLERS[self.current].label
                              if self.current in CONTROLLERS else ""),
            "finished": done, "total": len(self.controllers),
            "progress": self.progress, "elapsed_s": round(self.elapsed_s, 1),
            "error": self.error,
            "workload_id": getattr(self.workload, "workload_id", ""),
            "seed": self.seed,
        }


# ---------------------------------------------------------------------------
def job_key(workload_id: str, controllers: Sequence[str], seed: int) -> str:
    return "{}|{}|{}".format(workload_id, ",".join(sorted(controllers)), seed)


def start_job(workload, controllers: Sequence[str], seed: int,
              profile: str, plan: Optional[Dict[str, float]] = None,
              record: bool = True, save: bool = True) -> RunJob:
    """Start a job, or hand back the one already doing this exact work."""
    key = job_key(getattr(workload, "workload_id", ""), controllers, seed)
    with _LOCK:
        existing = _JOBS.get(key)
        if existing is not None and existing.alive:
            return existing
        job = RunJob(key=key, workload=workload,
                     controllers=[c for c in controllers], seed=int(seed),
                     profile=profile, plan=plan, record=record, save=save)
        _JOBS[key] = job
    return job.start()


def get_job(key: str) -> Optional[RunJob]:
    return _JOBS.get(key)


def forget_job(key: str) -> None:
    with _LOCK:
        _JOBS.pop(key, None)


def active_jobs() -> List[str]:
    return [key for key, job in list(_JOBS.items()) if job.alive]
