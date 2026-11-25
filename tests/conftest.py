"""
Shared test setup.

Two things live here that every test file used to do for itself, or not
do at all.

**The import path.** Fourteen files each began with the same four lines
inserting the repository root into `sys.path`. Doing it once here means
a new test file starts with its imports.

**Leftover background work.** The dashboard runs simulations on threads,
and a test that starts one and then fails leaves it running into the
next test - where it competes for cores, writes into the same scratch
directories and makes an unrelated failure look like a flake. Every test
session now ends by cancelling whatever is still going.
"""

from __future__ import annotations

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "sumo: starts a real SUMO simulation, so it is slow")
    config.addinivalue_line(
        "markers", "legacy: covers Stage 1-3, which the current product "
                   "does not use but still ships as reproducible evidence")


@pytest.fixture(autouse=True, scope="session")
def _no_stray_background_work():
    """Leave no simulation running behind us."""
    yield
    try:
        from simulation.hybrid_job import active_jobs, cancel_job, get_job
        for workload_id in list(active_jobs()):
            cancel_job(workload_id)
            job = get_job(workload_id)
            if job is not None and job._thread is not None:
                job._thread.join(timeout=20)
    except Exception:
        pass
    try:
        from dashboard import jobs
        for key in list(jobs.active_jobs()):
            job = jobs.get_job(key)
            if job is not None:
                job.cancel()
                if job._thread is not None:
                    job._thread.join(timeout=20)
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _committed_figures_are_not_touched():
    """The figures in the report must survive a test run.

    `results/plots` holds the five graphs the Stage 1 report uses. A test
    that regenerates them in place passes while quietly rewriting
    committed evidence, so this fails the test that does it rather than
    leaving it to be noticed in `git status`.
    """
    plots = os.path.join(PROJECT_ROOT, "results", "plots")
    before = {}
    for folder, _dirs, names in os.walk(plots):
        for name in names:
            path = os.path.join(folder, name)
            before[path] = os.path.getmtime(path)
    yield
    changed = [os.path.relpath(path, plots).replace("\\", "/")
               for path, stamp in before.items()
               if os.path.isfile(path) and os.path.getmtime(path) != stamp]
    assert not changed, (
        "this test rewrote committed figures in results/plots: {}. Pass an "
        "out_dir so it writes into tmp_path instead.".format(
            ", ".join(sorted(changed))))
