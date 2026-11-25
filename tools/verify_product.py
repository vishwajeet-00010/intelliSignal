"""
IntelliSignal
tools/verify_product.py

End-to-end checks against the real simulator, outside the test suite.

    python tools/verify_product.py                     # all of them
    python tools/verify_product.py --quick             # the cheap half

The test suite proves the parts behave. This proves the product does:
that every controller the dashboard offers really obeys the timing rules
in SUMO, that the model the registry names really is the one that runs,
that the numbers on a page are the numbers the harness produced, and
that a run saved by the dashboard reloads as the same run.

Each check prints PASS or FAIL with what it measured, and the script
exits non-zero if any of them failed - so it can be the last thing run
before a commit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RESULTS: List[Dict[str, Any]] = []


def check(name: str):
    """Register one check. The function returns (ok, detail)."""
    def wrap(function: Callable[[], Tuple[bool, str]]):
        function._check_name = name
        return function
    return wrap


# ===========================================================================
@check("every controller obeys the 15-30 s rule in real SUMO")
def controllers_obey_the_profile():
    from simulation.engine import (CONTROLLER_ORDER, model_status,
                                   preset_workload, run_controller)

    workload = preset_workload("typical", 300.0)
    usable = model_status().get("usable")
    lines = []
    ok = True
    for controller in CONTROLLER_ORDER:
        if controller == "dqn" and not usable:
            lines.append("dqn skipped: no validated model")
            continue
        result = run_controller(workload, controller, seed=70001)
        audit = result.audit
        good = bool(audit.get("compliant"))
        ok = ok and good
        lines.append("{:<13} greens {:>2}-{:>2} s, {} completed, {}".format(
            controller, int(audit.get("shortest_green") or 0),
            int(audit.get("longest_green") or 0),
            audit.get("greens_completed", 0),
            "clean" if good else "VIOLATIONS: {}".format(
                audit.get("violations"))))
    return ok, "; ".join(lines)


@check("the audit is independent: it catches a signal that breaks the rules")
def the_audit_can_fail():
    """A check that cannot fail proves nothing, so make it fail."""
    from rl.timing_profile import get_profile
    from simulation.recorder import EW_G, EW_Y, NS_G, NS_Y, RED, audit_signal

    profile = get_profile("standard")
    codes = ([NS_G] * 9 + [NS_Y] * 3 + [RED] * 2
             + [EW_G] * 31 + [EW_Y] * 3 + [RED] * 2 + [NS_G] * 15)
    report = audit_signal(codes, profile, 3.0, 2.0)
    rules = {v["rule"] for v in report["violations"]}
    wanted = {"minimum green", "maximum green"}
    return (wanted <= rules,
            "a 9 s and a 31 s green were reported as {}".format(
                sorted(rules) or "nothing"))


@check("the registered model is the file the registry names")
def the_registry_is_honest():
    from agents.phase_dqn_agent import (registered_member_paths,
                                        registered_profile_model, sha256_of)

    record = registered_profile_model("standard")
    if not record:
        return False, "no model is registered for the standard profile"
    expected = [m.get("sha256", "").lower()
                for m in (record.get("members") or [])] or \
        [str(record.get("sha256", "")).lower()]
    lines = []
    ok = True
    for index, relative in enumerate(registered_member_paths(record)):
        path = os.path.join(PROJECT_ROOT, relative)
        if not os.path.isfile(path):
            ok = False
            lines.append("{} is MISSING".format(relative))
            continue
        digest = sha256_of(path)
        want = expected[index] if index < len(expected) else ""
        same = digest == want
        ok = ok and same
        lines.append("{} {}".format(relative,
                                    "matches" if same else "DOES NOT MATCH"))
    return ok, "; ".join(lines)


@check("a lock-and-extend model is run as lock-and-extend")
def the_action_space_comes_from_the_checkpoint():
    """The defect this check exists for produced only the two shortest
    greens on the menu, in the dashboard and in Hybrid, for days."""
    from agents.phase_dqn_agent import load_registered_profile_model
    from rl.policy_rollout import (ACTION_SPACE_EXTEND, action_space_of,
                                   ladder_of)
    from simulation.engine import preset_workload, run_controller

    agent = load_registered_profile_model("standard")
    kind = action_space_of(agent)
    workload = preset_workload("typical", 300.0)
    result = run_controller(workload, "dqn", seed=70002)
    served = sorted({row["green_seconds"] for row in result.decisions
                     if row.get("completed", True)})
    if kind != ACTION_SPACE_EXTEND:
        return True, "the registered model is a duration menu; greens {}" \
            .format(served)
    allowed = set(ladder_of(agent))
    ok = set(served) <= allowed
    return ok, "ladder {} and it served {}".format(sorted(allowed), served)


@check("the dashboard's numbers are the harness's numbers")
def one_engine():
    from evaluation.harness import fixed_policy, run_policy
    from rl.timing_profile import get_profile
    from simulation.engine import preset_workload, run_controller

    profile = get_profile("standard")
    workload = preset_workload("offpeak", 300.0)
    through_engine = run_controller(workload, "fixed_time", seed=70003,
                                    plan={"ns": 22, "ew": 15}, record=False)
    through_harness = run_policy(workload, fixed_policy(profile, 22, 15),
                                 profile, 70003, controller="fixed",
                                 tag="_vfy")
    same = (abs(through_engine.delay - through_harness.mean_total_delay) < 1e-6
            and through_engine.completed == through_harness.completed)
    return same, "engine {:.4f} s against harness {:.4f} s".format(
        through_engine.delay, through_harness.mean_total_delay)


@check("two controllers meet exactly the same vehicles")
def identical_traffic():
    from custom_traffic.builder import build_route_file, scenario_name
    from simulation.engine import preset_workload

    workload = preset_workload("peak", 300.0)
    first = build_route_file(workload, scenario_name(workload, "_vfy1"))
    second = build_route_file(workload, scenario_name(workload, "_vfy2"))
    with open(first, encoding="utf-8") as a, open(second, encoding="utf-8") as b:
        same = [line for line in a if "<vehicle " in line] == \
               [line for line in b if "<vehicle " in line]
    for path in (first, second):
        try:
            os.remove(path)
        except OSError:
            pass
    return same, "{} vehicles written identically twice".format(
        workload.total_vehicles)


@check("a saved run reloads as the same run")
def saved_runs_round_trip():
    import tempfile

    from simulation.engine import (load_run, preset_workload, run_controller,
                                   save_run)

    workload = preset_workload("offpeak", 300.0)
    result = run_controller(workload, "fixed_time", seed=70004,
                            plan={"ns": 18, "ew": 18})
    with tempfile.TemporaryDirectory() as folder:
        save_run(result, directory=folder)
        back = load_run(workload.workload_id, "fixed_time", 70004,
                        directory=folder)
    if back is None:
        return False, "the run did not reload"
    same = (back.delay == result.delay and back.waiting == result.waiting
            and len(back.series) == len(result.series)
            and back.replay["steps"] == result.replay["steps"]
            and back.audit.get("compliant") == result.audit.get("compliant"))
    return same, "{:.3f} s delay, {} steps of replay, audit {}".format(
        back.delay, back.replay["steps"], back.audit.get("compliant"))


@check("a vehicle still inside at the end is charged for its delay")
def stranded_vehicles_are_charged():
    from simulation.engine import preset_workload, run_controller

    workload = preset_workload("peak", 300.0)
    result = run_controller(workload, "fixed_time", seed=70005,
                            plan={"ns": 30, "ew": 15}, record=False)
    inside = int(result.metrics.get("still_on_road", 0))
    if inside == 0:
        return True, "nothing was left inside, so nothing to charge"
    from evaluation.harness import fixed_policy, run_policy
    from rl.timing_profile import get_profile

    run = run_policy(workload, fixed_policy(get_profile("standard"), 30, 15),
                     get_profile("standard"), 70005, controller="fixed",
                     tag="_vfy2")
    charged = int(run.delay.get("unfinished_charged", 0))
    return charged == inside, \
        "{} vehicles were still inside and {} of them were charged".format(
            inside, charged)


@check("the model beats the tuned fixed plan on the traffic it is shown")
def the_model_actually_wins():
    from simulation.engine import (compare, model_status, preset_workload,
                                   reference_plan, run_controller)

    if not model_status().get("usable"):
        return False, "no validated model is registered"
    plan = reference_plan() or {"ns": 25, "ew": 15}
    wins, lines = 0, []
    for preset in ("offpeak", "typical", "peak"):
        workload = preset_workload(preset, 600.0)
        runs = [run_controller(workload, "fixed_time", seed=70006, plan=plan,
                               record=False),
                run_controller(workload, "dqn", seed=70006, record=False)]
        table = compare(runs)
        row = next(r for r in table["rows"] if r["controller"] == "dqn")
        wins += 1 if row["change_percent"] < 0 else 0
        lines.append("{} {:+.1f} %".format(preset, row["change_percent"]))
    return wins >= 2, "{} of 3 presets improved: {}".format(
        wins, ", ".join(lines))


@check("the dashboard starts and every page renders")
def the_dashboard_renders():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(os.path.join(PROJECT_ROOT, "dashboard.py"),
                            default_timeout=300)
    app.run()
    if app.exception:
        return False, str(app.exception)
    pages = app.sidebar.radio[0].options
    for page in pages:
        app.sidebar.radio[0].set_value(page).run()
        if app.exception:
            return False, "{} raised {}".format(page, app.exception)
    return True, "{} pages: {}".format(len(pages), ", ".join(pages))


QUICK = {"the audit is independent: it catches a signal that breaks the rules",
         "the registered model is the file the registry names",
         "two controllers meet exactly the same vehicles",
         "the dashboard starts and every page renders"}


# ===========================================================================
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="only the checks that do not run long "
                             "simulations")
    parser.add_argument("--out", default=None, help="write the results here")
    args = parser.parse_args(argv)

    import torch
    torch.set_num_threads(1)

    checks = [value for value in globals().values()
              if callable(value) and hasattr(value, "_check_name")]
    if args.quick:
        checks = [c for c in checks if c._check_name in QUICK]

    print("=" * 78)
    print("IntelliSignal - end-to-end verification ({} checks)".format(
        len(checks)))
    print("=" * 78)
    failed = 0
    for function in checks:
        started = time.time()
        try:
            ok, detail = function()
        except BaseException as exc:                     # noqa: BLE001
            ok, detail = False, "{}: {}".format(type(exc).__name__, exc)
        failed += 0 if ok else 1
        RESULTS.append({"check": function._check_name, "ok": bool(ok),
                        "detail": detail,
                        "seconds": round(time.time() - started, 1)})
        print("{}  {}".format("PASS" if ok else "FAIL", function._check_name))
        print("      {}".format(detail))
        print("      ({:.0f} s)".format(time.time() - started))

    print("=" * 78)
    print("{} of {} checks passed".format(len(checks) - failed, len(checks)))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(RESULTS, handle, indent=2)
        print("saved {}".format(args.out))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
