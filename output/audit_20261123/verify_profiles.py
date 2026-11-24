"""End-to-end verification of the timing profiles, through the demo path.

Five questions, answered by running things:

  1. Does every controller actually obey the profile in a real SUMO run?
  2. Does the dashboard refuse a model that does not fit the profile?
  3. Does a custom run under the standard profile save, reload and replay?
  4. Do the two measurements of the same run agree?
  5. Do the pre-existing Moderate and Heavy results still reproduce?

    python output/audit_20261123/verify_profiles.py
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.phase_dqn_agent import load_registered_profile_model
from custom_traffic.resolver import resolve
from custom_traffic.spec import CustomSpec
from evaluation.fixed_time_sweep import (cleanup_generated_inputs,
                                         run_fixed_time_plan)
from evaluation.harness import (fixed_policy, max_pressure_policy,
                                run_actuated, run_policy)
from rl.timing_profile import PROFILES, compatible, get_profile
from simulation.controller import MODE_DQN, MODE_FIXED_TIME, SimulationController
from simulation.custom_runs import load_custom_run

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print("  [{}] {:<54} {}".format(PASS if ok else FAIL, name, detail))
    return ok


def demo_workload(duration=600.0):
    spec = CustomSpec.empty(seed=6101)
    spec.name = "verify"
    spec.duration_s = duration
    north = spec.lane("north", 0)
    north.enabled, north.count = True, 140
    east = spec.lane("east", 0)
    east.enabled, east.count_min, east.count_max = True, 200, 240
    south = spec.lane("south", 1)
    south.enabled, south.count = True, 80
    resolution = resolve(spec)
    assert resolution.ok, [str(i) for i in resolution.errors]
    return resolution.workload


# ===========================================================================
def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    workload = demo_workload()
    print("workload {} - {} vehicles over {:.0f} s".format(
        workload.workload_id[:8], workload.total_vehicles,
        workload.duration_s))

    # --- 1. every controller obeys the profile ------------------------
    section("1. Timing compliance, in real SUMO runs")
    for profile_name in PROFILES:
        profile = get_profile(profile_name)
        runs = {
            "fixed": run_policy(workload,
                                fixed_policy(profile, profile.max_green,
                                             profile.min_green),
                                profile, 7001, controller="fixed", tag="_v1"),
            "max-pressure": run_policy(workload, max_pressure_policy(profile),
                                       profile, 7001,
                                       controller="pressure", tag="_v2"),
            "actuated": run_actuated(workload, profile, 7001,
                                     gap_seconds=3.0, tag="_v3"),
        }
        for name, run in runs.items():
            report = run.compliance
            ok = (report["greens_below_min"] == 0
                  and report["greens_above_max"] == 0
                  and report["red_limit_breaches"] == 0)
            check("{:<10} {:<13} greens {:g}-{:g} s".format(
                profile_name, name, report["min_green_seen"] or 0,
                report["max_green_seen"] or 0), ok,
                "{} truncated".format(report["truncated_phases"]))

    # --- 2. an incompatible model is refused --------------------------
    section("2. Model compatibility")
    standard = get_profile("standard")
    responsive = get_profile("responsive")
    controller = SimulationController()

    agent_std = controller.load_phase_agent("moderate", profile="standard")
    check("the standard profile loads a 15-30 s model",
          compatible(standard, agent_std.durations),
          "menu {}".format([int(d) for d in agent_std.durations]))

    agent_rsp = controller.load_phase_agent("moderate", profile="responsive")
    check("a responsive model is NOT valid under standard",
          not compatible(standard, agent_rsp.durations),
          "menu {} is refused".format([int(d) for d in agent_rsp.durations]))

    # --- 3. a custom run saves, reloads and carries its compliance ----
    section("3. Custom run under the standard profile")
    run = controller.run_custom(workload, mode=MODE_DQN, agent=agent_std,
                                seed=6101, profile="standard")
    compliance = run.metric("timing_compliance") or {}
    check("the run records its timing compliance",
          bool(compliance) and compliance.get("compliant") is True,
          "greens {:g}-{:g} s".format(compliance.get("min_green_seen") or 0,
                                      compliance.get("max_green_seen") or 0))
    check("the run records its profile",
          run.metric("timing_profile") == "standard",
          str(run.metric("timing_profile")))
    check("requested / entered / completed are all reported",
          run.metric("vehicles_requested") == workload.total_vehicles,
          "{} requested, {} entered, {} completed, {} waiting to enter".format(
              run.metric("vehicles_requested"), run.metric("vehicles_entered"),
              run.metric("vehicles_completed"),
              run.metric("vehicles_waiting_to_enter")))

    reloaded = load_custom_run(workload.workload_id, "dqn")
    check("the saved run reloads from disk", reloaded is not None)
    if reloaded:
        saved = reloaded["summary"]
        check("the reloaded numbers match what was displayed",
              saved["average_waiting_time"] == run.metric("average_waiting_time")
              and saved["vehicles_completed"] == run.metric("vehicles_completed"),
              "waiting {} s, completed {}".format(
                  saved["average_waiting_time"], saved["vehicles_completed"]))
        check("the replay timeseries was written",
              bool(reloaded.get("timeseries_csv")),
              reloaded.get("timeseries_csv", ""))

    # --- 4. the two measurements agree --------------------------------
    section("4. Cross-check between the two measurements")
    measured = run_policy(workload,
                          lambda obs, env: int(agent_std.act(obs)),
                          standard, 6101, controller="candidate", tag="_v4")
    cross = measured.crosscheck
    check("live collector and SUMO trip info agree on completions",
          cross["completed_agree"],
          "{} vs {}".format(cross["completed_live"],
                            cross["completed_tripinfo"]))
    check("total delay per requested vehicle is reported",
          measured.mean_total_delay > 0,
          "{:.2f} s, of which {:.2f} s is insertion delay".format(
              measured.mean_total_delay,
              measured.delay["mean_insertion_delay_per_requested"]))

    # --- 5. the historical results still reproduce --------------------
    section("5. The pre-existing results are unchanged")
    for scenario, ns, ew, expected in (("heavy", 7.0, 6.0, 6.670),
                                       ("moderate", 6.0, 4.0, 2.790)):
        try:
            row = run_fixed_time_plan(scenario, ns, 6101, ew_green=ew,
                                      tag="_verp")
        finally:
            cleanup_generated_inputs(scenario, "_verp")
        measured_wait = float(row["average_waiting_time"])
        check("{} ns{:g}/ew{:g} on seed 6101".format(scenario, ns, ew),
              abs(measured_wait - expected) < 0.01,
              "saved {:.3f} s, now {:.3f} s".format(expected, measured_wait))

    # ------------------------------------------------------------------
    section("Summary")
    failed = [name for name, ok, _ in results if not ok]
    print("  {} checks, {} passed, {} failed".format(
        len(results), len(results) - len(failed), len(failed)))
    for name in failed:
        print("    FAILED: {}".format(name))
    with open(os.path.join(HERE, "verification.json"), "w",
              encoding="utf-8") as handle:
        json.dump([{"check": n, "passed": ok, "detail": d}
                   for n, ok, d in results], handle, indent=2)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
