"""Post-cleanup verification, through the paths the demonstration uses.

Three questions, answered by running things rather than by reading them:

  1. Do Moderate and Heavy still produce the numbers they produced before
     the cleanup? Measured on seed 6101 against the saved per-seed results.
  2. Does a custom workload run end to end under both controllers, with the
     preview and the simulation agreeing?
  3. Does a saved run reload?

    python output/demo_check_20261123/verify_demo.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from custom_traffic.resolver import resolve                     # noqa: E402
from custom_traffic.spec import CustomSpec                      # noqa: E402
from evaluation.fixed_time_sweep import (cleanup_generated_inputs,  # noqa: E402
                                         run_fixed_time_plan)
from simulation.custom_runs import (load_custom_run, run_dqn,   # noqa: E402
                                    run_fixed, save_custom_run)

#: Measured on 22 Nov 2026 over seeds 6101-6120 and recorded in
#: output/dqn_phase_20261122/FINDINGS.md. Seed 6101 on its own is one draw
#: from that set, so it is compared against that seed's own recorded value.
EXPECTED_SEED_6101 = {
    # scenario: (ns_green, ew_green, mean waiting on seed 6101)
    "heavy": (7.0, 6.0, 6.670),
    "moderate": (6.0, 4.0, None),      # filled in from the saved JSON below
}

SAVED_MODERATE = os.path.join(
    PROJECT_ROOT, "output", "dqn_phase_20261122", "moderate_fixed_confirm.json")
SAVED_HEAVY = os.path.join(
    PROJECT_ROOT, "output", "dqn_phase_20261122", "heavy_fixed_confirm.json")


def saved_value(path, ns, ew, seed_index=0):
    with open(path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    for row in rows:
        if abs(row["ns"] - ns) < 1e-9 and abs(row["ew"] - ew) < 1e-9:
            return row["per_seed"][seed_index]
    return None


def check_scenarios() -> bool:
    print("=" * 74)
    print("1. Moderate and Heavy on seed 6101, against the saved results")
    print("=" * 74)
    ok = True
    for scenario, path in (("heavy", SAVED_HEAVY), ("moderate", SAVED_MODERATE)):
        ns, ew, _ = EXPECTED_SEED_6101[scenario]
        expected = saved_value(path, ns, ew)
        if expected is None:
            print("  {}: no saved value for ns={:g}/ew={:g} - skipped".format(
                scenario, ns, ew))
            continue
        try:
            row = run_fixed_time_plan(scenario, ns, 6101, ew_green=ew,
                                 tag="_verify")
        finally:
            cleanup_generated_inputs(scenario, "_verify")
        measured = float(row["average_waiting_time"])
        same = abs(measured - expected) < 0.01
        ok = ok and same
        print("  {:<9} ns={:g} ew={:g}  saved {:.3f} s  now {:.3f} s   {}"
              .format(scenario, ns, ew, expected, measured,
                      "MATCH" if same else "DIFFERS"))
    return ok


def check_custom() -> bool:
    print()
    print("=" * 74)
    print("2. A custom workload, end to end, under both controllers")
    print("=" * 74)

    # The brief's example: 200 cars on one lane, 300-350 on another, with
    # speed, turns and arrival windows left blank.
    spec = CustomSpec.empty(seed=6101)
    spec.name = "teacher_demo"
    spec.duration_s = 900.0
    north = spec.lane("north", 0)
    north.enabled, north.count = True, 200
    east = spec.lane("east", 0)
    east.enabled, east.count_min, east.count_max = True, 300, 350

    resolution = resolve(spec)
    if not resolution.ok:
        for issue in resolution.errors:
            print("  ERROR", issue)
        return False
    workload = resolution.workload
    print("  workload {}  {} vehicles over {:.0f} s".format(
        workload.workload_id, workload.total_vehicles, workload.duration_s))
    for row in workload.preview_rows():
        print("    {:<28} {:>4} veh  {:<14} {:<10} {}".format(
            row["Lane"], row["Vehicles"], row["Arrive between"],
            row["Rate"], row["Turns"]))

    results = {}
    summary, timeseries = run_fixed(workload, tag="_vfix")
    save_custom_run(workload, summary, timeseries)
    results["fixed_time"] = summary

    from agents.phase_dqn_agent import load_registered_phase_model
    agent = load_registered_phase_model(workload.base_scenario)
    dqn_summary, dqn_series, decisions = run_dqn(workload, agent, tag="_vdqn")
    save_custom_run(workload, dqn_summary, dqn_series, decisions)
    results["dqn"] = dqn_summary

    print()
    print("  {:<12} {:>8} {:>10} {:>10} {:>10} {:>12} {:>8}".format(
        "controller", "wait s", "requested", "entered", "completed",
        "on the road", "pending"))
    for name, summary in results.items():
        print("  {:<12} {:>8.2f} {:>10} {:>10} {:>10} {:>12} {:>8}".format(
            name, summary["average_waiting_time"],
            summary["vehicles_requested"], summary["vehicles_entered"],
            summary["vehicles_completed"], summary["vehicles_still_on_road"],
            summary["vehicles_waiting_to_enter"]))

    # The preview promised exactly this many vehicles; SUMO must agree.
    agreed = all(summary["vehicles_requested"] == workload.total_vehicles
                 for summary in results.values())
    entered_all = all(summary["vehicles_entered"] <= workload.total_vehicles
                      for summary in results.values())
    print()
    print("  preview total matches what was requested of SUMO: {}".format(
        "yes" if agreed else "NO"))
    print("  decisions recorded by the DQN run: {}".format(len(decisions)))

    reloaded = load_custom_run(workload.workload_id, "dqn")
    print("  saved run reloads from disk: {}".format(
        "yes" if reloaded else "NO"))
    return agreed and entered_all and reloaded is not None


def main() -> int:
    scenarios_ok = check_scenarios()
    custom_ok = check_custom()
    print()
    print("=" * 74)
    print("scenarios reproduce their saved results : {}".format(
        "yes" if scenarios_ok else "NO"))
    print("custom workload runs end to end          : {}".format(
        "yes" if custom_ok else "NO"))
    print("=" * 74)
    return 0 if (scenarios_ok and custom_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
