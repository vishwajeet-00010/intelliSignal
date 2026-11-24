"""
IntelliSignal
evaluation/compare_profiles.py

The fair comparison: every controller, one profile, one split, one table.

Discipline
----------
The fixed-time baseline is **tuned on the validation split and reported on
the test split**, exactly like the learned candidates. Tuning it on the
test split would be the same mistake as selecting a model there, and
leaving it untuned would be manufacturing a win by weakening the
opponent.

North-south and east-west greens are tuned independently, because the
demand is asymmetric and a symmetric sweep cannot find the optimum.

Everything else - the traffic, the clearance, the horizon, the timing
profile and the two measurements - is identical for every controller, by
construction: they all go through `evaluation/harness.py`.

What is reported
----------------
Mean waiting is kept because it is the historical headline, but the table
leads with **total delay per requested vehicle**, which charges the
insertion delay of vehicles that never got in. A controller cannot
improve that number by refusing traffic.

Completion, insertion backlog, queue, mean green and timing compliance
are all shown alongside, so a lower delay bought by excluding vehicles is
visible rather than hidden.

    python evaluation/compare_profiles.py --profile standard \\
        --candidate-a output/run_a/models/best.pt \\
        --candidate-b output/run_b/models/best.pt
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import sys
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.phase_dqn_agent import PhaseDQNAgent
from evaluation.harness import (ControllerRun, aggregate, fixed_policy,
                                greedy_agent_policy, max_pressure_policy,
                                run_actuated, run_extend_policy, run_policy)
from rl.phase_state import extractor_from_contract
from rl.timing_profile import compatible, get_profile
from training.workload_library import split, sumo_seeds


# ---------------------------------------------------------------------------
def tune_fixed(profile, workloads, seeds, step: int = 3,
               quiet: bool = False) -> Dict[str, Any]:
    """Sweep north-south and east-west green independently.

    Coordinate descent over the profile's own menu: hold one direction,
    sweep the other, keep the best, then swap. Measured on the workloads
    it is given - which must be the validation split, never the test one.
    """
    menu = list(profile.durations)
    best = {"ns": menu[0], "ew": menu[0], "delay": None}

    def measure(ns, ew):
        runs = [run_policy(workload, fixed_policy(profile, ns, ew), profile,
                           seeds[index % len(seeds)],
                           controller="fixed", tag="_tune")
                for index, workload in enumerate(workloads)]
        return statistics.fmean(run.mean_total_delay for run in runs)

    best["delay"] = measure(best["ns"], best["ew"])
    for axis in ("ns", "ew"):
        for value in menu:
            if value == best[axis]:
                continue
            trial = dict(best)
            trial[axis] = value
            delay = measure(trial["ns"], trial["ew"])
            if not quiet:
                print("    ns {:>4g} ew {:>4g} -> {:8.2f}".format(
                    trial["ns"], trial["ew"], delay), flush=True)
            if delay < best["delay"]:
                best = dict(trial, delay=delay)
    return best


# ---------------------------------------------------------------------------
def load_candidate(path: Optional[str], profile):
    """Load a checkpoint and refuse one that does not fit the profile."""
    if not path or not os.path.exists(path):
        return None, "no checkpoint at {}".format(path)
    agent = PhaseDQNAgent.load(path)
    if not compatible(profile, agent.durations):
        return None, (
            "checkpoint menu {} is outside the {} profile ({:g}-{:g} s); "
            "it is historical evidence, not a valid model for this profile"
            .format(agent.durations, profile.name,
                    profile.min_green, profile.max_green))
    return agent, ""


# ---------------------------------------------------------------------------
def evaluate_all(profile_name: str, candidate_a: Optional[str],
                 candidate_b: Optional[str], extension: float = 3.0,
                 duration_s: float = 900.0, tune: bool = True,
                 fixed_plan: Optional[Dict[str, float]] = None,
                 quiet: bool = False) -> Dict[str, Any]:
    profile = get_profile(profile_name)
    validation = split("validation", duration_s=duration_s)
    test = split("test", duration_s=duration_s)
    val_seeds = sumo_seeds("validation", len(validation))
    test_seeds = sumo_seeds("test", len(test))

    # --- the fixed baseline, tuned where the models were selected ------
    if fixed_plan is None and tune:
        if not quiet:
            print("tuning the fixed plan on the VALIDATION split "
                  "(never on test):")
        fixed_plan = tune_fixed(profile, validation, val_seeds, quiet=quiet)
        if not quiet:
            print("  best fixed plan: ns {:g} / ew {:g} "
                  "({:.2f} s delay on validation)".format(
                      fixed_plan["ns"], fixed_plan["ew"], fixed_plan["delay"]))
    elif fixed_plan is None:
        fixed_plan = {"ns": profile.durations[0], "ew": profile.durations[0],
                      "delay": None}

    # --- everything is measured on the TEST split, once ----------------
    results: Dict[str, List[ControllerRun]] = {}

    def record(name, runs):
        results[name] = runs
        if not quiet:
            summary = aggregate(runs)
            print("  {:<26} delay {:>8.2f} +- {:<6.2f} wait {:>7.2f} "
                  "completed {:>6.1f} pending {:>6.1f}".format(
                      name, summary["mean_total_delay"],
                      summary["sd_total_delay"], summary["mean_waiting"],
                      summary["mean_completed"],
                      summary["mean_waiting_to_enter"]), flush=True)

    if not quiet:
        print("\nmeasuring every controller on the TEST split "
              "({} workloads):".format(len(test)))

    record("fixed ns{:g}/ew{:g}".format(fixed_plan["ns"], fixed_plan["ew"]), [
        run_policy(workload, fixed_policy(profile, fixed_plan["ns"],
                                          fixed_plan["ew"]),
                   profile, test_seeds[index % len(test_seeds)],
                   controller="fixed", tag="_cmpf")
        for index, workload in enumerate(test)])

    record("actuated (gap 3s)", [
        run_actuated(workload, profile, test_seeds[index % len(test_seeds)],
                     gap_seconds=3.0, tag="_cmpa")
        for index, workload in enumerate(test)])

    record("max-pressure", [
        run_policy(workload, max_pressure_policy(profile), profile,
                   test_seeds[index % len(test_seeds)],
                   controller="max-pressure", tag="_cmpp")
        for index, workload in enumerate(test)])

    notes: List[str] = []
    agent_a, why = load_candidate(candidate_a, profile)
    if agent_a is not None:
        record("DQN candidate A", [
            run_policy(workload, greedy_agent_policy(agent_a), profile,
                       test_seeds[index % len(test_seeds)],
                       controller="candidate A",
                       extractor=extractor_from_contract(
                           agent_a.observation_contract), tag="_cmpA")
            for index, workload in enumerate(test)])
    elif why:
        notes.append("Candidate A was not evaluated: {}".format(why))

    agent_b, why = load_candidate(candidate_b, profile)
    if agent_b is not None:
        record("DQN candidate B", [
            run_extend_policy(workload, agent_b, profile,
                              test_seeds[index % len(test_seeds)],
                              extension=extension,
                              extractor=extractor_from_contract(
                                  agent_b.observation_contract), tag="_cmpB")
            for index, workload in enumerate(test)])
    elif why:
        notes.append("Candidate B was not evaluated: {}".format(why))

    table = [aggregate(runs) for runs in results.values()]
    table.sort(key=lambda row: row["mean_total_delay"])

    return {
        "profile": profile.to_dict(),
        "fixed_plan": fixed_plan,
        "split": "test",
        "workloads": [{"name": w.name, "id": w.workload_id[:8],
                       "vehicles": w.total_vehicles} for w in test],
        "seeds": test_seeds,
        "table": table,
        "per_workload": {
            name: [{"workload": w.name, "delay": run.mean_total_delay,
                    "waiting": run.mean_waiting, "completed": run.completed,
                    "pending": run.waiting_to_enter,
                    "compliant": run.compliant}
                   for run, w in zip(runs, test)]
            for name, runs in results.items()},
        "notes": notes,
    }


# ---------------------------------------------------------------------------
def render(result: Dict[str, Any]) -> None:
    profile = result["profile"]
    print("\n" + "=" * 104)
    print("{} profile  |  test split, {} workloads  |  green {:g}-{:g} s"
          .format(profile["name"].upper(), len(result["workloads"]),
                  profile["min_green"], profile["max_green"]))
    print("=" * 104)
    print("{:<26} {:>9} {:>7} {:>9} {:>10} {:>9} {:>8} {:>7} {:>6}".format(
        "controller", "delay/veh", "sd", "wait/veh", "completed",
        "pending", "queue", "green", "ok"))
    print("-" * 104)
    for row in result["table"]:
        print("{:<26} {:>9.2f} {:>7.2f} {:>9.2f} {:>10.1f} {:>9.1f} "
              "{:>8.2f} {:>7.1f} {:>6}".format(
                  row["controller"], row["mean_total_delay"],
                  row["sd_total_delay"], row["mean_waiting"],
                  row["mean_completed"], row["mean_waiting_to_enter"],
                  row["mean_queue"], row["mean_green"],
                  "yes" if row["all_compliant"] else "NO"))
    print("-" * 104)
    print("delay/veh is total delay per REQUESTED vehicle, insertion delay "
          "included. Lower is better and it cannot be improved by refusing "
          "vehicles.")

    fixed = next((row for row in result["table"]
                  if row["controller"].startswith("fixed")), None)
    for row in result["table"]:
        if fixed is None or row is fixed:
            continue
        change = 100.0 * (row["mean_total_delay"] - fixed["mean_total_delay"]) \
            / fixed["mean_total_delay"]
        verdict = "BEATS" if change < 0 else "loses to"
        print("  {:<26} {} the tuned fixed plan by {:+.1f} %".format(
            row["controller"], verdict, change))

    for note in result.get("notes", []):
        print("  NOTE: {}".format(note))
    print("=" * 104)


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="standard")
    parser.add_argument("--candidate-a", default=None)
    parser.add_argument("--candidate-b", default=None)
    parser.add_argument("--extension", type=float, default=3.0)
    parser.add_argument("--duration", type=float, default=900.0)
    parser.add_argument("--no-tune", action="store_true")
    parser.add_argument("--ns-green", type=float, default=None)
    parser.add_argument("--ew-green", type=float, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    plan = None
    if args.ns_green is not None and args.ew_green is not None:
        plan = {"ns": args.ns_green, "ew": args.ew_green, "delay": None}

    result = evaluate_all(args.profile, args.candidate_a, args.candidate_b,
                          extension=args.extension, duration_s=args.duration,
                          tune=not args.no_tune, fixed_plan=plan,
                          quiet=args.quiet)
    render(result)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, default=float)
        print("saved {}".format(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
