"""Build the headline comparison on the confirmatory seeds.

Every row is measured the same way: 1800 simulated seconds, seeds
6101-6120, `utils.metrics.MetricsCollector`. The fixed-time rows run SUMO's
own traffic-light program. The learned rows run their agent through the
environment it was trained in.

    python output/dqn_phase_20261122/final_table.py \
        --phase-model output/dqn_phase_20261122/runs/G3/models/best.pt \
        --stage4-model output/dqn_retraining_20261120/models/dqn_heavy/best.pt
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
for path in (PROJECT_ROOT, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from evaluation.evaluate_phase_dqn import (evaluate_agent,  # noqa: E402
                                           evaluate_fixed)
from pattern import make_pattern                            # noqa: E402
from probe import cleanup, parse_seeds, run                 # noqa: E402


# ---------------------------------------------------------------------------
def evaluate_pattern(ns, ew, scenario, seeds, tag) -> Dict[str, Any]:
    policy = make_pattern(ns, ew)
    rows = []
    try:
        for seed in seeds:
            policy.reset()
            rows.append(run(policy, scenario, seed, tag=tag))
    finally:
        cleanup(scenario, tag)
    waits = [float(r["average_waiting_time"]) for r in rows]
    return {
        "controller": "pattern ns[{}] ew[{}]".format(
            ",".join(map(str, ns)), ",".join(map(str, ew))),
        "seeds": [int(r["seed"]) for r in rows],
        "mean_waiting_time": statistics.fmean(waits),
        "sd_waiting_time": statistics.pstdev(waits),
        "mean_throughput": statistics.fmean(float(r["throughput"]) for r in rows),
        "mean_queue_length": statistics.fmean(
            float(r["average_queue_length"]) for r in rows),
        "mean_stranded": statistics.fmean(
            float(r["vehicles_still_in_network"]) for r in rows),
        "per_seed_waiting": waits,
    }


# ---------------------------------------------------------------------------
def evaluate_stage4(scenario: str, model: str, seeds) -> Optional[Dict[str, Any]]:
    """The deployed Stage 3/4 KEEP-SWITCH agent, through its own evaluator."""
    if not model or not os.path.exists(model):
        return None
    from evaluation.evaluate_dqn import aggregate, evaluate
    result = evaluate(scenario=scenario, model=model, seeds=list(seeds),
                      quiet=True, save=False)
    rows = result["rows"] if isinstance(result, dict) and "rows" in result else result
    waits = [float(r["average_waiting_time"]) for r in rows]
    return {
        "controller": "dqn_stage4_keep_switch",
        "seeds": [int(r.get("seed", s)) for r, s in zip(rows, seeds)],
        "mean_waiting_time": statistics.fmean(waits),
        "sd_waiting_time": statistics.pstdev(waits),
        "mean_throughput": statistics.fmean(float(r["throughput"]) for r in rows),
        "mean_queue_length": statistics.fmean(
            float(r["average_queue_length"]) for r in rows),
        "mean_stranded": statistics.fmean(
            float(r.get("vehicles_still_in_network", 0)) for r in rows),
        "per_seed_waiting": waits,
    }


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="heavy")
    parser.add_argument("--seeds", default="6101-6120")
    parser.add_argument("--phase-model", required=True)
    parser.add_argument("--stage4-model", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    seeds = [int(v) for v in parse_seeds(args.seeds)]
    results: List[Dict[str, Any]] = []

    print("fixed 30/30, the original Stage 1 baseline")
    results.append(evaluate_fixed(args.scenario, seeds, 30, 30, tag="_ft3030"))

    print("\nfixed 7/6, the tuned plan")
    fixed76 = evaluate_fixed(args.scenario, seeds, 7, 6, tag="_ft76")
    results.append(fixed76)

    print("\nbest repeating pattern ns[7,7] ew[5,6]")
    results.append(evaluate_pattern([7, 7], [5, 6], args.scenario, seeds, "_pat"))

    stage4 = evaluate_stage4(args.scenario, args.stage4_model, seeds)
    if stage4:
        print("\nStage 4 deployed KEEP/SWITCH DQN")
        results.append(stage4)

    print("\nphase-duration DQN")
    phase = evaluate_agent(args.phase_model, args.scenario, seeds, tag="_fin")
    results.append(phase)

    results.sort(key=lambda r: r["mean_waiting_time"])

    print("\n" + "=" * 90)
    print("heavy traffic, seeds {}-{}, 1800 s, MetricsCollector".format(
        seeds[0], seeds[-1]))
    print("=" * 90)
    print("{:<34} {:>9} {:>7} {:>11} {:>9} {:>9}".format(
        "controller", "wait", "sd", "throughput", "queue", "stranded"))
    print("-" * 90)
    for row in results:
        print("{:<34} {:>9.3f} {:>7.3f} {:>11.1f} {:>9.3f} {:>9.1f}".format(
            row["controller"][:34], row["mean_waiting_time"],
            row["sd_waiting_time"], row["mean_throughput"],
            row["mean_queue_length"], row["mean_stranded"]))
    print("-" * 90)

    delta = phase["mean_waiting_time"] - fixed76["mean_waiting_time"]
    diffs = [d - f for d, f in zip(phase["per_seed_waiting"],
                                   fixed76["per_seed_waiting"])]
    wins = sum(1 for d in diffs if d < 0)
    print("phase DQN against the tuned fixed plan: {:+.3f} s ({:+.1f} %), "
          "wins on {}/{} seeds".format(delta,
                                       100.0 * delta / fixed76["mean_waiting_time"],
                                       wins, len(diffs)))
    print("=" * 90)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, default=float)
        print("saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
