"""
IntelliSignal
evaluation/evaluate_phase_dqn.py

Measures a trained duration-choosing DQN against the fixed-time baseline.

    python evaluation/evaluate_phase_dqn.py \
        --model output/run/models/best.pt --seeds 6101-6120

Both controllers are run over the same seeds, for the same 1800 simulated
seconds, and measured by `utils.metrics.MetricsCollector` - the class that
produced the Stage 1 baseline. The fixed-time row is produced by
`evaluation/fixed_time_sweep.py`, which runs SUMO's own traffic-light
program rather than driving a controller through the environment, because
that is how a fixed-time plan actually runs.

What is printed, and why each column is there
---------------------------------------------
mean waiting      the headline: mean accumulated waiting per vehicle
sd                across seeds, so a win inside the noise is visible as one
throughput        vehicles that completed their route; a controller that
                  lowers waiting by stranding vehicles is not better, and
                  this is the column that shows it
stranded          vehicles still in the network at 1800 s, same reason
green histogram   how often each duration was chosen. A policy that puts
                  95 % of its mass on one entry has not learned to react to
                  anything - it has found a fixed plan the slow way, and the
                  project has shipped that mistake before.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.phase_dqn_agent import PhaseDQNAgent                  # noqa: E402
from evaluation.fixed_time_sweep import (cleanup_generated_inputs,  # noqa: E402
                                         parse_number_list,
                                         run_fixed_time_plan)
from rl.phase_action import PHASE_NS                              # noqa: E402
from rl.phase_environment import PhaseDurationEnv                 # noqa: E402
from rl.phase_state import extractor_from_contract                # noqa: E402
from utils.metrics import MetricsCollector                        # noqa: E402


# ---------------------------------------------------------------------------
def evaluate_agent(model_path: str, scenario: str, seeds: Sequence[int],
                   tag: str = "_eval") -> Dict[str, Any]:
    """Greedy rollouts of a trained policy, measured like the baseline."""
    agent = PhaseDQNAgent.load(model_path)
    extractor = extractor_from_contract(agent.observation_contract)
    env = PhaseDurationEnv(scenario=scenario, durations=agent.durations,
                           state_extractor=extractor, tag=tag)

    rows: List[Dict[str, Any]] = []
    greens: List[float] = []
    greens_ns: List[float] = []
    greens_ew: List[float] = []
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=int(seed))
            collector = MetricsCollector(step_length=env.step_length,
                                         conn=env.connection)
            env.step_callback = collector.collect_step
            while True:
                phase = env.controller.phase
                action = agent.act(observation, training=False)
                observation, _, terminated, truncated, info = env.step(action)
                served = float(info["green_served"])
                greens.append(served)
                (greens_ns if phase == PHASE_NS else greens_ew).append(served)
                if terminated or truncated:
                    break
            summary = collector.summary(scenario)
            summary["seed"] = int(seed)
            rows.append(summary)
            print("  seed {}  wait {:>8.3f}  throughput {:>6}".format(
                seed, summary["average_waiting_time"], summary["throughput"]),
                flush=True)
    finally:
        env.close()
        env.cleanup_generated_inputs()

    return _aggregate("dqn_phase", rows, greens, greens_ns, greens_ew,
                      durations=agent.durations, model=model_path)


# ---------------------------------------------------------------------------
def evaluate_fixed(scenario: str, seeds: Sequence[int], ns_green: float,
                   ew_green: float, tag: str = "_evalfixed") -> Dict[str, Any]:
    """The fixed plan, through SUMO's own traffic-light program."""
    rows = []
    try:
        for seed in seeds:
            row = run_fixed_time_plan(scenario, ns_green, int(seed),
                                      ew_green=ew_green, tag=tag)
            rows.append(row)
            print("  seed {}  wait {:>8.3f}  throughput {:>6}".format(
                seed, row["average_waiting_time"], row["throughput"]),
                flush=True)
    finally:
        cleanup_generated_inputs(scenario, tag)
    label = "fixed_ns{:g}_ew{:g}".format(ns_green, ew_green)
    return _aggregate(label, rows, [], [], [])


# ---------------------------------------------------------------------------
def _aggregate(name, rows, greens, greens_ns, greens_ew, **extra):
    waits = [float(r["average_waiting_time"]) for r in rows]
    result = {
        "controller": name,
        "seeds": [int(r["seed"]) for r in rows],
        "mean_waiting_time": statistics.fmean(waits),
        "sd_waiting_time": statistics.pstdev(waits) if len(waits) > 1 else 0.0,
        "min_waiting_time": min(waits),
        "max_waiting_time": max(waits),
        "mean_throughput": statistics.fmean(float(r["throughput"]) for r in rows),
        "mean_queue_length": statistics.fmean(
            float(r["average_queue_length"]) for r in rows),
        "mean_stranded": statistics.fmean(
            float(r["vehicles_still_in_network"]) for r in rows),
        "per_seed_waiting": waits,
    }
    if greens:
        counts = collections.Counter(greens)
        total = float(len(greens))
        result["green_histogram"] = {
            "{:g}".format(green): round(counts[green] / total, 4)
            for green in sorted(counts)}
        result["largest_green_share"] = max(counts.values()) / total
        result["mean_green"] = statistics.fmean(greens)
        result["mean_green_ns"] = statistics.fmean(greens_ns) if greens_ns else 0.0
        result["mean_green_ew"] = statistics.fmean(greens_ew) if greens_ew else 0.0
    result.update(extra)
    return result


# ---------------------------------------------------------------------------
def report(dqn: Dict[str, Any], fixed: Dict[str, Any]) -> None:
    print("\n" + "=" * 78)
    print("{:<26} {:>9} {:>7} {:>11} {:>9}".format(
        "controller", "wait", "sd", "throughput", "stranded"))
    print("-" * 78)
    for result in (fixed, dqn):
        print("{:<26} {:>9.3f} {:>7.3f} {:>11.1f} {:>9.1f}".format(
            result["controller"], result["mean_waiting_time"],
            result["sd_waiting_time"], result["mean_throughput"],
            result["mean_stranded"]))
    print("-" * 78)

    delta = dqn["mean_waiting_time"] - fixed["mean_waiting_time"]
    percent = 100.0 * delta / fixed["mean_waiting_time"]
    verdict = "BETTER than" if delta < 0 else "WORSE than"
    print("DQN is {} the fixed plan by {:.3f} s ({:+.1f} %)".format(
        verdict, abs(delta), percent))

    # Paired across seeds: the two controllers saw the same traffic, so the
    # per-seed difference is the honest comparison, not the two means.
    if dqn["seeds"] == fixed["seeds"]:
        diffs = [d - f for d, f in zip(dqn["per_seed_waiting"],
                                       fixed["per_seed_waiting"])]
        wins = sum(1 for d in diffs if d < 0)
        print("paired over {} seeds: DQN wins {}, loses {}, "
              "mean difference {:+.3f} s (sd {:.3f})".format(
                  len(diffs), wins, len(diffs) - wins,
                  statistics.fmean(diffs),
                  statistics.pstdev(diffs) if len(diffs) > 1 else 0.0))

    if "green_histogram" in dqn:
        print("\ngreen durations chosen (ns mean {:.2f} s, ew mean {:.2f} s):"
              .format(dqn["mean_green_ns"], dqn["mean_green_ew"]))
        for green, share in sorted(dqn["green_histogram"].items(),
                                   key=lambda kv: float(kv[0])):
            bar = "#" * int(round(share * 50))
            print("  {:>4} s  {:>6.1%}  {}".format(green, share, bar))
        if dqn["largest_green_share"] > 0.9:
            print("  NOTE: {:.0%} of decisions used one duration - this "
                  "policy is a fixed plan found the slow way."
                  .format(dqn["largest_green_share"]))
    print("=" * 78)


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--scenario", default="heavy")
    parser.add_argument("--seeds", default="6101-6120")
    parser.add_argument("--ns-green", type=float, default=7.0)
    parser.add_argument("--ew-green", type=float, default=6.0)
    parser.add_argument("--tag", default="_eval")
    parser.add_argument("--skip-fixed", action="store_true",
                        help="reuse a previously measured baseline from --out")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    seeds = [int(v) for v in parse_number_list(args.seeds)]

    print("DQN policy: {}".format(args.model))
    dqn = evaluate_agent(args.model, args.scenario, seeds, tag=args.tag)

    print("\nfixed plan ns={:g} ew={:g}, through SUMO's own TLS program:"
          .format(args.ns_green, args.ew_green))
    fixed = evaluate_fixed(args.scenario, seeds, args.ns_green, args.ew_green,
                           tag=args.tag + "f")

    report(dqn, fixed)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({"dqn": dqn, "fixed": fixed}, handle, indent=2,
                      default=float)
        print("saved {}".format(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
