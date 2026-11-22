"""
IntelliSignal
evaluation/calibrate_phase_observation.py

Measures the normalisation scales for `rl/phase_state.py`.

Why this is a script and not a constant
---------------------------------------
The observation divides queues, waiting times and vehicle counts by fixed
scales and clips to [0, 1]. Scales that are too large squash every real
reading into the bottom of the range, and the network then has to resolve the
difference between a five-vehicle queue and a six-vehicle queue inside a few
hundredths of an input. Heavy traffic and moderate traffic differ by more
than a factor of two in demand, so one set of constants cannot serve both:
moderate traffic under heavy traffic's scales uses about a fifth of the range.

This runs a fixed-time plan and reports the 99th percentile of each quantity,
which is what the scales should be set to - high enough that clipping is
rare, low enough that ordinary readings use most of the range.

    python evaluation/calibrate_phase_observation.py --scenario moderate \
        --ns-green 5 --ew-green 4 --seeds 3001-3005
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from typing import Dict, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.fixed_time_sweep import parse_number_list        # noqa: E402
from rl.phase_action import PHASE_EW, PHASE_NS                   # noqa: E402
from rl.phase_environment import PhaseDurationEnv                # noqa: E402
from rl.phase_state import PhaseStateExtractor                   # noqa: E402


def percentile(values: List[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(round(fraction * (len(ordered) - 1))), len(ordered) - 1)
    return float(ordered[index])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="moderate")
    parser.add_argument("--ns-green", type=float, default=7.0)
    parser.add_argument("--ew-green", type=float, default=6.0)
    parser.add_argument("--seeds", default="3001-3005")
    parser.add_argument("--durations", default="4,5,6,7,8,9,10,12,15,20")
    parser.add_argument("--tag", default="_calib")
    args = parser.parse_args(argv)

    durations = [float(d) for d in parse_number_list(args.durations)]
    seeds = [int(s) for s in parse_number_list(args.seeds)]
    ns_index = durations.index(args.ns_green)
    ew_index = durations.index(args.ew_green)

    env = PhaseDurationEnv(scenario=args.scenario, durations=durations,
                           state_extractor=PhaseStateExtractor(
                               max_green_seconds=max(durations)),
                           tag=args.tag)
    samples: Dict[str, List[float]] = {
        "queue_per_direction": [], "waiting_per_direction": [],
        "vehicles_per_direction": [], "queue_per_approach": []}
    try:
        for seed in seeds:
            env.reset(seed=seed)
            while True:
                raw = env.state_extractor.raw(env.connection)
                for phase in (PHASE_NS, PHASE_EW):
                    samples["queue_per_direction"].append(raw["queues"][phase])
                    samples["waiting_per_direction"].append(raw["waiting"][phase])
                    samples["vehicles_per_direction"].append(raw["vehicles"][phase])
                samples["queue_per_approach"].extend(
                    raw["approach_queues"].values())
                action = (ns_index if env.controller.phase == PHASE_NS
                          else ew_index)
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
    finally:
        env.close()
        env.cleanup_generated_inputs()

    print("\n{} under a fixed ns={:g} / ew={:g} plan, {} seeds".format(
        args.scenario, args.ns_green, args.ew_green, len(seeds)))
    print("{:<26} {:>8} {:>8} {:>8} {:>8}".format(
        "quantity", "mean", "p99", "max", "suggest"))
    print("-" * 62)
    suggestion = {}
    for name, values in samples.items():
        p99 = percentile(values, 0.99)
        # Round up to something a human would write down.
        step = 1.0 if p99 < 30 else 10.0
        suggested = max(step, float(int(p99 / step + 1) * step))
        suggestion[name] = suggested
        print("{:<26} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.0f}".format(
            name, statistics.fmean(values), p99, max(values), suggested))

    print("\n--obs-queue {:.0f} --obs-waiting {:.0f} --obs-vehicles {:.0f} "
          "--obs-approach-queue {:.0f}".format(
              suggestion["queue_per_direction"],
              suggestion["waiting_per_direction"],
              suggestion["vehicles_per_direction"],
              suggestion["queue_per_approach"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
