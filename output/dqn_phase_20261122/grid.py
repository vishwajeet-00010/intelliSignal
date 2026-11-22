"""Map the fixed-plan landscape on a (ns_green, ew_green) grid.

The demand in this project is generated with SUMO `<flow vehsPerHour=...>`
and no `random="true"`, so vehicles are inserted on an equidistant,
deterministic schedule. That makes the delay surface sharply structured
rather than smooth, and the measured optimum (ns=7 / ew=6, 5.754 s) sits in
a very narrow valley: one second either way roughly doubles the delay.

This script measures that surface so the learned controller has a target it
can be judged against, and so a near-miss is recognised as a near-miss.

One worker handles a slice of the north-south values; several run at once
with different `--tag` values so their generated SUMO inputs never collide.

    python output/dqn_phase_20261122/grid.py --ns 7,8 --ew 4-10 \
        --seeds 6101,6102 --tag w0 --out output/dqn_phase_20261122/grid_w0.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
for path in (PROJECT_ROOT, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from probe import cleanup, make_fixed, parse_seeds, run  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="heavy")
    parser.add_argument("--ns", required=True)
    parser.add_argument("--ew", required=True)
    parser.add_argument("--seeds", default="6101,6102")
    parser.add_argument("--tag", default="g")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    ns_values = [int(v) for v in parse_seeds(args.ns)]
    ew_values = [int(v) for v in parse_seeds(args.ew)]
    seeds = parse_seeds(args.seeds)

    rows = []
    try:
        for ns in ns_values:
            for ew in ew_values:
                policy = make_fixed(ns, ew)
                episodes = [run(policy, args.scenario, seed, tag=args.tag)
                            for seed in seeds]
                waits = [e["average_waiting_time"] for e in episodes]
                row = {
                    "ns_green": ns,
                    "ew_green": ew,
                    "cycle": ns + ew + 8,
                    "wait": round(statistics.fmean(waits), 3),
                    "wait_sd": round(statistics.pstdev(waits), 3),
                    "throughput": round(statistics.fmean(
                        e["throughput"] for e in episodes), 1),
                    "queue": round(statistics.fmean(
                        e["average_queue_length"] for e in episodes), 3),
                    "stranded": round(statistics.fmean(
                        e["vehicles_still_in_network"] for e in episodes), 1),
                    "seeds": len(seeds),
                }
                rows.append(row)
                print("ns={:<3d} ew={:<3d} cycle={:<3d} wait={:>8.3f} "
                      "thr={:>7.1f}".format(ns, ew, row["cycle"], row["wait"],
                                            row["throughput"]), flush=True)
    finally:
        cleanup(args.scenario, args.tag)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
