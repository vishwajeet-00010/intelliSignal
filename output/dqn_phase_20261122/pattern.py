"""Search repeating multi-cycle green patterns.

A fixed-time plan gives the same green every cycle. A learned controller does
not have to: it sees the queue state, which under this project's deterministic
`<flow vehsPerHour=...>` demand repeats on a cycle of its own, so it can serve
7 s, then 7 s, then 6 s, and so on.

This script measures whether any such repeating pattern beats the best single
fixed plan. It sets the ceiling the DQN should be judged against:

  * if the best pattern is clearly better than ns=7/ew=6, a learned policy
    has something real to find and the target is that number
  * if nothing beats it, then the fixed plan is at the optimum of this whole
    family and no amount of training will pass it

    python output/dqn_phase_20261122/pattern.py --ns-pool 6,7,8 \
        --ew-pool 5,6,7 --length 2 --seeds 6101 --tag p0 --out patterns.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
for path in (PROJECT_ROOT, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from probe import PHASE_NS, cleanup, parse_seeds, run  # noqa: E402


def make_pattern(ns_pattern, ew_pattern):
    """A controller that cycles through a fixed list of greens per direction."""
    counters = {"ns": 0, "ew": 0}

    def policy(conn, phase, state):
        if phase == PHASE_NS:
            value = ns_pattern[counters["ns"] % len(ns_pattern)]
            counters["ns"] += 1
        else:
            value = ew_pattern[counters["ew"] % len(ew_pattern)]
            counters["ew"] += 1
        return float(value)

    policy.name = "ns[{}] ew[{}]".format(
        ",".join(str(v) for v in ns_pattern),
        ",".join(str(v) for v in ew_pattern))
    policy.reset = lambda: counters.update(ns=0, ew=0)
    return policy


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="heavy")
    parser.add_argument("--ns-pool", default="6,7,8")
    parser.add_argument("--ew-pool", default="5,6,7")
    parser.add_argument("--length", type=int, default=2)
    parser.add_argument("--seeds", default="6101")
    parser.add_argument("--tag", default="p")
    parser.add_argument("--shard", default="0/1",
                        help="i/N - this worker takes every Nth pattern")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    ns_pool = [int(v) for v in parse_seeds(args.ns_pool)]
    ew_pool = [int(v) for v in parse_seeds(args.ew_pool)]
    seeds = parse_seeds(args.seeds)
    index, workers = (int(v) for v in args.shard.split("/"))

    combos = [
        (ns, ew)
        for ns in itertools.product(ns_pool, repeat=args.length)
        for ew in itertools.product(ew_pool, repeat=args.length)
    ]
    # A rotation of a pattern is the same cycle started at a different point;
    # keeping only the lexicographically smallest rotation of each removes
    # that duplication without removing any distinct pattern.
    def canonical(pattern):
        rotations = [tuple(pattern[i:] + pattern[:i])
                     for i in range(len(pattern))]
        return min(rotations)

    seen = set()
    unique = []
    for ns, ew in combos:
        key = (canonical(list(ns)), canonical(list(ew)))
        if key in seen:
            continue
        seen.add(key)
        unique.append((ns, ew))

    mine = unique[index::workers]
    print("{} patterns total, {} after removing rotations, {} for this worker"
          .format(len(combos), len(unique), len(mine)), flush=True)

    rows = []
    try:
        for ns, ew in mine:
            policy = make_pattern(list(ns), list(ew))
            episodes = []
            for seed in seeds:
                policy.reset()
                episodes.append(run(policy, args.scenario, seed, tag=args.tag))
            waits = [e["average_waiting_time"] for e in episodes]
            row = {
                "ns_pattern": "-".join(str(v) for v in ns),
                "ew_pattern": "-".join(str(v) for v in ew),
                "mean_cycle": (sum(ns) / len(ns)) + (sum(ew) / len(ew)) + 8,
                "wait": round(statistics.fmean(waits), 3),
                "wait_sd": round(statistics.pstdev(waits), 3),
                "throughput": round(statistics.fmean(
                    e["throughput"] for e in episodes), 1),
                "stranded": round(statistics.fmean(
                    e["vehicles_still_in_network"] for e in episodes), 1),
                "seeds": len(seeds),
            }
            rows.append(row)
            print("{:<12} {:<12} wait={:>8.3f} thr={:>7.1f}".format(
                row["ns_pattern"], row["ew_pattern"], row["wait"],
                row["throughput"]), flush=True)
    finally:
        cleanup(args.scenario, args.tag)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    rows.sort(key=lambda r: r["wait"])
    print("\nbest 10 for this worker:")
    for row in rows[:10]:
        print("  {:>8.3f}  ns[{}] ew[{}]".format(
            row["wait"], row["ns_pattern"], row["ew_pattern"]))
    print("saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
