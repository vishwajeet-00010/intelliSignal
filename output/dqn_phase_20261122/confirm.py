"""Confirm the promising green patterns on the full confirmatory seed set.

The pattern search ranks on a single seed, which is enough to find the
candidates and not enough to believe the ordering. This re-measures the
shortlist on seeds 6101-6120 - the same seeds the fixed-time baseline of
5.754 s was measured on - so the ceiling a learned policy is judged against
is itself measured properly.

    python output/dqn_phase_20261122/confirm.py --shard 0/8 --tag c0 \
        --out output/dqn_phase_20261122/confirm_0.csv
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

from pattern import make_pattern            # noqa: E402
from probe import cleanup, parse_seeds, run  # noqa: E402

#: The shortlist, plus the plain fixed plan as the row everything is read
#: against. "7-7 / 6-6" is the fixed 7/6 plan written as a pattern, and it
#: must come out at the fixed plan's number - it is the control.
SHORTLIST = [
    ([7, 7], [6, 6]),          # = fixed ns7/ew6, the control
    ([7, 7], [5, 6]),
    ([7, 7], [5, 7]),
    ([7, 7], [4, 7]),
    ([6, 7], [5, 5]),
    ([6, 7], [5, 6]),
    ([7, 8], [5, 6]),
    ([7, 8], [5, 5]),
    ([5, 7], [4, 5]),
    ([6, 8], [5, 5]),
    ([7, 9], [4, 6]),
    ([5, 8], [5, 5]),
    ([7, 7], [5, 5]),
    ([7, 7], [4, 6]),
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="heavy")
    parser.add_argument("--seeds", default="6101-6120")
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--tag", default="c")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    seeds = parse_seeds(args.seeds)
    index, workers = (int(v) for v in args.shard.split("/"))
    mine = SHORTLIST[index::workers]

    rows = []
    try:
        for ns, ew in mine:
            policy = make_pattern(ns, ew)
            episodes = []
            for seed in seeds:
                policy.reset()
                episodes.append(run(policy, args.scenario, seed, tag=args.tag))
            waits = [e["average_waiting_time"] for e in episodes]
            row = {
                "ns_pattern": "-".join(str(v) for v in ns),
                "ew_pattern": "-".join(str(v) for v in ew),
                "wait": round(statistics.fmean(waits), 3),
                "wait_sd": round(statistics.pstdev(waits), 3),
                "wait_min": round(min(waits), 3),
                "wait_max": round(max(waits), 3),
                "throughput": round(statistics.fmean(
                    e["throughput"] for e in episodes), 1),
                "queue": round(statistics.fmean(
                    e["average_queue_length"] for e in episodes), 3),
                "stranded": round(statistics.fmean(
                    e["vehicles_still_in_network"] for e in episodes), 1),
                "seeds": len(seeds),
            }
            rows.append(row)
            print("ns[{}] ew[{}]  wait {:>7.3f} (sd {:.3f})  thr {:>7.1f}"
                  .format(row["ns_pattern"], row["ew_pattern"], row["wait"],
                          row["wait_sd"], row["throughput"]), flush=True)
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
