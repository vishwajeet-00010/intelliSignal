"""Is there any headroom left beyond the symmetric 7 s plan on heavy traffic?

Heavy demand is not symmetric: 850 veh/h on each of north and south against
650 on each of east and west. The best fixed plan found so far gives both
directions 7 s, which was selected from a symmetric sweep only.

If an asymmetric plan beats 12.64 s, then there is something for an adaptive
controller to capture and the DQN is still leaving value on the table. If the
symmetric plan is already best, then a learned controller cannot beat a fixed
plan on this demand by timing alone, and the project's case has to rest on
something else - robustness to demand it was not tuned for.

Runs on a small seed set first, because this is a search, not a final measure.
"""

from __future__ import annotations

import statistics
import sys
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.fixed_time_sweep import (cleanup_generated_inputs,  # noqa: E402
                                         run_fixed_time_plan, write_rows)

SCENARIO = "heavy"
SEEDS = [6101, 6102, 6103, 6104, 6105]

#: north-south green, east-west green. Demand ratio is 850:650 = 1.31, so the
#: proportional split around a 14 s total is roughly 8:6.
PLANS = [
    (7, 7),      # the current best, symmetric
    (8, 6),      # proportional to demand
    (9, 5),
    (8, 7),
    (9, 6),
    (9, 7),
    (10, 6),
    (10, 7),
    (6, 6),
    (8, 8),
    (7, 6),
    (6, 7),
]


def main() -> int:
    rows = []
    try:
        for ns, ew in PLANS:
            for seed in SEEDS:
                rows.append(run_fixed_time_plan(
                    SCENARIO, float(ns), seed,
                    ew_green=float(ew) if ew != ns else None,
                    tag="_asym", quiet=True))
            waits = [float(r["average_waiting_time"]) for r in rows[-len(SEEDS):]]
            tps = [float(r["throughput"]) for r in rows[-len(SEEDS):]]
            print("ns={:>3g}s ew={:>3g}s cycle={:>4g}s  wait={:>8.3f}  tp={:>8.1f}"
                  .format(ns, ew, ns + ew + 8, statistics.fmean(waits),
                          statistics.fmean(tps)), flush=True)
    finally:
        cleanup_generated_inputs(SCENARIO, tag="_asym")

    write_rows(rows, str(OUT / "asymmetric_sweep.csv"))

    grouped = defaultdict(list)
    for row in rows:
        grouped[(float(row["ns_green"]), float(row["ew_green"]))].append(
            float(row["average_waiting_time"]))
    ranked = sorted(grouped.items(), key=lambda kv: statistics.fmean(kv[1]))

    print("\nranked, best first:")
    for (ns, ew), waits in ranked:
        print("  ns={:>3g} ew={:>3g}  wait={:>8.3f}".format(
            ns, ew, statistics.fmean(waits)))
    best, best_waits = ranked[0]
    symmetric = statistics.fmean(grouped[(7.0, 7.0)])
    print("\nbest plan ns={:g} ew={:g} at {:.3f} s".format(
        best[0], best[1], statistics.fmean(best_waits)))
    print("symmetric 7/7 at {:.3f} s".format(symmetric))
    gain = (symmetric - statistics.fmean(best_waits)) / symmetric
    print("headroom beyond the symmetric plan: {:.1%}".format(gain))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
