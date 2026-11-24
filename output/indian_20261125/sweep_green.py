"""Where does the best green sit once startup lost time is modelled?

The question this answers is the whole argument. Until now the measured
optimum for this junction was a seven-second green, which no traffic
engineer would deploy. The suspicion was that the model, not the method,
was wrong: a vehicle with no `startupDelay` begins moving the instant the
signal changes, so a four-second green produced nearly four seconds of
discharge and short cycles won.

With the Indian urban fleet - two-wheelers, cars, autos and buses, each
with a real pause before it moves off - a short green should mostly be
spent waiting for the queue to start. If that is right, the optimum moves
into the range practitioners actually use and the 15 s minimum stops
being a constraint at all.

    python output/indian_20261125/sweep_green.py --greens 8-60 --seeds 2
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from custom_traffic.resolver import workload_from_scenario
from simulation.hybrid import FixedPlan, run_fixed_plan


def parse_list(text):
    out = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if "-" in chunk[1:]:
            a, b = chunk.rsplit("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif chunk:
            out.append(int(chunk))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="indian_arterial")
    parser.add_argument("--greens", default="8,10,12,15,20,25,30,35,40,45,50,60")
    parser.add_argument("--ew", default=None,
                        help="fix the east-west green and sweep north-south")
    parser.add_argument("--duration", type=float, default=900.0)
    parser.add_argument("--seeds", default="6101,6102")
    parser.add_argument("--tag", default="_sw")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    greens = parse_list(args.greens)
    seeds = parse_list(args.seeds)
    workload = workload_from_scenario(args.scenario,
                                      duration_s=args.duration, seed=9001)
    yellow = float(workload.signal.get("yellow", 3.0))
    all_red = float(workload.signal.get("all_red", 2.0))

    print("{} - {} vehicles ({} PCU) over {:.0f} s, clearance {:.0f} s"
          .format(args.scenario, workload.total_vehicles,
                  workload.total_pcu(), args.duration, yellow + all_red))
    print("{:>6} {:>6} {:>7} {:>10} {:>8} {:>11} {:>9}".format(
        "ns", "ew", "cycle", "wait s", "sd", "completed", "pending"))
    print("-" * 64)

    rows = []
    for green in greens:
        ew = int(args.ew) if args.ew else green
        runs = []
        for seed in seeds:
            plan = FixedPlan(ns_green=float(green), ew_green=float(ew),
                             yellow=yellow, all_red=all_red, source="sweep")
            runs.append(run_fixed_plan(workload, plan, seed,
                                       tag="{}{}".format(args.tag, green)))
        waits = [r.mean_waiting_time for r in runs]
        row = {
            "ns_green": green, "ew_green": ew,
            "cycle": green + ew + 2 * (yellow + all_red),
            "wait": round(statistics.fmean(waits), 3),
            "sd": round(statistics.pstdev(waits), 3),
            "completed": round(statistics.fmean(r.completed for r in runs), 1),
            "pending": round(statistics.fmean(r.waiting_to_enter
                                              for r in runs), 1),
            "requested": runs[0].requested,
        }
        rows.append(row)
        print("{:>6} {:>6} {:>7.0f} {:>10.2f} {:>8.2f} {:>11.1f} {:>9.1f}"
              .format(row["ns_green"], row["ew_green"], row["cycle"],
                      row["wait"], row["sd"], row["completed"],
                      row["pending"]), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    best = min(rows, key=lambda r: r["wait"])
    print("\nbest: ns {} / ew {} (cycle {:.0f} s) at {:.2f} s mean waiting"
          .format(best["ns_green"], best["ew_green"], best["cycle"],
                  best["wait"]))
    print("saved", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
