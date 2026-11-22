"""Assemble the published comparison from the measurements already taken.

Every number here was produced by a separate run of the measurement code in
`evaluation/` and `output/dqn_phase_20261122/`; this script only collects
them into one table and one JSON per scenario, so the report and the
promotion record cannot disagree with each other.
"""

from __future__ import annotations

import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))

SEEDS = list(range(6101, 6121))

# Fixed plans, measured through SUMO's own traffic-light program.
FIXED = {
    "heavy": [
        ("fixed 30/30  (Stage 1 baseline)", 53.029, 1.854, 1349.8),
        ("fixed 7/7    (best symmetric)", 12.638, 0.814, 1441.3),
        ("fixed 7/6    (best fixed plan)", 5.754, 0.418, 1466.5),
        ("pattern ns[7,7] ew[5,6]", 5.680, 0.246, 1463.8),
    ],
    "moderate": [
        ("fixed 30/30  (Stage 1 baseline)", 9.863, 0.200, 692.9),
        ("fixed 5/5    (previous baseline)", 3.042, 0.112, 690.0),
        ("fixed 6/4    (best fixed plan)", 2.761, 0.109, 690.8),
    ],
}

TUNED_BASELINE = {"heavy": 5.754, "moderate": 2.761}
WINNER = {"heavy": "R143", "moderate": "M1"}


def load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    everything = {}
    for scenario in ("heavy", "moderate"):
        rows = [{"controller": name, "mean_waiting_time": wait,
                 "sd_waiting_time": sd, "mean_throughput": thr,
                 "family": "fixed_time"}
                for name, wait, sd, thr in FIXED[scenario]]

        stage4 = load(os.path.join(
            HERE, "eval", "stage4_{}.json".format(scenario)))
        rows.append({
            "controller": "DQN Stage 4 (KEEP/SWITCH, deployed)",
            "mean_waiting_time": stage4["mean_waiting_time"],
            "sd_waiting_time": stage4["sd_waiting_time"],
            "mean_throughput": stage4["mean_throughput"],
            "family": "keep_switch_dqn",
            "per_seed_waiting": stage4["per_seed_waiting"],
        })

        winner = load(os.path.join(HERE, "eval",
                                   "{}.json".format(WINNER[scenario])))
        rows.append({
            "controller": "dqn_phase (duration DQN, {})".format(WINNER[scenario]),
            "mean_waiting_time": winner["mean_waiting_time"],
            "sd_waiting_time": winner["sd_waiting_time"],
            "mean_throughput": winner["mean_throughput"],
            "family": "phase_duration",
            "per_seed_waiting": winner["per_seed_waiting"],
            "green_histogram": winner.get("green_histogram"),
            "mean_green_ns": winner.get("mean_green_ns"),
            "mean_green_ew": winner.get("mean_green_ew"),
            "largest_green_share": winner.get("largest_green_share"),
            "model": winner.get("model"),
            "durations": winner.get("durations"),
        })
        # The row `promote_phase_model.py` reads as the baseline.
        baseline = next(r for r in rows
                        if r["controller"].startswith("fixed")
                        and abs(r["mean_waiting_time"]
                                - TUNED_BASELINE[scenario]) < 1e-9)
        rows.append({
            "controller": "fixed_ns7_ew6" if scenario == "heavy"
                          else "fixed_ns6_ew4",
            "mean_waiting_time": baseline["mean_waiting_time"],
            "sd_waiting_time": baseline["sd_waiting_time"],
            "mean_throughput": baseline["mean_throughput"],
            "family": "fixed_time_tuned_alias",
            "seeds": SEEDS,
        })

        rows.sort(key=lambda r: r["mean_waiting_time"])
        everything[scenario] = rows

        out = os.path.join(HERE, "final_{}.json".format(scenario))
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2, default=float)

        print("\n" + "=" * 84)
        print("{} traffic - seeds {}-{}, 1800 s, MetricsCollector".format(
            scenario.upper(), SEEDS[0], SEEDS[-1]))
        print("=" * 84)
        print("{:<40} {:>9} {:>7} {:>12}".format(
            "controller", "wait (s)", "sd", "throughput"))
        print("-" * 84)
        for row in rows:
            if row["family"] == "fixed_time_tuned_alias":
                continue
            print("{:<40} {:>9.3f} {:>7.3f} {:>12.1f}".format(
                row["controller"], row["mean_waiting_time"],
                row["sd_waiting_time"], row["mean_throughput"]))
        print("-" * 84)

        dqn = next(r for r in rows if r["family"] == "phase_duration")
        base = TUNED_BASELINE[scenario]
        print("duration DQN against the best fixed plan: {:.3f} vs {:.3f} "
              "= {:+.1f} %".format(dqn["mean_waiting_time"], base,
                                   100.0 * (dqn["mean_waiting_time"] - base) / base))
        old = next(r for r in rows if r["family"] == "keep_switch_dqn")
        print("duration DQN against the deployed Stage 4 DQN: {:+.1f} %".format(
            100.0 * (dqn["mean_waiting_time"] - old["mean_waiting_time"])
            / old["mean_waiting_time"]))
        print("saved {}".format(out))

    # Robustness across heavy training seeds, measured on the selection set.
    print("\n" + "=" * 84)
    print("HEAVY robustness - every training run, selection seeds 4001-4020")
    print("(fixed 7/6 on those seeds: 6.040 s)")
    print("=" * 84)
    runs = []
    for name in ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8",
                 "R142", "R143", "R144", "H1", "H2"):
        path = os.path.join(HERE, "select", "{}.json".format(name))
        if os.path.exists(path):
            runs.append((name, load(path)["mean_waiting_time"]))
    beat = [n for n, w in runs if w < 6.040]
    print("  runs measured     : {}".format(len(runs)))
    print("  beat the best fixed plan : {} of {}".format(len(beat), len(runs)))
    print("  median waiting    : {:.3f} s".format(
        statistics.median(w for _, w in runs)))
    print("  the one failure   : {}".format(
        ", ".join(n for n, w in runs if w >= 6.040) or "none"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
