"""Rebuild the Stage 4 controller comparison with a correctly executed
tuned fixed-time baseline.

Reads, never writes, the original Stage 4 artifacts. Emits
`promotion_decision_corrected.json` beside this script.

What is corrected
-----------------
`output/dqn_stage4_20261120/stage4_eval.py:131-133,250-253` built the tuned
baseline as `FixedTimeAgent(ns_green=7, ew_green=7, yellow=3.0)` and ran it
through `run_agent_episode`, i.e. inside `IntelliSignalEnv`. The environment
quantises green duration to decision boundaries, so the plan it actually ran
was an 11-second green, not 7. Its `yellow=3.0` also contradicts the scenario
files, which set `signal.yellow: 4`.

The replacement rows in `*_fixed_time_tuned_confirmatory.csv` were measured
with `evaluation/fixed_time_sweep.py`, which runs SUMO's own TLS program on the
same confirmatory seeds (6101-6120) and takes yellow from the scenario.

Every other controller's numbers are taken unchanged from the original Stage 4
confirmatory rows. Only the tuned fixed-time row is replaced, and the original
value is preserved alongside it.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE4 = os.path.dirname(HERE)

METRICS = ("average_waiting_time", "average_queue_length", "max_queue_length",
           "throughput", "vehicles_still_in_network")
SCENARIOS = ("moderate", "heavy")

#: Lower is better for every metric except throughput.
HIGHER_IS_BETTER = {"throughput"}


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def means(rows: List[Dict[str, str]]) -> Dict[str, float]:
    out = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows if row.get(metric) not in (None, "")]
        if values:
            out[metric] = statistics.fmean(values)
    return out


def rank(controllers: Dict[str, Dict[str, float]]) -> str:
    """Best controller by mean waiting time, lower is better."""
    return min(controllers, key=lambda name: controllers[name]["average_waiting_time"])


def build_scenario(scenario: str) -> Dict[str, object]:
    original_rows = read_csv(os.path.join(
        STAGE4, scenario, "confirmatory", "confirmatory_rows.csv"))
    replacement_rows = read_csv(os.path.join(
        HERE, "{}_fixed_time_tuned_confirmatory.csv".format(scenario)))

    by_controller: Dict[str, List[Dict[str, str]]] = {}
    for row in original_rows:
        by_controller.setdefault(row["controller"], []).append(row)

    controllers = {name: means(rows) for name, rows in by_controller.items()}

    superseded = controllers.pop("fixed_time_tuned", None)
    corrected = means(replacement_rows)
    controllers["fixed_time_tuned"] = corrected

    green = float(replacement_rows[0]["green"])
    yellow = float(replacement_rows[0]["yellow"])
    seeds = sorted({int(row["seed"]) for row in replacement_rows})
    original_seeds = sorted({int(row["seed"]) for row in original_rows})

    best = rank(controllers)
    dqn = controllers.get("incumbent_dqn", {})
    gap = None
    if dqn and best != "incumbent_dqn":
        best_wait = controllers[best]["average_waiting_time"]
        gap = (dqn["average_waiting_time"] - best_wait) / best_wait

    return {
        "scenario": scenario,
        "confirmatory_seeds": seeds,
        "seed_sets_match": seeds == original_seeds,
        "tuned_fixed_time_plan": {
            "green_seconds": green,
            "yellow_seconds": yellow,
            "cycle_seconds": float(replacement_rows[0]["cycle"]),
            "execution": "SUMO TLS program via evaluation/fixed_time_sweep.py",
        },
        "superseded_fixed_time_tuned": {
            "means": superseded,
            "why": ("FixedTimeAgent driven through IntelliSignalEnv, which "
                    "quantised the 7 s request to an 11 s green, and ran a 3 s "
                    "yellow where the scenario defines 4 s."),
        },
        "all_controller_means": controllers,
        "recommend_best_overall_controller": best,
        "incumbent_dqn_gap_vs_best": gap,
    }


def main() -> int:
    scenarios = {name: build_scenario(name) for name in SCENARIOS}
    document = {
        "document": "Stage 4 promotion decision, corrected",
        "supersedes": "output/dqn_stage4_20261120/promotion_decision.json",
        "correction": "output/dqn_stage4_20261120/CORRECTION.md",
        "decision": "NO_CANDIDATE_PROMOTED",
        "decision_unchanged": True,
        "what_changed": ("Only the fixed_time_tuned row. No candidate passed the "
                         "validation gate in the original run and none does here; "
                         "the promotion outcome is unaffected. What changes is the "
                         "recommended best overall controller."),
        "scenarios": scenarios,
        "promotion_note": "No automatic checkpoint replacement was performed.",
    }
    out = os.path.join(STAGE4, "promotion_decision_corrected.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=1)
        handle.write("\n")

    for name, data in scenarios.items():
        print("\n=== {} (seeds {}-{}) ===".format(
            name, data["confirmatory_seeds"][0], data["confirmatory_seeds"][-1]))
        ordered = sorted(data["all_controller_means"].items(),
                         key=lambda item: item[1]["average_waiting_time"])
        for controller, values in ordered:
            print("  {:<22s} wait={:>8.3f}  throughput={:>8.2f}".format(
                controller, values["average_waiting_time"], values["throughput"]))
        print("  superseded fixed_time_tuned wait={:.3f}".format(
            data["superseded_fixed_time_tuned"]["means"]["average_waiting_time"]))
        print("  best = {}".format(data["recommend_best_overall_controller"]))
        if data["incumbent_dqn_gap_vs_best"] is not None:
            print("  incumbent_dqn is {:.1%} worse than best".format(
                data["incumbent_dqn_gap_vs_best"]))
    print("\nwrote {}".format(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
