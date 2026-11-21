"""Apply the predeclared Stage 4 gates to confirmatory per-seed outputs."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parent


def rows_for(scenario):
    path = OUT / scenario / "confirmatory" / "confirmatory_rows.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def mean(rows, key):
    return statistics.fmean(float(row[key]) for row in rows)


def assess(scenario, rows, manifest):
    by = {}
    for row in rows:
        by.setdefault(row["controller"], []).append(row)
    required = ("starting_dqn", "incumbent_dqn", "fixed_time_tuned", "actuated_tuned")
    absent = [name for name in required if name not in by]
    if absent:
        return {"scenario": scenario, "promotion": "NOT_ELIGIBLE",
                "reason": "Missing confirmatory controller(s): " + ", ".join(absent),
                "controllers_present": sorted(by)}
    if "stage4_candidate" not in by:
        return {"scenario": scenario, "promotion": "NOT_ELIGIBLE",
                "reason": "No training candidate passed the predeclared validation gate; candidate was not run on confirmatory seeds.",
                "all_controller_means": {
                    name: {metric: mean(values, metric) for metric in
                           ("average_waiting_time", "average_queue_length",
                            "max_queue_length", "throughput", "vehicles_still_in_network")}
                    for name, values in by.items()},
                "recommend_best_overall_controller": min(
                    ((name, mean(values, "average_waiting_time"))
                     for name, values in by.items()), key=lambda item: item[1])[0]}
    old = {int(row["seed"]): row for row in by["incumbent_dqn"]}
    new = {int(row["seed"]): row for row in by["stage4_candidate"]}
    seeds = sorted(set(old) & set(new))
    if len(seeds) != 20:
        raise ValueError(f"Expected exactly 20 paired test seeds for {scenario}; got {len(seeds)}")
    win_count = sum(float(new[s]["average_waiting_time"]) <
                    float(old[s]["average_waiting_time"]) for s in seeds)
    gates = manifest["promotion_gates"]
    comparisons = {
        "mean_waiting_improvement_at_least_2_percent":
            mean(by["stage4_candidate"], "average_waiting_time") <=
            mean(by["incumbent_dqn"], "average_waiting_time") *
            (1 - float(gates["min_wait_improvement_percent"]) / 100),
        "paired_waiting_wins_at_least_11_of_20":
            win_count >= int(gates["min_paired_wait_wins_of_20"]),
        "average_queue_regression_at_most_2_percent":
            mean(by["stage4_candidate"], "average_queue_length") <=
            mean(by["incumbent_dqn"], "average_queue_length") *
            (1 + float(gates["max_average_queue_regression_percent"]) / 100),
        "max_queue_regression_at_most_5_percent":
            mean(by["stage4_candidate"], "max_queue_length") <=
            mean(by["incumbent_dqn"], "max_queue_length") *
            (1 + float(gates["max_max_queue_regression_percent"]) / 100),
        "throughput_regression_at_most_1_percent":
            mean(by["stage4_candidate"], "throughput") >=
            mean(by["incumbent_dqn"], "throughput") *
            (1 - float(gates["max_throughput_regression_percent"]) / 100),
        "zero_blocked_switches":
            sum(int(row["blocked_switches"]) for row in by["stage4_candidate"]) == 0,
        "zero_teleports":
            sum(int(row["teleports"]) for row in by["stage4_candidate"]) == 0,
        "beats_tuned_fixed_time_on_mean_waiting":
            mean(by["stage4_candidate"], "average_waiting_time") <
            mean(by["fixed_time_tuned"], "average_waiting_time"),
        "beats_tuned_pressure_actuated_on_mean_waiting":
            mean(by["stage4_candidate"], "average_waiting_time") <
            mean(by["actuated_tuned"], "average_waiting_time"),
    }
    promotes = all(comparisons[key] for key in list(comparisons)[:7])
    # Report benchmark superiority as a separate claim gate.
    return {
        "scenario": scenario,
        "promotion": "PROMOTE_ELIGIBLE" if promotes else "DO_NOT_PROMOTE",
        "candidate_vs_incumbent": {
            "waiting_mean": mean(by["stage4_candidate"], "average_waiting_time"),
            "waiting_sd": statistics.stdev(float(r["average_waiting_time"])
                                             for r in by["stage4_candidate"]),
            "incumbent_waiting_mean": mean(by["incumbent_dqn"], "average_waiting_time"),
            "waiting_delta": mean(by["stage4_candidate"], "average_waiting_time") -
                             mean(by["incumbent_dqn"], "average_waiting_time"),
            "paired_seed_wins": win_count,
            "average_queue_mean": mean(by["stage4_candidate"], "average_queue_length"),
            "incumbent_average_queue_mean": mean(by["incumbent_dqn"], "average_queue_length"),
            "throughput_mean": mean(by["stage4_candidate"], "throughput"),
            "incumbent_throughput_mean": mean(by["incumbent_dqn"], "throughput"),
            "tuned_fixed_time_wait_mean": mean(by["fixed_time_tuned"], "average_waiting_time"),
            "tuned_pressure_actuated_wait_mean": mean(by["actuated_tuned"], "average_waiting_time"),
            "starting_dqn_wait_mean": mean(by["starting_dqn"], "average_waiting_time"),
        },
        "gates": comparisons,
        "recommend_best_overall_controller": min(
            ((name, mean(values, "average_waiting_time"))
             for name, values in by.items()), key=lambda item: item[1])[0],
        "all_controller_means": {
            name: {metric: mean(values, metric) for metric in
                   ("average_waiting_time", "average_queue_length",
                    "max_queue_length", "throughput", "vehicles_still_in_network")}
            for name, values in by.items()},
        "candidate_seed_regressed_vs_incumbent": [
            seed for seed in seeds if float(new[seed]["average_waiting_time"]) >
            float(old[seed]["average_waiting_time"])],
    }


def main():
    manifest_path = OUT / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    decisions = {}
    for scenario in ("moderate", "heavy"):
        decisions[scenario] = assess(scenario, rows_for(scenario), manifest)
    output = {"decision": "NO_CANDIDATE_PROMOTED",
              "scenarios": decisions,
              "promotion_note": "No automatic checkpoint replacement was performed."}
    if all(value["promotion"] == "PROMOTE_ELIGIBLE"
           for value in decisions.values()):
        output["decision"] = "CANDIDATES_MEET_PROMOTION_GATES"
    decision_path = OUT / "promotion_decision.json"
    decision_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    manifest["confirmatory_decision"] = str(decision_path)
    manifest["confirmatory"] = decisions
    manifest["status"] = "evaluation_complete_no_auto_promotion"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
