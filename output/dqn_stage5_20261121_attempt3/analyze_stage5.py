"""Summarize Stage 5 validation and apply frozen promotion gates."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def mean(rows, key):
    return statistics.fmean(float(row[key]) for row in rows)


def analyze(scenario, manifest):
    folder = OUT / scenario
    selection = read_json(folder / "validation" / "validation_selection.json")
    confirm_path = folder / "confirmatory" / "confirmatory_rows.csv"
    confirm_rows = read_rows(confirm_path) if confirm_path.exists() else []
    by = {}
    for row in confirm_rows:
        by.setdefault(row["controller"], []).append(row)
    output = {
        "scenario": scenario,
        "configuration_checks": selection["configuration_checks"],
        "selected_candidate": selection.get("selected_candidate"),
        "confirmatory_candidate_included": "stage5_candidate" in by,
        "confirmatory_rows": str(confirm_path) if confirm_rows else None,
    }
    checks = selection["candidate_checks"]
    paired = {}
    for seed in manifest["seed_plan"]["training"]:
        control = checks.get(f"control_seed{seed}")
        treatment = checks.get(f"treatment_seed{seed}")
        if (control and treatment and control.get("status") == "evaluated" and
                treatment.get("status") == "evaluated"):
            paired[str(seed)] = {
                "treatment_minus_control_wait_seconds":
                    treatment["mean_waiting_time"] - control["mean_waiting_time"],
                "treatment_minus_control_queue":
                    treatment["mean_average_queue"] - control["mean_average_queue"],
                "treatment_minus_control_throughput":
                    treatment["mean_throughput"] - control["mean_throughput"],
                "control_passed": control["passes_validation_gate"],
                "treatment_passed": treatment["passes_validation_gate"],
            }
    output["paired_training_seed_validation_deltas"] = paired
    if paired:
        output["mean_paired_treatment_minus_control_wait_seconds"] = statistics.fmean(
            item["treatment_minus_control_wait_seconds"] for item in paired.values())
        output["treatment_wait_wins_of_five"] = sum(
            item["treatment_minus_control_wait_seconds"] < 0 for item in paired.values())

    if "incumbent_dqn" in by:
        output["confirmatory_controller_means"] = {
            name: {metric: mean(rows, metric) for metric in
                   ("average_waiting_time", "average_queue_length", "max_queue_length",
                    "throughput", "vehicles_still_in_network")}
            for name, rows in by.items()}
        candidate = by.get("stage5_candidate")
        if candidate:
            current = by["incumbent_dqn"]
            old = {int(row["seed"]): row for row in current}
            new = {int(row["seed"]): row for row in candidate}
            seeds = sorted(set(old) & set(new))
            wins = sum(float(new[seed]["average_waiting_time"]) <
                       float(old[seed]["average_waiting_time"]) for seed in seeds)
            gates = manifest["promotion_gates"]
            promotion_gates = {
                "all_20_sealed_seed_pairs_present":
                    seeds == list(manifest["seed_plan"]["confirmatory"]),
                "mean_wait_improvement_at_least_2_percent":
                    mean(candidate, "average_waiting_time") <=
                    mean(current, "average_waiting_time") * .98,
                "paired_waiting_wins_at_least_11_of_20": wins >= gates["min_paired_wait_wins_of_20"],
                "average_queue_regression_at_most_2_percent":
                    mean(candidate, "average_queue_length") <= mean(current, "average_queue_length") * 1.02,
                "max_queue_regression_at_most_5_percent":
                    mean(candidate, "max_queue_length") <= mean(current, "max_queue_length") * 1.05,
                "throughput_regression_at_most_1_percent":
                    mean(candidate, "throughput") >= mean(current, "throughput") * .99,
                "zero_blocked_switches":
                    sum(int(row.get("blocked_switches") or 0) for row in candidate) == 0,
                "zero_teleports":
                    sum(int(row.get("teleports") or 0) for row in candidate) == 0,
                "full_test_suite_passed": manifest.get("tests", {}).get("status") == "passed",
            }
            output["candidate_confirmatory"] = {
                "waiting_mean": mean(candidate, "average_waiting_time"),
                "incumbent_waiting_mean": mean(current, "average_waiting_time"),
                "paired_waiting_wins": wins,
                "paired_seeds": len(seeds), "gates": promotion_gates,
                "promotion_eligible": all(promotion_gates.values()),
                "beats_tuned_fixed_time_waiting": mean(candidate, "average_waiting_time") <
                    mean(by["fixed_time_tuned"], "average_waiting_time"),
                "beats_tuned_pressure_waiting": mean(candidate, "average_waiting_time") <
                    mean(by["actuated_tuned"], "average_waiting_time"),
            }
            output["best_waiting_controller"] = min(
                ((name, mean(rows, "average_waiting_time")) for name, rows in by.items()),
                key=lambda pair: pair[1])[0]
    else:
        output["confirmatory_status"] = "not_run"
    if selection.get("selected_candidate") is None:
        output["scenario_decision"] = "NO_VALIDATION_ELIGIBLE_CHECKPOINT"
    elif output.get("candidate_confirmatory", {}).get("promotion_eligible"):
        output["scenario_decision"] = "PROMOTION_GATES_PASS_MANUAL_REVIEW_REQUIRED"
    elif output.get("confirmatory_candidate_included"):
        output["scenario_decision"] = "DO_NOT_PROMOTE"
    else:
        output["scenario_decision"] = "CONFIRMATORY_PENDING"
    return output


def main():
    manifest_path = OUT / "experiment_manifest.json"
    manifest = read_json(manifest_path)
    scenarios = {name: analyze(name, manifest) for name in ("moderate", "heavy")
                 if (OUT / name / "validation" / "validation_selection.json").exists()}
    report = {"scenarios": scenarios, "next_stage_allowed": False,
              "stage_signoff": "requires_explicit_user_confirmation_after_completion"}
    path = OUT / "stage5_decision.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest["decision"] = str(path)
    manifest["scenario_decisions"] = scenarios
    if len(scenarios) == 2 and all(value["scenario_decision"] != "CONFIRMATORY_PENDING"
                                   for value in scenarios.values()):
        manifest["status"] = "evaluation_complete_no_automatic_promotion"
    else:
        manifest["status"] = "stage5_incomplete_or_evaluation_pending"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
