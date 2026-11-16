"""Report completed isolated experiments without running SUMO or changing models."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
METRICS = {
    "average_waiting_time": ("Waiting (s/vehicle)", "lower"),
    "average_queue_length": ("Average queue (vehicles)", "lower"),
    "max_queue_length": ("Maximum queue (vehicles)", "lower"),
    "throughput": ("Completed vehicles", "higher"),
    "vehicles": ("Inserted vehicles", "context"),
    "vehicles_still_in_network": ("Vehicles remaining", "context"),
    "teleports": ("Teleports", "lower"),
}
REQUIRED = ["training_summary.json"] + [
    f"{group}_{model}.json"
    for group in ("reference", "holdout")
    for model in ("candidate", "incumbent")
]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite metric: {value!r}")
    return result


def indexed_rows(payload, controller, seeds, scenario, duration):
    rows = {}
    for row in payload["rows"]:
        if row["controller"] != controller:
            continue
        seed = int(row["seed"])
        if seed in rows:
            raise ValueError(f"Duplicate {controller} seed {seed}")
        if row["scenario"] != scenario:
            raise ValueError(f"Scenario mismatch: {row['scenario']} != {scenario}")
        if not math.isclose(number(row["simulation_time"]), duration):
            raise ValueError(f"Unexpected simulation duration for {scenario}/{seed}")
        for metric in METRICS:
            number(row[metric])
        rows[seed] = row
    if set(rows) != set(seeds):
        raise ValueError(f"Seed mismatch for {scenario}/{controller}: {sorted(rows)}")
    return rows


def sample_std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def describe_pair(scenario, group, incumbent, candidate, fixed, untrained, seeds):
    result = []
    for metric, (label, direction) in METRICS.items():
        old = [number(incumbent[seed][metric]) for seed in seeds]
        new = [number(candidate[seed][metric]) for seed in seeds]
        differences = [b - a for a, b in zip(old, new)]
        old_mean, new_mean = statistics.fmean(old), statistics.fmean(new)
        delta = statistics.fmean(differences)
        improvement = None
        if old_mean and direction != "context":
            improvement = (delta if direction == "higher" else -delta) / old_mean * 100
        result.append({
            "scenario": scenario, "seed_set": group, "metric": metric,
            "label": label, "preferred_direction": direction, "n": len(seeds),
            "incumbent_mean": old_mean, "incumbent_std": sample_std(old),
            "candidate_mean": new_mean, "candidate_std": sample_std(new),
            "candidate_minus_incumbent_mean": delta,
            "paired_difference_std": sample_std(differences),
            "improvement_percent_vs_incumbent": improvement,
            "candidate_lower_seeds": sum(b < a for a, b in zip(old, new)),
            "candidate_equal_seeds": sum(b == a for a, b in zip(old, new)),
            "candidate_higher_seeds": sum(b > a for a, b in zip(old, new)),
            "fixed_time_mean": statistics.fmean(number(fixed[s][metric]) for s in seeds),
            "untrained_mean": statistics.fmean(number(untrained[s][metric]) for s in seeds),
        })
    return result


def mean_history(rows, key):
    values = [number(row[key]) for row in rows if row.get(key) not in (None, "")]
    return statistics.fmean(values) if values else None


def fmt(value):
    return "n/a" if value is None else f"{value:.3f}"


def summarize_scenario(scenario, manifest):
    folder = OUT / scenario
    summary = read_json(folder / "training_summary.json")
    history_path = Path(summary["history_csv"])
    if not history_path.is_absolute():
        history_path = ROOT / history_path
    with history_path.open(newline="", encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    expected_episodes = int(manifest["episodes"])
    if len(history) != expected_episodes or int(summary["episodes"]) != expected_episodes:
        raise ValueError(f"{scenario}: training history is incomplete")
    if [int(r["episode"]) for r in history] != list(range(1, expected_episodes + 1)):
        raise ValueError(f"{scenario}: training episodes are not contiguous")

    rows, sections = [], []
    sections += [f"## {scenario.title()} traffic", "",
                 f"Training: {summary['episodes']} episodes, seed {summary['seed']}, "
                 f"{summary['wall_clock_seconds']:.1f} seconds. Selected `best.pt` is from "
                 f"episode {summary['best_window_episode']} using the existing five-episode "
                 "mean training-return rule. No evaluation results were used to select this checkpoint.", ""]
    for group in ("reference", "holdout"):
        seeds = manifest[f"{group}_seeds"]
        candidate_payload = read_json(folder / f"{group}_candidate.json")
        incumbent_payload = read_json(folder / f"{group}_incumbent.json")
        duration = number(manifest["duration_seconds"])
        args = (seeds, scenario, duration)
        candidate = indexed_rows(candidate_payload, "dqn", *args)
        incumbent = indexed_rows(incumbent_payload, "dqn", *args)
        fixed = indexed_rows(candidate_payload, "fixed_time", *args)
        untrained = indexed_rows(candidate_payload, "dqn_untrained", *args)
        paired = describe_pair(scenario, group, incumbent, candidate, fixed, untrained, seeds)
        rows.extend(paired)
        by_metric = {row["metric"]: row for row in paired}
        wait = by_metric["average_waiting_time"]
        sections += [f"### {group.title()} seeds: {', '.join(map(str, seeds))}", "",
                     "Means over matched traffic seeds; delta is candidate minus incumbent. "
                     "Standard deviations and per-metric seed counts are in `paired_metrics.csv`.", "",
                     "| Metric | Fixed-time | Untrained | Incumbent | Candidate | Delta |",
                     "|---|---:|---:|---:|---:|---:|"]
        for row in paired:
            sections.append("| " + " | ".join([
                row["label"], fmt(row["fixed_time_mean"]), fmt(row["untrained_mean"]),
                fmt(row["incumbent_mean"]), fmt(row["candidate_mean"]),
                fmt(row["candidate_minus_incumbent_mean"]),
            ]) + " |")
        sections += ["", f"Waiting-time wins/ties/losses: **{wait['candidate_lower_seeds']}/"
                     f"{wait['candidate_equal_seeds']}/{wait['candidate_higher_seeds']}**. "
                     f"Mean waiting improvement versus incumbent: "
                     f"**{fmt(wait['improvement_percent_vs_incumbent'])}%**.", "",
                     "| Seed | Incumbent waiting | Candidate waiting | Delta |",
                     "|---|---:|---:|---:|"]
        for seed in seeds:
            old, new = number(incumbent[seed]["average_waiting_time"]), number(candidate[seed]["average_waiting_time"])
            sections.append(f"| {seed} | {old:.3f} | {new:.3f} | {new-old:+.3f} |")
        sections.append("")
        if group == "holdout":
            checks = {
                "Lower mean waiting": wait["candidate_minus_incumbent_mean"] < 0,
                "Waiting wins on more than half of holdout seeds": wait["candidate_lower_seeds"] > len(seeds) / 2,
                "No worse mean queue": by_metric["average_queue_length"]["candidate_minus_incumbent_mean"] <= 1e-9,
                "No worse mean throughput": by_metric["throughput"]["candidate_minus_incumbent_mean"] >= -1e-9,
                "No worse mean teleports": by_metric["teleports"]["candidate_minus_incumbent_mean"] <= 1e-9,
            }
            passed = all(checks.values())
            sections += ["### Promotion assessment", "",
                         "**" + ("PASS: candidate meets every declared holdout guardrail."
                                    if passed else "DO NOT PROMOTE: candidate fails one or more declared holdout guardrails.") + "**", ""]
            sections.extend(f"- {'PASS' if passed_check else 'FAIL'}: {label}." for label, passed_check in checks.items())
            sections += ["", "This report does not replace any model. The original checkpoint remains the dashboard default.", ""]

    sections += ["### Training trend", "",
                 "Descriptive 50-episode windows; higher return means less negative return. "
                 "Traffic seeds and exploration change during training, so these trends do not prove convergence.", "",
                 "| Episodes | Mean return | Mean waiting | Mean queue | Mean loss | Mean epsilon |",
                 "|---|---:|---:|---:|---:|---:|"]
    for start in range(0, len(history), 50):
        window = history[start:start + 50]
        values = [mean_history(window, key) for key in
                  ("total_reward", "average_waiting_time", "average_queue_length", "loss", "epsilon")]
        sections.append(f"| {start+1}–{start+len(window)} | " + " | ".join(map(fmt, values)) + " |")
    previous, last = mean_history(history[-100:-50], "total_reward"), mean_history(history[-50:], "total_reward")
    sections += ["", f"Mean return changed by **{last-previous:+.3f}** between the last two "
                 "50-episode windows. A single run cannot establish that longer training will improve the policy.", "",
                 f"Artifacts: `{scenario}/models/`, `{scenario}/training/`, `{scenario}/evaluation/`, "
                 f"and `{scenario}/plots/`.", ""]
    return rows, sections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("heavy", "moderate"))
    args = parser.parse_args()
    manifest = read_json(OUT / "manifest.json")
    requested = [args.scenario] if args.scenario else ["heavy", "moderate"]
    available = [scenario for scenario in requested
                 if all((OUT / scenario / name).is_file() for name in REQUIRED)]
    if not available:
        print("No requested scenario has completed training and both evaluation sets; no report written.")
        return 2
    missing = [scenario for scenario in requested if scenario not in available]
    report = ["# 300-episode DQN experiment", "",
              "Completed scenarios in this report: **" + ", ".join(available) + "**.", "",
              "Fresh training uses seed 42, traffic seeds 42–341, 300 episodes, and the default "
              "epsilon schedule stretched to 27,000 decisions. This compares complete training "
              "recipes, not a continuation of the old checkpoint or an isolated change in episode count.", "",
              "Reference seeds 42/43/44 overlap training and are reproduction checks. Holdout seeds "
              "1001–1010 are outside the training range. Every evaluation uses 1800 simulated seconds, "
              "greedy actions, and the same metric collector. The candidate comparison includes fixed-time "
              "and an untrained network; the incumbent is evaluated on the same seeds.", "",
              "Declared rule from the experiment manifest: " + manifest["promotion_rule"], "",
              "Only **one training seed** is used per scenario. Holdout traffic seeds assess demand "
              "variability, not reproducibility across independent training runs; no statistical "
              "significance or general superiority is claimed.", "",
              "Inserted and remaining vehicle counts are context metrics: changing admission into the "
              "network can change the remaining count. Read them together with throughput and waiting.", ""]
    if missing:
        report += ["Pending scenarios omitted: " + ", ".join(missing) + ".", ""]
    integrity_path = OUT / "integrity.json"
    if integrity_path.is_file():
        integrity = read_json(integrity_path)
        changed = integrity["changed"]
        report += [f"Protected-artifact check: {integrity['protected_files']} files checked; "
                   f"{len(changed)} changed." + (" Changed paths: " + ", ".join(changed) if changed else ""), ""]
    else:
        report += ["Final protected-artifact integrity check is pending while the experiment runs.", ""]
    all_rows = []
    for scenario in available:
        rows, sections = summarize_scenario(scenario, manifest)
        all_rows.extend(rows)
        report.extend(sections)
    with (OUT / "paired_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    (OUT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Analyzed {', '.join(available)}: {OUT / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
