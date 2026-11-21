"""Build a concise Markdown report and PDF content from frozen experiment data.

This script analyzes saved artifacts only. It never launches SUMO, changes a
checkpoint, or chooses a recipe. Run after final analysis and independent review.
"""
from pathlib import Path
import csv
import json

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
SCENARIOS = ("heavy", "moderate")
LABELS = {
    "average_waiting_time": "Waiting (s/vehicle)",
    "average_queue_length": "Mean queue (vehicles)",
    "max_queue_length": "Maximum queue",
    "throughput": "Completed vehicles",
    "vehicles_departed": "Inserted vehicles",
    "vehicles_still_in_network": "Remaining vehicles",
    "teleports": "Teleports",
}


def read(name):
    return json.loads((RUN / name).read_text(encoding="utf-8"))


def rows(name):
    with (RUN / name).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def fmt(value, digits=3):
    return f"{float(value):.{digits}f}"


def md_table(headers, body):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(str(x) for x in row) + " |" for row in body],
    ])


def text(value, kind="body"):
    return {"type": kind, "text": value}


def table(headers, body, widths=None):
    return {"type": "table", "headers": headers, "rows": body,
            **({"widths": widths} if widths else {})}


def main():
    result = read("final_decision.json")
    selection = read("selection.json")
    diagnostics = read("diagnostics.json")
    frozen = read("freeze.json")
    paired = rows("paired_final_metrics.csv")
    variability = rows("training_traffic_variability.csv")
    validation = rows("validation_metrics.csv")
    agreement_rows = rows("policy_agreement.csv")
    checks = {(r["scenario"], r["comparison"], r["metric"]): r for r in paired}
    diag = {(r["scenario"], r["policy"]): r for r in diagnostics["policies"]}
    trained = [s for s in SCENARIOS if selection[s]["new_training_needed"]]
    validation_rows = []
    for scenario in SCENARIOS:
        for policy in ("incumbent_best", "incumbent_final", "prior_best", "prior_final", "prior_ep100", "prior_ep200"):
            group = [r for r in validation if r["scenario"] == scenario and r["policy"] == policy]
            means = [sum(float(r[k]) for r in group) / len(group)
                     for k in ("waiting_unrounded", "queue_unrounded", "throughput")]
            eligible = selection[scenario]["all_candidate_gates"].get(policy)
            validation_rows.append([scenario, policy, *[fmt(v) for v in means],
                                    "baseline" if eligible is None else ("PASS" if eligible["eligible"] else "FAIL")])
    if trained:
        training_note = (
            "Fresh control and lower-learning-rate recipes use independent training seeds "
            "42, 43 and 44, 300 episodes per run, 1800 seconds per episode, and "
            "the same 27000-decision exploration schedule. Only learning rate changes. "
            "The report keeps training-seed uncertainty separate from traffic-seed uncertainty."
        )
    else:
        training_note = (
            "No new training was run: existing checkpoints passed the validation gate "
            "in both scenarios, satisfying the predeclared condition for avoiding retraining. "
            "The conditional control/lower-learning-rate experiment was therefore not triggered. "
            "The evaluated historical checkpoints use training seed 42 only; training-seed "
            "variability is unavailable, not zero."
        )
    outcome = "Incumbent models remain unchanged. " + " ".join(
        f"{s.capitalize()}: " + (
            "passes the declared final criteria; eligible for human consideration only."
            if result[s]["promotion_criteria_passed"] else
            "fails the declared final criteria; retain the incumbent."
        ) for s in SCENARIOS
    )
    final_rows = []
    pdf_final_rows = []
    role_names = {"incumbent": "Incumbent", "selected": "Frozen challenger",
                  "fixed_time": "Fixed-time", "untrained": "Untrained (seed 42)"}
    if "untrained_historical0" in result["heavy"]["means"]:
        role_names["untrained_historical0"] = "Untrained (seed 0)"
    for scenario in SCENARIOS:
        for role, label in role_names.items():
            m = result[scenario]["means"][role]
            vals = [fmt(m[k]) for k in (
                "average_waiting_time", "average_queue_length", "throughput",
                "vehicles_departed", "vehicles_still_in_network", "teleports")]
            final_rows.append([scenario, label, *vals])
            pdf_final_rows.append([f"{scenario.capitalize()} / {label}", *vals])
    paired_rows = []
    for scenario in SCENARIOS:
        for metric, label in LABELS.items():
            p = checks[scenario, "selected_minus_incumbent", metric]
            paired_rows.append([scenario.capitalize(), label,
                                f"{float(p['paired_mean_delta']):+.3f}",
                                fmt(p["paired_traffic_seed_sd"])])
    gates = []
    for scenario in SCENARIOS:
        c = result[scenario]["incumbent_comparison"]
        gates.append([scenario.capitalize(), result[scenario]["selected_recipe"],
                      f"{c['waiting_wins']}/{c['waiting_ties']}/{c['waiting_losses']}",
                      ", ".join(k for k, v in c["checks"].items() if not v) or "None"])
    good_bad = []
    for scenario in SCENARIOS:
        r = result[scenario]
        m = r["means"]
        improvement = 100 * (m["fixed_time"]["average_waiting_time"] -
                             m["selected"]["average_waiting_time"]) / m["fixed_time"]["average_waiting_time"]
        c = r["incumbent_comparison"]
        bad = ", ".join(k.replace("_", " ") for k, v in c["checks"].items() if not v)
        delta = c['mean_deltas']
        good_bad.append(
            f"<b>{scenario.capitalize()} - good:</b> frozen-challenger waiting is "
            f"{abs(improvement):.2f}% {'below' if improvement >= 0 else 'above'} fixed-time on the final seeds. "
            f"Against the incumbent, waiting changes by {delta['average_waiting_time']:+.3f} s "
            f"and completed vehicles by {delta['throughput']:+.2f}. "
            f"Maximum queue changes by {delta['max_queue_length']:+.2f} vehicles; "
            f"throughput is {abs(r['fixed_time_comparison']['mean_deltas']['throughput']):.2f} "
            f"{'below' if r['fixed_time_comparison']['mean_deltas']['throughput'] < 0 else 'above'} fixed-time. "
            f"<b>Bad / limit:</b> " + (
                f"failed incumbent guardrails: {bad}." if bad else
                "passing this finite test does not establish general superiority."
            )
        )
    diagnostic_rows = []
    q_rows = []
    reward_notes = []
    for scenario in SCENARIOS:
        for policy in ("incumbent_best", selection[scenario]["candidate"]):
            d = diag[scenario, policy]
            short_name = {"incumbent_best": "Incumbent best", "incumbent_final": "Incumbent final",
                          "prior_best": "Prior best", "prior_final": "Prior final",
                          "prior_ep100": "Prior 100", "prior_ep200": "Prior 200"}[policy]
            diagnostic_rows.append([
                scenario.capitalize() + " / " + short_name,
                str(d["requested_switches"]), str(d["effective_switches"]),
                str(d["blocked_switches"]), str(d["forced_switches"]),
                f"{d['completed_green_duration_mean']:.1f} "
                f"({d['completed_green_duration_min']:.0f}-{d['completed_green_duration_max']:.0f})",
            ])
            clips = d["clip_fraction_by_dim"]
            q_rows.append([
                scenario.capitalize() + " / " + short_name,
                f"{100 * max(clips[:4]):.2f}% / {100 * max(clips[4:8]):.2f}%",
                f"{100 * clips[9]:.2f}%",
                f"{d['q_min']:.2f} to {d['q_max']:.2f}",
                f"{d['q_gap_mean']:.3f}",
                f"{d['td_abs_mean']:.3f} / {d['td_abs_p95']:.3f}",
            ])
        d = diag[scenario, "incumbent_best"]
        corr = diagnostics["reward_correlations"][scenario]
        reward_notes.append(
            f"{scenario.capitalize()}: current-waiting penalties account for "
            f"{100*d['waiting_fraction_of_reward_cost']:.2f}% of incumbent reward cost. "
            f"Descriptive reward correlations: waiting {corr['reward_vs_waiting_pearson']:.3f}, "
            f"queue {corr['reward_vs_queue_pearson']:.3f}, throughput "
            f"{corr['reward_vs_throughput_pearson']:.3f}."
        )
    agreement = selection["moderate"]["incumbent_prior_best_effective_common_observation_agreement"]
    agreement_note = (
        f"Moderate incumbent/prior-best effective actions agree on {100*agreement:.3f}% "
        "of common-observation probes from their two rollout corpora. This measures "
        "decisions in the tested states, not equality on all possible states. "
        "weight_comparison.json separately records tensor and closed-loop trace comparisons."
    )
    weights = read("weight_comparison.json")["moderate"]
    if not weights["incumbent_prior_best_all_policy_tensors_equal"]:
        agreement_note += " Their weights differ; identical effective decisions do not imply identical networks."
    final_probe_rows = [r for r in agreement_rows if r["scenario"] == "moderate"
                        and r["a"] == "incumbent_best" and r["b"] == "incumbent_final"]
    if final_probe_rows:
        probe_count = sum(int(r["decisions"]) for r in final_probe_rows)
        final_agreement = sum(int(r["decisions"]) * float(r["effective_agreement"])
                              for r in final_probe_rows) / probe_count
        agreement_note += (
            f" Best versus incumbent-final effective agreement is {100*final_agreement:.3f}% "
            f"over {probe_count} evaluated state visits across all diagnostic rollouts; "
            "the best-checkpoint comparison does not describe every moderate policy."
        )
    interpretation = (
        "Measured: reward uses interval-average current waiting on incoming lanes; "
        "reported waiting uses accumulated waiting per entered vehicle. Queue samples "
        "are shared, but throughput is not rewarded directly. "
        f"There were {diagnostics['rank_reversal_count']} paired checkpoint/seed reward-waiting "
        "ranking reversals in this validation corpus. Possible explanation: these objective "
        "differences and similar action choices can limit the benefit of more training. "
        "The measurements do not prove the cause of any performance difference."
    )
    methodology = (
        "Validation: seeds 2001-2010, greedy actions, six predeclared existing checkpoints "
        "per scenario plus fixed-time and seed-42 untrained controls. The repository seed "
        "audit found no prior recorded use of validation or final seeds; deleted/external "
        "runs cannot be ruled out. Historical seeds 1001-1010 were not treated as untouched. "
        "Training traffic, if used, is confined to seeds 42-343."
    )
    selection_note = (
        "Validation eligibility requires lower mean waiting, more than 5/10 waiting wins, "
        "and no worse mean queue, throughput or teleports against the incumbent. Eligible "
        "checkpoints rank by waiting, queue, throughput and declared order. If none qualify, "
        "the challenger is diagnostic only. No selection uses exploratory training return alone."
    )
    final_note = (
        f"freeze.json fixes every checkpoint hash, source hash and recipe before the "
        f"{frozen['evaluation_budget_episodes']} final evaluation episodes on seeds 3001-3020. "
        "Promotion requires lower waiting, more than 10/20 waiting wins and no worse mean "
        "queue, throughput or teleports. A newly trained recipe must also beat its matched "
        "validation-selected control, improve in at least two of three training seeds, and "
        "have a negative upper bound for the predeclared 95% t interval across training-seed "
        "mean waiting differences. No thresholds were relaxed; no model is replaced automatically."
    )
    selected_paths = "Frozen challengers: " + "; ".join(
        f"{scenario}: `{next(e['path'] for e in frozen['entries'][scenario] if e.get('role') == 'selected')}`"
        for scenario in SCENARIOS
    ) + ". Full checkpoint hashes are in freeze.json. Final results do not reopen the training or selection branch."
    limits = (
        "Traffic SD describes variability over 20 matched traffic seeds. Shared traffic "
        "seeds and multiple checkpoints are correlated observations. They do not create "
        "independent training replicates. Fixed-time waiting/queue retain the collector's "
        "two-decimal per-run precision; learned-policy guardrails use unrounded values. "
        "Inserted and remaining vehicles are context measures and must be read with "
        "throughput. Zero observed teleports is a simulator result."
    )
    diag_note = (
        "Each diagnostic row covers 1800 decisions over ten validation episodes. "
        "Green values are completed durations; censored final greens are retained in "
        "diagnostics.json. Clip rates above are the maximum across the four queue/wait "
        "dimensions, not pooled rates. Exact-boundary saturation is separate. The binary "
        "phase is categorical. Q gaps and TD values are descriptive; TD is offline residual "
        "error under the stored target network and checkpoint gamma, bootstrapping at "
        "time limits. Original training-minibatch TD errors were not logged."
    )
    integrity = read("integrity.json")
    preservation = (
        f"Protected files: {integrity['protected_files']}; changed: {len(integrity['changed'])}; "
        f"missing: {len(integrity['missing'])}. Scenario settings, safety logic, baselines, "
        "incumbent models and all prior experiment artifacts were preserved. Greedy "
        "evaluation verifies RNG, weights, replay and learning-counter invariance. "
        "Independent review and reproducible full-state/retry checks are retained."
    )
    variability_rows = []
    for scenario in SCENARIOS:
        for label in role_names:
            group = [r for r in variability if r["scenario"] == scenario and r["label"] == label
                     and r["metric"] == "average_waiting_time"]
            if group:
                variability_rows.append([scenario, role_names[label],
                                         fmt(group[0]["traffic_seed_mean"]),
                                         fmt(group[0]["traffic_seed_sd"])])
        uncertainty = result[scenario].get("training_uncertainty")
        if uncertainty:
            reward_notes.append(
                f"{scenario.capitalize()} training-seed waiting deltas vs control: "
                f"{uncertainty['training_seed_waiting_deltas']}; mean {uncertainty['mean']:.3f}, "
                f"SD {uncertainty['sample_sd']:.3f}, 95% t interval {uncertainty['ci95_t_df2']}."
            )
    md = [
        "# IntelliSignal controlled experiment - 19-20 November 2026", "", outcome, "",
        "## Frozen final comparison", "", "Twenty paired final traffic seeds, 1800 seconds per episode.", "",
        md_table(["Scenario", "Controller", "Waiting", "Queue", "Completed", "Inserted", "Remaining", "Teleports"], final_rows), "",
        md_table(["Scenario", "Frozen challenger", "Waiting W/T/L", "Failed incumbent criteria"], gates), "",
        "## Paired differences and variability", "", "Challenger minus incumbent; SD is over paired traffic-seed differences.", "",
        md_table(["Scenario", "Metric", "Mean delta", "Paired SD"], paired_rows), "",
        md_table(["Scenario", "Controller", "Waiting mean", "Traffic SD"], variability_rows), "",
        training_note, "", limits, "", "## Diagnosis before training", "",
        md_table(["Scenario / model", "Requested switches", "Effective switches", "Blocked", "Forced", "Green mean (min-max) s"], diagnostic_rows), "",
        md_table(["Scenario / model", "Max queue / wait clip rate", "Elapsed clip", "Q range", "Mean Q gap", "Abs TD mean / p95"], q_rows), "",
        diag_note, "", *reward_notes, "", agreement_note, "", interpretation, "",
        "## Selection, safeguards and decision", "", methodology, "", selection_note, "", final_note, "", selected_paths, "",
        md_table(["Scenario", "Validation checkpoint", "Waiting", "Queue", "Completed", "Gate"], validation_rows), "",
        preservation, "", "Good and bad: " + " ".join(s.replace("<b>", "").replace("</b>", "") for s in good_bad), "",
        "## Reproduce and inspect", "",
        "See COMMANDS.md for exact execution and verification commands. Primary artifacts: "
        "manifest.json, selection.json, freeze.json, diagnostics.json, weight_comparison.json, "
        "validation_metrics.csv, final_metrics.csv, paired_final_metrics.csv, "
        "training_traffic_variability.csv, final_decision.json, integrity.json and REVIEW.md. "
        "Every per-seed measurement and decision trace is retained under validation/ and final/.", "",
    ]
    (RUN / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    panels = []
    for scenario in SCENARIOS:
        untrained_chart_role = "untrained_historical0" if "untrained_historical0" in role_names else "untrained"
        chart_rows = [{"label": role_names[r].replace("Frozen challenger", "Challenger"),
                       "value": result[scenario]["means"][r]["average_waiting_time"]}
                      for r in ("fixed_time", "incumbent", "selected", untrained_chart_role)]
        maximum = 120 if scenario == "heavy" else 15
        panels.append({"title": scenario.capitalize() + " traffic", "maximum": maximum, "rows": chart_rows})
    pages = [
        {"title": "IntelliSignal<br/>Controlled comparison", "subtitle": "Good results, bad results, and the evidence for the decision",
         "blocks": [text(outcome), {"type": "chart", "panels": panels},
                    text("Frozen final test: 20 paired traffic seeds", "heading"),
                    table(["Scenario / controller", "Wait s", "Queue", "Done", "In", "Left", "TP"],
                          pdf_final_rows, [31, 13, 12, 11, 11, 11, 8]),
                    text("Done: completed; In: inserted; Left: remaining; TP: teleports. Means across 1800-second episodes. Read admission and remaining counts together with throughput.", "small"),
                    text("Untrained initialization 0 reproduces the historical control definition. Initialization 42 is the additional diagnostic control; both are preserved. Neither control affects selection or promotion.", "small")]},
        {"title": "The good and bad", "subtitle": "Challenger minus incumbent; negative waiting and queue deltas are preferable",
         "blocks": [*[text(t) for t in good_bad],
                    table(["Scenario", "Metric", "Mean delta", "Paired SD"], paired_rows, [18, 41, 20, 21]),
                    table(["Scenario", "Waiting wins / ties / losses", "Failed criteria"],
                          [[g[0], g[2], g[3].replace("_", " ")] for g in gates], [18, 35, 47]),
                    text(limits, "small")]},
        {"title": "What the policies do", "subtitle": "Measured on validation traffic before any new training",
         "blocks": [table(["Scenario / model", "Req. switch", "Effective", "Blocked", "Forced", "Green mean (range), s"],
                          diagnostic_rows, [28, 12, 12, 12, 12, 24]),
                    table(["Scenario / model", "Queue / wait clip", "Elapsed clip", "Q range", "Q gap", "Abs TD mean / p95"],
                          q_rows, [25, 18, 13, 17, 10, 17]),
                    text(diag_note, "small"), text(agreement_note),
                    *[text(t, "small") for t in reward_notes], text(interpretation)]},
        {"title": "How to trust the result", "subtitle": "Fixed rules, preserved originals, and reproducible evidence",
         "blocks": [text("Selection and final-test separation", "heading"),
                    text(methodology), text(selection_note), text(final_note),
                    text("Checks and limitations", "heading"), text(preservation),
                    text(training_note),
                    text("Evidence and commands", "heading"),
                    text("Experiment: <b>experiments/controlled_20261119/</b><br/>"
                         "REPORT.md and final_decision.json: outcomes and criteria.<br/>"
                         "paired_final_metrics.csv and training_traffic_variability.csv: paired effects and variability.<br/>"
                         "diagnostics.json and weight_comparison.json: reward, actions, clipping, Q and TD evidence.<br/>"
                         "manifest.json, selection.json and freeze.json: predeclared rules and frozen hashes.<br/>"
                         "REVIEW.md and integrity.json: independent review and preservation.<br/>"
                         "COMMANDS.md: exact commands; verification scripts inspect saved records.<br/>"
                         "validation/ and final/: complete per-seed measurements and traces.", "small"),
                    text("No monotonic improvement is promised. Training completion and increased reward are not promotion evidence.", "small")]},
    ]
    (RUN / "pdf_content.json").write_text(json.dumps({"pages": pages}, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(RUN / 'REPORT.md'), "pdf_content": str(RUN / 'pdf_content.json'),
                      "final_episodes": frozen['evaluation_budget_episodes']}))


if __name__ == "__main__":
    main()
