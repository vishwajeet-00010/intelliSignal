"""Build the Stage 4 evidence report and polished PDF from frozen outputs."""
from __future__ import annotations

import ast
import csv
import json
import math
from pathlib import Path
import statistics

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak, KeepTogether)
from reportlab.graphics.shapes import Drawing, String, Line, PolyLine, Rect

OUT = Path(__file__).resolve().parent
NAVY = colors.HexColor("#16324F")
BLUE = colors.HexColor("#2E86AB")
GREEN = colors.HexColor("#3A9D77")
RED = colors.HexColor("#C65353")
GRAY = colors.HexColor("#617080")
WHITE = colors.white


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(rows: list[dict], metric: str) -> float:
    return statistics.fmean(float(row[metric]) for row in rows)


def sd(rows: list[dict], metric: str) -> float:
    values = [float(row[metric]) for row in rows]
    return statistics.stdev(values) if len(values) > 1 else 0.0


def fmt(value, digits=2):
    return "n/a" if value is None else f"{value:.{digits}f}"


def grouped(rows: list[dict]) -> dict[str, list[dict]]:
    result = {}
    for row in rows:
        result.setdefault(row["controller"], []).append(row)
    return result


def paired_bootstrap_ci(old: list[dict], new: list[dict], metric: str,
                        reps=10000) -> tuple[float, float]:
    """Deterministic percentile interval for paired (new - old) seed deltas."""
    import numpy as np
    old_by = {int(row["seed"]): float(row[metric]) for row in old}
    new_by = {int(row["seed"]): float(row[metric]) for row in new}
    seeds = sorted(set(old_by) & set(new_by))
    delta = np.array([new_by[seed] - old_by[seed] for seed in seeds], dtype=float)
    rng = np.random.default_rng(41027)
    samples = np.empty(reps, dtype=float)
    for i in range(reps):
        samples[i] = np.mean(rng.choice(delta, size=len(delta), replace=True))
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def training_chart(scenario: str, width=505, height=180):
    drawing = Drawing(width, height)
    left, right, bottom, top = 48, width - 12, 26, height - 27
    paths = []
    for seed in (55, 56):
        file = OUT / scenario / f"seed{seed}" / "training" / \
            f"dqn_{scenario}_seed{seed}_training.csv"
        rows = read_csv(file)
        values = [(int(row["episode"]), float(row["average_waiting_time"]))
                  for row in rows]
        if values:
            paths.append((seed, values))
    validations = []
    for seed in (55, 56):
        path = OUT / scenario / f"seed{seed}" / "models" / "validation_history.json"
        if path.exists():
            validations.append((seed, read_json(path)))
    all_y = [value for _, points in paths for _, value in points]
    all_y += [float(r["mean_waiting_time"]) for _, rows in validations for r in rows]
    if not all_y:
        drawing.add(String(left, height / 2, "No training history yet", fillColor=GRAY))
        return drawing
    ymin, ymax = min(all_y), max(all_y)
    if math.isclose(ymin, ymax):
        ymax += 1.0
    ymin = max(0, ymin - (ymax - ymin) * .08)
    ymax += (ymax - ymin) * .08
    max_x = max([points[-1][0] for _, points in paths] + [1])
    drawing.add(Line(left, bottom, right, bottom, strokeColor=GRAY, strokeWidth=.7))
    drawing.add(Line(left, bottom, left, top, strokeColor=GRAY, strokeWidth=.7))
    for i in range(4):
        y = bottom + (top - bottom) * i / 3
        val = ymin + (ymax - ymin) * i / 3
        drawing.add(Line(left, y, right, y, strokeColor=colors.HexColor("#E6EBEF"), strokeWidth=.5))
        drawing.add(String(2, y - 3, f"{val:.1f}", fontSize=7, fillColor=GRAY))
    palette = (BLUE, GREEN)
    for index, (seed, points) in enumerate(paths):
        coords = []
        for episode, value in points:
            x = left + (right - left) * episode / max_x
            y = bottom + (top - bottom) * (value - ymin) / (ymax - ymin)
            coords.extend((x, y))
        drawing.add(PolyLine(coords, strokeColor=palette[index], strokeWidth=1.1))
        drawing.add(String(right - 100, top - 9 * index, f"Training seed {seed}",
                            fontSize=7, fillColor=palette[index]))
    # Validation points are plotted as squares; each run has its own color.
    for index, (seed, rows) in enumerate(validations):
        for row in rows:
            episode = int(row["episode"])
            value = float(row["mean_waiting_time"])
            x = left + (right - left) * episode / max_x
            y = bottom + (top - bottom) * (value - ymin) / (ymax - ymin)
            drawing.add(Rect(x - 2, y - 2, 4, 4, fillColor=palette[index],
                             strokeColor=WHITE, strokeWidth=.4))
    drawing.add(String(left, 6, "Training episode", fontSize=7, fillColor=GRAY))
    drawing.add(String(width - 142, 6, "Lower waiting is better", fontSize=7, fillColor=GRAY))
    return drawing


def styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle(name="Title2", parent=s["Title"], fontName="Helvetica-Bold",
                         fontSize=23, leading=27, textColor=NAVY, alignment=TA_CENTER,
                         spaceAfter=10))
    s.add(ParagraphStyle(name="Section2", parent=s["Heading1"], fontSize=17,
                         leading=20, textColor=NAVY, spaceBefore=3, spaceAfter=8))
    s.add(ParagraphStyle(name="Body2", parent=s["BodyText"], fontSize=9.2,
                         leading=13, textColor=colors.HexColor("#243746"), spaceAfter=6))
    s.add(ParagraphStyle(name="Small2", parent=s["BodyText"], fontSize=7.3,
                         leading=9.5, textColor=GRAY))
    s.add(ParagraphStyle(name="Callout", parent=s["BodyText"], fontSize=10.2,
                         leading=14, textColor=NAVY, backColor=colors.HexColor("#EAF3F7"),
                         borderColor=BLUE, borderWidth=.8, borderPadding=8, spaceAfter=9))
    return s


def table(data, widths=None, header=True, font=7.5, repeat=1):
    result = Table(data, colWidths=widths, repeatRows=repeat if header else 0,
                   hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#D7E0E7")),
        ("FONTSIZE", (0, 0), (-1, -1), font),
    ]
    if header:
        commands += [("BACKGROUND", (0, 0), (-1, 0), NAVY),
                     ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                     ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    result.setStyle(TableStyle(commands))
    return result


def scenario_section(scenario: str, styleset, manifest: dict):
    folder = OUT / scenario
    conf_path = folder / "confirmatory" / "confirmatory_rows.csv"
    rows = read_csv(conf_path)
    by = grouped(rows)
    selection_path = folder / "validation" / "validation_selection.json"
    selection = read_json(selection_path) if selection_path.exists() else {}
    story = [Paragraph(f"{scenario.title()} traffic", styleset["Section2"])]
    story.append(Paragraph(
        "Confirmatory evaluation uses 20 fresh, matched traffic seeds and 1,800 simulated seconds per run. "
        "Checkpoint selection and actuated tuning use only the separate five-seed validation set.",
        styleset["Body2"]))
    summary = [["Controller", "Waiting mean ± SD (s)", "Avg queue", "Max queue",
                "Throughput", "Vehicles left", "Blocked / teleports"]]
    labels = [("starting_dqn", "Starting DQN"), ("incumbent_dqn", "Current DQN"),
              ("stage4_candidate", "Stage 4 candidate"),
              ("fixed_time_original", "Original fixed-time"),
              ("fixed_time_tuned", "Tuned fixed-time"),
              ("actuated_tuned", "Tuned actuated")]
    for key, label in labels:
        values = by.get(key)
        if not values:
            continue
        summary.append([label,
                        f"{mean(values,'average_waiting_time'):.2f} ± {sd(values,'average_waiting_time'):.2f}",
                        fmt(mean(values, "average_queue_length")),
                        fmt(mean(values, "max_queue_length")),
                        fmt(mean(values, "throughput"), 1),
                        fmt(mean(values, "vehicles_still_in_network"), 1),
                        f"{sum(int(r.get('blocked_switches') or 0) for r in values)} / "
                        f"{sum(int(r.get('teleports') or 0) for r in values)}"])
    story.append(table(summary, widths=[1.75*inch, 1.65*inch, .95*inch, .95*inch,
                                       1.15*inch, 1.05*inch, 1.4*inch], font=7.0))
    story.append(Spacer(1, 7))
    if "starting_dqn" in by and "incumbent_dqn" in by:
        start_wait = mean(by["starting_dqn"], "average_waiting_time")
        current_wait = mean(by["incumbent_dqn"], "average_waiting_time")
        wait_change = (start_wait - current_wait) / start_wait * 100 if start_wait else 0.0
        start_throughput = mean(by["starting_dqn"], "throughput")
        current_throughput = mean(by["incumbent_dqn"], "throughput")
        throughput_change = (current_throughput - start_throughput) / start_throughput * 100 if start_throughput else 0.0
        story.append(Paragraph(
            f"Current DQN versus starting DQN: waiting is down {wait_change:.1f}% "
            f"({start_wait:.2f} to {current_wait:.2f} s); throughput is "
            f"{throughput_change:+.1f}% ({start_throughput:.1f} to "
            f"{current_throughput:.1f} vehicles). These are the existing models; "
            "the new Stage 4 candidates did not pass the two-run validation gate.",
            styleset["Callout"]))
    if "stage4_candidate" in by and "incumbent_dqn" in by:
        lo, hi = paired_bootstrap_ci(by["incumbent_dqn"], by["stage4_candidate"],
                                     "average_waiting_time")
        old, new = by["incumbent_dqn"], by["stage4_candidate"]
        delta = mean(new, "average_waiting_time") - mean(old, "average_waiting_time")
        wins = sum(float(n["average_waiting_time"]) < float(o["average_waiting_time"])
                   for o, n in zip(sorted(old, key=lambda r: int(r["seed"])),
                                   sorted(new, key=lambda r: int(r["seed"]))))
        story.append(Paragraph(
            f"Stage 4 candidate paired waiting-time change versus current DQN: {delta:+.2f} s "
            f"(95% paired bootstrap interval {lo:+.2f} to {hi:+.2f}); it wins on {wins}/20 seeds.",
            styleset["Callout"]))
    elif not selection.get("confirmatory_candidate_allowed", False):
        story.append(Paragraph(
            "No Stage 4 candidate passed the predeclared validation gate, so no candidate was "
            "evaluated on confirmatory seeds or promoted. The current DQN remains the rollback target.",
            styleset["Callout"]))
    story.append(training_chart(scenario))
    story.append(Spacer(1, 8))

    if rows:
        story.append(PageBreak())
        story.append(Paragraph("Confirmatory per-seed waiting and throughput", styleset["Heading2"]))
        seed_rows = [["Seed", "Starting DQN", "Current DQN", "Stage 4 DQN",
                      "Tuned fixed", "Actuated", "DQN throughput"]]
        indexes = {name: {int(r["seed"]): r for r in values}
                   for name, values in by.items()}
        for seed in sorted(int(r["seed"]) for r in by.get("incumbent_dqn", [])):
            seed_rows.append([
                seed,
                fmt(float(indexes["starting_dqn"][seed]["average_waiting_time"]))
                if "starting_dqn" in indexes else "n/a",
                fmt(float(indexes["incumbent_dqn"][seed]["average_waiting_time"]))
                if "incumbent_dqn" in indexes else "n/a",
                fmt(float(indexes["stage4_candidate"][seed]["average_waiting_time"]))
                if "stage4_candidate" in indexes else "not tested",
                fmt(float(indexes["fixed_time_tuned"][seed]["average_waiting_time"]))
                if "fixed_time_tuned" in indexes else "n/a",
                fmt(float(indexes["actuated_tuned"][seed]["average_waiting_time"]))
                if "actuated_tuned" in indexes else "n/a",
                fmt(float(indexes["incumbent_dqn"][seed]["throughput"]), 0)
                if "incumbent_dqn" in indexes else "n/a",
            ])
        story.append(table(seed_rows, widths=[.42*inch, .75*inch, .75*inch,
                                              .75*inch, .75*inch, .75*inch,
                                              .80*inch], font=6.5))
    story.append(Spacer(1, 8))
    story.append(PageBreak())
    story.append(Paragraph("Validation and action checks", styleset["Heading2"]))
    checks = [["Check", "Result"]]
    checks.append(["Two independent training seeds passed validation",
                   "PASS" if selection.get("all_independent_candidates_pass") else "FAIL"])
    checks.append(["Validation-tuned actuated controller selected",
                   selection.get("selected_actuated", {}).get("selected_label", "None met guardrails")])
    checks.append(["Confirmatory candidate evaluation permitted",
                   "YES" if selection.get("confirmatory_candidate_allowed") else "NO"])
    checks.append(["Candidate blocked switches / teleports",
                   ("0 / 0" if "stage4_candidate" in by and all(
                       int(row["blocked_switches"]) == 0 and int(row["teleports"]) == 0
                       for row in by["stage4_candidate"]) else "No candidate test")])
    if "stage4_candidate" in by:
        candidate_rows = by["stage4_candidate"]
        total_decisions = sum(int(row["decisions"]) for row in candidate_rows)
        total_switches = sum(int(row["switch_count"]) for row in candidate_rows)
        ratios = total_switches / total_decisions if total_decisions else 0.0
        green_durations = set()
        for row in candidate_rows:
            raw = row.get("completed_green_duration_counts", "")
            if raw:
                green_durations.update(ast.literal_eval(raw).keys())
        checks.append(["Candidate switch / decision ratio", f"{ratios:.3f}"])
        checks.append(["Distinct completed green durations", str(len(green_durations))])
    story.append(table(checks, widths=[3.7*inch, 2.4*inch], font=7.2))
    story.append(Spacer(1, 6))
    story.append(Paragraph("Observation range diagnostics", styleset["Heading2"]))
    story.append(Paragraph(
        "Raw state values were sampled on the separate calibration seeds 5201–5210. "
        "The ratios below compare the 99th percentile with the fixed observation scales "
        "(queue/wait: 3/14 for moderate; 16/142 for heavy). Ratios above 1 indicate "
        "that some inputs exceed the nominal scale; this is diagnostic evidence, not a "
        "post-hoc scale adjustment.", styleset["Small2"]))
    raw_stats = selection.get("observation_stats", {})
    scales = {"moderate": {"queues": 3.0, "waiting": 14.0},
              "heavy": {"queues": 16.0, "waiting": 142.0}}[scenario]
    obs_rows = [["Training seed", "Signal", "Approach", "p95 raw", "p99 raw",
                 "p99 / scale", "Maximum"]]
    for seed, entry in sorted(raw_stats.items(), key=lambda item: int(item[0])):
        path = Path(entry.get("path", ""))
        if not path.is_absolute():
            path = OUT / path
        if not path.exists():
            continue
        stats = read_json(path)
        for kind in ("queues", "waiting"):
            for approach, values in stats.get("percentiles", {}).get(kind, {}).items():
                p99 = float(values["p99"])
                obs_rows.append([seed, "Queue" if kind == "queues" else "Waiting",
                                 approach.title(), fmt(float(values["p95"]), 1),
                                 fmt(p99, 1), fmt(p99 / scales[kind], 2),
                                 fmt(float(values["max"]), 1)])
    if len(obs_rows) > 1:
        story.append(table(obs_rows, widths=[.75*inch, .82*inch, .8*inch,
                                             .85*inch, .85*inch, .9*inch,
                                             .85*inch], font=6.8))
    else:
        story.append(Paragraph("Calibration observation statistics are unavailable.",
                               styleset["Small2"]))
    return story


def build_pdf():
    manifest_path = OUT / "experiment_manifest.json"
    manifest = read_json(manifest_path)
    s = styles()
    target = OUT / "stage4_validation_report.pdf"
    doc = SimpleDocTemplate(str(target), pagesize=landscape(letter),
                            rightMargin=.45*inch, leftMargin=.45*inch,
                            topMargin=.48*inch, bottomMargin=.45*inch,
                            title="IntelliSignal Stage 4 DQN Validation Report",
                            author="IntelliSignal project")
    story = [Spacer(1, .36*inch),
             Paragraph("IntelliSignal | Stage 4", s["Title2"]),
             Paragraph("Robust DQN validation and improvement", s["Heading2"]),
             Spacer(1, 8),
             Paragraph("Decision-focused evaluation across independent training runs and fresh traffic seeds",
                       s["Body2"]),
             Spacer(1, 10)]
    audit = (OUT / "audit.md").read_text(encoding="utf-8")
    story.append(Paragraph(
        "<b>Evaluation discipline.</b> Candidate training used seeds 55 and 56. "
        "Fresh validation seeds 5101-5105 selected checkpoints and tuned a pressure-actuated "
        "heuristic. Confirmatory seeds 6101-6120 were reserved for the final comparison. "
        "No candidate was automatically copied to the incumbent model location.", s["Callout"]))
    prior = OUT.parents[0] / "dqn_retraining_20261120" / "experiment_manifest.json"
    previous = read_json(prior)
    story.append(Paragraph("Verified starting reference", s["Heading2"]))
    refs = [["Scenario", "Starting DQN wait", "Latest DQN wait", "Tuned fixed wait",
             "Stage 4 outcome"]]
    for scenario in ("moderate", "heavy"):
        values = previous["measured_results"][scenario]
        refs.append([scenario.title(), f"{values['starting_dqn_waiting']:.2f} s",
                     f"{values['mean_waiting']:.2f} s",
                     f"{values['tuned_fixed_waiting']:.2f} s",
                     "See scenario pages"])
    story.append(table(refs, widths=[1.05*inch, 1.4*inch, 1.35*inch, 1.35*inch, 1.45*inch]))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Historical failure addressed: the prior 300-episode heavy candidate was selected by "
        "training return and failed its fresh holdout waiting, queue, and throughput gates. "
        "Stage 4 uses validation-selected checkpoints, independent training seeds, explicit "
        "degradation stopping, and a sealed final seed set.", s["Body2"]))
    story.append(Paragraph(
        "The tuned pressure-actuated comparator is a project-specific queue/wait heuristic, "
        "not a certified signal-control product or a universal industry baseline.", s["Small2"]))
    for scenario in ("moderate", "heavy"):
        story.append(PageBreak())
        story.extend(scenario_section(scenario, s, manifest))
    story.append(PageBreak())
    story.append(Paragraph("Protocol, promotion gates, and reproducibility", s["Section2"]))
    gate_rows = [["Gate", "Predeclared threshold"]]
    for key, value in manifest.get("promotion_gates", {}).items():
        gate_rows.append([key.replace("_", " ").title(), str(value)])
    story.append(table(gate_rows, widths=[3.2*inch, 3.6*inch], font=7.4))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "The manifest records commands, package versions, seed sets, checkpoint paths, "
        "hashes, tests, gates, and final decision. Full raw per-seed evaluation rows and "
        "training histories are retained beside this PDF. A failed validation or confirmatory "
        "gate leaves the previous DQN unchanged.", s["Body2"]))
    story.append(Paragraph(f"Full repository audit: {audit.splitlines()[0]}", s["Small2"]))

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D7E0E7"))
        canvas.line(.45*inch, .32*inch, 10.55*inch, .32*inch)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(GRAY)
        canvas.drawString(.48*inch, .18*inch, "IntelliSignal Stage 4 | Evidence-based results")
        canvas.drawRightString(10.52*inch, .18*inch, f"Page {document.page}")
        canvas.restoreState()
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(target)


if __name__ == "__main__":
    build_pdf()
