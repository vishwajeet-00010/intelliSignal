"""Build the final Stage 5 evidence report PDF from frozen experiment outputs."""
from __future__ import annotations

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
from reportlab.graphics.shapes import Drawing, String, Line, PolyLine

OUT = Path(__file__).resolve().parent
REPORT = OUT.parents[1] / "output" / "pdf" / "intellisignal_stage5_dqn_stability_report.pdf"
NAVY = colors.HexColor("#16324F")
BLUE = colors.HexColor("#2E86AB")
GREEN = colors.HexColor("#3A9D77")
GRAY = colors.HexColor("#617080")
WHITE = colors.white
TRAIN_SEEDS = (6207, 6208, 6209, 6210, 6211)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def fmt(value, digits=2):
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def styles():
    result = getSampleStyleSheet()
    result.add(ParagraphStyle(name="T", parent=result["Title"], fontName="Helvetica-Bold",
                              fontSize=23, leading=27, textColor=NAVY, alignment=TA_CENTER,
                              spaceAfter=10))
    result.add(ParagraphStyle(name="H", parent=result["Heading1"], fontSize=16,
                              leading=19, textColor=NAVY, spaceBefore=2, spaceAfter=8))
    result.add(ParagraphStyle(name="B", parent=result["BodyText"], fontSize=9,
                              leading=12, textColor=colors.HexColor("#243746"), spaceAfter=6))
    result.add(ParagraphStyle(name="S", parent=result["BodyText"], fontSize=7.2,
                              leading=9, textColor=GRAY, spaceAfter=5))
    result.add(ParagraphStyle(name="C", parent=result["BodyText"], fontSize=10,
                              leading=14, textColor=NAVY, backColor=colors.HexColor("#EAF3F7"),
                              borderColor=BLUE, borderWidth=.8, borderPadding=8, spaceAfter=9))
    return result


def make_table(data, widths, font=7.2):
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#D7E0E7")),
        ("FONTSIZE", (0, 0), (-1, -1), font),
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    return table


def training_chart(scenario: str, width=690, height=210):
    drawing = Drawing(width, height)
    left, right, bottom, top = 48, width - 15, 28, height - 28
    arms = {"control": (BLUE, "Control: legacy scale"),
            "treatment": (GREEN, "Treatment: full 60 s scale")}
    series = {}
    validation = {}
    for arm in arms:
        paths = []
        val_runs = []
        for seed in TRAIN_SEEDS:
            base = OUT / scenario / arm / f"seed{seed}"
            history = read_csv(base / "training" / f"dqn_{scenario}_seed{seed}_training.csv")
            if history:
                paths.append({int(row["episode"]): float(row["average_waiting_time"])
                              for row in history})
            val_path = base / "models" / "validation_history.json"
            if val_path.exists():
                val_runs.append(read_json(val_path))
        if paths:
            steps = sorted(set().union(*(set(p) for p in paths)))
            series[arm] = [(step, statistics.fmean(p[step] for p in paths if step in p))
                           for step in steps]
        if val_runs:
            by_ep = {}
            for run in val_runs:
                for item in run:
                    by_ep.setdefault(int(item["episode"]), []).append(
                        float(item["mean_waiting_time"]))
            validation[arm] = [(ep, statistics.fmean(vals)) for ep, vals in sorted(by_ep.items())]
    all_y = [y for points in series.values() for _, y in points]
    all_y += [y for points in validation.values() for _, y in points]
    if not all_y:
        drawing.add(String(left, height / 2, "Training data not available", fillColor=GRAY))
        return drawing
    ymin, ymax = min(all_y), max(all_y)
    if math.isclose(ymin, ymax):
        ymax = ymin + 1.0
    pad = .08 * (ymax - ymin)
    ymin, ymax = max(0.0, ymin - pad), ymax + pad
    max_x = max([max(points)[0] for points in series.values()] + [1])
    drawing.add(Line(left, bottom, right, bottom, strokeColor=GRAY, strokeWidth=.7))
    drawing.add(Line(left, bottom, left, top, strokeColor=GRAY, strokeWidth=.7))
    for i in range(4):
        y = bottom + (top - bottom) * i / 3
        val = ymin + (ymax - ymin) * i / 3
        drawing.add(Line(left, y, right, y, strokeColor=colors.HexColor("#E6EBEF"), strokeWidth=.5))
        drawing.add(String(4, y - 3, f"{val:.1f}", fontSize=7, fillColor=GRAY))
    for index, (arm, (color, label)) in enumerate(arms.items()):
        pts = series.get(arm, [])
        coords = []
        for ep, value in pts:
            coords.extend((left + (right-left)*ep/max_x,
                           bottom + (top-bottom)*(value-ymin)/(ymax-ymin)))
        if len(coords) >= 4:
            drawing.add(PolyLine(coords, strokeColor=color, strokeWidth=1.3))
        for ep, value in validation.get(arm, []):
            x = left + (right-left)*ep/max_x
            y = bottom + (top-bottom)*(value-ymin)/(ymax-ymin)
            drawing.add(Line(x-2.5, y-2.5, x+2.5, y+2.5, strokeColor=color, strokeWidth=1))
            drawing.add(Line(x-2.5, y+2.5, x+2.5, y-2.5, strokeColor=color, strokeWidth=1))
        drawing.add(String(right-175, top-10*index, label, fontSize=7, fillColor=color))
    drawing.add(String(left, 8, "Training episode; validation checkpoints marked", fontSize=7, fillColor=GRAY))
    drawing.add(String(width-130, 8, "Lower waiting is better", fontSize=7, fillColor=GRAY))
    return drawing


def scenario_pages(scenario, styleset, manifest, decision):
    folder = OUT / scenario
    selection = read_json(folder / "validation" / "validation_selection.json") if (folder / "validation" / "validation_selection.json").exists() else {}
    result = decision.get("scenarios", {}).get(scenario, {})
    story = [Paragraph(f"{scenario.title()} traffic: training stability", styleset["H"])]
    story.append(Paragraph(
        "Five matched training seeds per arm; five separate validation seeds per checkpoint; "
        "1,800 simulated seconds per evaluation. Only phase-age scaling differs between arms.",
        styleset["B"]))
    config = selection.get("configuration_checks", {})
    config_table = [["Training arm", "Validation passes / 5", "Eligible", "Mean validation wait"]]
    for arm, title in (("control", "Legacy scale (20 s)"),
                       ("treatment", "Full legal scale (60 s)")):
        val = config.get(arm, {})
        config_table.append([title, f"{val.get('runs_passed', 'n/a')} / 5",
                             "YES" if val.get("validation_eligible") else "NO",
                             fmt(val.get("mean_waiting_time_across_runs"))])
    story.append(make_table(config_table, [2.1*inch, 1.5*inch, 1.1*inch, 1.8*inch]))
    paired_mean = result.get("mean_paired_treatment_minus_control_wait_seconds")
    wins = result.get("treatment_wait_wins_of_five")
    story.append(Spacer(1, 7))
    story.append(Paragraph(
        f"Matched-seed validation effect (treatment minus control): "
        f"{fmt(paired_mean)} seconds; treatment wins on {wins if wins is not None else 'n/a'}/5 training seeds. "
        f"Decision: {result.get('scenario_decision', 'pending')}.", styleset["C"]))
    story.append(training_chart(scenario))
    means = result.get("confirmatory_controller_means", {})
    if means:
        story.append(Paragraph("Confirmatory controller means", styleset["Heading2"]))
        rows = [["Controller", "Wait (s)", "Avg queue", "Max queue", "Throughput", "Vehicles left"]]
        order = [("starting_dqn", "Starting DQN"), ("incumbent_dqn", "Current DQN"),
                 ("stage5_candidate", "Selected Stage 5 DQN"),
                 ("fixed_time_original", "Original fixed-time"),
                 ("fixed_time_tuned", "Tuned fixed-time"),
                 ("actuated_tuned", "Validation-tuned pressure heuristic")]
        for key, label in order:
            if key in means:
                item = means[key]
                rows.append([label, fmt(item["average_waiting_time"]),
                             fmt(item["average_queue_length"]), fmt(item["max_queue_length"]),
                             fmt(item["throughput"], 1), fmt(item["vehicles_still_in_network"], 1)])
        story.append(make_table(rows, [2.3*inch, .9*inch, .9*inch, .9*inch, 1.15*inch, 1.2*inch], 6.8))
    elif selection.get("selected_candidate") is None:
        story.append(Paragraph(
            "No configuration met the four-of-five validation threshold. The candidate was "
            "not included in the confirmatory cohort; the incumbent remains unchanged.", styleset["C"]))

    story.append(PageBreak())
    story.append(Paragraph(f"{scenario.title()} traffic: seed-level results and diagnostics", styleset["H"]))
    checks = selection.get("candidate_checks", {})
    rows = [["Arm", "Train seed", "Episodes", "Best episode", "Best train validation wait",
             "Evaluation wait", "Avg queue", "Throughput", "Gate"]]
    for arm, title in (("control", "Control"), ("treatment", "Treatment")):
        for seed in TRAIN_SEEDS:
            key = f"{arm}_seed{seed}"
            item = checks.get(key, {})
            training = manifest.get("training_runs", {}).get(f"{scenario}_{arm}_seed{seed}", {})
            summary_path = training.get("summary")
            summary = read_json(Path(summary_path)) if summary_path and Path(summary_path).exists() else {}
            rows.append([title, seed, summary.get("episodes", "n/a"),
                         summary.get("best_window_episode", "n/a"),
                         fmt(summary.get("best_validation_mean_waiting_time")),
                         fmt(item.get("mean_waiting_time")),
                         fmt(item.get("mean_average_queue")), fmt(item.get("mean_throughput"), 1),
                         "PASS" if item.get("passes_validation_gate") else
                         ("MISSING" if item.get("status") != "evaluated" else "FAIL")])
    story.append(make_table(rows, [.82*inch, .72*inch, .68*inch, .72*inch, 1.32*inch,
                                   .92*inch, .82*inch, .9*inch, .62*inch], 6.2))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Phase-age and observation diagnostics", styleset["Heading2"]))
    cal = selection.get("calibration", {})
    cal_rows = [["Arm", "Phase-age scale", "Raw phase p99", "Normalized p99",
                 "Phase clipped", "Queue / wait clipping"]]
    for arm, title in (("control", "Control"), ("treatment", "Treatment")):
        stats = cal.get(arm, {}).get("raw_and_normalized_percentiles", {})
        phase = stats.get("phase_elapsed", {})
        qclip = [v.get("fraction_clipped", 0) for v in stats.get("queues", {}).values()]
        wclip = [v.get("fraction_clipped", 0) for v in stats.get("waiting", {}).values()]
        cal_rows.append([title, fmt(phase.get("normalization_scale_seconds"), 0),
                         fmt(phase.get("p99")), fmt(phase.get("normalized_p99"), 3),
                         fmt(phase.get("fraction_clipped", 0)*100, 2)+"%",
                         (fmt(statistics.fmean(qclip)*100, 2)+"% / "+
                          fmt(statistics.fmean(wclip)*100, 2)+"%") if qclip and wclip else "n/a"])
    story.append(make_table(cal_rows, [1.1*inch, 1.2*inch, 1.1*inch, 1.25*inch, 1.2*inch, 1.8*inch]))
    story.append(Paragraph(
        "Calibration samples diagnose clipping and scale coverage; they were not used to tune "
        "the declared 60-second treatment divisor. Per-approach percentiles and full sample "
        "details are retained in the calibration JSON artifacts.", styleset["S"]))
    return story


def build():
    manifest = read_json(OUT / "experiment_manifest.json")
    decision = read_json(OUT / "stage5_decision.json") if (OUT / "stage5_decision.json").exists() else {"scenarios": {}}
    s = styles()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(REPORT), pagesize=landscape(letter),
                            leftMargin=.48*inch, rightMargin=.48*inch,
                            topMargin=.48*inch, bottomMargin=.45*inch,
                            title="IntelliSignal Stage 5 DQN Stability Study",
                            author="IntelliSignal project")
    story = [Spacer(1, .25*inch),
             Paragraph("IntelliSignal | Stage 5", s["T"]),
             Paragraph("DQN stability and robustness study", s["Heading2"]),
             Paragraph("Controlled test of phase-age observation scaling", s["B"]),
             Spacer(1, 10),
             Paragraph("Study decision", s["H"])]
    overall = "Stage 5 evaluation is not complete yet."
    if decision.get("scenarios"):
        values = [v.get("scenario_decision") for v in decision["scenarios"].values()]
        overall = ("No Stage 5 model was promoted; the existing incumbent remains unchanged."
                   if all(v not in ("CONFIRMATORY_PENDING", None) for v in values)
                   else "Stage 5 evaluation is incomplete; no later stage may begin.")
    story.append(Paragraph(overall, s["C"]))
    story.append(Paragraph(
        "Only one factor changed: phase_elapsed normalization. The legacy control divides by "
        "20 seconds and saturates even though legal green can last 60 seconds; treatment maps "
        "the full 60-second legal interval to the observation range. DQN algorithm settings, "
        "traffic, reward, action safeguards, and training seeds are matched.", s["B"]))
    train = manifest.get("training", {})
    rows = [["Scenario", "Control pass rate", "Treatment pass rate", "Selected checkpoint", "Decision"]]
    for scenario in ("moderate", "heavy"):
        val = decision.get("scenarios", {}).get(scenario, {})
        checks = val.get("configuration_checks", {})
        selected = val.get("selected_candidate") or {}
        rows.append([scenario.title(),
                     f"{checks.get('control', {}).get('runs_passed', 'n/a')}/5",
                     f"{checks.get('treatment', {}).get('runs_passed', 'n/a')}/5",
                     f"{selected.get('arm', 'none')} seed {selected.get('seed', '')}".strip(),
                     val.get("scenario_decision", "pending")])
    story.append(make_table(rows, [1.35*inch, 1.45*inch, 1.55*inch, 1.8*inch, 2.2*inch]))
    story.append(Spacer(1, 10))
    tests = manifest.get("tests") or {}
    story.append(Paragraph(
        f"Test suite: {tests.get('result', 'not run')}. Training status: "
        f"{train.get('status', 'not started')}. No automatic model replacement is permitted.", s["B"]))
    story.append(Paragraph(
        "Interpretation limit: a successful result applies to this SUMO intersection and these "
        "traffic scenarios. It does not establish deployment readiness or performance at other sites.", s["S"]))
    for scenario in ("moderate", "heavy"):
        story.append(PageBreak())
        story.extend(scenario_pages(scenario, s, manifest,
                                    decision.get("scenarios", {}).get(scenario, {})))
    story.append(PageBreak())
    story.append(Paragraph("Frozen protocol, test and sign-off status", s["H"]))
    gates = manifest.get("validation_gates", {})
    gate_rows = [["Validation gate", "Threshold"]] + [[key.replace("_", " ").title(), str(value)]
                                                         for key, value in gates.items()]
    story.append(make_table(gate_rows, [3.4*inch, 4.2*inch]))
    story.append(Spacer(1, 9))
    story.append(Paragraph(
        "Stage completion checklist: training cells, validation, confirmatory evaluation when "
        "eligible, full test suite, integrity comparison, documentation, and this report must all "
        "be recorded. Stage 6 is explicitly blocked until the user confirms Stage 5 is clear and complete.",
        s["C"]))
    story.append(Paragraph(
        "The machine-readable manifest records seeds, commands, runtime versions, checkpoints, "
        "metrics, hashes, tests, gates and the decision. Raw per-seed CSV, JSON and training histories "
        "remain alongside the experiment.", s["B"]))

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D7E0E7"))
        canvas.line(.48*inch, .32*inch, 10.52*inch, .32*inch)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(GRAY)
        canvas.drawString(.5*inch, .18*inch, "IntelliSignal Stage 5 | Stability evidence")
        canvas.drawRightString(10.5*inch, .18*inch, f"Page {document.page}")
        canvas.restoreState()
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(REPORT)


if __name__ == "__main__":
    build()
