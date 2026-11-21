"""Build the final-seed DQN comparison PDF from recorded experiment outputs."""
from __future__ import annotations

import csv
import json
import os
import statistics

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak, KeepTogether)
from reportlab.graphics.shapes import Drawing, String, Line, Rect
from reportlab.platypus.flowables import Flowable

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.dirname(__file__)
SEEDS = list(range(3001, 3021))
NAVY = colors.HexColor("#16324F")
BLUE = colors.HexColor("#2E86AB")
GREEN = colors.HexColor("#3A9D77")
ORANGE = colors.HexColor("#E08E45")
GRAY = colors.HexColor("#667085")


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def historic(scenario, controller):
    rows = []
    for seed in SEEDS:
        data = read_json(os.path.join(ROOT, "experiments", "controlled_20261119",
                                      "final", scenario, controller,
                                      "{}.json".format(seed)))
        rows.append(data["summary"])
    return rows


def stats(rows, key):
    values = [float(row[key]) for row in rows]
    return statistics.fmean(values), statistics.stdev(values)


def chart(title, labels, series, width=720, height=205):
    drawing = Drawing(width, height)
    left, right, bottom, top = 42, width - 15, 42, height - 30
    maxval = max(v for _, _, values in series for v in values) * 1.08 or 1
    drawing.add(String(left, height - 17, title, fontName="Helvetica-Bold",
                       fontSize=10, fillColor=NAVY))
    drawing.add(Line(left, bottom, right, bottom, strokeColor=GRAY, strokeWidth=0.6))
    groups = len(labels)
    groupw = (right - left) / max(groups, 1)
    bw = max(1.0, min(5.0, groupw * 0.72 / len(series)))
    palette = [GRAY, BLUE, GREEN, ORANGE]
    for index, label in enumerate(labels):
        gx = left + index * groupw + groupw * 0.12
        for sidx, (_, _, vals) in enumerate(series):
            value = vals[index]
            barh = (top - bottom) * value / maxval
            x = gx + sidx * bw
            drawing.add(Rect(x, bottom, bw - 0.5, barh, fillColor=palette[sidx],
                             strokeColor=None))
        if groups <= 22:
            drawing.add(String(left + index * groupw + groupw * 0.06, 22,
                               str(label), fontName="Helvetica", fontSize=5.5,
                               fillColor=GRAY))
    lx = left
    for idx, (name, _, _) in enumerate(series):
        drawing.add(Rect(lx, height - 28, 7, 7, fillColor=palette[idx], strokeColor=None))
        drawing.add(String(lx + 10, height - 27, name, fontSize=7, fillColor=GRAY))
        lx += min(170, 45 + len(name) * 4)
    return drawing


def training_chart(scenario, metric, title, width=720, height=150):
    rows = read_csv(os.path.join(OUT, "training",
                                 "dqn_{}_seed42_training.csv".format(scenario)))
    xvals = [int(r["episode"]) for r in rows]
    yvals = [float(r[metric]) for r in rows]
    ymin, ymax = min(yvals), max(yvals)
    drawing = Drawing(width, height)
    left, right, bottom, top = 45, width - 20, 28, height - 35
    drawing.add(String(left, height - 18,
                       title,
                       fontName="Helvetica-Bold", fontSize=10, fillColor=NAVY))
    drawing.add(Line(left, bottom, right, bottom, strokeColor=GRAY, strokeWidth=0.6))
    span = max(ymax - ymin, 1e-6)
    points = []
    for i, val in enumerate(yvals):
        x = left + (right - left) * i / max(len(yvals) - 1, 1)
        y = bottom + (top - bottom) * (val - ymin) / span
        points.append((x, y))
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        drawing.add(Line(x1, y1, x2, y2, strokeColor=BLUE, strokeWidth=0.8))
    floor_ep = next((i for i, row in enumerate(rows, 1)
                     if float(row["epsilon"]) <= 0.05), None)
    if floor_ep:
        x = left + (right - left) * (floor_ep - 1) / max(len(rows) - 1, 1)
        drawing.add(Line(x, bottom, x, top, strokeColor=ORANGE, strokeWidth=0.8,
                         strokeDashArray=[3, 2]))
        drawing.add(String(min(x + 3, right - 80), top - 8, "epsilon floor",
                           fontSize=7, fillColor=ORANGE))
    drawing.add(String(3, top, "{:.1f}".format(ymax), fontSize=7, fillColor=GRAY))
    drawing.add(String(3, bottom, "{:.1f}".format(ymin), fontSize=7, fillColor=GRAY))
    return drawing


def validation_chart(scenario, width=720, height=150):
    rows = read_json(os.path.join(OUT, "models", "dqn_{}".format(scenario),
                                  "validation_history.json"))
    yvals = [float(row["mean_waiting_time"]) for row in rows]
    ymin, ymax = min(yvals), max(yvals)
    drawing = Drawing(width, height)
    left, right, bottom, top = 45, width - 20, 25, height - 32
    drawing.add(String(left, height - 17,
                       "Held-out validation waiting time (2001-2003; lower is better)",
                       fontName="Helvetica-Bold", fontSize=9, fillColor=NAVY))
    drawing.add(Line(left, bottom, right, bottom, strokeColor=GRAY, strokeWidth=0.6))
    span = max(ymax - ymin, 1e-6)
    points = [(left + (right-left)*i/max(len(yvals)-1, 1),
               bottom + (top-bottom)*(v-ymin)/span) for i, v in enumerate(yvals)]
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        drawing.add(Line(x1, y1, x2, y2, strokeColor=GREEN, strokeWidth=1.2))
    for x, y in points:
        drawing.add(Rect(x-1.5, y-1.5, 3, 3, fillColor=GREEN, strokeColor=None))
    drawing.add(String(3, top, "{:.1f}".format(ymax), fontSize=7, fillColor=GRAY))
    drawing.add(String(3, bottom, "{:.1f}".format(ymin), fontSize=7, fillColor=GRAY))
    return drawing


def make_report():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"],
                              fontSize=24, leading=29, textColor=NAVY,
                              alignment=TA_CENTER, spaceAfter=8))
    styles.add(ParagraphStyle(name="Deck", parent=styles["Normal"],
                              fontSize=10, leading=14, textColor=GRAY,
                              alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading1"],
                              fontSize=16, textColor=NAVY, spaceBefore=4, spaceAfter=7))
    styles.add(ParagraphStyle(name="Small", parent=styles["Normal"],
                              fontSize=7, leading=9, textColor=GRAY))
    doc = SimpleDocTemplate(os.path.join(OUT, "dqn_training_comparison.pdf"),
                            pagesize=landscape(letter), rightMargin=30,
                            leftMargin=30, topMargin=26, bottomMargin=28)
    story = [Spacer(1, 0.55 * inch),
             Paragraph("IntelliSignal DQN Training Results", styles["ReportTitle"]),
             Paragraph("Measured comparison across 20 frozen final seeds, 3001-3020",
                       styles["Deck"]), Spacer(1, 0.35 * inch),
             Paragraph("What changed", styles["Section"]),
             Paragraph("The DQN was retrained for 100 episodes per scenario after reducing the decision interval to 5 seconds, masking unavailable switches, storing executed actions, and applying measured scenario-specific observation scales. Checkpoints were selected by held-out waiting time on seeds 2001-2003. Incumbent checkpoints and historical experiments were preserved.", styles["BodyText"]),
             Spacer(1, 12)]

    for scenario in ("moderate", "heavy"):
        new = read_csv(os.path.join(OUT, "evaluation", scenario,
                                    "dqn_{}_evaluation.csv".format(scenario)))
        old = historic(scenario, "incumbent")
        baseline = historic(scenario, "fixed_time")
        tuned = read_csv(os.path.join(OUT, "fixed_time",
                                      "{}_tuned.csv".format(scenario)))
        wait_old = stats(old, "average_waiting_time")
        wait_new = stats(new, "average_waiting_time")
        wait_base = stats(baseline, "average_waiting_time")
        wait_tuned = stats(tuned, "average_waiting_time")
        thru_new = stats(new, "throughput")
        scenario_pass = (wait_new[0] < (4 if scenario == "moderate" else 25))
        blocked = sum(int(float(row.get("blocked_switches", 0))) for row in new)
        switch_ratio = statistics.fmean(
            float(row["switch_count"]) / float(row["decisions"]) for row in new)
        obs = read_json(os.path.join(OUT, "{}_posttrain_observation_stats.json".format(scenario)))
        queue_scale = 3.0 if scenario == "moderate" else 16.0
        waiting_scale = 14.0 if scenario == "moderate" else 142.0
        normalized_p99 = [entry["p99"] / scale
                          for kind, scale in (("queues", queue_scale),
                                              ("waiting", waiting_scale))
                          for entry in obs["percentiles"][kind].values()]
        obs_min, obs_max = min(normalized_p99), max(normalized_p99)
        story += [PageBreak(), Paragraph(scenario.title() + " traffic", styles["Section"]),
                  Paragraph("All values are measured on identical frozen traffic seeds. Lower waiting is better; throughput is vehicles completed.", styles["Small"]), Spacer(1, 7)]
        summary = [["Controller", "Mean waiting ± SD (s)", "Mean throughput ± SD", "Mean queue", "Mean vehicles left"],
                   ["Starting DQN", "{:.2f} ± {:.2f}".format(*wait_old),
                    "{:.0f} ± {:.0f}".format(*stats(old, "throughput")),
                    "{:.2f}".format(stats(old, "average_queue_length")[0]),
                    "{:.1f}".format(stats(old, "vehicles_still_in_network")[0])],
                   ["Trained DQN", "{:.2f} ± {:.2f}".format(*wait_new),
                    "{:.0f} ± {:.0f}".format(*thru_new),
                    "{:.2f}".format(stats(new, "average_queue_length")[0]),
                    "{:.1f}".format(stats(new, "vehicles_still_in_network")[0])],
                   ["Historical fixed time", "{:.2f} ± {:.2f}".format(*wait_base),
                    "{:.0f} ± {:.0f}".format(*stats(baseline, "throughput")),
                    "{:.2f}".format(stats(baseline, "average_queue_length")[0]),
                    "{:.1f}".format(stats(baseline, "vehicles_still_in_network")[0])],
                   ["Tuned fixed time", "{:.2f} ± {:.2f}".format(*wait_tuned),
                    "{:.0f} ± {:.0f}".format(*stats(tuned, "throughput")),
                    "{:.2f}".format(stats(tuned, "average_queue_length")[0]),
                    "{:.1f}".format(stats(tuned, "vehicles_still_in_network")[0])]]
        table = Table(summary, colWidths=[130, 150, 150, 100, 130], repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5F8")]),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D4DCE4")),
            ("FONTSIZE", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
        gates = [["Acceptance check", "Measured", "Result"],
                 ["Mean waiting below {} s".format(4 if scenario == "moderate" else 25),
                  "{:.2f} s".format(wait_new[0]), "PASS" if scenario_pass else "FAIL"],
                 ["Heavy throughput at least 1,390" if scenario == "heavy" else "Moderate waiting target",
                  "{:.0f} vehicles".format(thru_new[0]) if scenario == "heavy" else "See waiting gate above",
                  "PASS" if scenario == "heavy" and thru_new[0] >= 1390 else ("N/A" if scenario == "moderate" else "FAIL")],
                 ["Blocked switches are zero", "{} / {} decisions".format(blocked, sum(int(float(r["decisions"])) for r in new)), "PASS" if blocked == 0 else "FAIL"],
                 ["Switches / decisions below 0.90", "{:.2f}".format(switch_ratio), "PASS" if switch_ratio < .90 else "FAIL"],
                 ["Traffic p99 normalized range 0.60-1.00", "{:.2f}-{:.2f}".format(obs_min, obs_max), "PASS" if obs_min >= .60 and obs_max <= 1.0 else "PARTIAL"]]
        gate_table = Table(gates, colWidths=[250, 220, 85])
        gate_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5F8")]),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D4DCE4")),
            ("FONTSIZE", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
        story += [table, Spacer(1, 6), gate_table, Spacer(1, 8),
                  Paragraph("Change versus starting DQN: {:+.1f}% waiting; throughput {:+.1f} vehicles.".format(
                      100 * (wait_new[0] - wait_old[0]) / wait_old[0],
                      thru_new[0] - stats(old, "throughput")[0]), styles["BodyText"]),
                  Spacer(1, 8),
                  chart("Per-seed average waiting time (seconds)", SEEDS,
                        [("Starting DQN", None, [float(r["average_waiting_time"]) for r in old]),
                         ("Trained DQN", None, [float(r["average_waiting_time"]) for r in new]),
                         ("Tuned fixed", None, [float(r["average_waiting_time"]) for r in tuned])]),
                  Paragraph("Training and validation curves appear in the learning checks section. Final comparisons use greedy 1800-second evaluations.", styles["Small"]),
                  PageBreak(), Paragraph(scenario.title() + " per-seed results", styles["Section"])]
        old_by = {int(r["seed"]): r for r in old}
        new_by = {int(r["seed"]): r for r in new}
        base_by = {int(r["seed"]): r for r in baseline}
        tuned_by = {int(r["seed"]): r for r in tuned}
        seedtable = [["Seed", "Start DQN wait", "Trained wait", "Change", "Tuned fixed", "Original fixed", "Trained throughput"]]
        for seed in SEEDS:
            o, n = old_by[seed], new_by[seed]
            delta = float(n["average_waiting_time"]) - float(o["average_waiting_time"])
            seedtable.append([seed, "{:.2f}".format(float(o["average_waiting_time"])),
                              "{:.2f}".format(float(n["average_waiting_time"])),
                              "{:+.2f}".format(delta),
                              "{:.2f}".format(float(tuned_by[seed]["average_waiting_time"])),
                              "{:.2f}".format(float(base_by[seed]["average_waiting_time"])),
                              "{:.0f}".format(float(n["throughput"]))])
        table = Table(seedtable, colWidths=[48, 95, 95, 75, 90, 95, 105], repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5F8")]),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#D4DCE4")),
            ("FONTSIZE", (0, 0), (-1, -1), 7), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
        regressions = [str(seed) for seed in SEEDS if
                       float(new_by[seed]["average_waiting_time"]) > float(old_by[seed]["average_waiting_time"])]
        story += [table, Spacer(1, 6), chart("Per-seed throughput (vehicles completed)", SEEDS,
                    [("Starting DQN", None, [float(r["throughput"]) for r in old]),
                     ("Trained DQN", None, [float(r["throughput"]) for r in new]),
                     ("Tuned fixed", None, [float(r["throughput"]) for r in tuned])], height=145),
                  Spacer(1, 9), Paragraph("Seeds with higher waiting after training: {}.".format(
                      ", ".join(regressions) if regressions else "none"), styles["BodyText"]),
                  PageBreak(), Paragraph(scenario.title() + " learning checks", styles["Section"]),
                  training_chart(scenario, "average_waiting_time",
                                 "Training episode waiting time", height=120),
                  Spacer(1, 6), training_chart(scenario, "total_reward",
                                               "Training return", height=120),
                  Spacer(1, 6), validation_chart(scenario, height=120),
                  Paragraph("This report compares a single training seed (42) against 20 final traffic seeds. Validation seeds 2001-2003 selected checkpoints; the frozen final set was not used during training or selection. Tuned fixed-time uses the measured 5-second moderate and 7-second heavy greens; original fixed-time preserves the historical 30/30-second plan.", styles["Small"])]

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D4DCE4"))
        canvas.line(30, 22, 762, 22)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(GRAY)
        canvas.drawString(30, 12, "IntelliSignal | DQN experiment | 20 final seeds")
        canvas.drawRightString(762, 12, "Page {}".format(document.page))
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    make_report()
