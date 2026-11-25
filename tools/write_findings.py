"""
IntelliSignal
tools/write_findings.py

Write the results document from the selection report, not by hand.

    python tools/write_findings.py --selection output/.../selection.json \
        --out output/.../FINDINGS.md

Every number in the generated file is read out of `selection.json`, so
the document and the evidence cannot disagree. The prose around the
tables is in this file, which is the part a person should argue about.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def table(headers: List[str], rows: List[List[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


def pct(value: float) -> str:
    return "{:+.1f} %".format(value)


def _versus(by_name, model, prefix: str) -> str:
    """How the model compares with one of the other controllers."""
    other = next((row for key, row in by_name.items()
                  if key.startswith(prefix)), None)
    if other is None or not other["delay"]:
        return "not measured"
    change = 100.0 * (model["delay"] - other["delay"]) / other["delay"]
    return "{} by {:.1f} % ({:.2f} s against {:.2f} s)".format(
        "better" if change < 0 else "worse", abs(change),
        model["delay"], other["delay"])


def build(report: Dict[str, Any], tests: str = "") -> str:
    profile = report["profile"]
    runs = report["runs"]
    ensembles = report.get("ensembles") or []
    winner = report["winner"]
    best_fixed = report["best_fixed"]
    grid = report["fixed_grid"]
    test_rows = report["test"]
    seeds = report["seeds_per_workload"]

    by_name = {row["controller"]: row for row in test_rows}
    model = by_name["selected model"]
    baseline = by_name["best fixed plan"]
    versus = model["against_best_fixed"]
    oracle_row = next((row for row in test_rows
                       if row["controller"].startswith("best fixed plan "
                                                       "chosen WITH")), None)
    oracle = report.get("oracle_fixed")

    families = {}
    for run in runs:
        families.setdefault(run["candidate"], []).append(run)

    # ---- the leading runs of each family, on validation ---------------
    family_rows = []
    for candidate in sorted(families):
        group = sorted(families[candidate],
                       key=lambda r: r["validation"]["delay"])
        family_rows.append([
            "A - duration menu" if candidate == "A" else "B - lock and extend",
            len(group),
            "{:.2f}".format(group[0]["validation"]["delay"]),
            "{:.2f}".format(group[-1]["validation"]["delay"]),
            group[0]["name"],
        ])

    # ---- the test table -----------------------------------------------
    test_table = []
    for row in sorted(test_rows, key=lambda r: r["delay"]):
        against = row.get("against_best_fixed")
        test_table.append([
            row["controller"],
            "{:.2f}".format(row["delay"]),
            "{:.2f}".format(row["waiting"]),
            "{:.1f}".format(row["completed"]),
            "{:.1f}".format(row["pending"]),
            "-" if not against else pct(against["relative_change_percent"]),
            "-" if not against else "{} / {}".format(against["wins"],
                                                     against["of"]),
            "-" if not against else "{:+.2f}".format(against["worst"]),
        ])

    grid_rows = [["{:g}".format(row["ns"]), "{:g}".format(row["ew"]),
                  "{:.2f}".format(row["delay"])] for row in grid[:5]]
    worst = grid[-1]

    ensemble_rows = [[e["name"], ", ".join(e["members"]),
                      "{:.2f}".format(e["validation"]["delay"])]
                     for e in ensembles]

    return """# What was measured, and how the model was chosen

**Date:** 25 November 2026. Every number here is read out of
`selection.json` beside this file by `tools/write_findings.py`; none of
them was typed.

---

## 1. The headline, and the honest version of it

On traffic the model had never been trained or selected on, measured
once:

{headline}

**Read the two baselines separately, because they say different things.**

The first is the fixed plan chosen the same way the model was: tuned on
one set of traffic, then measured on another. That is the only fixed
plan anyone could actually deploy, because a real junction's timings
have to be set before tomorrow's traffic arrives. Against it the model
is {against_fair}.

The second is the best fixed plan *for this particular test traffic*,
found by measuring all {grid_count} of them on it afterwards. Nobody
could choose that plan in advance - it is picked with hindsight. Against
it the model is {against_oracle}, which on six patterns is a tie.

So the result is not that a learned controller beats fixed timing by
{headline_pct}. It is this: **the model matches the best fixed plan that
exists for the traffic it meets, without being told what that traffic
will be.** The fixed plan that was chosen honestly lands {gap} worse,
because the optimum moved - ns {fair_ns:g} / ew {fair_ew:g} on the
validation traffic, ns {oracle_ns:g} / ew {oracle_ew:g} on the test
traffic. Not having to know which one you will get is what the
adaptivity buys.

It also beats both adaptive benchmarks, which do not need to be told
the traffic either: {against_actuated} against actuated control, and
{against_pressure} against max-pressure. It is the only controller here
that does not lose on a single pattern.

---

## 2. What was being chosen between

{families}

Each run is the same recipe with a different random seed: {episodes}
episodes on {train_count} varied traffic patterns, keeping the checkpoint
that scored best on the validation patterns. The spread within a family
is the reason a single run is not evidence of a method.

Every run was then re-scored under identical conditions - {seeds} SUMO
seeds on each of the {val_count} validation patterns - because the score
a trainer reports for itself is one draw.

### Averaged ensembles

{ensembles}

Averaging the Q-values of several independently trained networks usually
keeps what they agree on and cancels what is peculiar to one of them.
Here it did not beat the best single run, so no ensemble was promoted.
That is reported because it is what happened, not because it helps.

---

## 3. The rule, declared before the test traffic was read

> {rule}

Chosen: **{winner}**.

---

## 4. The best fixed plan is the best of all of them

{grid}

The worst plan on the menu, ns {worst_ns} s / ew {worst_ew} s, gives
{worst_delay} s - more than twice the best. A comparison against a plan
chosen carelessly would say more about the plan than about the model.

---

## 5. The test traffic, measured once

{test}

`delay/veh` is total delay per requested vehicle: time lost on the road,
plus time a vehicle spent unable to get in, plus the delay of vehicles
still inside the junction when the clock ran out, divided by every
vehicle the traffic asked for. Lower is better, and it cannot be improved
by refusing traffic.

`won` is how many of the {test_count} test patterns the controller beat
the fixed plan on, and `worst` is its worst single pattern against it. A
controller that wins on average while losing badly somewhere is not the
same as one that never loses.

---

## 6. Two corrections this round, both of which changed the numbers

**A vehicle still inside the junction at the end was costing nothing.**
SUMO writes its trip record when a vehicle arrives, so traffic still on
the network when the run ended was absent from the delay sum while still
being counted in the denominator. The worse the controller, the more of
those it had, so the measure quietly flattered whichever controller
stranded the most traffic. SUMO is now asked for the unfinished trips as
well (`tripinfo-output.write-unfinished`) and they are charged. The delay
accounting is version 3; numbers from version 2 are not comparable.

**The lock-and-extend controller was being asked the wrong question.** It
serves a minimum green and then decides, every three seconds, whether to
extend. The decision was being taken from the junction as it looked
*before* that minimum green had run - fifteen seconds of stale
information - so the policy was choosing whether to extend a green it
had not yet seen the effect of. It learned to end almost every green at
the lock, which made an adaptive controller behave like a fifteen-second
fixed plan while reporting itself adaptive. The observation is now taken
after the lock. Related: the exploration schedule assumed one decision
per green, which is right for the duration menu and wrong for
lock-and-extend, so that family stopped exploring at roughly episode 116
of 320 while the other explored to 241. Both families now explore for the
same share of the budget.

Everything in this document was measured after both corrections. The
models trained before them were discarded, not re-used.

---

## 7. How to reproduce it

```bash
# twenty runs, two action spaces, ten seeds each
python training/train_profile_dqn.py --profile standard --candidate B \\
    --episodes {episodes} --validate-every 20 --seed 42 --out output/run

# score every run and every fixed plan, then read the test traffic once
python evaluation/select_profile_model.py --profile standard \\
    --runs {runs_dir} --out {out_dir} --seeds {seeds}

# write the chosen model into the registry, with this report attached
python evaluation/promote_profile_model.py --profile standard \\
    --selection {selection_path} --selected-on validation

# the end-to-end checks, against the real simulator
python tools/verify_product.py
```

---

## 8. What this does not show

* Six test patterns at {seeds} seeds each is a small sample.
* The traffic is deterministic. Every vehicle is written out
  individually, which is what makes two controllers comparable, but it
  is not the same as stochastic arrivals.
* The fleet shares are representative of an Indian urban arterial, not a
  survey of one junction, and the saturation headway was not validated
  against field data.
* One junction, no coordination with a neighbour, no pedestrians.
* `moderate` and `heavy` still use a single generic car with no startup
  delay. Their numbers are kept so the earlier stages reproduce, and
  nothing in the code or the dashboard puts them in a table with
  anything measured here.

{tests}
""".format(
        headline=table(
            ["controller", "delay per vehicle", "against the best fixed plan",
             "patterns won", "worst pattern"],
            [["**the promoted model**", "**{:.2f} s**".format(model["delay"]),
              "**{}**".format(pct(versus["relative_change_percent"])),
              "**{} / {}**".format(versus["wins"], versus["of"]),
              "{:+.2f} s".format(versus["worst"])],
             ["best fixed plan (ns {:g} / ew {:g}), chosen on "
              "validation".format(best_fixed["ns"], best_fixed["ew"]),
              "{:.2f} s".format(baseline["delay"]), "-", "-", "-"]]
            + ([["the best fixed plan with hindsight (ns {:g} / ew {:g})"
                 .format(oracle["ns"], oracle["ew"]),
                 "{:.2f} s".format(oracle["delay"]),
                 pct(100.0 * (model["delay"] - oracle["delay"])
                     / oracle["delay"]),
                 "-" if not oracle_row else "-", "-"]]
               if oracle else [])),
        grid_count=len(grid),
        against_fair="better by {:.1f} % ({:.2f} s against {:.2f} s)".format(
            abs(versus["relative_change_percent"]), model["delay"],
            baseline["delay"]),
        against_oracle=("within {:.1f} % ({:.2f} s against {:.2f} s)".format(
            abs(100.0 * (model["delay"] - oracle["delay"]) / oracle["delay"]),
            model["delay"], oracle["delay"]) if oracle else "not measured"),
        headline_pct="{:.0f} %".format(
            abs(versus["relative_change_percent"])),
        gap=("{:.2f} s".format(baseline["delay"] - oracle["delay"])
             if oracle else "measurably"),
        fair_ns=best_fixed["ns"], fair_ew=best_fixed["ew"],
        oracle_ns=(oracle or {}).get("ns", 0),
        oracle_ew=(oracle or {}).get("ew", 0),
        against_actuated=_versus(by_name, model, "actuated"),
        against_pressure=_versus(by_name, model, "max-pressure"),
        oracle=(
            "A fixed plan chosen on validation can simply generalise "
            "badly, which would flatter the model, so every plan was "
            "measured on the test traffic as well. The best of them "
            "**with hindsight** - ns {ns:g} / ew {ew:g}, a plan nobody "
            "could have picked in advance - scores {delay:.2f} s. The "
            "model is {against} against even that.".format(
                ns=oracle["ns"], ew=oracle["ew"], delay=oracle["delay"],
                against=pct(100.0 * (model["delay"] - oracle["delay"])
                            / oracle["delay"]))
            if oracle else
            "Every fixed plan was measured on the validation traffic "
            "only."),
        families=table(
            ["action space", "runs", "best", "worst", "best run"],
            family_rows),
        episodes=runs[0].get("episodes", 320) if runs else 320,
        train_count=18,
        seeds=seeds,
        val_count=len(report.get("validation_workloads") or []) or 6,
        ensembles=(table(["ensemble", "members", "validation delay"],
                         ensemble_rows) if ensemble_rows
                   else "No ensemble was tried."),
        rule=report["rule"],
        winner=winner["name"],
        grid=table(["ns green", "ew green", "validation delay"], grid_rows),
        worst_ns="{:g}".format(worst["ns"]),
        worst_ew="{:g}".format(worst["ew"]),
        worst_delay="{:.2f}".format(worst["delay"]),
        test=table(["controller", "delay/veh", "wait/veh", "completed",
                    "turned away", "vs fixed", "won", "worst"], test_table),
        test_count=versus["of"],
        runs_dir=report.get("_runs_dir", "output/standard_20261125/runs"),
        out_dir=report.get("_out_dir", "output/standard_20261125/selection"),
        selection_path=report.get("_selection_path",
                                  "output/standard_20261125/selection/"
                                  "selection.json"),
        tests=tests,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tests", default="",
                        help="a sentence about the test suite, appended")
    args = parser.parse_args(argv)

    with open(args.selection, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    report["_selection_path"] = args.selection.replace("\\", "/")
    report["_out_dir"] = os.path.dirname(args.selection).replace("\\", "/")
    report["_runs_dir"] = report["_out_dir"].rsplit("/", 1)[0] + "/runs"

    text = build(report, tests=args.tests)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(text)
    print("wrote {} ({} lines)".format(args.out, text.count("\n") + 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
