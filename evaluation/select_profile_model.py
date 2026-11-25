"""
IntelliSignal
evaluation/select_profile_model.py

Choosing the model for a timing profile from every seed that was trained,
and measuring it against a baseline that is as strong as a fixed plan gets.

    python evaluation/select_profile_model.py --profile standard \\
        --runs output/indian_20261125/runs --out output/selection

Why this exists
---------------
`compare_profiles.py` compares two hand-picked checkpoints with a fixed plan
found by one pass of coordinate descent, on one SUMO seed per workload. That
was enough to show the idea worked. It is not enough to pick a model from
twenty runs, for three reasons this script addresses:

1. **The baseline has to be the best fixed plan, not a good one.** One pass
   of coordinate descent from (15, 15) stops at the first plan neither
   single-axis move improves. Here EVERY north-south / east-west pair on
   the profile's menu is measured on the validation split, so "the DQN
   beats tuned fixed time" means it beats the best of all sixty-four.

2. **One seed per workload is one draw.** Every measurement here is the
   mean over several SUMO seeds per workload.

3. **Twenty runs need a declared rule.** Each run's own `best.pt` is
   re-scored on the validation split under identical conditions, the
   leading runs of each family are also tried as an averaged ensemble, and
   the lowest validation delay wins - decided before the test split is
   touched. The test split is then measured once, for the winner and the
   baselines.

The selection metric is the project's honest one: mean total delay per
REQUESTED vehicle (time lost on the road plus time kept waiting to enter),
from SUMO's own trip information.

Everything is run through `evaluation/harness.py`, in parallel worker
processes, and every row is written out so the result can be re-derived.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SELECTION_VERSION = 1
METRIC = "mean total delay per requested vehicle"

_WORKLOADS: Dict[Tuple[str, float], list] = {}
_POLICIES: Dict[Tuple[str, ...], Any] = {}


# ---------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------
def _workloads(split_name: str, duration_s: float):
    from training.workload_library import split

    key = (split_name, float(duration_s))
    if key not in _WORKLOADS:
        _WORKLOADS[key] = split(split_name, duration_s=duration_s)
    return _WORKLOADS[key]


def _policy(paths: Sequence[str]):
    from agents.phase_ensemble import load_policy

    key = tuple(paths)
    if key not in _POLICIES:
        _POLICIES[key] = load_policy(list(paths))
    return _POLICIES[key]


def _measure(task: Dict[str, Any]) -> Dict[str, Any]:
    """One controller on one workload with one seed. Runs in a worker."""
    import torch

    torch.set_num_threads(1)
    from evaluation.harness import (fixed_policy, max_pressure_policy,
                                    run_actuated, run_agent, run_policy)
    from rl.timing_profile import get_profile

    profile = get_profile(task["profile"])
    workload = _workloads(task["split"], task["duration"])[task["workload"]]
    seed = int(task["seed"])
    kind = task["kind"]

    if kind == "fixed":
        run = run_policy(workload,
                         fixed_policy(profile, task["ns"], task["ew"]),
                         profile, seed, controller="fixed", tag="_sel")
    elif kind == "actuated":
        run = run_actuated(workload, profile, seed,
                           gap_seconds=float(task.get("gap", 3.0)),
                           tag="_sel")
    elif kind == "max_pressure":
        run = run_policy(workload, max_pressure_policy(profile), profile,
                         seed, controller="max-pressure", tag="_sel")
    elif kind == "model":
        run = run_agent(workload, _policy(task["paths"]), profile, seed,
                        controller="model", tag="_sel")
    else:                                             # pragma: no cover
        raise ValueError("unknown task kind {!r}".format(kind))

    greens = run.greens
    return {
        "key": task["key"], "kind": kind, "split": task["split"],
        "workload": int(task["workload"]), "workload_name": workload.name,
        "seed": seed,
        "delay": run.mean_total_delay,
        "waiting": run.mean_waiting,
        "completed": run.completed,
        "requested": int(run.delay.get("requested", workload.total_vehicles)),
        "pending": run.waiting_to_enter,
        "queue": float(run.summary.get("average_queue_length", 0.0)),
        "mean_green": statistics.fmean(greens) if greens else 0.0,
        "compliant": bool(run.compliant),
        "red_breaches": int(run.compliance.get("red_limit_breaches", 0) or 0),
        "teleports": int(run.summary.get("teleports", 0) or 0),
        "crosscheck_ok": bool(run.crosscheck.get("completed_agree", True)),
    }


# ---------------------------------------------------------------------------
# Coordinator side
# ---------------------------------------------------------------------------
def run_tasks(tasks: List[Dict[str, Any]], workers: int,
              label: str = "", quiet: bool = False) -> List[Dict[str, Any]]:
    """Run every task, in parallel, and return their rows."""
    if not tasks:
        return []
    started = time.time()
    rows: List[Dict[str, Any]] = []
    if workers <= 1:
        for task in tasks:
            rows.append(_measure(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_measure, task) for task in tasks]
            for index, future in enumerate(as_completed(futures), 1):
                rows.append(future.result())
                if not quiet and index % max(len(tasks) // 10, 1) == 0:
                    print("    {} {}/{}  ({:.0f} s)".format(
                        label, index, len(tasks), time.time() - started),
                        flush=True)
    if not quiet:
        print("    {} done: {} runs in {:.0f} s".format(
            label, len(tasks), time.time() - started), flush=True)
    return rows


def expand(key: str, kind: str, split_name: str, profile: str,
           duration: float, seeds: Sequence[int], n_workloads: int,
           **extra) -> List[Dict[str, Any]]:
    return [dict(key=key, kind=kind, split=split_name, profile=profile,
                 duration=float(duration), workload=index, seed=int(seed),
                 **extra)
            for index in range(n_workloads) for seed in seeds]


def summarise(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean over workloads of the per-workload mean over seeds."""
    by_workload: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        by_workload.setdefault(int(row["workload"]), []).append(row)

    def per_workload(field):
        return [statistics.fmean(r[field] for r in group)
                for _, group in sorted(by_workload.items())]

    delay = per_workload("delay")
    waiting = per_workload("waiting")
    return {
        "runs": len(rows),
        "workloads": len(by_workload),
        "delay": round(statistics.fmean(delay), 4),
        "delay_sd_across_workloads": round(
            statistics.pstdev(delay) if len(delay) > 1 else 0.0, 4),
        "waiting": round(statistics.fmean(waiting), 4),
        "completed": round(statistics.fmean(per_workload("completed")), 2),
        "pending": round(statistics.fmean(per_workload("pending")), 2),
        "queue": round(statistics.fmean(per_workload("queue")), 3),
        "mean_green": round(statistics.fmean(per_workload("mean_green")), 2),
        "all_compliant": all(r["compliant"] for r in rows),
        "red_breaches": sum(r["red_breaches"] for r in rows),
        "teleports": sum(r["teleports"] for r in rows),
        "crosscheck_ok": all(r["crosscheck_ok"] for r in rows),
        "per_workload_delay": [round(v, 4) for v in delay],
        "per_workload_waiting": [round(v, 4) for v in waiting],
    }


def paired(candidate: Dict[str, Any], baseline: Dict[str, Any]
           ) -> Dict[str, Any]:
    """Per-workload differences in delay, candidate minus baseline."""
    diffs = [a - b for a, b in zip(candidate["per_workload_delay"],
                                   baseline["per_workload_delay"])]
    return {
        "wins": sum(1 for d in diffs if d < 0),
        "of": len(diffs),
        "mean_difference": round(statistics.fmean(diffs), 4),
        "best": round(min(diffs), 4),
        "worst": round(max(diffs), 4),
        "relative_change_percent": round(
            100.0 * (candidate["delay"] - baseline["delay"])
            / baseline["delay"], 2) if baseline["delay"] else 0.0,
    }


# ---------------------------------------------------------------------------
def discover_runs(runs_dir: str) -> List[Dict[str, Any]]:
    """Every finished training run under `runs_dir` that has a best.pt."""
    found = []
    for summary_path in sorted(glob.glob(os.path.join(runs_dir, "*",
                                                      "summary.json"))):
        run_dir = os.path.dirname(summary_path)
        best = os.path.join(run_dir, "models", "best.pt")
        if not os.path.isfile(best):
            continue
        try:
            with open(summary_path, "r", encoding="utf-8") as handle:
                summary = json.load(handle)
        except (OSError, ValueError):
            continue
        if summary.get("pilot"):
            continue
        found.append({
            "name": os.path.basename(run_dir),
            "path": os.path.relpath(best, PROJECT_ROOT).replace("\\", "/"),
            "candidate": str(summary.get("candidate", "")),
            "seed": int(summary.get("seed", 0)),
            "episodes": int(summary.get("episodes", 0)),
            "training_validation_delay": summary.get(
                "best_validation_total_delay"),
            "training_validation_episode": summary.get(
                "best_validation_episode"),
        })
    return found


def select(profile_name: str = "standard",
           runs_dir: str = "output/indian_20261125/runs",
           out_dir: str = "output/selection", seeds_per_workload: int = 3,
           duration_s: float = 900.0, workers: int = 8,
           ensemble_sizes: Sequence[int] = (3, 5), quiet: bool = False
           ) -> Dict[str, Any]:
    from rl.timing_profile import get_profile
    from training.workload_library import describe, split, sumo_seeds

    started = time.time()
    profile = get_profile(profile_name)
    os.makedirs(out_dir, exist_ok=True)

    validation = split("validation", duration_s=duration_s)
    test = split("test", duration_s=duration_s)
    val_seeds = sumo_seeds("validation", seeds_per_workload)
    test_seeds = sumo_seeds("test", seeds_per_workload)
    menu = [float(d) for d in profile.durations]
    runs = discover_runs(runs_dir)
    if not runs:
        raise SystemExit("No finished training runs under {}".format(runs_dir))

    def say(text=""):
        if not quiet:
            print(text, flush=True)

    say("=" * 78)
    say("Model selection | {} profile | {} runs | {} seeds per workload".format(
        profile.name, len(runs), seeds_per_workload))
    say("  validation seeds {}   test seeds {}".format(val_seeds, test_seeds))
    say("  selection metric: {}".format(METRIC))
    say("=" * 78)

    all_rows: List[Dict[str, Any]] = []

    # ---- 1. every fixed plan on the menu, on validation ---------------
    say("\n[1/5] every fixed plan on the menu ({} x {}) on VALIDATION".format(
        len(menu), len(menu)))
    tasks = []
    for ns, ew in itertools.product(menu, menu):
        tasks += expand("fixed ns{:g}/ew{:g}".format(ns, ew), "fixed",
                        "validation", profile.name, duration_s, val_seeds,
                        len(validation), ns=ns, ew=ew)
    rows = run_tasks(tasks, workers, "fixed grid", quiet)
    all_rows += rows
    grid = []
    for ns, ew in itertools.product(menu, menu):
        key = "fixed ns{:g}/ew{:g}".format(ns, ew)
        summary = summarise([r for r in rows if r["key"] == key])
        grid.append({"ns": ns, "ew": ew, "key": key, **summary})
    grid.sort(key=lambda row: row["delay"])
    best_fixed = grid[0]
    say("  best fixed plan on validation: ns {:g} / ew {:g}  -> {:.2f} s "
        "delay, {:.2f} s waiting".format(best_fixed["ns"], best_fixed["ew"],
                                         best_fixed["delay"],
                                         best_fixed["waiting"]))
    for row in grid[1:5]:
        say("      next: ns {:g} / ew {:g} -> {:.2f} s".format(
            row["ns"], row["ew"], row["delay"]))
    best_by_waiting = min(grid, key=lambda row: row["waiting"])

    # ---- 2. every run's best.pt, re-scored on validation --------------
    say("\n[2/5] every trained run re-scored on VALIDATION")
    tasks = []
    for run in runs:
        tasks += expand(run["name"], "model", "validation", profile.name,
                        duration_s, val_seeds, len(validation),
                        paths=[run["path"]])
    rows = run_tasks(tasks, workers, "runs", quiet)
    all_rows += rows
    for run in runs:
        run["validation"] = summarise([r for r in rows
                                       if r["key"] == run["name"]])
    runs.sort(key=lambda run: run["validation"]["delay"])
    for run in runs:
        say("  {:<8} cand {}  seed {:>3}  validation delay {:>7.2f}  "
            "wait {:>6.2f}  compliant {}".format(
                run["name"], run["candidate"], run["seed"],
                run["validation"]["delay"], run["validation"]["waiting"],
                "yes" if run["validation"]["all_compliant"] else "NO"))

    # ---- 3. ensembles of the leading runs of each family --------------
    say("\n[3/5] averaged ensembles of the leading runs, on VALIDATION")
    ensembles: List[Dict[str, Any]] = []
    tasks = []
    for candidate in sorted({run["candidate"] for run in runs}):
        family = [run for run in runs if run["candidate"] == candidate]
        for size in ensemble_sizes:
            if len(family) < size:
                continue
            members = family[:size]
            name = "ensemble {}x{}".format(candidate, size)
            ensembles.append({
                "name": name, "candidate": candidate, "size": size,
                "members": [m["name"] for m in members],
                "paths": [m["path"] for m in members]})
            tasks += expand(name, "model", "validation", profile.name,
                            duration_s, val_seeds, len(validation),
                            paths=[m["path"] for m in members])
    rows = run_tasks(tasks, workers, "ensembles", quiet)
    all_rows += rows
    for ensemble in ensembles:
        ensemble["validation"] = summarise(
            [r for r in rows if r["key"] == ensemble["name"]])
        say("  {:<14} of {}  validation delay {:>7.2f}  wait {:>6.2f}".format(
            ensemble["name"], ", ".join(ensemble["members"]),
            ensemble["validation"]["delay"],
            ensemble["validation"]["waiting"]))

    # ---- 4. the declared rule ------------------------------------------
    contenders = (
        [{"name": run["name"], "kind": "single", "candidate": run["candidate"],
          "paths": [run["path"]], "members": [run["name"]],
          "validation": run["validation"]} for run in runs]
        + [{"name": e["name"], "kind": "ensemble", "candidate": e["candidate"],
            "paths": e["paths"], "members": e["members"],
            "validation": e["validation"]} for e in ensembles])
    eligible = [c for c in contenders if c["validation"]["all_compliant"]]
    eligible.sort(key=lambda c: c["validation"]["delay"])
    winner = eligible[0]
    best_single = next(c for c in eligible if c["kind"] == "single")
    say("\n[4/5] selected on VALIDATION, before the test split is read:")
    say("  WINNER      {:<16} {:.2f} s".format(winner["name"],
                                              winner["validation"]["delay"]))
    say("  best single {:<16} {:.2f} s".format(
        best_single["name"], best_single["validation"]["delay"]))
    say("  best fixed  ns {:g} / ew {:g}    {:.2f} s".format(
        best_fixed["ns"], best_fixed["ew"], best_fixed["delay"]))

    # ---- 5. the test split, once ---------------------------------------
    say("\n[5/5] the TEST split, measured once")
    report = [
        ("selected model", "model", {"paths": winner["paths"]}),
        ("best fixed plan", "fixed", {"ns": best_fixed["ns"],
                                      "ew": best_fixed["ew"]}),
        ("actuated (gap 3 s)", "actuated", {"gap": 3.0}),
        ("max-pressure", "max_pressure", {}),
    ]
    if best_single["name"] != winner["name"]:
        report.append(("best single run ({})".format(best_single["name"]),
                       "model", {"paths": best_single["paths"]}))
    for candidate in sorted({run["candidate"] for run in runs}):
        lead = next(c for c in eligible if c["kind"] == "single"
                    and c["candidate"] == candidate)
        label = "best candidate {} run ({})".format(candidate, lead["name"])
        if lead["name"] not in (winner["name"], best_single["name"]):
            report.append((label, "model", {"paths": lead["paths"]}))
    if (best_by_waiting["ns"], best_by_waiting["ew"]) != \
            (best_fixed["ns"], best_fixed["ew"]):
        report.append(("fixed plan with least waiting", "fixed",
                       {"ns": best_by_waiting["ns"],
                        "ew": best_by_waiting["ew"]}))

    # Every fixed plan measured on the TEST traffic as well. It is not a
    # baseline anyone could have chosen - it is picked with hindsight -
    # but without it a plan that happened to generalise badly would be
    # reported as the thing the model beat, and the improvement would be
    # the plan's bad luck rather than the model's work.
    for ns, ew in itertools.product(menu, menu):
        report.append(("oracle fixed ns{:g}/ew{:g}".format(ns, ew), "fixed",
                       {"ns": ns, "ew": ew}))

    tasks = []
    for label, kind, extra in report:
        tasks += expand(label, kind, "test", profile.name, duration_s,
                        test_seeds, len(test), **extra)
    rows = run_tasks(tasks, workers, "test", quiet)
    all_rows += rows
    test_table = []
    oracle_rows = []
    for label, kind, extra in report:
        summary = summarise([r for r in rows if r["key"] == label])
        entry = {"controller": label, "kind": kind, "detail": extra,
                 **summary}
        (oracle_rows if label.startswith("oracle fixed")
         else test_table).append(entry)
    oracle_rows.sort(key=lambda row: row["delay"])
    oracle = dict(oracle_rows[0]) if oracle_rows else None
    if oracle is not None:
        oracle["controller"] = (
            "best fixed plan chosen WITH hindsight on the test traffic "
            "(ns {:g} / ew {:g})".format(oracle["detail"]["ns"],
                                         oracle["detail"]["ew"]))
        test_table.append(oracle)
    baseline = next(row for row in test_table
                    if row["controller"] == "best fixed plan")
    for row in test_table:
        row["against_best_fixed"] = (paired(row, baseline)
                                     if row is not baseline else None)
    test_table.sort(key=lambda row: row["delay"])

    say("")
    say("{:<34} {:>9} {:>9} {:>10} {:>8} {:>7}  {}".format(
        "controller (test split)", "delay/veh", "wait/veh", "completed",
        "pending", "green", "vs best fixed"))
    say("-" * 104)
    for row in test_table:
        versus = row["against_best_fixed"]
        say("{:<34} {:>9.2f} {:>9.2f} {:>10.1f} {:>8.1f} {:>7.1f}  {}".format(
            row["controller"], row["delay"], row["waiting"],
            row["completed"], row["pending"], row["mean_green"],
            "-" if versus is None else "{:+.1f} %  wins {}/{}  worst {:+.2f} s"
            .format(versus["relative_change_percent"], versus["wins"],
                    versus["of"], versus["worst"])))

    result = {
        "selection_version": SELECTION_VERSION,
        "profile": profile.to_dict(),
        "metric": METRIC,
        "rule": ("Lowest validation mean of the metric among single runs and "
                 "averaged ensembles whose every run obeyed the timing "
                 "profile. Decided before the test split was measured."),
        "seeds_per_workload": seeds_per_workload,
        "validation_seeds": val_seeds, "test_seeds": test_seeds,
        "duration_s": duration_s,
        "validation_workloads": describe(validation),
        "test_workloads": describe(test),
        "fixed_grid": grid,
        "fixed_grid_on_test": [
            {"ns": row["detail"]["ns"], "ew": row["detail"]["ew"],
             "delay": row["delay"], "waiting": row["waiting"]}
            for row in oracle_rows],
        "oracle_fixed": ({"ns": oracle["detail"]["ns"],
                          "ew": oracle["detail"]["ew"],
                          "delay": oracle["delay"],
                          "waiting": oracle["waiting"]}
                         if oracle is not None else None),
        "best_fixed": {"ns": best_fixed["ns"], "ew": best_fixed["ew"],
                       "validation": {k: best_fixed[k] for k in
                                      ("delay", "waiting", "completed",
                                       "pending")}},
        "runs": runs,
        "ensembles": ensembles,
        "winner": winner,
        "best_single": best_single,
        "test": test_table,
        "wall_clock_s": round(time.time() - started, 1),
    }
    with open(os.path.join(out_dir, "selection.json"), "w",
              encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=float)
    with open(os.path.join(out_dir, "selection_rows.json"), "w",
              encoding="utf-8") as handle:
        json.dump(all_rows, handle, default=float)
    say("\nsaved {}  ({:.0f} s)".format(
        os.path.join(out_dir, "selection.json"), time.time() - started))
    return result


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="standard")
    parser.add_argument("--runs", default="output/indian_20261125/runs")
    parser.add_argument("--out", default="output/selection")
    parser.add_argument("--seeds", type=int, default=3,
                        help="SUMO seeds per workload")
    parser.add_argument("--duration", type=float, default=900.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ensembles", default="3,5",
                        help="ensemble sizes to try, comma separated")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    select(profile_name=args.profile, runs_dir=args.runs, out_dir=args.out,
           seeds_per_workload=args.seeds, duration_s=args.duration,
           workers=args.workers,
           ensemble_sizes=[int(s) for s in args.ensembles.split(",") if s],
           quiet=args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
