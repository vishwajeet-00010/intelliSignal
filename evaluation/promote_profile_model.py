"""
IntelliSignal
evaluation/promote_profile_model.py

Registers the model for a timing profile.

    python evaluation/promote_profile_model.py --profile standard \\
        --model output/run/models/best.pt \\
        --result output/run/compare_standard.json \\
        --selected-on validation

Three gates, each of which has caught a real mistake in this project:

1. **The checkpoint must fit the profile.** A menu outside the profile's
   green limits is refused. A 4 s-green policy under a 15 s minimum is
   not a cautious fallback, it is an invalid controller.

2. **The measurement must show it winning.** A promotion needs a results
   file in which the model beats the tuned fixed plan. Promoting against
   a mislabelled baseline has happened here before - see
   `output/dqn_stage4_20261120/CORRECTION.md`.

3. **The selection must not have been made on the reported split.**
   `--selected-on` is recorded, and promoting a model chosen on the test
   split needs `--force` and says so in the record.

The record is written to `profile_models` in `models/active_models.json`,
which is separate from both `active_models` (the Stage 3 KEEP/SWITCH
models the dashboard reads) and `phase_duration_models` (the per-scenario
duration models). Nothing that reads those two changes behaviour.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import date
from typing import Any, Dict, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.phase_dqn_agent import (PROFILE_REGISTRY_SECTION, REGISTRY_PATH,
                                    PhaseDQNAgent)
from rl.timing_profile import compatible, get_profile


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_result(path: Optional[str], controller: str) -> Dict[str, Any]:
    """Pull this controller's row and the fixed baseline out of a comparison."""
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    table = payload.get("table") or []
    mine = next((row for row in table
                 if controller.lower() in str(row.get("controller", "")).lower()),
                None)
    fixed = next((row for row in table
                  if str(row.get("controller", "")).startswith("fixed")), None)
    if mine is None:
        return {}

    per_workload = (payload.get("per_workload") or {})
    paired = None
    fixed_key = next((k for k in per_workload if k.startswith("fixed")), None)
    mine_key = next((k for k in per_workload
                     if controller.lower() in k.lower()), None)
    if fixed_key and mine_key:
        base = {row["workload"]: row["delay"] for row in per_workload[fixed_key]}
        deltas = [row["delay"] - base[row["workload"]]
                  for row in per_workload[mine_key]]
        paired = {"workloads": len(deltas),
                  "wins": sum(1 for d in deltas if d < 0),
                  "worst_delta": round(max(deltas), 3),
                  "mean_delta": round(sum(deltas) / len(deltas), 3)}

    return {
        "split": payload.get("split"),
        "profile": (payload.get("profile") or {}).get("name"),
        "mean_total_delay": mine.get("mean_total_delay"),
        "sd_total_delay": mine.get("sd_total_delay"),
        "mean_waiting": mine.get("mean_waiting"),
        "mean_completed": mine.get("mean_completed"),
        "mean_waiting_to_enter": mine.get("mean_waiting_to_enter"),
        "all_compliant": mine.get("all_compliant"),
        "baseline_controller": (fixed or {}).get("controller"),
        "baseline_total_delay": (fixed or {}).get("mean_total_delay"),
        "baseline_completed": (fixed or {}).get("mean_completed"),
        "paired": paired,
    }


def read_selection(path: Optional[str]) -> Dict[str, Any]:
    """The whole selection run, from `evaluation/select_profile_model.py`.

    It carries what a promotion needs and a comparison file cannot: which
    model won, out of how many, against a fixed baseline that is the best
    of every plan on the menu rather than the end of one descent.
    """
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    winner = payload.get("winner") or {}
    rows = {row.get("controller"): row for row in (payload.get("test") or [])}
    mine = rows.get("selected model")
    baseline = rows.get("best fixed plan")
    if not winner or mine is None or baseline is None:
        return {}
    versus = mine.get("against_best_fixed") or {}
    return {
        "paths": list(winner.get("paths") or []),
        "winner": winner.get("name", ""),
        "kind": winner.get("kind", "single"),
        "candidates": len(payload.get("runs") or []),
        "measurement": {
            "split": "test",
            "profile": (payload.get("profile") or {}).get("name"),
            "seeds_per_workload": payload.get("seeds_per_workload"),
            "mean_total_delay": mine.get("delay"),
            "sd_total_delay": mine.get("delay_sd_across_workloads"),
            "mean_waiting": mine.get("waiting"),
            "mean_completed": mine.get("completed"),
            "mean_waiting_to_enter": mine.get("pending"),
            "all_compliant": mine.get("all_compliant"),
            "baseline_controller": "best fixed plan ns {:g} / ew {:g}".format(
                (payload.get("best_fixed") or {}).get("ns", 0),
                (payload.get("best_fixed") or {}).get("ew", 0)),
            "baseline_total_delay": baseline.get("delay"),
            "baseline_completed": baseline.get("completed"),
            "paired": {
                "workloads": versus.get("of"),
                "wins": versus.get("wins"),
                "worst_delta": versus.get("worst"),
                "mean_delta": versus.get("mean_difference"),
            } if versus else None,
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--model", default=None,
                        help="the checkpoint; taken from --selection if "
                             "that is given instead")
    parser.add_argument("--selection", default=None,
                        help="selection.json from "
                             "evaluation/select_profile_model.py")
    parser.add_argument("--result", default=None)
    parser.add_argument("--controller", default="candidate",
                        help="which row of the results table is this model")
    parser.add_argument("--selected-on", default="validation",
                        choices=["validation", "test", "training"])
    parser.add_argument("--label", default="")
    parser.add_argument("--note", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    profile = get_profile(args.profile)
    selection = read_selection(args.selection)
    paths = selection.get("paths") or ([args.model] if args.model else [])
    if not paths:
        raise SystemExit(
            "nothing to promote: pass --model, or --selection with the "
            "selection.json of a finished selection run.")
    models = [os.path.abspath(path) for path in paths]
    for model in models:
        if not os.path.exists(model):
            raise SystemExit("no such checkpoint: {}".format(model))

    for model in models:
        agent = PhaseDQNAgent.load(model)
        if not compatible(profile, agent.durations):
            raise SystemExit(
                "refusing to promote: the checkpoint's greens are {} s, "
                "outside the {} profile ({:g}-{:g} s).".format(
                    agent.durations, profile.name, profile.min_green,
                    profile.max_green))
    model = models[0]

    measurement = selection.get("measurement") or read_result(
        args.result, args.controller)
    learned = measurement.get("mean_total_delay")
    baseline = measurement.get("baseline_total_delay")
    if learned is None or baseline is None:
        if not args.force:
            raise SystemExit(
                "refusing to promote without a measurement. Pass --result "
                "with the JSON from evaluation/compare_profiles.py.")
    elif learned >= baseline and not args.force:
        raise SystemExit(
            "refusing to promote: total delay per requested vehicle is "
            "{:.2f} s against the tuned fixed plan's {:.2f} s.".format(
                learned, baseline))

    if args.selected_on == "test" and not args.force:
        raise SystemExit(
            "refusing to promote a model selected on the test split. "
            "Select on validation and report on test.")

    # What this controller can actually do, not what its checkpoint
    # happens to store. A lock-and-extend model keeps the profile's menu
    # in `durations` and can only ever produce its extension ladder, so
    # recording the menu here told every reader the wrong thing.
    from rl.policy_rollout import action_space_of, describe, extension_of

    story = describe(agent)
    record = {
        "label": args.label or "Profile DQN ({})".format(profile.name),
        "profile": profile.to_dict(),
        "controller_family": "phase_duration",
        "candidate": (agent.scenario.split("candidate:")[-1]
                      if "candidate:" in (agent.scenario or "") else ""),
        "action_space": action_space_of(agent),
        "extension": extension_of(agent) if action_space_of(agent) ==
        "lock_extend" else 0.0,
        "reachable_greens": story["greens"],
        "how_it_works": story["summary"],
        "path": os.path.relpath(model, PROJECT_ROOT).replace("\\", "/"),
        "sha256": sha256_of(model),
        "durations": list(agent.durations),
        "architecture": agent.architecture(),
        "observation_contract": dict(agent.observation_contract),
        "selected_on": args.selected_on,
        "measurement": measurement,
        "promoted_on": date.today().isoformat(),
    }
    if len(models) > 1:
        # An ensemble: every member is recorded with its own fingerprint,
        # and `path` stays the first of them so a reader that only knows
        # about single models still finds a real checkpoint.
        record["members"] = [{
            "path": os.path.relpath(m, PROJECT_ROOT).replace("\\", "/"),
            "sha256": sha256_of(m)} for m in models]
    if args.selection:
        record["selection_report"] = os.path.relpath(
            os.path.abspath(args.selection), PROJECT_ROOT).replace("\\", "/")
        record["selected_from"] = selection.get("candidates")
        record["selection_winner"] = selection.get("winner")
    if args.note:
        record["note"] = args.note

    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        payload = {}
    payload.setdefault(PROFILE_REGISTRY_SECTION, {})[profile.name] = record
    with open(REGISTRY_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print("registered {} under '{}' -> {}".format(
        profile.name, PROFILE_REGISTRY_SECTION, record["path"]))
    if learned is not None and baseline is not None:
        print("  {:.2f} s against the tuned fixed plan's {:.2f} s ({:+.1f} %)"
              .format(learned, baseline, 100.0 * (learned - baseline) / baseline))
    if measurement.get("paired"):
        paired = measurement["paired"]
        print("  paired: wins on {}/{} workloads, worst case {:+.2f} s"
              .format(paired["wins"], paired["workloads"],
                      paired["worst_delta"]))
    print("  selected on the {} split".format(args.selected_on))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
