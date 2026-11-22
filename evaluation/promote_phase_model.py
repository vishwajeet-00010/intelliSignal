"""
IntelliSignal
evaluation/promote_phase_model.py

Registers a duration-choosing checkpoint as the promoted model for a scenario.

    python evaluation/promote_phase_model.py --scenario heavy \
        --model output/run/models/best.pt \
        --result output/run/final_table.json \
        --note "beat the tuned fixed plan on seeds 6101-6120"

What promotion means here
-------------------------
It writes an entry under `phase_duration_models` in
`models/active_models.json` holding the path, the sha256 of the exact file
measured, the duration menu, the observation contract the policy was trained
with, and the measurement that justified the promotion.

It is deliberately NOT written into `active_models`. That section is read by
`agents/dqn_agent.py` and the dashboard, both of which expect a KEEP/SWITCH
checkpoint whose observation contract carries a `decision_interval`. A
duration-menu model there would not be a promotion, it would be a crash.

A promotion is refused unless the recorded measurement actually shows the
model beating its fixed-time baseline. The project has promoted a model on a
mislabelled baseline once already (see
`output/dqn_stage4_20261120/CORRECTION.md`); the gate is cheap and that
mistake was not.
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

from agents.phase_dqn_agent import (REGISTRY_PATH,          # noqa: E402
                                    REGISTRY_SECTION, PhaseDQNAgent)


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_measurement(path: Optional[str]) -> Dict[str, Any]:
    """Pull the DQN row and its fixed-time baseline out of a results file."""
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    rows = payload if isinstance(payload, list) else list(payload.values())
    learned = next((r for r in rows
                    if str(r.get("controller", "")).startswith("dqn_phase")), None)
    # The baseline is whichever row the results file marks as the tuned
    # fixed plan, falling back to the best fixed row. Matching a specific
    # plan by name would quietly find nothing on a scenario whose optimum is
    # not the one that name was written for.
    baseline = next((r for r in rows
                     if r.get("family") == "fixed_time_tuned_alias"), None)
    if baseline is None:
        fixed = [r for r in rows
                 if str(r.get("controller", "")).startswith("fixed")]
        baseline = min(fixed, key=lambda r: r["mean_waiting_time"]) if fixed else None
    if learned is None:
        return {}
    out = {
        "seeds": learned.get("seeds"),
        "mean_waiting_time": learned.get("mean_waiting_time"),
        "sd_waiting_time": learned.get("sd_waiting_time"),
        "mean_throughput": learned.get("mean_throughput"),
    }
    if baseline is not None:
        out["baseline_controller"] = baseline.get("controller")
        out["baseline_mean_waiting_time"] = baseline.get("mean_waiting_time")
        out["baseline_mean_throughput"] = baseline.get("mean_throughput")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--result", default=None,
                        help="JSON written by final_table.py or "
                             "evaluate_phase_dqn.py --out")
    parser.add_argument("--label", default="Phase-duration DQN")
    parser.add_argument("--stage", default="dqn_phase_20261122")
    parser.add_argument("--note", default="")
    parser.add_argument("--force", action="store_true",
                        help="register even though the recorded measurement "
                             "does not beat the baseline")
    args = parser.parse_args(argv)

    model = os.path.abspath(args.model)
    if not os.path.exists(model):
        raise SystemExit("no such checkpoint: {}".format(model))

    agent = PhaseDQNAgent.load(model)
    measurement = read_measurement(args.result)

    learned = measurement.get("mean_waiting_time")
    baseline = measurement.get("baseline_mean_waiting_time")
    if learned is not None and baseline is not None:
        if learned >= baseline and not args.force:
            raise SystemExit(
                "refusing to promote: the recorded mean waiting time "
                "({:.3f} s) is not better than the baseline's ({:.3f} s). "
                "Use --force only if you mean it.".format(learned, baseline))
    elif not args.force:
        raise SystemExit(
            "refusing to promote without a measurement. Pass --result with "
            "the JSON from the confirmatory evaluation, or --force.")

    record = {
        "label": args.label,
        "stage": args.stage,
        "scenario": args.scenario,
        "controller_family": "phase_duration",
        "path": os.path.relpath(model, PROJECT_ROOT).replace("\\", "/"),
        "sha256": sha256_of(model),
        "durations": list(agent.durations),
        "gamma_per_second": agent.gamma_per_second,
        "double_dqn": agent.double_dqn,
        "architecture": agent.architecture(),
        "observation_contract": dict(agent.observation_contract),
        "measurement": measurement,
        "promoted_on": date.today().isoformat(),
    }
    if args.note:
        record["note"] = args.note

    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        payload = {}
    if not isinstance(payload, dict):
        raise SystemExit("{} is not a JSON object".format(REGISTRY_PATH))

    payload.setdefault(REGISTRY_SECTION, {})[args.scenario] = record
    with open(REGISTRY_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print("registered {} under '{}' -> {}".format(
        args.scenario, REGISTRY_SECTION, record["path"]))
    if learned is not None and baseline is not None:
        print("  {:.3f} s against the baseline's {:.3f} s ({:+.1f} %)".format(
            learned, baseline, 100.0 * (learned - baseline) / baseline))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
