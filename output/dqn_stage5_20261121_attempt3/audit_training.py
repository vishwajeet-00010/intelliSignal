"""Audit every new Stage 5 cell against the frozen training contract."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent
SCENARIOS = ("moderate", "heavy")
ARMS = {"control": 20.0, "treatment": 60.0}
SEEDS = range(6207, 6212)
VALIDATION = list(range(6301, 6306))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit_cell(scenario: str, arm: str, seed: int) -> dict:
    base = OUT / scenario / arm / f"seed{seed}"
    result = {"scenario": scenario, "arm": arm, "seed": seed,
              "status": "not_started", "errors": []}
    if not base.exists():
        return result
    result["status"] = "partial"
    summary_path = base / "training_summary.json"
    if not summary_path.exists():
        result["errors"].append("training_summary.json missing")
        if (base / "training_failure.json").exists():
            result["status"] = "failed"
        return result
    summary = read_json(summary_path)
    episodes = int(summary.get("episodes", 0))
    result["episodes"] = episodes
    result["stopped_early"] = bool(summary.get("stopped_early"))
    if episodes != 100 and not (0 < episodes < 100 and episodes % 10 == 0
                                and summary.get("stopped_early")
                                and summary.get("stop_reason")):
        result["errors"].append("episode count is neither 100 nor a documented checkpoint stop")
    if summary.get("scenario") != scenario or int(summary.get("seed", -1)) != seed:
        result["errors"].append("summary scenario/seed mismatch")
    if summary.get("phase_elapsed_scale") != ARMS[arm]:
        result["errors"].append("summary phase-age scale mismatch")
    if summary.get("validation_seeds") != VALIDATION:
        result["errors"].append("summary validation seeds mismatch")
    config = summary.get("config", {})
    if config.get("episodes") != 100 or config.get("checkpoint_every") != 10:
        result["errors"].append("frozen episode/checkpoint config mismatch")
    frozen_path = base / "frozen_run_config.json"
    if not frozen_path.exists():
        result["errors"].append("frozen_run_config.json missing")
    else:
        frozen = read_json(frozen_path)
        if (frozen.get("scenario"), frozen.get("arm"), frozen.get("seed"),
            frozen.get("phase_elapsed_scale")) != (scenario, arm, seed, ARMS[arm]):
            result["errors"].append("frozen run identity or scale mismatch")
        if frozen.get("dqn_config") != config:
            result["errors"].append("frozen run config differs from summary")
    history_path = base / "training" / f"dqn_{scenario}_seed{seed}_training.csv"
    if not history_path.exists():
        result["errors"].append("episode history missing")
    else:
        with history_path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        result["history_rows"] = len(rows)
        if len(rows) != episodes or [int(r["episode"]) for r in rows] != list(range(1, episodes + 1)):
            result["errors"].append("episode history count or sequence mismatch")
        for row in rows:
            try:
                if any(not math.isfinite(float(row[key])) for key in
                       ("total_reward", "average_queue_length", "average_waiting_time",
                        "throughput", "max_queue_length", "vehicles_still_in_network")):
                    result["errors"].append("nonfinite training metric")
                    break
                if int(row["teleports"]) != 0 or int(row["blocked_switches"]) != 0:
                    result["errors"].append("training safety outcome nonzero")
                    break
            except (KeyError, ValueError, TypeError):
                result["errors"].append("invalid training metric")
                break
    model_dir = base / "models"
    expected_episodes = list(range(10, episodes + 1, 10))
    missing_checkpoints = [ep for ep in expected_episodes if not
                           (model_dir / f"checkpoint_ep{ep:03d}.pt").is_file()]
    if missing_checkpoints:
        result["errors"].append(f"missing checkpoints: {missing_checkpoints}")
    for name in ("best.pt", "final.pt", "metadata.json", "validation_history.json"):
        if not (model_dir / name).is_file():
            result["errors"].append(f"{name} missing")
    validation_path = model_dir / "validation_history.json"
    if validation_path.exists():
        checks = read_json(validation_path)
        result["validation_checks"] = len(checks)
        if [int(c["episode"]) for c in checks] != expected_episodes:
            result["errors"].append("validation checkpoint cadence mismatch")
        if any(c.get("seeds") != VALIDATION or len(c.get("rows", [])) != 5
               for c in checks):
            result["errors"].append("validation seed set or row count mismatch")
    for name in ("model_dir", "best_model", "final_model", "history_csv"):
        value = summary.get(name)
        if not value or not Path(value).resolve().is_relative_to(OUT.resolve()):
            result["errors"].append(f"{name} escapes attempt directory")
    result["status"] = "complete" if not result["errors"] else "invalid"
    return result


def main() -> int:
    cells = [audit_cell(scenario, arm, seed) for scenario in SCENARIOS
             for seed in SEEDS for arm in ARMS]
    factor_checks = []
    for scenario in SCENARIOS:
        for seed in SEEDS:
            paths = {arm: OUT / scenario / arm / f"seed{seed}" / "frozen_run_config.json"
                     for arm in ARMS}
            if all(path.exists() for path in paths.values()):
                configs = {arm: read_json(path) for arm, path in paths.items()}
                passed = (configs["control"].get("dqn_config") ==
                          configs["treatment"].get("dqn_config") and
                          configs["control"].get("phase_elapsed_scale") == 20.0 and
                          configs["treatment"].get("phase_elapsed_scale") == 60.0)
            else:
                passed = False
            factor_checks.append({"scenario": scenario, "seed": seed,
                                  "only_phase_age_divisor_differs": passed})
    counts = {status: sum(cell["status"] == status for cell in cells)
              for status in ("complete", "partial", "failed", "invalid", "not_started")}
    report = {"attempt": OUT.name, "parent_attempt": "dqn_stage5_20261121",
              "counts": counts, "cells": cells, "factor_checks": factor_checks,
              "all_20_cells_valid": counts["complete"] == 20 and
                                    all(check["only_phase_age_divisor_differs"]
                                        for check in factor_checks)}
    path = OUT / "training_artifact_audit.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(counts), flush=True)
    return 0 if report["all_20_cells_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
