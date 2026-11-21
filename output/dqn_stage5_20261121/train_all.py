"""Serially run every frozen Stage 5 training cell without overwriting."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
import subprocess
import sys

OUT = Path(__file__).resolve().parent
PYTHON = sys.executable
SCENARIOS = ("moderate", "heavy")
ARMS = ("control", "treatment")
SEEDS = (6207, 6208, 6209, 6210, 6211)


def jobs():
    # Interleave matched seeds/arms so wall-clock or software drift is less
    # confounded with one arm always running first.
    for scenario in SCENARIOS:
        for seed in SEEDS:
            for arm in ARMS:
                yield scenario, arm, seed


def write_status(path: Path, jobs_list: list[dict], runner_pid) -> None:
    payload = {
        "runner_pid": runner_pid,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "jobs": jobs_list,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    manifest_path = OUT / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs_list = [{"scenario": s, "arm": a, "seed": seed, "status": "queued"}
                 for s, a, seed in jobs()]
    existing = [OUT / item["scenario"] / item["arm"] / f"seed{item['seed']}"
                for item in jobs_list
                if (OUT / item["scenario"] / item["arm"] /
                    f"seed{item['seed']}").exists()]
    if existing:
        print("Refusing to start the frozen batch because run directories already "
              "exist. Preserve them and create a separately named attempt before "
              "rerunning:", flush=True)
        for path in existing:
            print("  {}".format(path), flush=True)
        return 2

    manifest["training"]["status"] = "running"
    manifest["status"] = "training_in_progress"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    status_path = OUT / "training_batch_status_final.json"
    write_status(status_path, jobs_list, os.getpid())
    failed = 0
    for item in jobs_list:
        item["status"] = "running"
        write_status(status_path, jobs_list, os.getpid())
        command = [PYTHON, str(OUT / "train_candidate.py"),
                   "--scenario", item["scenario"], "--arm", item["arm"],
                   "--seed", str(item["seed"])]
        print("\n=== Stage 5 cell: {} {} seed {} ===".format(
            item["scenario"], item["arm"], item["seed"]), flush=True)
        result = subprocess.run(command, cwd=OUT.parents[1], check=False)
        item["return_code"] = result.returncode
        item["status"] = "completed" if result.returncode == 0 else "failed"
        failed += int(result.returncode != 0)
        write_status(status_path, jobs_list, os.getpid())
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["training"]["status"] = "complete_with_failures" if failed else "complete"
    manifest["training"]["completed_cells"] = sum(
        item["status"] == "completed" for item in jobs_list)
    manifest["training"]["failed_cells"] = failed
    manifest["status"] = "training_complete" if not failed else "training_batch_has_failures"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_status(status_path, jobs_list, None)
    print("Stage 5 training batch finished: {} complete, {} failed".format(
        manifest["training"]["completed_cells"], failed), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
