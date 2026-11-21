"""Run the remaining predeclared independent candidate jobs sequentially."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

OUT = Path(__file__).resolve().parent
SCRIPT = OUT / "train_candidate.py"
JOBS = (("moderate", 56), ("heavy", 55), ("heavy", 56))


def main() -> int:
    status_path = OUT / "remaining_training_status.json"
    status = {"jobs": []}
    for scenario, seed in JOBS:
        job = {"scenario": scenario, "seed": seed, "status": "running"}
        status["jobs"].append(job)
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        completed = subprocess.run([
            sys.executable, str(SCRIPT), "--scenario", scenario,
            "--seed", str(seed)], cwd=str(OUT.parents[1]), check=False)
        job["return_code"] = completed.returncode
        job["status"] = "completed" if completed.returncode == 0 else "failed"
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return 0 if all(job["return_code"] == 0 for job in status["jobs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
