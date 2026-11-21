"""Refresh manifest candidate records from completed training summaries."""
from __future__ import annotations

import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent


def main():
    manifest_path = OUT / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for scenario in ("moderate", "heavy"):
        for seed in (55, 56):
            run_dir = OUT / scenario / f"seed{seed}"
            summary_path = run_dir / "training_summary.json"
            failure_path = run_dir / "training_failure.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                key = f"{scenario}_seed{seed}"
                manifest.setdefault("candidates", {})[key] = {
                    "status": "completed_early_stopped" if summary.get("stopped_early") else "completed",
                    "training_summary": str(summary_path),
                    "checkpoint": str(run_dir / "models" / "best.pt"),
                    "history": str(run_dir / "training" /
                                    f"dqn_{scenario}_seed{seed}_training.csv"),
                    "validation_history": str(run_dir / "models" / "validation_history.json"),
                    "episodes_completed": summary["episodes"],
                    "best_validation_episode": summary["best_window_episode"],
                    "best_validation_waiting": summary["best_validation_mean_waiting_time"],
                    "stop_reason": summary.get("stop_reason"),
                    "config": summary["config"],
                }
                manifest.setdefault("training_runs_started", []).append(key)
            elif failure_path.exists():
                manifest.setdefault("candidates", {})[f"{scenario}_seed{seed}"] = json.loads(
                    failure_path.read_text(encoding="utf-8"))
    manifest["training_runs_started"] = list(dict.fromkeys(
        manifest.get("training_runs_started", [])))
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Updated {manifest_path}")


if __name__ == "__main__":
    main()
