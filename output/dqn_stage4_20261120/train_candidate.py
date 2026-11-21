"""Train one isolated Stage 4 candidate. Existing artifacts are never replaced."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))

from agents.dqn_config import load_dqn_config
from training.train_dqn import train


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("moderate", "heavy"), required=True)
    parser.add_argument("--seed", type=int, choices=(55, 56), required=True)
    args = parser.parse_args()
    run_dir = OUT / args.scenario / f"seed{args.seed}"
    model_dir, history_dir = run_dir / "models", run_dir / "training"
    summary_path = run_dir / "training_summary.json"
    if run_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing run directory: {run_dir}")
    model_dir.mkdir(parents=True)
    history_dir.mkdir(parents=True)
    config = load_dqn_config(episodes=100, seed=args.seed, gamma=0.98,
                             checkpoint_every=10)
    try:
        summary = train(
            scenario=args.scenario, config=config,
            model_dir=str(model_dir), results_dir=str(history_dir),
            quiet=False, progress_every=10, save_plots=False,
            validation_seeds=(5101, 5102, 5103, 5104, 5105),
            degradation_tolerance=0.10, degradation_patience=3)
    except Exception as exc:
        failure = {"scenario": args.scenario, "seed": args.seed,
                   "status": "failed", "error": repr(exc),
                   "config": config.as_dict()}
        (run_dir / "training_failure.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8")
        raise
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest_path = OUT / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = f"{args.scenario}_seed{args.seed}"
    manifest.setdefault("candidates", {})[run_id] = {
        "status": "completed_early_stopped" if summary["stopped_early"] else "completed",
        "training_summary": str(summary_path),
        "checkpoint": str(model_dir / "best.pt"),
        "history": str(history_dir / f"dqn_{args.scenario}_seed{args.seed}_training.csv"),
        "validation_history": str(model_dir / "validation_history.json"),
        "episodes_completed": summary["episodes"],
        "best_validation_episode": summary["best_window_episode"],
        "best_validation_waiting": summary["best_validation_mean_waiting_time"],
        "stop_reason": summary["stop_reason"],
        "config": summary["config"],
    }
    manifest.setdefault("training_runs_started", []).append(run_id)
    manifest["training_runs_started"] = list(dict.fromkeys(
        manifest["training_runs_started"]))
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Stage 4 candidate saved: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
