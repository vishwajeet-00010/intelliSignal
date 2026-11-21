"""Run one isolated, predeclared Stage 5 training cell."""
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

SEEDS = (6207, 6208, 6209, 6210, 6211)
# Stage 5's control is the frozen legacy scale even though new standalone
# training now defaults to the full legal phase-age range.
ARMS = {"control": 20.0, "treatment": 60.0}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("moderate", "heavy"), required=True)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    run_id = f"{args.scenario}_{args.arm}_seed{args.seed}"
    run_dir = OUT / args.scenario / args.arm / f"seed{args.seed}"
    if run_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing run: {run_dir}")
    model_dir, history_dir = run_dir / "models", run_dir / "training"
    model_dir.mkdir(parents=True)
    history_dir.mkdir(parents=True)
    scale = ARMS[args.arm]
    config = load_dqn_config(episodes=100, seed=args.seed, gamma=0.98,
                             learning_rate=0.001, batch_size=64,
                             target_update_frequency=500,
                             checkpoint_every=10, exploration_fraction=0.5)
    config_payload = config.as_dict()
    (run_dir / "frozen_run_config.json").write_text(json.dumps({
        "scenario": args.scenario, "arm": args.arm, "seed": args.seed,
        "phase_elapsed_scale": scale,
        "changed_factor": "phase_elapsed observation divisor only",
        "dqn_config": config_payload,
    }, indent=2), encoding="utf-8")
    try:
        summary = train(
            scenario=args.scenario, config=config,
            model_dir=str(model_dir), results_dir=str(history_dir),
            quiet=False, progress_every=10, save_plots=False,
            validation_seeds=(6301, 6302, 6303, 6304, 6305),
            degradation_tolerance=0.10, degradation_patience=3,
            phase_elapsed_scale=scale)
    except Exception as exc:
        failure = {"run_id": run_id, "scenario": args.scenario,
                   "arm": args.arm, "seed": args.seed,
                   "status": "failed", "error": repr(exc),
                   "phase_elapsed_scale": scale, "config": config_payload}
        (run_dir / "training_failure.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8")
        manifest_path = OUT / "experiment_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("training_runs", {})[run_id] = failure
        manifest["status"] = "training_failed_cell_recorded"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise
    summary.update({"arm": args.arm, "phase_elapsed_scale": scale})
    summary_path = run_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest_path = OUT / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("training_runs", {})[run_id] = {
        "status": "completed_early_stopped" if summary["stopped_early"] else "completed",
        "summary": str(summary_path),
        "checkpoint": str(model_dir / "best.pt"),
        "history": str(history_dir / f"dqn_{args.scenario}_seed{args.seed}_training.csv"),
        "validation_history": str(model_dir / "validation_history.json"),
        "best_validation_episode": summary["best_window_episode"],
        "best_validation_waiting": summary["best_validation_mean_waiting_time"],
        "episodes_completed": summary["episodes"],
        "stopped_early": summary["stopped_early"],
        "stop_reason": summary["stop_reason"],
        "phase_elapsed_scale": scale,
        "config": config_payload,
    }
    manifest["training_runs_started"] = list(dict.fromkeys(
        manifest.get("training_runs_started", []) + [run_id]))
    manifest["status"] = "training_in_progress"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Stage 5 cell saved: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
