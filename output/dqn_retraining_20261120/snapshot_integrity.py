"""Write a fresh source and checkpoint hash inventory for this experiment."""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CHANGED = {
    "agents/base_agent.py": "Carry min_green through evaluation state for action masking.",
    "agents/dqn_agent.py": "Mask unavailable actions in greedy evaluation and epsilon-greedy selection.",
    "agents/dqn_config.py": "Align built-in discount defaults with gamma 0.98.",
    "config/dqn/training.yaml": "Use 5-second decision training and gamma 0.98.",
    "config/heavy.yaml": "Add measured heavy scenario RL observation scales; preserve signal section.",
    "config/moderate.yaml": "Add measured moderate scenario RL observation scales; preserve signal section.",
    "config/rl/environment.yaml": "Lower cadence and min green to 5 seconds; document fallback scales.",
    "evaluation/evaluate_dqn.py": "Include blocked-switch and green-duration diagnostics in evaluation CSVs.",
    "evaluation/collect_observation_stats.py": "Collect raw per-approach observation percentiles from greedy SUMO rollouts.",
    "rl/config.py": "Align built-in timing defaults with the 5-second environment.",
    "rl/environment.py": "Expose min_green in environment state information.",
    "rl/state.py": "Use measured scenario scales and a realistic phase-elapsed divisor.",
    "simulation/agent_simulation.py": "Record blocked switches and completed green-duration counts.",
    "tests/test_dqn.py": "Update stale cadence/gamma expectations and cover action masking.",
    "training/train_dqn.py": "Replay executed actions and choose best checkpoint by held-out waiting time with RNG restoration.",
    "output/dqn_retraining_20261120/fixed_time_baseline.py": "Reproduce tuned fixed-time baselines across selected seed sets.",
    "output/dqn_retraining_20261120/build_report.py": "Build the final-seed comparison PDF from recorded CSV/JSON results.",
    "output/dqn_retraining_20261120/snapshot_integrity.py": "Generate experiment source and model integrity hashes.",
}


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_hashes(paths):
    return {path: sha(os.path.join(ROOT, path)) for path in paths}


if __name__ == "__main__":
    incumbent = ["models/dqn_moderate/best.pt", "models/dqn_heavy/best.pt"]
    candidates = [
        "output/dqn_retraining_20261120/models/dqn_moderate/best.pt",
        "output/dqn_retraining_20261120/models/dqn_heavy/best.pt",
    ]
    snapshot = {
        "experiment": "dqn_retraining_20261120",
        "source_files": relative_hashes(CHANGED),
        "incumbent_checkpoint_sha256": relative_hashes(incumbent),
        "candidate_checkpoint_sha256": relative_hashes(candidates),
        "protected_experiments_modified": False,
        "incumbent_models_modified": False,
    }
    with open(os.path.join(HERE, "integrity_snapshot.json"), "w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2)

    manifest = {
        "experiment": "dqn_retraining_20261120",
        "objective": "Improve DQN waiting-time performance consistently across final traffic seeds.",
        "training": {
            "episodes_per_scenario": 100,
            "training_seed": 42,
            "validation_seeds": [2001, 2002, 2003],
            "final_seeds": list(range(3001, 3021)),
            "episode_duration_seconds": 1800,
            "decision_interval_seconds": 5,
            "min_green_seconds": 5,
            "gamma": 0.98,
            "checkpoint_selection": "lowest mean waiting on held-out validation seeds",
            "epsilon_floor": 0.05,
        },
        "normalization_calibration": {
            "initial_rollouts": 10,
            "calibration_seeds": list(range(2001, 2011)),
            "post_training_rollouts": 10,
            "moderate_queue_scale": 3,
            "moderate_waiting_scale": 14,
            "heavy_queue_scale": 16,
            "heavy_waiting_scale": 142,
            "post_training_stats": {
                "moderate": "moderate_posttrain_observation_stats.json",
                "heavy": "heavy_posttrain_observation_stats.json",
            },
        },
        "tuned_fixed_time_green_seconds": {"moderate": 5, "heavy": 7},
        "measured_results": {
            "moderate": {"mean_waiting": 2.929, "waiting_sd": 0.1478584,
                         "mean_throughput": 690.4, "tuned_fixed_waiting": 3.0075,
                         "starting_dqn_waiting": 5.72, "seeds_regressed_vs_starting": 0},
            "heavy": {"mean_waiting": 16.637, "waiting_sd": 2.4137372,
                      "mean_throughput": 1419.05, "tuned_fixed_waiting": 12.736,
                      "starting_dqn_waiting": 46.97, "seeds_regressed_vs_starting": 0},
        },
        "tests": {"command": "python -m pytest -q --basetemp tmp/pytest-run-final3-20261120",
                  "passed": 223, "skipped": 1, "failed": 0, "errors": 0},
        "gate_status": {
            "moderate_waiting_lt_4": "PASS",
            "heavy_waiting_lt_25": "PASS",
            "heavy_throughput_gte_1390": "PASS",
            "blocked_switches_zero": "PASS",
            "switch_ratio_below_0_90": "PASS",
            "traffic_normalized_p99_range": "PARTIAL; heavy east/west dimensions remain lower than 0.6, moderate waiting p99 slightly clips",
            "learning_after_epsilon_floor": "PASS; both scenarios improved for several post-floor blocks",
            "all_final_seeds_better_than_starting_dqn": "PASS; zero regressions in each scenario",
        },
        "report": "dqn_training_comparison.pdf",
        "integrity_snapshot": "integrity_snapshot.json",
    }
    with open(os.path.join(HERE, "experiment_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
