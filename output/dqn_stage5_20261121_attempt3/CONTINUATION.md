# Stage 5 attempt 3 continuation

This is the fresh primary attempt. Its parent is `output/dqn_stage5_20261121/`; the sandbox-blocked attempt 2 is preserved separately. The copied protocol, training runner, candidate trainer, evaluation script, and hash script match the parent byte for byte. Only the copied post-training/report workflow was adjusted for attempt-specific paths and stronger evidence checks.

At setup, the 20-cell serial training runner and a post-training watcher were started outside the filesystem/network sandbox, because SUMO/TraCI local sockets failed with WinError 10013 inside it. The watcher waits for terminal cell statuses, audits every cell, runs the full test suite, validation, eligible confirmatory evaluation, integrity comparison, and builds and renders an attempt-specific PDF. It does not claim Stage 5 complete before visual review.

Check `training_batch_status_final.json` and each run's `training_summary.json`, history CSV, checkpoint files, and validation history. The batch status can be stale after interruption, so use `training_artifact_audit.json` and `run_reconciliation.md` when available. The first watcher probe falsely reported an absent runner because Windows blocked a PID query; that probe is retained as `run_reconciliation_pipeline_probe1.md`. The corrected watcher uses recent training artifacts as a heartbeat.

After automation reaches a terminal result, inspect every image under `pdf_render/`, verify the report against `stage5_decision.json`, review `training_artifact_audit.json`, the manifest, test log, and before/after integrity files, then update the checklist and README/HANDOFF without erasing existing text. `handoff_update.md` is a generated draft for that review. Do not promote a model or start Stage 6 automatically.

If the runner stops before all 20 valid cells exist, preserve this attempt. Training replay and RNG state are not saved, so a partial cell cannot be resumed exactly. Any replacement must use another clean, separately named attempt.
