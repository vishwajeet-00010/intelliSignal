# Experiment continuation

- Active run: `experiments/dqn_300_20261117`; no newer experiment found. Original interrupted run `dqn_300_20261116` is preserved.
- Latest recovery: 18 Nov 2026, 10:47 IST. Launched hidden runner PID **13012**; see `launch.json`. Confirm actual process/lock and recent file progress before any relaunch.
- Validated recovery boundary: **heavy 157/300**, moderate not started. Snapshot: 28,260 decisions/replay entries, 27,761 updates, epsilon 0.05, best episode 154. Continue at 158, never restart completed episodes.
- Method: unchanged manifest; 300 episodes each, 1800 seconds, training seed 42; reference 42/43/44 and holdout 1001–1010, fixed-time/untrained controls plus incumbent comparison. No automatic promotion.
- Command: `.\.venv-stage3\Scripts\python.exe -u experiments/dqn_300_20261117/run_experiment.py` (requires local TraCI socket access outside sandbox). Runner holds `run.lock`, resumes full state, and skips completed stages.
- Monitor: `status.json`, `run.log`, `stderr-recovery157.log`, each scenario's `training/dqn_<scenario>_seed42_training.csv`, and `resume.pt` timestamp. Full snapshots save atomically after every episode.
- Recovery repair: Windows denied replacement of resume.pt at episode157. Specialist validated complete resume.pt.tmp157 against CSV, manifest, tensors and RNG; both156/157 preserved in `heavy/recovery_157/`, then157 committed. Added bounded PermissionError retries to atomic replacements; cause of Windows lock unconfirmed.
- Validation: independent specialist reviewed recovery/evaluation; 157 contiguous finite history rows, manifest/environment match, all163 protected files unchanged. Exact-resume synthetic test and transient/exhausted-retry checks passed after fix; do not repeat absent changes.
- Next: finish heavy training/evaluation, then moderate training/evaluation; runner generates `REPORT.md`, `paired_metrics.csv`, and `integrity.json`. Main owns execution/edits; reuse specialist for final evidence review. Preserve incumbent models regardless of candidate result until explicitly authorized otherwise.
- Milestone: heavy199 episodes validated after recovery; history checked contiguous/finite and current error log empty. Read live CSV/resume snapshot for exact latest episode.
