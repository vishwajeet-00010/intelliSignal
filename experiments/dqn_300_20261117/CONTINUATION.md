# Experiment continuation

- COMPLETE: 19 Nov 2026, 10:52 IST. Run: experiments/dqn_300_20261117. Do not restart completed training/evaluation.
- Both scenarios: 300 contiguous episodes, 1800 seconds each, seed42; final checkpoints episode300. Selected best: heavy273, moderate248. Full-state recovery snapshots retained.
- Execution identity: launcher13012 / Python12376, launched18Nov10:47; status.json now complete. Prior interrupted dqn_300_20261116 preserved.
- Evaluation: 104 episodes total; reference42/43/44 and holdout1001-1010, candidate/incumbent plus fixed-time and untrained controls. No tuning against holdouts or automatic promotion.
- Holdout results (incumbent -> candidate): heavy waiting46.676 ->47.725s, queue36.805 ->37.667, throughput1345.5 ->1344.9; moderate waiting5.726 ->5.726s, queue2.082 ->2.082, throughput690.6 ->690.6. All teleports zero. Neither candidate passes promotion; retain both incumbents.
- Validation: specialist reviewed coverage, durations, finite metrics, contiguous histories and report/CSV consistency. Best/final weights load and are finite (final_validation.json). All163 protected files unchanged (integrity.json). Exact-resume and file-lock retry tests passed after recovery fix.
- Recovery history: resumed full state at86; Windows replacement failure at157 recovered from validated complete temporary snapshot, preserving both156/157 copies in heavy/recovery_157. Added bounded PermissionError retries; no architecture, reward, hyperparameter or selection changes.
- Artifacts: REPORT.md, paired_metrics.csv, integrity.json, final_validation.json; heavy/ and moderate/ contain models, training, evaluation, plots and resume.pt. Logs and manifest retained.
- Commands (reference only): .\.venv-stage3\Scripts\python.exe -u experiments/dqn_300_20261117/run_experiment.py ; analysis-only regeneration: .\.venv-stage3\Scripts\python.exe -B experiments/dqn_300_20261117/analyze_results.py
- Limitations: one training seed; reference seeds overlap training; identical recorded moderate metrics do not prove identical networks. Recorded elapsed times include long interruptions and are not compute benchmarks.
- Next: no required experiment work remains. Report results and preserve incumbent dashboard defaults.
