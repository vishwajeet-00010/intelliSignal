# Controlled experiment continuation

- Final simulations COMPLETE: 20 November 2026, 12:53:39 IST. Both scenario workers exited 0; 200 final episodes were saved. Do not restart training or simulations. Analysis was completed later on 20 November from saved records only.
- Validation: 160 episodes; six predeclared historical checkpoints plus fixed-time and initialization-seed-42 untrained control for each scenario, seeds 2001-2010. All episodes are 1800 seconds. The reviewer independently checked the full validation corpus.
- Selection: heavy `prior_best` (prior 300-episode run, episode 273), moderate `incumbent_final` (original 100-episode run, episode 100). Both passed the validation gate, so the predeclared optional retraining branch was not triggered. No learning-rate, discount or Double-DQN comparison was run. Training-seed variability remains unavailable for these single-seed historical models.
- Final freeze preceded all final results: SHA256 `f93c14addd160fadbdc7315a690fcd4039206d4a61659471fd63a296282c916a`. Final seeds 3001-3020 are now examined evidence and cannot be reused as a fresh untouched test.
- Final coverage: incumbent, frozen challenger, fixed-time, historical untrained initialization 0, and additional untrained initialization 42; 20 paired traffic seeds per controller and scenario. The additional control was documented before final access and never affected selection or promotion thresholds.
- Heavy FAIL: incumbent to challenger waiting 46.973960 to 47.623870 s; average queue 37.032833 to 37.546889; throughput 1345.00 to 1345.85; waiting wins/ties/losses 7/0/13. Retain incumbent.
- Moderate PASS under declared criteria: waiting 5.718732 to 5.431197 s; average queue 2.076917 to 1.999333; throughput 690.00 to 690.35; waiting wins/ties/losses 18/0/2. Eligible for human consideration only. Mean maximum queue worsened from 7.1 to 7.6; this tradeoff must remain visible. No model was installed or replaced.
- All final teleports zero. Inserted/remaining values and paired variability are in the final CSVs; read admission and remaining counts with throughput.
- Preservation after final analysis: 745 pre-existing files checked, zero changed/missing/conflicting hashes (`integrity.json`). Frozen source/checkpoint hashes remain valid. The original recovery code and file-lock retry protections were preserved and passed the independent synthetic verification.
- Runtime evidence: `launches/final_1795155701305132000.json`, its `_outcome.json`, `final-heavy.log`, `final-moderate.log`. Launcher 32252; recorded children 25976/21384, both exited 0. Elapsed time includes interruptions and is not a compute benchmark.
- Final derived evidence: `final_decision.json`, `final_metrics.csv`, `paired_final_metrics.csv`, `training_traffic_variability.csv`. Diagnostics: `diagnostics.json`, `policy_agreement.csv`, `weight_comparison.json`. Reproduction: `COMMANDS.md`.
- Delivery COMPLETE on 20 November 2026: independent final review passed all 200 episodes and 28,800 learned/untrained-policy decisions, including final-decision recomputation. The parent generated `REPORT.md` and the four-page comparison PDF, rendered all four pages, and passed visual QA. Final artifacts: `output/pdf/IntelliSignal_Controlled_Comparison_20261119.pdf` and this experiment's `REPORT.md`. No post-final tuning or further training is authorized by this protocol; the predeclared validation gate had already stopped the optional training branch before final testing.

Analysis-only commands (no new SUMO runs):

```powershell
.\.venv-stage3\Scripts\python.exe -B experiments/controlled_20261119/final_evaluate.py --analyze
.\.venv-stage3\Scripts\python.exe -B experiments/controlled_20261119/verify_integrity.py
```
