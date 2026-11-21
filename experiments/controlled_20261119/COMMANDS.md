# Reproduction and continuation

Run from `C:\Users\Vishwajeet\OneDrive\Desktop\project\intelliSignal` with the existing `.venv-stage3` Python. All invocations use `-B` to avoid modifying historical bytecode caches. SUMO/TraCI needs access to a local loopback socket; the restricted sandbox's initial attempt failed with `WinError 10013`, and the successful validation launch used elevated execution permission. This is local simulation, with no cloud service.

## Executed sequence

The predeclaration was written before new validation traffic was opened. The independent reviewer approved the shortlist, seed audit, safety-preserving instrumentation, and RNG isolation before execution. Preparation refuses to overwrite its existing manifest or protection snapshots:

```powershell
.\.venv-stage3\Scripts\python.exe -B experiments/controlled_20261119/run_controlled.py --prepare
.\.venv-stage3\Scripts\python.exe -B experiments/controlled_20261119/run_controlled.py --validation
```

`manifest.json` is the original predeclaration. `prevalidation_amendment.json` records the expanded seed-field audit and checkpoint-specific diagnostic discount factor; `validation_source_freeze.json` pins the final validation runner before successful execution. Historical seeds 1001-1010 are not reused as a new test set. Validation uses 2001-2010. Final seeds 3001-3020 must remain unopened until `freeze.json` exists.

## Continue interrupted validation safely

Do not launch a second process while validation is running. The original validation command was used for the first execution. Its original completion marker is exclusive, so use the separate continuation driver for a completed or interrupted rerun:

```powershell
.\.venv-stage3\Scripts\python.exe -B experiments/controlled_20261119/resume_validation.py
```

This verifies the frozen runner and checkpoint hashes, skips only complete matching 1800-second episodes, retains malformed/incomplete files under adjacent `recovery/` directories, writes new records atomically with bounded file-lock retries, and verifies no learning/replay/weight mutation. A completed continuation requires no new SUMO episodes.

## Analyze validation, then follow the fixed branch

The analysis requires all 160 validation episodes. It never opens final-test results:

```powershell
.\.venv-stage3\Scripts\python.exe -B experiments/controlled_20261119/analyze_controlled.py
```

`selection.json` records the original validation gates and conditional training decision for each scenario. If it says no training is needed, skip directly to freeze. Otherwise the bounded branch below runs only the triggered scenarios. It compares learning rate 0.001 against 0.0003, with every other learning setting fixed; three training seeds, 300 episodes per run, 1800 seconds per episode, and the same 27000-decision exploration schedule. Do not add recipes or extend the budget after inspecting results.

```powershell
# Conditional only; preparation is exclusive and must run once before --run.
.\.venv-stage3\Scripts\python.exe -B experiments/controlled_20261119/train_and_select.py --predeclare
.\.venv-stage3\Scripts\python.exe -B -u experiments/controlled_20261119/train_and_select.py --run
```

There are at most three worker processes, one per training seed, each with one Torch thread, private SUMO generated-input directories, private model/history/console files and full-state episode-boundary recovery. Control workers finish before the lower-LR group starts. Validation runs in a separate process. Checkpoints best/100/200/final receive the same greedy validation selection for both recipes. Rerunning `--run` resumes preserved complete state and skips fully selected runs; do not delete snapshots or recovery evidence. The preserved recovery implementation is imported from the previous experiment and hash-pinned in the new training predeclaration.

## Freeze and final evaluation

Only after selection is complete and independently reviewed:

```powershell
.\.venv-stage3\Scripts\python.exe -B experiments/controlled_20261119/final_evaluate.py --freeze
.\.venv-stage3\Scripts\python.exe -B -u experiments/controlled_20261119/run_final_parallel.py
.\.venv-stage3\Scripts\python.exe -B experiments/controlled_20261119/final_evaluate.py --analyze
```

Freeze records recipe, all selected replicate checkpoint paths and hashes, source hashes, seed list, numerical precision and fixed budget. A pre-final provenance amendment retains the validation initialization-seed-42 untrained control and adds the historical initialization-seed-0 control as a separate final controller, adding 40 final episodes across both scenarios. The original active validation source and records are preserved. Neither untrained control affects checkpoint selection or promotion thresholds. These two fixed initializations do not estimate a distribution of random policies. The evaluator refuses missing/changed provenance. A completed rerun validates and skips existing episodes. Final evidence cannot be used to revise selection or thresholds. Once evaluated, these seeds are examined evidence forever; a replay is a reproducibility check, not a fresh untouched test.

The frozen launcher permits two worker processes, one per scenario, with one Torch thread each and separate generated inputs. It writes `final-heavy.log`, `final-moderate.log`, and unique process/exit records under `launches/`. The final budget is 200 episodes. After interruption, inspect the recorded worker PIDs before restarting; an orphaned child may outlive the launcher, and no same-scenario duplicate should run. The launcher lock prevents concurrent launchers while its owner remains alive. For a serial continuation, the frozen evaluator also supports `--evaluate --scenario heavy` or `--evaluate --scenario moderate` after confirming no corresponding worker is alive.

Validation completed all 160 episodes. The saved-record analysis selected `prior_best` for heavy and `incumbent_final` for moderate; both satisfied the declared validation gates, so no new training was triggered. The safe validation-continuation command was run afterward and performed zero new simulations. Freeze was written before any final episode, with SHA256 `F93C14ADDD160FADBDC7315A690FCD4039206D4A61659471FD63A296282C916A`.

The final launcher was executed on 20 November 2026 using the command above and approved local-socket access. Both workers exited 0, completing all 200 final episodes at 12:53:39 IST. The attempt is recorded in `launches/final_1795155701305132000.json` and its `_outcome.json`. Final analysis and preservation verification were then executed from saved artifacts. Heavy failed the declared final gate; moderate passed and is eligible for human consideration only. Both incumbent models remain installed unchanged. The optional training commands were not executed; they remain documentation of the untriggered bounded branch.

## Independent verification and preservation

```powershell
.\.venv-stage3\Scripts\python.exe -B experiments/controlled_20261119/verify_review.py
.\.venv-stage3\Scripts\python.exe -B experiments/controlled_20261119/verify_records.py --partition validation
.\.venv-stage3\Scripts\python.exe -B experiments/controlled_20261119/verify_records.py --partition final
.\.venv-stage3\Scripts\python.exe -B experiments/controlled_20261119/verify_integrity.py
```

Consult each verifier's `--help` for the exact supported arguments. `verify_review.py` exercises exact recovery, lock retries and artifact continuation behavior with isolated temporary fixtures. `verify_records.py` recomputes observation normalization, greedy Q/action choices, reward/queue identities, signal resolution and offline TD arithmetic against checkpoint weights. Offline TD measurements are not the unavailable original training minibatch errors.

`integrity.json` compares every preexisting file in the two protection snapshots, including incumbent models, historical experiments, baseline results, scenario/network settings, safety modules and prior output artifacts. New experiment files are intentionally outside that protected historical set. Neither freeze nor evaluation installs a model or changes a dashboard default.

For a fresh replay, copy only the experiment driver `.py` files into a new sibling directory under `experiments/`, then prepare and run there. Keep this entire completed directory untouched. Update explicit output destinations in report-only commands for the new run. Replaying the same final seeds does not create new confirmatory evidence.

## Rebuild the report and PDF from saved evidence

These commands generate report deliverables only; they do not simulate, select checkpoints or change models. The runtime path below is the verified bundle used for the delivered PDF. Re-run dependency discovery if that installed bundle changes.

```powershell
.\.venv-stage3\Scripts\python.exe -B experiments/controlled_20261119/write_report.py
& 'C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1119-907-0_26.1119.907.0_x64__3k8sg7r9htsxt/dependencies/python/python.exe' -B experiments/controlled_20261119/build_comparison_pdf.py
pdftoppm -r 95 -png output/pdf/IntelliSignal_Controlled_Comparison_20261119.pdf tmp/pdfs/controlled_review/page
```

The delivered PDF has four pages. Each rendered page was visually inspected. Report regeneration replaces only the current report/PDF deliverables; previous experiments and their PDFs are protected and remain unchanged.
