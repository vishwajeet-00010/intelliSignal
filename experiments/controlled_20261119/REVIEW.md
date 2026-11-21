# Independent review - controlled IntelliSignal experiment

Reviewer: `/root/review`, separate from the execution agent. Initial review was
performed before new validation or final-test results were available. Status:
**in progress; this document is not an approval of unexecuted phases.**

## Guidance and historical evidence

Read the root `HANDOFF.md` and the prior experiment's `manifest.json`,
`REPORT.md`, and `CONTINUATION.md`. No `AGENTS.md` was found in the repository
or its ancestor directories. Existing models, scenario settings, safety rules,
baseline measurements, and experiment artifacts must remain intact.

The old heavy candidate failed its declared promotion rule; moderate summary
metrics tied. Those observations do not justify automatic promotion or prove
that the moderate networks implement identical functions. Seeds 1001-1010
are previously examined historical evidence.

## Independent checks before results

- Independently scanned 454 historical CSV, JSON, YAML, Markdown, and Python
  artifacts under `experiments/`, `verification/`, `models/`, and `results/`,
  excluding this new experiment. Parsed CSV seed columns and inspected
  seed-labelled text. Found no seed-field evidence of validation seeds
  2001-2010 or final-test seeds 3001-3020. This is a repository-scoped audit,
  not proof about deleted files or runs outside the repository. The execution
  agent must retain its more structured audit and checkpoint metadata check.
- `rl/reward.py` penalizes interval mean current waiting on incoming lanes
  and incoming queue counts; fairness weight is zero. `utils/metrics.py`
  reports latest accumulated waiting per observed vehicle, averaged over all
  observed vehicles. Throughput is not an explicit reward component. These
  quantities should be related empirically, not equated.
- `training/train_dqn.py::_run_episode` stores **terminated**, not truncated,
  as replay done. Diagnostic TD residuals must preserve time-limit
  bootstrapping. Historical training-batch TD residuals were not logged and
  cannot be reconstructed from aggregate loss alone.
- `DQNAgent.load()` restores network weights and counters but does not
  automatically replace `agent.config` with the saved checkpoint config.
  Diagnostic gamma must therefore come from the checkpoint itself.
- Existing evaluation helpers reseed global RNG, and constructing networks
  consumes Torch RNG. A standalone evaluation process, or complete RNG
  capture/restoration including agent creation, is required when selecting
  checkpoints during training.
- `ActionController.resolve()` enforces max green at decision boundaries.
  With 10-second decisions and four seconds of clearance, some uninterrupted
  green runs may reach 66 seconds. Report measured completed and censored
  green durations without changing existing signal semantics.
- Count clipping from raw normalized features outside [0, 1]; an ordinary
  binary phase value of one is not clipping. Separate saturation at exactly
  one from values actually clipped.

## Items awaiting execution evidence

The execution agent proposed six checkpoints per scenario, greedy validation,
fixed-time/untrained controls, a bounded heavy-only learning-rate comparison
if needed, and recipe/checkpoint freezing before final tests. The final
manifest, code, actual seed coverage, recovery verification, protected hashes,
selection calculation, final pairing and interpretation remain to be audited.

Requested clarifications before results: exact validation gate and tie-breaks;
whether incumbent-final is an eligible existing candidate; final comparison of
all three training seeds without treating their matched traffic results as
60 independent traffic seeds; and explicit limits on any unperformed phase.

## Pre-run code review and recovery verification

Reviewed the written manifest and `run_controlled.py` before validation opened.
The manifest includes incumbent-final as a candidate, filters eligible
checkpoints before ranking, fixes validation/final guardrails, and limits the
one-factor experiment to learning rate 0.001 versus 0.0003. It reserves seeds
3001-3020 and records checkpoint hashes and initial source hash. The initial
maximum budget is 160 existing-checkpoint/control validation episodes, plus up
to six 300-episode runs per scenario only if the declared triggers apply.

Validation instrumentation uses greedy argmax, original environment/safety
logic, separate generated SUMO input directories, checkpoint-metadata gamma,
and masks only true termination in its offline chronological TD diagnostic.
All reviewed checkpoint configs use gamma 0.9. The RNG fingerprint assertion
covers global Python, NumPy, and CPU Torch RNG during policy episodes.

Requested a supplementary seed audit including actual `sumo_seed` training
history columns and `training_traffic_seeds` fields, plus enforcement of
manifest checkpoint hashes and inference-state invariance. The original seed
parser omitted those field names. This omission does not demonstrate leakage,
but the initial audit alone cannot fully support the stated training-seed
coverage claim. Supplementary evidence should preserve the original manifest.

Independent command run:

```powershell
.\.venv-stage3\Scripts\python.exe -B experiments/dqn_300_20261117/verify_resume.py
```

Passed: uninterrupted versus interrupted/resumed CPU weights, optimizer,
target network, replay contents, global/action/replay RNG, action traces, best
checkpoint rollback, final checkpoint equality, completed rerun without extra
episodes, and config/scenario mismatch refusal. The synthetic test uses no
SUMO runs and writes only temporary artifacts.

An independent in-memory monkeypatch test of `replace_with_retry` also passed:
two transient `PermissionError`s were retried and the third replacement
succeeded; exhausting retries preserved both complete temporary source and
the prior destination. This tests the retained implementation directly,
without modifying prior experiment files.

The execution agent addressed the pre-run requests before validation:
`seed_audit_expanded.json` includes the omitted seed fields and reports no
reserved-seed hits; `prevalidation_amendment.json` preserves the original
manifest while documenting the instrumentation changes. Checkpoint-direct
gamma, manifest checkpoint digest assertions, and before/after inference
weight/counter/replay comparisons are implemented. Independently verified
`run_controlled.py` SHA256
`56bc1766d1f7b21bdcbf4a025595b99cc8fff9e856290dd702da2f749faff7cb`
matches `validation_source_freeze.json`. Validation was approved under the
unchanged gate, shortlist, and budget. Final-test evaluation is not yet
approved or assessed.

The direct checks above are now reproducible together:

```powershell
.\.venv-stage3\Scripts\python.exe -B experiments/controlled_20261119/verify_review.py
```

Executed successfully; machine-readable evidence is in
`review_verification.json`. This verifier runs the inherited recovery check
and the transient/exhausted retry checks using temporary files. Synthetic
CPU recovery evidence does not by itself verify a real SUMO interruption.

## Validation record audit (partial; run still in progress)

Initial completed validation records passed the independent
`verify_records.py --partition validation` audit. The latest run at this
review update covered 11 episodes and 1,980 decisions. The machine-readable
`review_data_audit_validation.json` records exactly which episodes were
checked and can be regenerated after all episodes finish.

Checks include 1,800 measured seconds and time-series rows; greedy requested
actions; minimum-green blocking and maximum-green forcing; effective actions;
raw-feature clipping; phase and green-elapsed transitions; reward-component
identity; the exact queue reward relationship; and TD residual arithmetic
using termination-only masking. Float32 rounding uses a scale-aware bound
derived from machine epsilon, without changing the learning formula.

The independent verifier additionally reloads each recorded checkpoint and
recomputes all saved Q values plus 179 of 180 next-state target-network values
per episode. The final next-observation was not persisted, so its target
cannot be independently recomputed from this artifact alone; its recorded
target still participates in the residual arithmetic check. This limitation
does not affect traffic metric collection or checkpoint selection.

No training or final-test claim is approved by this partial record audit.

## Downstream protocol review before final access

Reviewed `train_and_select.py`, `artifact_io.py`, and `final_evaluate.py`
before their experiment phases ran. Corrected draft issues included a stale
analysis-module import, eager fallback evaluation in the metrics helper,
shortlist tie-break order, and unrounded final waiting/queue guardrails.
These were resolved before any final traffic seed was opened.

The optional training driver compares only learning rate 0.001 versus
0.0003; each recipe uses seeds 42, 43, and 44, 300 episodes, 1,800 seconds,
the same 27,000-decision exploration decay, and one Torch thread. Separate
worker processes redirect generated routes, additional files, and SUMO
configuration to per-scenario/recipe/seed directories. The shared network is
read-only. Validation runs in a separate child process. The retained recovery
module is hashed and unchanged. The prospective training traffic union is
42-343 and is disjoint from both new evaluation partitions.

The final driver freezes selection, model digests, source digests, original
manifest digest, numerical precision, seed list, and episode budget. It
checks those digests on evaluation and analysis. Gates use unrounded learned
policy waiting and queue metrics; fixed-time's two-decimal collector values
are descriptive comparisons. For training experiments, it first averages
replicates within each of 20 matched traffic seeds and separately reports
training-seed mean deltas and the predeclared t interval. It does not treat
60 observations as independent traffic seeds. No model replacement code is
introduced.

The new atomic artifact helper passed independent synthetic checks for
idempotent identical completed writes, refusal to overwrite different valid
artifacts, and preservation of malformed/null partial JSON before retry.
These results are included in the updated `review_verification.json`.

Downstream execution is approved **conditional on complete validation and
the declared selection calculation, followed by a frozen selection before
final access**. This approval concerns experimental procedure, not evidence
of improvement. The original active validation runner is unchanged; its
direct JSON writes and exclusive completion marker mean interruption or a
completed rerun needs careful artifact validation rather than blindly
trusting file existence. The downstream drivers use the reviewed atomic
helper and preserve invalid partials.

## Untrained-control provenance correction before final testing

Parent review found that the active runner's comment claiming a historical
seed-42 untrained match was incorrect. Independently confirmed
`evaluation/compare_controllers.py` constructs the historical random-weight
control with initialization seed **0**, not 42. The new runner constructs
seed 42. This distinction can materially change greedy untrained behavior;
the controls must not be described as identical.

Approved a transparent correction before final freeze: preserve the original
frozen runner and all seed-42 validation artifacts; label seed 42 as a new
initialization control; predeclare and add a separately named historical
seed-0 final control (40 additional episodes across the two scenarios); and
report both. The historical comparison may use seed 0 as its primary
untrained control. Neither untrained control determines checkpoint selection
or promotion, and no gate is relaxed. This correction does not establish
variability across all possible random initializations or prove that every
untrained policy is poor.

Also verified a pre-final eligibility correction requested by parent review:
the freeze records validation eligibility and diagnostic-only status, and
final promotion consideration requires that eligibility as well as all
unchanged final numerical guardrails. A checkpoint retained solely as a
diagnostic challenger cannot become promotable after its final results are
seen. The driver reports validation eligibility and final guardrails
separately.

The expanded protected-artifact check covers 745 unique pre-existing files,
including production code, previous experiment artifacts, baseline/scenario
files, and both existing PDFs under `output/pdf/`. The pre-final check found
no changed or missing files (`integrity_before_final.json`). A final check is
still required after all work finishes.

## Continued independent review after session interruption

The replacement independent reviewer `/root/review_continued` read the inherited
guidance and review evidence and repeated the complete saved-record audit after
the session resumed. At that point all 160 validation records were present,
and no final records or final freeze existed. The audit passed for all 160
episodes and 25,200 learned/untrained-controller decisions. It independently
recomputed saved Q values and 179 of 180 next-state target values per episode,
in addition to the reward, queue, duration, clipping, TD and action checks
described above. Evidence: `review_data_audit_validation.json`.

Independent recomputation directly from the raw validation summaries confirmed
the exact eligibility gates and shortlist tie-breaks: heavy selects
`prior_best` (mean waiting delta -0.888346 seconds, queue -0.588 vehicles,
throughput +4.7 vehicles, seven of ten waiting wins); moderate selects
`incumbent_final` (waiting -0.319088 seconds, queue -0.093722 vehicles,
throughput +0.9 vehicles, nine of ten waiting wins). Both satisfy validation
eligibility. The moderate tie among eligible incumbent-final, prior-final and
prior-episode-200 is resolved by the declared shortlist order. No final
results informed these calculations.

Both scenarios therefore take the predeclared no-new-training branch.
This does not provide variability across independent training seeds: these
historical selected checkpoints are seed-42 models. Do not report missing
training variability as zero or claim the optional learning-rate comparison
was executed. Final testing and final protection checks remain required.

### Final access approval

Before any final record existed, independently verified `freeze.json` SHA256
`f93c14addd160fadbdc7315a690fcd4039206d4a61659471fd63a296282c916a`, all five
frozen source hashes, four incumbent/selected checkpoint hashes, and the
manifest, selection and untrained-control amendment hashes. Both candidates
are validation-eligible. The fixed budget is 200 episodes: two scenarios,
20 traffic seeds, and five controllers per scenario (incumbent, selected,
historical untrained initialization 0, additional initialization 42, and
fixed-time). Approved final access under unchanged promotion thresholds.

The reviewed launcher uses at most two independent processes, one per
scenario, each with one Torch thread and separate generated inputs and logs.
It performs no training or selection. Source hashes and this concurrency
limit are frozen before final results. After an abrupt launcher interruption,
inspect saved child PIDs before restarting because the launcher lock covers
the parent lifetime and does not prove its children exited.

## Final independent acceptance - 20 November 2026

The final reviewer `/root/review_finish` repeated the complete saved-record
audit after both final workers exited successfully. The command
`python -B experiments/controlled_20261119/verify_records.py --partition final`
passed for **200 episodes and 28,800 learned/untrained decisions**. The audit
independently recomputed Q values, 179/180 next-state targets per episode,
reward/queue identities, clipping, signal safety resolution, TD arithmetic,
1800-second coverage, controller identity and final paired decision metrics.
Evidence is retained in `review_data_audit_final.json`. All frozen source
and checkpoint hashes still match. The protection check reports 745 files,
zero changed, zero missing and zero conflicting snapshot entries.

Independent paired recomputation confirms the frozen heavy challenger fails
the original promotion rule: waiting increases by 0.649910 seconds and mean
queue by 0.514056 vehicles, with seven waiting wins and thirteen losses.
Throughput increases by 0.85 vehicles and teleports remain zero; those gains
do not override the failed waiting/queue requirements. Retain the heavy
incumbent.

The frozen moderate challenger passes all original incumbent guardrails:
waiting decreases by 0.287536 seconds, mean queue by 0.077583 vehicles,
throughput increases by 0.35 vehicles, and waiting improves on eighteen of
twenty seeds. Teleports remain zero. Its mean maximum queue increases by
0.5 vehicles, which is a real tradeoff although maximum queue was not a
promotion guardrail. Both challengers complete fewer vehicles than fixed-time
(heavy -5.7; moderate -2.5), so lower waiting must not be presented as an
improvement in every metric. Moderate is eligible for human consideration;
this experiment did not replace either incumbent.

No new training was performed because both scenarios met the predeclared
validation gate using existing checkpoints. Therefore the optional lower-LR
versus control experiment was not run, and training-seed variability remains
unavailable. Traffic-seed variability is reported separately. This is one
selected historical training seed per scenario, not evidence of repeatability
across training seeds 42, 43 and 44. The selection and promotion rules were
not relaxed after final results. The final seeds are now examined evidence
and may not be described as untouched in later work.

**Experimental result accepted.** Report/PDF presentation review follows
after those deliverables are generated; this acceptance covers the saved
measurements, frozen selection and decision calculation.

### Deliverable review

Read the generated `REPORT.md` against `final_decision.json`, paired metrics,
validation diagnostics and the frozen selection. Tables, waiting win counts,
conditional no-training explanation and training-seed limitations agree with
the measured evidence. The prose explicitly includes moderate's +0.5 maximum
queue tradeoff and both challengers' lower throughput than fixed-time.

Independently inspected all four rendered pages of
`output/pdf/IntelliSignal_Controlled_Comparison_20261119.pdf` through
`tmp/pdfs/controlled_review/page-1.png` to `page-4.png`. Tables, charts, labels
and footers are legible with no observed clipping or overlap. The PDF separates
validation diagnostics from final measurements, identifies both untrained
initializations and states that no incumbent was replaced. **Report and PDF
accepted for delivery.**
