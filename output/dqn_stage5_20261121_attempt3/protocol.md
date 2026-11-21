# IntelliSignal Stage 5 protocol: phase-age observability and DQN stability

Protocol initially frozen on 21 Nov 2026, before Stage 5 code changes or training. Two dated run-set amendments below record interrupted batches and the primary seed changes. Stage 5 outputs are isolated in this directory. The Stage 4 incumbent and historical outputs are protected.

## Question and single-factor hypothesis

Stage 4 reports run-to-run validation failures in both traffic scenarios. The 10th observation feature, `phase_elapsed`, is divided by `min(max_green, 4 * min_green)`. With the current environment values (`max_green=60 s`, `min_green=5 s`), it saturates at 20 s. The agent therefore cannot distinguish a green phase that has lasted 20 s from one approaching the enforced 60 s maximum. This is a plausible observability limitation, not a proven cause of Stage 4 variance.

Compare two otherwise identical configurations:

| Arm | Phase elapsed divisor | All other settings |
|---|---:|---|
| Control | Existing behavior: 20 s | Stage 4 settings unchanged |
| Treatment | Full legal phase age: 60 s | Stage 4 settings unchanged |

Only the divisor for observation feature 10 changes. Keep reward, network architecture, optimizer, replay, target update frequency, exploration schedule, episodes, environment and scenario files unchanged. Keep a selectable legacy default so existing models and tests retain their behavior.

## Fixed settings and seed splits

- Scenarios: moderate and heavy, each at the existing 1,800 simulated seconds.
- Independent training seeds, matched across arms: 6207–6211.
- Validation seeds: 6301–6305.
- Calibration seeds for raw/normalized feature diagnostics only: 6401–6410.
- Sealed confirmatory seeds: 6501–6520.
- Stage 5 training: 100 episodes/run; checkpoint and validation every 10 episodes; episode traffic varies by the existing seed schedule.
- Stage 4 degradation safeguard: stop after 3 consecutive validation checks more than 10% above the best waiting mean; preserve the best validation checkpoint and write all run/failure details.
- All seed sets were checked against Stage 3 and Stage 4 training, validation, calibration, and confirmatory sets, including 42–44, 55–56, 1001–1010, 2001–2010, 3001–3020, 5101–5210, and 6101–6120. No overlap is allowed.

Five seeds per arm and scenario are retained despite the estimated runtime (roughly 11 hours based on Stage 4 training durations). If execution is interrupted or the environment makes this impractical, record the incomplete cells and do not call the study robust or Stage 5 complete.

## Training, evaluation and gates

1. Capture hashes of current incumbent models, configs, and baseline results before code edits.
2. Train all matched control/treatment seeds in isolated directories. Reject runs with non-finite metrics or invalid simulator/safety outcomes. Save episode history, validation history, summary, config, best checkpoint and any failure details.
3. Greedily evaluate each of the 10 checkpoints on the five validation seeds, alongside the unchanged current DQN, starting DQN, original and tuned fixed-time controls, and a pressure-actuated heuristic tuned only on validation seeds. Preserve per-seed metrics: waiting, average/max queue, throughput, vehicles remaining, blocked switches, teleports and action diagnostics.
4. A configuration is validation-eligible only when at least 4 of its 5 independent runs each meet all per-run limits versus current DQN: mean waiting no more than +3%, average queue no more than +2%, throughput no more than -1%, zero blocked switches and zero teleports.
5. Select at most one eligible checkpoint per scenario from validation results only. If both arms qualify, report their paired per-seed validation comparison and select only by the frozen rule (lowest validation waiting among eligible checkpoints; control may be selected). Do not use calibration or confirmatory results for selection.
6. If a checkpoint is eligible, evaluate it once on the sealed, matched 20-seed confirmatory set against the current DQN, starting DQN, original/tuned fixed-time, and the validation-tuned actuated controller.
7. Promotion requires every Stage 4 confirmatory gate: at least 2% lower mean waiting than current DQN; at least 11/20 paired waiting wins; average queue regression <=2%; maximum queue regression <=5%; throughput regression <=1%; zero blocked switches and teleports; and full suite with no failures. Comparisons to tuned baselines are reported separately. Never replace an incumbent automatically.

## Diagnostics and reporting

On calibration seeds, report the raw p50/p95/p99/max of `phase_elapsed`, queue and waiting inputs per scenario/approach; the normalized phase-age distribution; and the fraction of observations clipped at 1.0, by arm. Calibration data may diagnose scale coverage but must not retune the treatment scale. Report paired arm differences and every independent run, including failed runs and stopping episodes.

## Runtime interruption addendum (frozen before replacement runs)


The first batch interruption notes initially described moderate/treatment seed 6201 as partial. Direct artifact inspection later confirmed that both moderate/control and moderate/treatment seed 6201 have complete 100-episode summaries and histories. They remain excluded as pilots because the primary seed set was amended before replacement training.

### Run-set amendment 2 — interrupted replacement batch

The second batch stopped during moderate/control seed 6202 at episode 78 of 100. Because replay and RNG state are not saved, this run cannot be resumed exactly; the partial run is preserved and excluded. Its paired moderate/treatment seed 6202 was never started. No other replacement cells ran. Replace the primary matched training set with seeds 6207–6211, preserving five complete pairs without overwriting prior artifacts. Validation, calibration, confirmatory seeds, and all gates remain unchanged. This addendum was recorded before the final batch was started.

### Retrospective artifact correction — 21 Nov 2026

Subsequent inspection found 100-episode `training_summary.json` files for both moderate seed-6202 arms, so the interruption paragraph above records an intermediate observation, not the final artifact state. The archived replacement-batch status still marks the treatment and later jobs failed (exit code 2), which conflicts with the seed-6202 treatment summary. Seed 6202 remains excluded from the primary analysis; the raw status log is preserved. A generated reconciliation table must list both status and artifact evidence without silently changing either record.

### Implementation and status reconciliation — 21 Nov 2026

The general DQN training entry point now defaults new training to a 60-second phase-age divisor. Stage 5 keeps its frozen control at 20 seconds and treatment at 60 seconds explicitly; this prevents the general training default from changing the experiment arms. The primary run status was reconciled from the preserved batch snapshots and per-cell artifacts: moderate/control seed 6207 is complete, moderate/treatment seed 6207 is partial, and the other 18 primary cells have no artifacts. No training runner was live at reconciliation. The manifest now reports Stage 5 as incomplete and preserves the former counters under `reported_counts_before_reconciliation`; the status JSON and logs are unchanged. See `run_reconciliation.md`.

## Completion gates

Stage 5 is complete only when all 20 training cells (2 scenarios x 2 arms x 5 seeds), required validation and any eligible confirmatory comparisons are accounted for; tests, integrity comparison, manifest, README/HANDOFF updates and visually inspected PDF exist; and the final decision is explicit. No Stage 6 work may start until the user explicitly confirms that the Stage 5 checklist is clear and complete.
