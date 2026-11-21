# Stage 5 attempt 3 — parked on 21 Nov 2026

**Status: parked, not abandoned and not complete.** No result is claimed. All artifacts,
the protocol, the lineage notes and the batch status are preserved exactly as they were.

## Where it stopped

`training_batch_status_final.json` (updated 2026-11-21T15:51:03+05:30) records **1 of 20
cells complete**:

| cell | status |
|---|---|
| moderate / control / 6207 | completed |
| moderate / treatment / 6207 | running |
| the other 18 cells | queued |

Per `CONTINUATION.md`, replay and RNG state are not saved, so a partial cell cannot be
resumed exactly. Any continuation must be a new, separately named attempt. Nothing in this
directory should be reused as a result.

## Why it was parked

Stage 5 compares the `phase_elapsed` observation divisor at 20 s (control) against 60 s
(treatment). The protocol is sound — frozen before training, matched seeds, sealed
confirmatory set, honest amendments. The problem is priority, not method:

1. **The feature being studied barely varies.** 99 % of moderate greens and 92 % of heavy
   greens are exactly the 6 s minimum, so `phase_elapsed` is almost always ≈ 6 s. Under the
   control arm that normalises to 0.30 and under the treatment arm to 0.10. Neither arm
   saturates, and `forced_switches` is **0** in every recorded run, so the 60 s `max_green`
   is never approached. There is very little signal for this experiment to resolve.

2. **A larger constraint is binding.** The action space can only produce greens of
   {6, 11, 16, 21, …}, because a switch leaves `decision_interval - yellow = 1 s` already
   served. The measured heavy optimum is a **7 s** green, which is not in that set. A plain
   7 s fixed-time plan reaches 12.736 s mean waiting against the Stage 4 incumbent's
   16.469 s. Until the action space can express that region, arm-level tuning of one
   observation feature cannot move the headline result.

3. **The experiment may be invalidated by the fix.** The recommended change to the action
   space alters what `phase_elapsed` means and how often it varies, so results gathered now
   would not transfer.

## Defect found in this attempt's evaluator — fix before any re-run

`stage5_eval.py:220-221` and `:324-325` construct the tuned fixed-time baseline as:

```python
FixedTimeAgent(ns_green=green, ew_green=green, yellow=3.0)
```

and pass it to `run_policy`, which routes everything except `fixed_time_original` through
`run_agent_episode`, i.e. through `IntelliSignalEnv`.

Two independent errors, inherited from `output/dqn_stage4_20261120/stage4_eval.py`:

1. **The environment quantises the green.** A 7 s request becomes an 11 s green. In Stage 4
   this produced a baseline of 36.799 s where the real 7 s plan measures 12.736 s. Meanwhile
   `fixed_time_original` runs through SUMO's own TLS program, so one comparison table mixes
   two execution mechanisms.
2. **`yellow=3.0` contradicts the scenario files**, which set `signal.yellow: 4`. Every other
   controller in the comparison uses 4 s.

**These lines were deliberately left unedited.** The protocol states that this attempt's
scripts match the parent byte for byte, and editing a frozen experiment's code in place
would destroy that guarantee. The correction belongs in the next attempt.

A corrected, reusable implementation now exists at **`evaluation/fixed_time_sweep.py`**,
which runs fixed-time plans through SUMO's own TLS program and takes yellow from the
scenario. Any new attempt must use it — and must not call `FixedTimeAgent` through
`run_agent_episode` for a baseline that is meant to represent a real fixed-time plan.

## Before resuming

1. Land the action-space fix so greens near 7 s are reachable
   (`STAGE5_AUDIT_FIX_BRIEF.md`, finding F3).
2. Re-calibrate the observation scales afterwards — the queue distribution changes.
3. Re-ask whether the `phase_elapsed` divisor still matters once greens actually vary.
4. Start a **new** attempt directory. Do not write into this one.

See `output/dqn_stage4_20261120/CORRECTION.md` for the corrected Stage 4 comparison and
`STAGE5_AUDIT_FIX_BRIEF.md` for the full audit.
