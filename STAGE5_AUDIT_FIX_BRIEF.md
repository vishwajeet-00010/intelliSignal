# IntelliSignal — full-stack audit (Stage 5 → Stage 1) and implementation brief

**Audience:** a coding agent (Codex) implementing the fixes.
**Audit date:** 2026-11-21, against `origin/main` at commit `8b8fa34`.
**Supersedes:** `DQN_PLATEAU_FIX_BRIEF.md` (2026-11-20). FIX 1–5 and 7 of that brief were
implemented; this brief records what worked, what is still wrong, and what to do next.
**Do not re-derive the measurements.** They are reproducible via Appendix A of the previous
brief and the evidence paths cited inline.

---

## 0. Verdict in one page

The previous brief's fixes **worked, and the project improved enormously**:

| scenario | before (Stage 3 incumbent) | after (Stage 4 incumbent) | change |
|---|---|---|---|
| moderate mean waiting | 5.72 s | **2.918 s** | −49 % |
| heavy mean waiting | 46.97 s | **16.469 s** | −65 % |
| heavy throughput | 1345 | **1420.6** | +75 vehicles |

**But three defects remain, and one of them inverts a headline conclusion.**

| # | Severity | Finding |
|---|---|---|
| **F1** | **Critical** | Stage 4's `fixed_time_tuned` is an **11-second** controller, not the 7-second one it claims. The real 7 s plan measures **12.736 s**; Stage 4 reports **36.799 s**. Its conclusion "best overall controller = incumbent_dqn" is **wrong for heavy**. |
| **F2** | **Critical** | The switch-ratio gate is **mathematically incapable of failing**. It reports PASS while the true value is 0.95 (heavy) / 0.99 (moderate) — a clear FAIL. The agent is still pinned at the switching ceiling. |
| **F3** | **Critical** | The action space **cannot express the heavy optimum**. Reachable greens are {6, 11, 16, 21…}; the optimum is 7 s. |
| F4 | Medium | Observation scales are per-scenario but not per-approach; heavy E/W dimensions use only 20–25 % of range. |
| F5 | Medium | On heavy the DQN beats a 20-line heuristic by **1.4 %**. Report this honestly. |
| F6 | Low | `-inf * 0 = NaN` hazard in the masked target. Currently unreachable, but silent if it ever fires. |
| F7 | Low | Stage 5 is spending ~11 h on a hypothesis that cannot matter much while F3 is the binding constraint. |
| F8 | Blocker | `.venv-stage3` is still unusable on this machine — tests and training cannot run. |

**Bottom line for heavy: a plain 7-second fixed-time plan (12.736 s) still beats the
trained DQN (16.469 s) by 29 %, with higher throughput (1441 vs 1421).** That is not
visible in the Stage 4 report because the baseline it compares against is misconfigured.

---

## 1. What was implemented since the last brief — all verified correct

Confirmed by reading the diff `c7707b6..8b8fa34`:

| previous FIX | status | evidence |
|---|---|---|
| FIX 1 — break the 16 s floor | **Done** | `config/rl/environment.yaml:29,33` now `decision_interval: 5`, `min_green: 5`; `rl/config.py:33-34` defaults match |
| FIX 2 — store executed action | **Done, and extended** | `training/train_dqn.py:377` stores `info["effective_action"]`; `agents/dqn_agent.py` adds `valid_actions` masking; `agents/replay_buffer.py` adds `next_action_mask`; masked target at `agents/dqn_agent.py:775-778` |
| FIX 3 — per-scenario observation scales | **Done (partial)** | `config/heavy.yaml` `rl.observation` = 16/142; `config/moderate.yaml` = 3/14; new `evaluation/collect_observation_stats.py` |
| FIX 4 — longer horizon | **Done** | `config/dqn/training.yaml:50` `gamma: 0.98` |
| FIX 5 — select on real objective | **Done** | checkpoint chosen by "lowest mean waiting on held-out validation seeds"; RNG restoration added |
| FIX 7 — tune the baseline | **Done then broken** | measured correctly in `output/dqn_retraining_20261120/fixed_time/`, then re-implemented wrongly in Stage 4 (see F1) |

Additional good work not asked for, which should be kept:

- `models/active_models.json` records an explicit **observation contract** per model
  (scales, `phase_elapsed_scale`, `decision_interval`, `min_green`, yellow). This is the
  right way to stop a checkpoint being loaded under the wrong normalisation.
- `requirements-win-py312.lock` pins the exact working environment.
- Blocked switches are now **zero everywhere** — the masking works.
- `output/dqn_retraining_20261120/experiment_manifest.json` honestly records
  `"tuned_fixed_waiting": 12.736` against `"mean_waiting": 16.637` for heavy. The data is
  right; only Stage 4's re-measurement went wrong.

---

## 2. Findings

### F1 (CRITICAL) — The tuned fixed-time baseline in Stage 4 is an 11-second controller

**Where:** `output/dqn_stage4_20261120/stage4_eval.py:131-133` and `:250-253`

```python
green = 5.0 if scenario == "moderate" else 7.0
run_rows += run_policy(scenario, "fixed_time_tuned",
                       FixedTimeAgent(ns_green=green, ew_green=green,
                                      yellow=3.0), VALIDATION_SEEDS)
```

Two separate errors in these three lines.

**Error 1 — wrong execution path.** `run_policy` sends every label except
`fixed_time_original` through `run_agent_episode`, which drives `IntelliSignalEnv`
(`simulation/agent_simulation.py:247`). The environment quantises green duration to
decision boundaries and enforces `min_green`. A `FixedTimeAgent` asking for a 7 s green
inside a 5 s-cadence environment cannot produce one.

Measured proof, from `completed_green_duration_counts` in
`output/dqn_stage4_20261120/heavy/validation/validation_rows.csv`:

| controller | actual completed greens |
|---|---|
| heavy `fixed_time_tuned` | **11 s × 99 %**, 10 s × 1 % |
| heavy `incumbent_dqn` | 6 s × 92 %, 11 s × 7 %, 16 s × 1 % |
| moderate `fixed_time_tuned` | 6 s × 99 % |

Its switch count is **120 over 360 decisions** = one phase change every 15 s = an 11 s
green plus 4 s yellow. The sweep in the previous brief measured green = 11 at **37.13 s**.
Stage 4 recorded **37.70 s** (validation) / **36.799 s** (confirmatory). The numbers match
an 11-second controller exactly.

Meanwhile `fixed_time_original` goes through `run_fixed_time_episode`, i.e. SUMO's own TLS
program. **One comparison table therefore mixes two different execution mechanisms.**

**Error 2 — `yellow=3.0` contradicts the scenario.** Both scenario files set
`signal.yellow: 4`. `fixed_time_original`, every DQN run and the previous brief's sweep all
use 4 s. Only `fixed_time_tuned` uses 3 s. Even if Error 1 were fixed, the baseline would
still be running a different clearance interval from everything it is compared against.

**Consequence.** `output/dqn_stage4_20261120/promotion_decision.json` states
`"recommend_best_overall_controller": "incumbent_dqn"` for heavy. The correct ranking on
heavy, using the properly executed 7 s plan already measured in
`output/dqn_retraining_20261120/fixed_time/heavy_tuned.csv` (20 seeds):

| controller | mean waiting | throughput |
|---|---|---|
| **tuned fixed-time, 7 s green, SUMO TLS** | **12.736** | **1441.0** |
| actuated_tuned (heuristic) | 16.700 | 1439.1 |
| incumbent_dqn | 16.469 | 1420.6 |
| starting_dqn | 19.411 | 1406.2 |
| fixed_time_tuned *as Stage 4 ran it* | 36.799 | 1359.2 |
| fixed_time_original (30 s) | 53.030 | 1349.8 |

Stage 4's own pre-training audit already said this —
`output/dqn_stage4_20261120/audit.md`: *"the heavy DQN remains behind tuned fixed-time on
mean waiting"* — and the conclusion then contradicted it.

**On moderate the conclusion survives**: DQN 2.918 vs true 5 s plan 3.007. A genuine, if
small, win.

#### Fix F1

1. In `stage4_eval.py` (and any copy in Stage 5), run **every** fixed-time baseline through
   the same SUMO TLS path as `fixed_time_original`, parameterised by green duration —
   reuse `output/dqn_retraining_20261120/fixed_time_baseline.py::run_one`, which is already
   correct, rather than `FixedTimeAgent` inside the env.
2. Delete the `yellow=3.0` override. Take yellow from the scenario config.
3. If an in-environment fixed-time controller is genuinely wanted as a *separate* row,
   keep it but label it for what it is, e.g. `fixed_time_tuned_in_env_11s`, and state in
   the report that it is quantised by the decision cadence.
4. Regenerate `promotion_decision.json` and `stage4_validation_report.pdf`. The
   `recommend_best_overall_controller` field must be recomputed; for heavy it becomes the
   tuned fixed-time plan, not the DQN.
5. **Do not overwrite the existing Stage 4 artifacts in place.** Write a corrected
   companion (e.g. `promotion_decision_corrected.json`) and a short
   `CORRECTION.md` explaining what changed and why, so the original record survives.

---

### F2 (CRITICAL) — The switch-ratio gate cannot fail

**Where:** `output/dqn_retraining_20261120/build_report.py:178-179` and `:222`

```python
switch_ratio = statistics.fmean(
    float(row["switch_count"]) / float(row["decisions"]) for row in new)
...
["Switches / decisions below 0.90", "{:.2f}".format(switch_ratio),
 "PASS" if switch_ratio < .90 else "FAIL"],
```

The previous brief's Gate B was:

> `effective_switches / max_possible_switches` is below 0.90

**`decisions` is the wrong denominator.** With `decision_interval=5` and `yellow=4`, after
a switch `green_elapsed = 5 - 4 = 1`, which is below `min_green = 5`, so the next decision
is always blocked. **A switch is possible on at most every second decision**, so
`switch_count / decisions` can never exceed **0.5** and the test always passes.
`experiment_manifest.json` duly records `"switch_ratio_below_0_90": "PASS"`.

The correct denominator is `decisions / 2`. Actual values, from the Stage 4 validation rows:

| scenario | controller | switches | ÷ 180 | gate |
|---|---|---|---|---|
| moderate | incumbent_dqn | 178.8 | **0.993** | **FAIL** |
| moderate | candidate_seed55 | 179.2 | **0.996** | **FAIL** |
| moderate | candidate_seed56 | 178.8 | **0.993** | **FAIL** |
| heavy | incumbent_dqn | 171.0 | **0.950** | **FAIL** |
| heavy | candidate_seed55 | 171.6 | **0.953** | **FAIL** |
| heavy | candidate_seed56 | 168.2 | **0.934** | **FAIL** |

**The agent is still pinned against the switching ceiling.** The floor moved from 16 s to
6 s, which is why performance improved so much, but the policy is still essentially
"switch as fast as the safety layer allows": 99 % of moderate greens and 92 % of heavy
greens are exactly the 6 s minimum.

The previous brief anticipated exactly this: *"If it is still pinned at the ceiling, the
agent has merely found a new floor and FIX 6 is required."*

#### Fix F2

In `build_report.py`, and every copy of the gate in `output/dqn_stage*/build_report.py`:

```python
# BEFORE
switch_ratio = statistics.fmean(
    float(row["switch_count"]) / float(row["decisions"]) for row in new)

# AFTER — the ceiling is one switch per two decisions, because a switch leaves
# green_elapsed = decision_interval - yellow, which is always below min_green.
def _max_switches(row):
    decisions = float(row["decisions"])
    interval = float(row.get("decision_interval", 5.0))
    yellow = float(row.get("yellow_time", 4.0))
    min_green = float(row.get("min_green", 5.0))
    blocked_after_switch = 1 if (interval - yellow) < min_green else 0
    return decisions / (1.0 + blocked_after_switch)

switch_ratio = statistics.fmean(
    float(row["switch_count"]) / _max_switches(row) for row in new)
```

and relabel the row **"Switches / maximum possible switches below 0.90"**. Persist
`decision_interval`, `yellow_time` and `min_green` into the per-run CSVs so the denominator
is derived, not hardcoded. Re-run the gate; expect FAIL, and say so in the report.

---

### F3 (CRITICAL) — The action space cannot express the heavy optimum

**Where:** `rl/action.py:151-160` (`commit`) combined with
`config/rl/environment.yaml:29,33`.

After a switch, `green_elapsed = decision_interval - yellow - all_red = 5 - 4 - 0 = 1`.
Greens therefore land on `1 + 5k`:

```
reachable green durations = {6, 11, 16, 21, 26, ...}
```

Confirmed by the measured distributions in §F1 — every controller in the environment,
learned or scripted, produces only 5 s, 6 s, 10 s, 11 s, 16 s, 21 s, 25 s, 26 s greens.
**7 s is not in the set, and 7 s is the heavy optimum.**

From the previous brief's sweep (5 seeds each, real SUMO TLS):

| green | heavy mean waiting |
|---|---|
| 6 s | 21.40 |
| **7 s** | **12.46** |
| 8 s | 24.13 |
| 11 s | 37.13 |

The DQN mixes 6 s (92 %) and 11 s (7 %) and reaches **16.47** — better than pure 6 s
*or* pure 11 s, which is **real adaptive gain and should be credited**. But no mixture of
6 and 11 reaches 12.74. The actuated heuristic, under the same constraint, plateaus at the
same 15–17 s. **The quantisation is the binding constraint, not the learning.**

#### Fix F3 — two options

**Option A (smaller, recommended first).** Decouple the yellow from the decision interval
so the interval can shrink without starving the green. In `rl/action.py:131-148`, `plan()`
currently serves yellow *inside* the interval. Change a SWITCH to serve
`yellow + all_red` **in addition to** a full interval of green, set
`green_elapsed = decision_interval` in `commit()`, and then:

```yaml
decision_interval: 1
min_green: 5
```

Reachable greens become {5, 6, 7, 8, …} — 7 s is expressible, and `min_green` becomes a
genuine safety parameter rather than an accidental quantiser. Cost: 1800 decisions per
episode instead of 360.

**Option B (the proper fix).** Replace the binary action space with green-duration
selection, as described in FIX 6 of the previous brief:

```python
GREEN_DURATIONS = [5, 6, 7, 8, 10, 12, 16, 20, 26, 34]   # seconds
NUM_ACTIONS = len(GREEN_DURATIONS)
```

At the start of each green the agent picks its duration; the environment runs that green
then yellow + all-red and asks again. Every useful duration is reachable, there are no
blocked actions, each decision is meaningful, and `max_green` is enforced by capping the
menu. Files: `rl/action.py`, `rl/environment.py` (action space, step timing, truncation
accounting), `rl/state.py` (`phase_elapsed` semantics). `agents/dqn_network.py` sizes
itself from `env.action_space.n`, so the network adapts automatically.

**Either option changes the observation contract**, so bump `schema_version` in
`models/active_models.json` and re-calibrate the scales (F4) afterwards.

---

### F4 (MEDIUM) — Observation scales are per-scenario but not per-approach

**Where:** `config/heavy.yaml` `rl.observation`, `rl/state.py:104-110`

One divisor serves all four approaches, but heavy demand is asymmetric
(N/S = 850, E/W = 650 veh/h) and the queues diverge much further than the demand does.
From `output/dqn_retraining_20261120/heavy_posttrain_observation_stats.json`:

| dimension | p99 | divisor | ratio |
|---|---|---|---|
| north queue | 15 | 16 | 0.94 |
| south queue | 14 | 16 | 0.88 |
| **east queue** | 4 | 16 | **0.25** |
| **west queue** | 4 | 16 | **0.25** |
| north waiting | 85 | 142 | 0.60 |
| **east waiting** | 29 | 142 | **0.20** |

Their own gate already reports this as
`"traffic_normalized_p99_range": "PARTIAL; heavy east/west dimensions remain lower than 0.6"`.

#### Fix F4

Give `StateExtractor` per-approach divisors instead of one shared pair. Keep a scalar
accepted for backward compatibility:

```yaml
rl:
  observation:
    max_queue_per_approach: {north: 16, south: 15, east: 5, west: 5}
    max_waiting_per_approach: {north: 90, south: 88, east: 32, west: 32}
```

Re-measure with `evaluation/collect_observation_stats.py` **after** F3 lands, since F3
changes the queue distribution. Do not hand-pick these numbers.

---

### F5 (MEDIUM) — On heavy, the DQN barely beats a 20-line heuristic

From `output/dqn_stage4_20261120/heavy/confirmatory/confirmatory_rows.csv` (20 seeds):

| controller | waiting | throughput |
|---|---|---|
| actuated_tuned (`PressureActuatedAgent`, ~25 lines) | 16.700 | 1439.1 |
| incumbent_dqn | 16.469 | 1420.6 |

**1.4 % better on waiting, and 18 vehicles/hour worse on throughput.** On moderate the DQN
is a clearer 12 % ahead (2.918 vs 3.331).

This is not a bug, but it must appear in the report. A reader comparing only against the
30 s fixed-time plan would conclude the DQN is transformative on heavy; against the right
baselines it currently is not. Expect this to change once F3 lands — that is the test of
whether the learned controller earns its complexity.

#### Fix F5

Add `actuated_tuned` and the correctly-executed tuned fixed-time plan to the headline
comparison table of every report, not only to an appendix. State the DQN's margin against
the **strongest** baseline, not the weakest.

---

### F6 (LOW) — `-inf * 0 = NaN` hazard in the masked target

**Where:** `agents/dqn_agent.py:775-780`

```python
next_q_values = next_q_values.masked_fill(~next_action_masks, -torch.inf)
next_q = next_q_values.max(dim=1).values
target_q = rewards + self.config.gamma * next_q * (1.0 - dones)
```

If any mask row were all-False, `next_q` would be `-inf` and `-inf * 0.0` is **NaN**, which
would silently poison the entire batch — including the transitions that were fine.

It is currently unreachable: `ReplayBuffer.push` raises on an empty `next_valid_actions`,
and `training/train_dqn.py:378-384` always produces at least one action. So this is
robustness, not an active bug. But the failure mode is silent, and a future action-space
change (F3) is exactly the kind of edit that could introduce an empty mask.

#### Fix F6

```python
# AFTER — never multiply a possibly-infinite value by zero
next_q = next_q_values.max(dim=1).values
target_q = torch.where(dones.bool(), rewards,
                       rewards + self.config.gamma * next_q)
```

Add a unit test that pushes a transition with `done=True` and a single-action mask and
asserts the resulting loss is finite.

---

### F7 (LOW) — Stage 5 is testing a hypothesis that cannot matter much yet

`output/dqn_stage5_20261121_attempt3/protocol.md` compares `phase_elapsed` divisor 20 s
(control) against 60 s (treatment), across 20 training cells, estimated ~11 hours.

The protocol is rigorous — frozen before training, matched seeds, sealed confirmatory set,
documented amendments. The objection is to its **priority**, not its method:

- 92–99 % of greens are 6 s, so `phase_elapsed` is almost always ≈ 6/20 = 0.3 under the
  control arm and 6/60 = 0.1 under the treatment arm. Neither saturates.
- `forced_switches` is **0** in every run, so the 60 s `max_green` is never approached. The
  agent has no behaviour at long phase ages to observe, because it never produces one.
- The 20 s divisor it is questioning came from a suggestion in the previous brief
  (`min(max_green, 4 * min_green)`); testing it is reasonable, but it is a second-order
  knob while F3 caps the whole system.

**Current status:** 1 of 20 cells complete, 1 running, 18 queued
(`training_batch_status_final.json`, updated 2026-11-21T15:51).

#### Fix F7

Stop or park Stage 5, land F1–F3, then decide whether the phase-age question is still
interesting — under Option B of F3 the feature's meaning changes entirely, which would
invalidate the experiment anyway. Preserve the attempt directories and the lineage notes;
the record-keeping there is good and should not be deleted.

---

### F8 (BLOCKER) — The local virtual environment is still unusable

`.venv-stage3/pyvenv.cfg`:

```
home = C:\Users\Vishwajeet\AppData\Roaming\uv\python\cpython-3.12.13-windows-x86_64-none
```

That interpreter does not exist on this machine. The system Python is 3.11.6 and has
neither `torch` nor `gymnasium`, so **the test suite and all training are unrunnable here**.
The 223/224-passing results in the manifests could not be re-verified during this audit.

`requirements-win-py312.lock` was added and documents the fix. Execute it:

```bash
py -3.12 -m venv .venv-stage3
.venv-stage3/Scripts/python.exe -m pip install -r requirements-win-py312.lock
.venv-stage3/Scripts/python.exe -m pytest -q
```

Do this **first** — none of the fixes below can be validated without it.

---

## 3. Order of work

1. **F8** — rebuild the venv, run the suite, confirm the 223/224 baseline.
2. **F2** — fix the gate denominator and re-run it on existing Stage 4 data. Expect FAIL.
   This is a few lines and immediately corrects the record.
3. **F1** — correct the fixed-time baseline, regenerate the Stage 4 comparison as a
   companion correction. This changes a published conclusion; do it before any new training.
4. **F6** — the two-line NaN guard plus its test, while touching that file anyway.
5. **F7** — park Stage 5.
6. **F3 Option A** — decouple yellow from the interval, `decision_interval: 1`,
   `min_green: 5`. Train **100 episodes**, not more.
7. **F4** — re-calibrate per-approach scales from the new rollouts.
8. Re-evaluate against the corrected baselines; only then consider F3 Option B.

**Acceptance gates for the F3 run** (on sealed seeds, greedy, same collector):

| gate | threshold | why |
|---|---|---|
| heavy mean waiting | **< 12.7 s** | must beat the 7 s fixed-time plan, the real bar |
| heavy throughput | **≥ 1441** | the 7 s plan's level |
| moderate mean waiting | **< 3.0 s** | must beat the 5 s fixed-time plan |
| switches ÷ **max possible** | **< 0.90** | F2's corrected gate — the agent must stop living on the floor |
| distinct completed greens | **≥ 4 values at ≥ 5 % each** | proves the policy is state-dependent, not periodic |
| margin vs `actuated_tuned` | **≥ 5 %** on heavy | the DQN must earn its complexity |

If the switch ratio is still ≥ 0.90 after Option A, go to Option B. Do not respond by
training longer — Stage 4 already demonstrated that 100 episodes is sufficient to reach
whatever the action space permits.

---

## 4. Things that are NOT broken — do not "fix" these

- **`rl/reward.py`** — still ranks the true optimum correctly. Unchanged and correct.
- **The masked-DQN implementation** — `valid_actions`, `next_action_mask` and the masked
  target are correctly done and backward-compatible. Blocked switches are zero everywhere.
- **`models/active_models.json`** — the observation contract is good engineering. Extend
  it (bump `schema_version`) rather than replacing it.
- **Stage 5's record-keeping** — the lineage notes, amendments and the refusal to claim
  completion on a partial run are exemplary. Park the experiment; keep the discipline.
- **Checkpoint selection by validation waiting** — correct, with RNG restoration.
- **Episode count** — 100 episodes is enough. Nothing here is solved by training longer.

---

## 5. Evidence index

| claim | file |
|---|---|
| Stage 4 controller means | `output/dqn_stage4_20261120/promotion_decision.json` |
| Stage 4 per-seed rows, green distributions, switch counts | `output/dqn_stage4_20261120/{heavy,moderate}/{validation,confirmatory}/*.csv` |
| True tuned fixed-time, 20 seeds | `output/dqn_retraining_20261120/fixed_time/{heavy,moderate}_tuned.csv` |
| Retraining results and gate statuses | `output/dqn_retraining_20261120/experiment_manifest.json` |
| Observation p99s after training | `output/dqn_retraining_20261120/{heavy,moderate}_posttrain_observation_stats.json` |
| Vacuous gate implementation | `output/dqn_retraining_20261120/build_report.py:178,222` |
| Mislabelled baseline | `output/dqn_stage4_20261120/stage4_eval.py:131-133,250-253` |
| Stage 5 status and protocol | `output/dqn_stage5_20261121_attempt3/{protocol.md,training_batch_status_final.json}` |
| Fixed-time green sweep (previous brief) | `DQN_PLATEAU_FIX_BRIEF.md` §2.4 + Appendix A |
