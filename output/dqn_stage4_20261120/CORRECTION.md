# Stage 4 correction — the tuned fixed-time baseline was an 11-second controller

**Issued:** 21 Nov 2026. **Applies to:** `output/dqn_stage4_20261120/`.

The original Stage 4 artifacts are **unchanged**. This correction is additive:

| file | role |
|---|---|
| `promotion_decision.json` | original, preserved as issued |
| `promotion_decision_corrected.json` | corrected comparison |
| `correction/build_correction.py` | regenerates the corrected file |
| `correction/{heavy,moderate}_fixed_time_tuned_confirmatory.csv` | the replacement measurements |

**The promotion decision does not change: `NO_CANDIDATE_PROMOTED`.** No candidate passed the
validation gate before, and none does now. What changes is the **recommended best overall
controller for heavy traffic**.

---

## What was wrong

`stage4_eval.py:131-133` and `:250-253`:

```python
green = 5.0 if scenario == "moderate" else 7.0
run_rows += run_policy(scenario, "fixed_time_tuned",
                       FixedTimeAgent(ns_green=green, ew_green=green,
                                      yellow=3.0), VALIDATION_SEEDS)
```

Two independent defects.

### 1. The baseline ran through the wrong execution path

`run_policy` sends every label except `fixed_time_original` through
`run_agent_episode`, i.e. through `IntelliSignalEnv`. The environment only decides at
`decision_interval` boundaries and refuses a switch before `min_green`, so green duration is
**quantised**. After a switch `green_elapsed = decision_interval - yellow = 5 - 4 = 1 s`,
which is below `min_green = 5 s`, so the next decision is always refused and the reachable
greens are **{6, 11, 16, 21, …}**.

A 7-second request cannot be honoured. It became an 11-second green:

| evidence | value |
|---|---|
| `completed_green_duration_counts`, heavy `fixed_time_tuned` | **11 s × 99 %**, 10 s × 1 % |
| switch count | **120** over 360 decisions = one phase change every 15 s |
| independent sweep, green = 11 | 37.13 s mean waiting |
| Stage 4 recorded | **36.799 s** |

Meanwhile `fixed_time_original` ran through SUMO's own TLS program. **One comparison table
therefore mixed two different execution mechanisms**, and only the tuned row was handicapped.

### 2. The clearance interval did not match the scenario

`yellow=3.0` contradicts `config/heavy.yaml` and `config/moderate.yaml`, which both set
`signal.yellow: 4`. `fixed_time_original`, every DQN run and every sweep use 4 s. Even with
the execution path fixed, a baseline running a different clearance interval is not
comparable to the controllers it is measured against.

---

## The corrected measurement

Re-measured with `evaluation/fixed_time_sweep.py`, which runs SUMO's own TLS program and
takes yellow from the scenario, on the **same confirmatory seeds 6101–6120** used by the
original Stage 4 confirmatory evaluation. Every other controller's numbers are taken
unchanged from `confirmatory_rows.csv`.

### Heavy — the conclusion changes

| controller | mean waiting (s) | throughput |
|---|---|---|
| **fixed_time_tuned — 7 s green, SUMO TLS** | **12.638** | **1441.30** |
| incumbent_dqn | 16.469 | 1420.55 |
| actuated_tuned | 16.700 | 1439.05 |
| starting_dqn | 19.411 | 1406.20 |
| fixed_time_original — 30 s green | 53.029 | 1349.80 |
| ~~fixed_time_tuned, as originally run~~ | ~~36.799~~ | ~~1359.20~~ |

`promotion_decision.json` states `"recommend_best_overall_controller": "incumbent_dqn"`.
**That is wrong for heavy.** The trained DQN is **30.3 % worse** than a plain 7-second
fixed-time plan, and moves ~21 fewer vehicles.

Stage 4's own pre-training audit already said so —
`audit.md`: *"the heavy DQN remains behind tuned fixed-time on mean waiting"* — and
`output/dqn_retraining_20261120/experiment_manifest.json` records
`"tuned_fixed_waiting": 12.736` against `"mean_waiting": 16.637`. The underlying data was
right throughout; only Stage 4's re-measurement of the baseline went wrong.

### Moderate — the conclusion holds

| controller | mean waiting (s) | throughput |
|---|---|---|
| **incumbent_dqn** | **2.918** | **690.80** |
| fixed_time_tuned — 5 s green, SUMO TLS | 3.042 | 689.95 |
| actuated_tuned | 3.331 | 689.80 |
| fixed_time_original — 30 s green | 9.863 | 692.90 |
| starting_dqn | 29.540 | 680.40 |
| ~~fixed_time_tuned, as originally run~~ | ~~3.560~~ | ~~689.55~~ |

The DQN is genuinely the best moderate controller, by 4.1 % over the correct baseline rather
than the 17.9 % the original comparison implied.

---

## What this does and does not mean

**It does not undo Stage 4's gains.** Against the starting DQN the retrained incumbent still
cuts mean waiting by 90.1 % on moderate (29.54 → 2.92) and 15.2 % on heavy (19.41 → 16.47).
Those figures stand.

**It does mean the heavy result is not yet a win.** The honest statement is: *on moderate the
learned controller beats every baseline tested; on heavy it does not yet beat a correctly
tuned fixed-time plan.*

**The likely cause is structural, not a training shortfall.** The same quantisation that
broke this baseline also constrains the agent. Reachable greens are {6, 11, 16, 21, …} and
the measured heavy optimum is **7 s**, which the action space cannot express. The incumbent
mixes 6 s greens (92 %) with 11 s greens (7 %) and reaches 16.47 — better than a pure 6 s
plan (21.40) or a pure 11 s plan (37.13), which is real adaptive gain — but no mixture of 6
and 11 reaches 12.64. The pressure-actuated heuristic plateaus at the same 15–17 s under the
same constraint.

See `STAGE5_AUDIT_FIX_BRIEF.md` finding F3 for the action-space fix.

---

## Related fixes landed with this correction

| fix | where |
|---|---|
| Correct fixed-time baseline runner | `evaluation/fixed_time_sweep.py` (new) |
| Switch-saturation gate with the right denominator | `evaluation/gates.py` (new), `tests/test_gates.py` |
| NaN guard in the masked DQN target | `agents/dqn_agent.py`, `tests/test_dqn.py` |
| Stage 5 parked, with the same defect recorded | `output/dqn_stage5_20261121_attempt3/PARKED.md` |

`stage5_eval.py:220-221,324-325` carries the identical `yellow=3.0` defect. Those lines were
**not** edited, because that attempt's protocol guarantees its scripts match the parent byte
for byte. Any new attempt must use `evaluation/fixed_time_sweep.py` instead.

## Reproducing this correction

```bash
python evaluation/fixed_time_sweep.py --scenario heavy --greens 7 \
    --seeds 6101-6120 \
    --out output/dqn_stage4_20261120/correction/heavy_fixed_time_tuned_confirmatory.csv
python evaluation/fixed_time_sweep.py --scenario moderate --greens 5 \
    --seeds 6101-6120 \
    --out output/dqn_stage4_20261120/correction/moderate_fixed_time_tuned_confirmatory.csv
python output/dqn_stage4_20261120/correction/build_correction.py
```

These runs need SUMO/TraCI but not PyTorch, because no learned policy is evaluated.
