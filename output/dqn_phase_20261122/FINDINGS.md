# The DQN now beats the tuned fixed-time plan, on both scenarios

**Date:** 22 Nov 2026.
**Measurement:** 1800 simulated seconds, confirmatory seeds 6101–6120, greedy
policy, the project's own `utils.metrics.MetricsCollector`. Fixed-time rows
run SUMO's own traffic-light program through
`evaluation/fixed_time_sweep.py`.

---

## 1. Result

### Heavy traffic

| controller | mean waiting (s) | sd | throughput |
|---|---|---|---|
| **duration DQN** (`R143`) | **5.151** | 0.333 | **1468.5** |
| best repeating pattern, ns[7,7] ew[5,6] | 5.680 | 0.246 | 1463.8 |
| fixed 7/6 — the best fixed plan | 5.754 | 0.418 | 1466.5 |
| fixed 7/7 — the best *symmetric* plan | 12.638 | 0.814 | 1441.3 |
| DQN Stage 4 (KEEP/SWITCH, the deployed model) | 16.469 | 2.659 | 1420.5 |
| fixed 30/30 — the original Stage 1 baseline | 53.029 | 1.854 | 1349.8 |

**−10.5 %** against the best fixed plan, **−68.7 %** against the deployed DQN,
and the highest throughput of any controller measured.

Paired across the 20 seeds — both controllers saw identical traffic on each —
the DQN wins on **17 of 20**, mean difference **−0.603 s**. Its worst seed
loses by 0.150 s.

### Moderate traffic

| controller | mean waiting (s) | sd | throughput |
|---|---|---|---|
| **duration DQN** (`M1`) | **2.237** | 0.142 | **691.8** |
| fixed 6/4 — the best fixed plan | 2.761 | 0.109 | 690.8 |
| DQN Stage 4 (KEEP/SWITCH, the deployed model) | 2.918 | 0.123 | 690.8 |
| fixed 5/5 — the plan previously used as the baseline | 3.042 | 0.112 | 690.0 |
| fixed 30/30 — the original Stage 1 baseline | 9.863 | 0.200 | 692.9 |

**−19.0 %** against the best fixed plan, and it wins on **20 of 20** seeds,
mean difference −0.524 s.

---

## 2. A correction to the previous moderate-traffic claim

The project has been reporting that the moderate DQN beats fixed-time. It
does beat the plan it was compared against — the symmetric 5/5 plan, 3.042 s —
but that is not the best fixed plan. Demand is 400 veh/h on each of north and
south against 300 on each of east and west, and a sweep over asymmetric plans
finds **ns=6 / ew=4 at 2.761 s**.

Against that, the deployed moderate DQN at 2.918 s is **5.7 % worse**, not
better. Both of this project's deployed models were losing to a properly
tuned fixed plan; only the heavy-traffic one was losing visibly.

The same correction applies on heavy and was already recorded: the optimum is
ns=7/ew=6 at 5.754 s, not the symmetric 7/7 at 12.638 s.

---

## 3. Why the old formulation could not win

Three causes, all properties of the formulation rather than of the training.

**The best plan was not expressible.** With KEEP/SWITCH the green is a whole
number of decision intervals, so the two directions can only differ by a whole
interval. A one-second difference needs a one-second interval, which the
environment forbade until `clearance_outside_interval` (commit `fb3eb2b`).

**A one-second interval destroys the credit assignment.** It produces ~1100
decisions per episode, nearly all KEEP, each with an almost identical reward.

**The transitions stop being Markov.** KEEP advanced 1 s and SWITCH 5 s, while
the update charged both the same mean reward and discounted both by one factor
of gamma. Training diverged: validation 17.4 → 112 → 146 → 17.0.

---

## 4. What was changed

`rl/phase_action.py`, `rl/phase_state.py`, `rl/phase_environment.py`,
`agents/phase_dqn_agent.py`, `training/train_phase_dqn.py`,
`evaluation/evaluate_phase_dqn.py`. Full description in
[`docs/phase_duration_control.md`](../../docs/phase_duration_control.md).

1. **One decision per green, choosing that green's duration** from a menu
   bounded by `min_green` and `max_green`. 7 s against 6 s is two actions,
   not a timing pattern held across eleven decisions. ~160 decisions per
   episode instead of ~1100.
2. **The reward is an integral over the decision's seconds**, not a mean, so
   a 24-second decision is not charged like an 8-second one.
3. **The future is discounted by elapsed time**, `gamma ** seconds`, with
   gamma as a per-second rate — the standard semi-Markov treatment. The
   congestion *inside* a decision is discounted too, by how far into the
   decision it happens; without that, long greens are systematically
   overcharged and the policy drifts to the shortest action on the menu.
4. **Double Q-learning.** With a menu whose entries differ by very little, the
   over-estimation bias of `max_a Q_target` is large relative to the signal.
5. **A phase-relative observation**, written from the served direction's point
   of view, so "give more time to the busier direction" is one rule instead of
   two mirror-image ones.

Safety is unchanged and is now structural rather than enforced: the menu
cannot express a green below `min_green` or above `max_green`, every action
ends with the Stage 1 yellow clearance, and every action hands the road to the
other direction next, so no approach can be starved.

---

## 5. The policies are genuinely adaptive

Every previous heavy model collapsed onto a single green duration — 92–99 % of
decisions at one value. These do not.

**Heavy (`R143`)** — mean green 6.95 s north-south, 5.49 s east-west:

| green | share |
|---|---|
| 5 s | 37.4 % |
| 7 s | 61.4 % |
| 4 s / 6 s | 1.2 % |

It holds north-south at 7 s and alternates east-west between 5 s and 6 s —
which is the structure the exhaustive pattern search independently found to be
the best repeating plan (ns[7,7] ew[5,6], 5.680 s). The DQN reaches 5.151 s,
better than the best pattern in that family, because it chooses from the
traffic state rather than on a fixed cycle.

**Moderate (`M1`)** — mean green 5.77 s north-south, 4.59 s east-west, spread
over 4 s (58.9 %), 5 s (28.3 %), 7 s (4.4 %), 15 s (3.8 %) and a tail. The
long greens are rare and are the response to an occasional build-up.

---

## 6. Robustness

Thirteen heavy training runs, differing in seed, discount, menu, n-step
return, reward weights, network size and observation scales, measured on
**selection seeds 4001–4020** (fixed 7/6 scores 6.040 s there):

* **12 of 13 beat the best fixed plan**; median 5.469 s.
* The one failure, `H1`, converged to 5.5 s / 4.2 s greens and reached
  11.034 s. It is the only run combining the tighter re-calibrated
  observation scales with gamma = 0.99; `H2`, the same scales at gamma =
  0.995, is among the best. Tight observation scales clip heavily while the
  policy is still exploring, and the gradient signal goes with them.
* The three runs that differ **only** in training seed (`R142`, `R143`,
  `R144`) scored 5.405, 5.148 and 5.675 — all wins.

The headline is therefore not one lucky run.

---

## 7. How the model was selected

Four disjoint seed sets, and nothing is selected on the seeds it is reported
on:

| set | seeds | used for |
|---|---|---|
| training | 42–641, 142–561 | episodes |
| validation | 2001–2005 | choosing `best.pt` inside a run |
| selection | 4001–4020 | choosing which run to promote |
| confirmatory | 6101–6120 | the numbers reported above |

The selection step exists because the headline had been chosen on the reported
seeds, which biases it. `R143` and `M1` are the best runs on 4001–4020, and
their 6101–6120 numbers are then an honest out-of-sample measurement.

---

## 8. Promotion

Both models are registered in `models/active_models.json` under
`phase_duration_models`, with their sha256, duration menu and observation
contract. That section is deliberately separate from `active_models`, which
`agents/dqn_agent.py` and the dashboard read and which expects a KEEP/SWITCH
checkpoint with a `decision_interval` — a duration-menu model there would be
a crash, not a promotion. The Stage 4 incumbents remain registered where they
were, so nothing that reads the old section changes behaviour.

---

## 9. Reproducing

```bash
# the best fixed plan, through SUMO's own traffic-light program
python evaluation/fixed_time_sweep.py --scenario heavy --greens 7 \
    --seeds 6101-6120 --out heavy_7.csv

# train
python training/train_phase_dqn.py --scenario heavy --episodes 420 \
    --seed 143 --gamma 0.99 --n-step 2 --threads 1 \
    --validation-seeds 2001-2005 --out output/run

# evaluate against the fixed plan on the confirmatory seeds
python evaluation/evaluate_phase_dqn.py \
    --model output/run/models/best.pt --seeds 6101-6120

# rebuild the tables in section 1
python output/dqn_phase_20261122/assemble.py
```

Environment: Python 3.12.10, torch 2.14.1, gymnasium 1.4.0, SUMO 1.28.0.
Test suite at the time of writing: **317 passed, 1 skipped**.

One thing to watch: `evaluation/evaluate_dqn.py` rewrites the PNGs under
`results/plots/` even when called with `save=False`. Running it to measure
the Stage 4 incumbents modified eleven committed historical plots; they were
restored with `git checkout -- results/plots/` and are unchanged in this
commit. If you re-run that evaluator, check `git status` afterwards.

---

## 10. One thing to keep in mind about this benchmark

Demand is generated with SUMO `<flow vehsPerHour=...>` and **no
`random="true"`**, so vehicles are inserted on an equidistant, deterministic
schedule. That is why the delay surface is so sharply structured — one second
either side of ns=7/ew=6 roughly doubles the delay — and it means part of what
an adaptive controller exploits here is a repeating arrival pattern rather
than random burstiness.

The comparison is fair: every controller in section 1 faces exactly the same
traffic on the same seeds. But a claim about *real* traffic would need demand
with `random="true"` or a measured arrival trace, and that experiment has not
been run. The honest statement of this result is: **on this benchmark, as it
has always been defined, the learned controller now beats the best fixed plan
that exists for it, on both scenarios, on waiting time and on throughput at
once.**
