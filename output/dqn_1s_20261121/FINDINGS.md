# Heavy traffic: the action space is no longer the limit — the learning is

**Date:** 21 Nov 2026. **Scope:** heavy scenario, confirmatory seeds 6101–6120,
1800 s episodes, greedy evaluation, the project's own `MetricsCollector`.

## Where heavy traffic actually stands

| controller | mean waiting (s) | throughput | notes |
|---|---|---|---|
| **fixed-time ns=7 / ew=6** | **5.754** | **1466.5** | best known plan, sd 0.429 |
| fixed-time 7 / 7 | 12.638 | 1441.0 | best *symmetric* plan |
| DQN, 11 s cadence | 13.300 | 1438.5 | identical to always-switch |
| DQN, Stage 4 (5 s cadence) | 16.469 | 1420.6 | the deployed incumbent |
| DQN, 1 s resolution | 17.332 | 1450.0 | this experiment |
| fixed-time 30 / 30 | 53.029 | 1349.8 | the original baseline |

The deployed incumbent is **2.9×** worse than a plain asymmetric fixed plan.

## What changed, and what it proved

Previous briefs concluded the heavy optimum was a symmetric 7 s green at
12.64 s. **That was wrong** — it came from a sweep that only varied both
directions together. Demand is 850 veh/h on each of north and south against
650 on each of east and west, and the best plan reflects that asymmetry:

| plan | waiting | throughput | stranded |
|---|---|---|---|
| ns=7 ew=6 | **5.72** | **1466.6** | **37.4** |
| ns=8 ew=6 | 11.62 | 1449.0 | 55.0 |
| ns=7 ew=7 | 12.82 | 1442.0 | 62.0 |
| ns=6 ew=6 | 19.89 | 1402.8 | 100.8 |
| ns=6 ew=7 | 27.14 | 1362.2 | 109.2 |

It is not a seed artifact: 7/6 also clears the most vehicles and strands the
fewest, across 20 seeds with sd 0.429.

**The environment could not express it.** With the clearance served inside the
decision interval, a switch leaves `interval - yellow` seconds of green already
served, so reachable greens are `{interval-4, 2*interval-4, …}` and the two
directions can only ever differ by a whole interval. A one-second difference
needs a one-second interval, which the old rule forbade.

`clearance_outside_interval` (commit `fb3eb2b`) fixes that, and the fix is
verified end to end: a scripted 7/6 policy *inside the environment* reproduces
the SUMO traffic-light program seed for seed — 6.670 / 5.890 / 5.170 against
6.67 / 5.89 / 5.17 — with 86 greens of 7 s and 85 of 6 s per episode.

**So the action space is no longer the constraint. The learning is.**

## What this training run showed

60 episodes, seed 42, `decision_interval: 1`, `min_green: 5`,
`clearance_outside_interval: true`, `gamma: 0.995`, ~1088 decisions per episode.

**Good news — the policy is no longer degenerate.** Every previous heavy DQN
collapsed onto a single green duration (92–99 % at the floor). This one spreads
across the range:

| green | share |
|---|---|
| 5 s | 35.1 % |
| 6 s | 13.5 % |
| 7 s | 13.6 % |
| 8 s | 13.1 % |
| 9 s | 10.3 % |
| 10 s | 5.6 % |
| 11–16 s | 6.3 % |

It is genuinely reacting to state. Throughput is also the second-highest of any
learned controller (1450.0).

**Bad news — training diverged.** Greedy validation on seeds 2001–2003:

| episode | 10 | 20 | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|---|
| waiting | 17.44 | **112.12** | **146.39** | **144.95** | 28.70 | 16.99 |

Mean Huber loss rose from 0.30 to 3.78 over the same span. The run recovered by
episode 60, but a policy that passes through 146 s is not converging.

## Why it diverged — a semi-Markov mismatch

With the clearance outside the interval, decisions no longer last the same
amount of simulated time:

* **KEEP** advances the simulation by `decision_interval` = **1 s**
* **SWITCH** advances it by `decision_interval + yellow + all_red` = **5 s**

`agents/dqn_agent.py::learn` discounts both by a single factor of `gamma`:

```python
target_q = torch.where(dones.bool(), rewards,
                       rewards + self.config.gamma * next_q)
```

A switch therefore compresses five seconds of consequence into one discount
step, which makes switching look systematically cheaper than it is. Combined
with `gamma = 0.995` — an effective horizon of 200 steps, so Q values on the
order of hundreds — that is a plausible route to the divergence observed.

This is a real modelling error, not a hyperparameter accident. The transition
is semi-Markov and the update treats it as Markov.

## The fix, in order

1. **Discount by elapsed time.** Store each transition's duration and use
   `gamma ** duration` in the target. `rl/environment.py` already tracks
   `self._simulated_seconds`; the per-step duration is `len(samples) *
   step_length`. This needs a field on `Transition`, a column in `Batch`, and
   one line in `learn()`. It is the correct fix and is contained.
2. **Lower the horizon while testing.** `gamma = 0.99` at a 1 s cadence is a
   100-second horizon, which is ample for a 20-second cycle and keeps Q values
   an order of magnitude smaller.
3. **Re-calibrate the observation scales.** `config/heavy.yaml` carries
   `max_queue_per_approach: 16`, measured under the 5 s cadence where mean
   queue was ~13.5. Under a 7/6 plan it is ~4.6, so the queue inputs now sit
   near 0.3 of their range. Re-measure with
   `evaluation/collect_observation_stats.py` after step 1.
4. **Consider one decision per green instead.** Choosing a duration from a menu
   at the start of each green removes the variable-duration problem entirely,
   gives ~170 meaningful decisions per episode instead of ~1100 noisy ones, and
   makes 7/6 a single action rather than a timing pattern the agent must hold.
   It is the larger change — the action space is referenced in about 74 places —
   but it is the cleaner formulation.

## Repository state

`config/heavy.yaml` has been **restored to the 5 s cadence**. The deployed
incumbent in `models/active_models.json` records
`observation_contract.decision_interval: 5.0`, and running it under any other
cadence would evaluate it outside the environment it learned. The 1 s settings
live only in this document and are reproduced by the command below.

No model was promoted. The Stage 4 incumbent remains the active heavy model.

## Reproducing

```bash
# the measured optimum, through SUMO's own traffic-light program
.venv-stage3/Scripts/python.exe evaluation/fixed_time_sweep.py \
    --scenario heavy --greens 7 --seeds 6101-6120 --out heavy_7s.csv

# the asymmetric search
.venv-stage3/Scripts/python.exe output/dqn_cadence_20261121/asymmetric_sweep.py

# this training run: add to config/heavy.yaml under `rl:`
#   environment: {decision_interval: 1, min_green: 5,
#                 clearance_outside_interval: true}
.venv-stage3/Scripts/python.exe training/train_dqn.py --scenario heavy \
    --episodes 60 --seed 42 --gamma 0.995 \
    --model-dir output/dqn_1s_20261121/models/dqn_heavy \
    --results-dir output/dqn_1s_20261121/training
```

Environment: Python 3.12.10, torch 2.14.1, gymnasium 1.4.0, SUMO 1.28.0.
Suite at the time of writing: 260 passed, 1 skipped.
