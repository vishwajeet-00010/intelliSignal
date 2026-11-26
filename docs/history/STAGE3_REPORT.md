# IntelliSignal — Stage 3 Implementation Report

**Stage:** 3 of 3 — DQN agent, training, evaluation and Fixed-Time vs DQN comparison
**Status:** complete
**Scope rule followed:** build on Stages 1 and 2, change nothing that was working.

Every number in this report came out of a SUMO run performed by the code in
this repository. Nothing is estimated, rounded up or borrowed from literature.
Where the DQN performed worse than the fixed-time baseline, the report says so.

---

## 1. Files created

### The agent and its parts

| File | Lines | What it is |
|---|---|---|
| `agents/dqn_network.py` | 123 | The Q-network. Input size from `env.observation_space.shape`, output size from `env.action_space.n`. |
| `agents/replay_buffer.py` | 137 | Fixed-capacity ring buffer with uniform random mini-batch sampling. |
| `agents/dqn_config.py` | 244 | `DQNConfig`, YAML loading, command-line overrides and validation. |
| `config/dqn/training.yaml` | 118 | Every hyperparameter, documented in place. |

`agents/dqn_agent.py` was a not-implemented stub and is now the real agent
(405 lines) — listed under *modified* below.

### Training

| File | Lines | What it is |
|---|---|---|
| `training/__init__.py` | 13 | Package marker. |
| `training/train_dqn.py` | 480 | The training loop and its CLI. |
| `training/plot_training.py` | 215 | Training curves from the recorded history CSV. |

### Evaluation and comparison

| File | Lines | What it is |
|---|---|---|
| `evaluation/__init__.py` | 20 | Package marker and the two rules the package enforces. |
| `evaluation/evaluate_dqn.py` | 330 | Greedy evaluation, multi-seed aggregation, mean/std. |
| `evaluation/compare_controllers.py` | 390 | Fixed-Time vs DQN on identical traffic, improvement percentages. |
| `evaluation/plot_comparison.py` | 250 | The comparison charts. |

### The shared episode runner

| File | Lines | What it is |
|---|---|---|
| `simulation/agent_simulation.py` | 390 | Runs one episode under one controller. Both controllers get byte-identical SUMO inputs and the same metrics collector. This file is the fairness guarantee of the whole comparison. |

### Tests and documentation

| File | What it is |
|---|---|
| `tests/test_dqn.py` | The Stage 3 test suite (96 tests). |
| `docs/dqn_training.md` | The DQN explained end to end, including every limitation. |
| `STAGE3_REPORT.md` | This file. |

### Generated output (not source)

```
models/dqn_moderate/          best.pt, final.pt, checkpoint_ep010…100.pt, metadata.json
models/dqn_heavy/             the same for the heavy scenario
models/dqn_moderate_seed43/   second training seed, kept separate
models/dqn_heavy_seed43/      second training seed, kept separate
results/training/             per-episode training history, one CSV per run
results/evaluation/           raw, aggregated and improvement CSVs + per-second time series
results/plots/training/       reward, loss, epsilon, queue, waiting-time curves
results/plots/comparison/     Fixed-Time vs DQN bar charts and queue-over-time
```

---

## 2. Files modified

| File | Change | Backwards compatible? |
|---|---|---|
| `agents/dqn_agent.py` | The stub became the real agent: epsilon-greedy selection, the TD update, the target network, checkpoints, `explain()` with Q-values. | The class and its name are unchanged; `DQN_IMPLEMENTED` is now `True`. |
| `agents/base_agent.py` | Added `TrafficState.observation()` and an optional `observation=` argument to `from_env_info()`. Updated the module docstring. | Yes — the observation travels in the existing `extra` dict and defaults to absent. |
| `agents/__init__.py` | Exports the Stage 3 names; `list_agents()` now reads DQN availability from the filesystem. | Yes. |
| `utils/metrics.py` | `MetricsCollector(conn=...)`, defaulting to the `traci` module. | Yes — a test asserts the default is still the module, so Stage 1 behaviour is untouched. |
| `rl/environment.py` | Added `step_callback`, a `connection` property, `extra_sumo_args`, and lifted `episode_scenario_config()` to module level. | Yes — all optional; the extracted function has identical behaviour. |
| `simulation/controller.py` | `control_modes(scenario)` gates the DQN on a real checkpoint; `run()` gained a DQN branch; `load_results(scenario, mode)`; model discovery, model status, training history, evaluation results, live comparison. | Yes — every new parameter has a default. |
| `dashboard.py` | Two new pages (DQN Model, Comparison), controller labelling, recorded Q-values on the Simulation page, rewritten About page. | — |
| `dashboard/controls.py` | Per-scenario mode list; model and seed pickers shown only in DQN mode. | — |
| `dashboard/components.py` | Q-value block in the decision panel; `comparison_table`, `improvement_badge`, `controller_badge`. | Yes — additive. |
| `dashboard/charts.py` | `controller_metric_chart`, `queue_by_controller_chart`, `training_curve_chart`. | Yes — additive. |
| `dashboard/state.py` | Two new pages, `seed` and `model_name` state, mode-aware `current_run()`. | Yes — `mode` defaults. |
| `tests/test_dashboard.py` | Six tests that asserted "the DQN does not exist" rewritten to assert the Stage 3 rule that replaced them; four new tests added. No test deleted. | See section 7. |
| `requirements.txt` | Added `torch>=2.0`, with notes on the CPU wheel and the Python-version constraint. | — |
| `README.md`, `docs/architecture.md`, `docs/rl_environment.md`, `docs/dashboard.md` | Updated to describe what is now built. | — |

### Not modified, on purpose

`sumo/` (the road network), `config/moderate.yaml`, `config/heavy.yaml`,
`config/rl/environment.yaml`, `rl/state.py`, `rl/action.py`, `rl/reward.py`,
`rl/config.py`, `rl/sumo_rl_adapter.py`, `rl/run_environment.py`,
`simulation/run_simulation.py`, `simulation/run_scenario.py`,
`simulation/plot_results.py`, `utils/config_loader.py`,
`utils/scenario_builder.py`, `agents/fixed_time_agent.py`,
`dashboard/theme.py`, `tests/test_simulation.py`,
`tests/test_rl_environment.py`, and **`results/csv/`** — the recorded Stage 1
baseline.

The observation, the action meaning, the reward and the signal-safety rules
are exactly as Stage 2 left them. Changing the reward between the baseline and
the learned controller would have made the comparison meaningless.

---

## 3. DQN architecture

```
10  (from env.observation_space.shape)
 |
Linear(10, 64) -> ReLU
 |
Linear(64, 64) -> ReLU
 |
Linear(64, 2)
 |
 2  (from env.action_space.n)  ->  [Q(keep), Q(switch)]
```

`10 -> 64 -> 64 -> 2`, **5,058 parameters**. The input and output sizes are
read from the environment's spaces, never typed in; a test builds the network
from an 11-value observation space and a 3-action space to prove it follows.

The agent returns only `0 = keep` or `1 = switch`. `rl/action.py` turns that
into a legal signal change: a switch always runs the Stage 1 4-second yellow,
a switch before `min_green` (10 s) is refused and recorded as refused, and a
`keep` after `max_green` (60 s) is overridden into a switch. A test drives the
network with pure exploration for 300 simulated seconds and asserts that no
green → green transition appears anywhere in the recorded signal states.

---

## 4. Hyperparameters actually used

From `config/dqn/training.yaml`, recorded in every checkpoint's metadata:

| Setting | Value |
|---|---|
| hidden layers | [64, 64] |
| episodes | 100 |
| episode duration | 1800 s (= 180 decisions at a 10 s decision interval) |
| batch size | 64 |
| learning rate | 0.001 (Adam) |
| gamma (discount) | **0.9** — the project's planned value, unchanged |
| loss | Huber (`smooth_l1`) |
| gradient clipping | max norm 10.0 |
| replay capacity | 50,000 |
| learning starts | 500 transitions |
| train frequency | every environment step |
| target update frequency | every 500 gradient steps |
| epsilon | 1.0 → 0.05, linear over 9,000 steps (= 50 % of the run) |
| seed | 42 (also 43 for the second training run) |
| checkpoint every | 10 episodes |
| best.pt selection | best mean return over the last 5 episodes |

The epsilon schedule length is derived from the number of episodes requested,
so `--episodes 20` decays over 1,800 steps instead of inheriting a
100-episode schedule and becoming greedy after two episodes.

---

## 5. Training commands

```bash
# the runs that produced the models in this repository
python training/train_dqn.py --scenario moderate --episodes 100 --seed 42
python training/train_dqn.py --scenario heavy    --episodes 100 --seed 42
python training/train_dqn.py --scenario moderate --episodes 100 --seed 43
python training/train_dqn.py --scenario heavy    --episodes 100 --seed 43

# a quick demonstration (a few minutes)
python training/train_dqn.py --scenario moderate --episodes 20 --seed 42
```

A run with seed 42 writes to `models/dqn_<scenario>/`; any other seed gets
`models/dqn_<scenario>_seed<seed>/`, so two runs never overwrite each other.

---

## 6. Evaluation and comparison commands

```bash
# greedy evaluation, epsilon = 0, no learning
python evaluation/evaluate_dqn.py --scenario moderate --seeds 42 43 44
python evaluation/evaluate_dqn.py --scenario heavy    --seeds 42 43 44

# Fixed-Time vs DQN on identical traffic, with the untrained control
python evaluation/compare_controllers.py --scenario moderate --seeds 42 43 44 --include-untrained
python evaluation/compare_controllers.py --scenario heavy    --seeds 42 43 44 --include-untrained

# redraw the charts from the recorded CSVs
python training/plot_training.py --scenario moderate --seed 42
python evaluation/plot_comparison.py --scenario moderate
```

---

## 7. Test result

```
python -m pytest -q
```

```
222 passed, 0 skipped, 0 failed in 157 s
```

| Suite | Tests | Status |
|---|---|---|
| `tests/test_simulation.py` — Stage 1 | 24 | unchanged, all passing |
| `tests/test_rl_environment.py` — Stage 2 | 47 | unchanged, all passing |
| `tests/test_dashboard.py` — dashboard / agents / controller | 55 (was 51) | all passing |
| `tests/test_dqn.py` — Stage 3 | 96 | all passing |

### No regression in Stages 1 and 2

`tests/test_simulation.py` and `tests/test_rl_environment.py` were **not
touched**. All 71 of their tests pass against the Stage 3 code, which is the
evidence that the additive seams in `utils/metrics.py` and `rl/environment.py`
changed no existing behaviour.

### Tests in `tests/test_dashboard.py` that were rewritten, and why

Six tests asserted an architecture that Stage 3 deliberately replaced. None
was deleted; each now asserts the rule that took its place.

| Test (before) | Asserted | Now asserts |
|---|---|---|
| `test_dqn_is_declared_but_not_implemented` | `DQN_IMPLEMENTED is False` | `DQN_IMPLEMENTED is True` and `is_learned is True` (renamed `…_is_implemented_in_stage_3`) |
| `test_creating_a_dqn_agent_refuses_with_an_explanation` | raises "not implemented" | raises `DQNModelNotFoundError` naming the training command, when no checkpoint exists |
| `test_agent_registry_reports_availability_honestly` | `dqn.available is False` always | availability is read from the filesystem: `False` with no checkpoint, `True` with one |
| `test_control_modes_expose_fixed_time_and_an_unavailable_dqn` | the mode is never available | the mode is gated on a real checkpoint, and the refusal says "train a model first" |
| `test_running_the_dqn_mode_is_refused` | always refused | refused **when no model exists** |
| `test_dqn_mode_is_offered_disabled_and_explained` | the Run button is always disabled | enabled with a model (and the model picker appears), disabled and explained without one |
| `test_about_page_states_that_dqn_is_not_implemented` | "no neural network" | the status table matches reality in both directions, including the future-scope row |

Four tests were added: the DQN mode forced into its no-model state on a
machine that *does* have a model, the DQN Model page with and without a model,
and the Comparison page with no evaluation files on disk.

### What the Stage 3 suite covers

Network (shapes, finite Q-values, sizing from the spaces, rejection of
impossible sizes); replay buffer (insertion, eviction, batch dimensions and
dtypes, refusal to over-sample, reproducibility); agent (valid actions,
greedy determinism, epsilon = 1 explores both actions, epsilon = 0 never
explores, learning waits for `learning_starts`, the policy network moves, the
target network starts identical and is refreshed on schedule, no gradient
reaches the target network, both loss functions, both action conventions,
Q-values in `explain()`, checkpoint round-trip reproducing identical
decisions, checkpoint completeness, refusal without weights); fairness (the
fixed-time comparison run reproducing the recorded Stage 1 baseline exactly,
identical simulated seconds, no unsafe transition, the Stage 1 collector
default unchanged, refusal to write into `results/csv/`); training (a real
short run end to end, checkpoints, metadata, history columns, no NaN loss,
curves generated, no leaked SUMO process, baseline untouched); evaluation
(model loads, greedy, **no gradient step and no buffer change**, multiple
seeds, no fabricated standard deviation for one seed, reproducibility, raw and
aggregated kept separate, refusal without a model); comparison (both
controllers on identical traffic, every run kept, improvement signs, zero
denominator, the baseline byte-identical, the untrained control, charts);
dashboard (mode gated per scenario, refusal without a model, honest model
status, a DQN run really running the network and writing its own files, no
fallback to the baseline, the live comparison refusing mismatched durations,
the Comparison page reading real evaluation CSVs).

---

## 8. Training result

Real numbers from `results/training/`. Returns are the sum of the
per-decision rewards, which are negative congestion costs, so **closer to zero
is better**.

| Run | Episodes | Gradient steps | Target updates | Mean return, first 5 | Mean return, last 5 | Best episode | Final mean loss | Final epsilon | Wall clock |
|---|---|---|---|---|---|---|---|---|---|
| moderate, seed 42 | 100 | 17,501 | 35 | −98.9 | **−32.4** | −26.8 | 0.0100 | 0.05 | 1,098 s |
| moderate, seed 43 | 100 | 17,501 | 35 | −100.6 | **−31.3** | −29.2 | 0.0100 | 0.05 | 748 s |
| heavy, seed 42 | 100 | 17,501 | 35 | −1,316.0 | **−621.9** | −546.8 | 1.2548 | 0.05 | 1,492 s |
| heavy, seed 43 | 100 | 17,501 | 35 | -1201.0 | **-597.3** | -535.4 | 1.3091 | 0.05 | 1,472 s |

Traffic metrics measured during training, same collector as the baseline:

| Run | Waiting time, episode 1 | Waiting time, last 5 episodes | Queue, last 5 episodes |
|---|---|---|---|
| moderate, seed 42 | 11.11 s | **5.90 s** | 2.15 |
| moderate, seed 43 | 11.25 s | **5.70 s** | 2.07 |
| heavy, seed 42 | 59.07 s | **49.37 s** | 39.00 |
| heavy, seed 43 | 55.04 s | **47.37 s** | 37.50 |

Both scenarios improved during training, and both pairs of training seeds
finished close together — moderate −32.4 and −31.3 (within 4 %), heavy −621.9
and −597.3 (within 4 %) — so the learning is not a single-seed accident.

Curves: `results/plots/training/dqn_<scenario>_seed<seed>_{reward,loss,epsilon,queue,waiting,overview}.png`.
They are drawn from the recorded CSV values with no smoothing of the data; a
moving average is overlaid on the raw curve and labelled as such.

---

## 9. Evaluation result — Fixed-Time vs DQN

Mean ± sample standard deviation over evaluation seeds 42, 43 and 44.
1800 simulated seconds per run. For each seed, one set of SUMO inputs was
built and both controllers ran on it.

### Moderate traffic — 1400 veh/h

| Metric | Fixed-Time | DQN | DQN improvement |
|---|---|---|---|
| Average waiting time (s/vehicle) | 9.91 ± 0.20 | **5.71 ± 0.10** | **+42.3 %** |
| Average queue length (vehicles) | 3.66 ± 0.07 | **2.08 ± 0.05** | **+43.0 %** |
| Maximum queue length (vehicles) | 9.7 ± 0.6 | **7.0 ± 0.0** | **+27.6 %** |
| Throughput (vehicles completed) | 693 ± 0 | 691 ± 2 | **−0.3 %** |
| Vehicles still in network | 9 ± 0 | 11 ± 2 | **−22.2 %** |
| Teleports | 0 ± 0 | 0 ± 0 | n/a |

### Heavy traffic — 3000 veh/h

| Metric | Fixed-Time | DQN | DQN improvement |
|---|---|---|---|
| Average waiting time (s/vehicle) | 52.60 ± 1.46 | **47.91 ± 2.00** | **+8.9 %** |
| Average queue length (vehicles) | 40.98 ± 1.19 | **37.75 ± 1.48** | **+7.9 %** |
| Maximum queue length (vehicles) | 71.3 ± 3.1 | **69.7 ± 2.3** | **+2.3 %** |
| Throughput (vehicles completed) | 1351 ± 4 | 1344 ± 6 | **−0.6 %** |
| Vehicles still in network | 96 ± 4 | 116 ± 2 | **−20.9 %** |
| Teleports | 0 ± 0 | 0 ± 0 | n/a |

### Every individual run (nothing averaged away)

Moderate:

| Seed | Controller | Waiting (s) | Queue | Max queue | Throughput | Still in network | Switches |
|---|---|---|---|---|---|---|---|
| 42 | Fixed-Time | 10.04 | 3.70 | 10 | 693 | 9 | 52 |
| 42 | DQN | 5.78 | 2.10 | 7 | 689 | 13 | 88 |
| 43 | Fixed-Time | 9.68 | 3.58 | 10 | 693 | 9 | 52 |
| 43 | DQN | 5.60 | 2.03 | 7 | 692 | 10 | 88 |
| 44 | Fixed-Time | 10.01 | 3.69 | 9 | 693 | 9 | 52 |
| 44 | DQN | 5.76 | 2.12 | 7 | 692 | 10 | 88 |

Heavy:

| Seed | Controller | Waiting (s) | Queue | Max queue | Throughput | Inserted | Still in network |
|---|---|---|---|---|---|---|---|
| 42 | Fixed-Time | 51.90 | 40.39 | 72 | 1351 | 1445 | 94 |
| 42 | DQN | 47.92 | 37.79 | 71 | 1347 | 1462 | 115 |
| 43 | Fixed-Time | 51.62 | 40.21 | 68 | 1355 | 1448 | 93 |
| 43 | DQN | 49.90 | 39.21 | 71 | 1337 | 1455 | 118 |
| 44 | Fixed-Time | 54.27 | 42.35 | 74 | 1348 | 1448 | 100 |
| 44 | DQN | 45.91 | 36.25 | 67 | 1347 | 1461 | 114 |

Raw CSVs: `results/evaluation/comparison_<scenario>_raw.csv`.

### The fairness check

At seed 42 the fixed-time comparison run reproduced the **recorded Stage 1
baseline exactly** — moderate 702 vehicles / 10.04 s / 3.70 / 693, heavy 1445
vehicles / 51.90 s / 40.39 / 1351 / max queue 72 / 0 teleports. A test asserts
this. If it had not matched, every improvement percentage would have been
measured against the wrong reference.

### The untrained control

Same architecture, random weights, same traffic:

| Average waiting time (s) | Fixed-Time | **DQN (trained)** | DQN (untrained control) |
|---|---|---|---|
| Moderate | 9.91 | **5.71** | 10.91 |
| Heavy | 52.60 | **47.91** | 88.21 |

The DQN acts every 10 s and may switch far more often than the fixed-time
plan, so part of the improvement could in principle have come from the shorter
decision interval rather than from learning. It did not: the untrained network
is **worse than fixed-time** in both scenarios, and under heavy demand it is
far worse (88.21 s) while also failing to admit traffic (about 1,200 vehicles
enter instead of 1,445). The learning is what produced the improvement.

Charts: `results/plots/comparison/<scenario>_{waiting_time,queue_length,throughput,max_queue,queue_over_time,summary}.png`.

---

## 10. Important limitations

### Did the DQN outperform Fixed-Time?

**Yes on delay, and clearly so — but not on every metric, and much less under
heavy demand.**

* **Waiting time and queues: yes.** 42 % and 43 % better under moderate
  demand, 9 % and 8 % better under heavy demand. Consistent across all three
  evaluation seeds: the worst DQN seed still beat the best fixed-time seed on
  both metrics in both scenarios.
* **Maximum queue: yes, modestly.** 27 % better under moderate demand, 2 %
  under heavy — essentially a tie in heavy traffic.
* **Throughput: no.** 0.3–0.6 % *lower*. That is a handful of vehicles, and it
  is not a win. Throughput here is bounded by demand (693 of 702 vehicles
  complete in the moderate scenario), so there is almost no headroom to gain.
* **Vehicles still in network: no.** 11 instead of 9 under moderate demand —
  two vehicles. Under heavy demand, 116 instead of 96, but the DQN also
  *admitted more traffic*: 1,462 vehicles entered under the DQN against 1,445
  under fixed-time at seed 42, because a less congested approach lets SUMO
  insert vehicles it had been holding back. More vehicles in means more
  unfinished at the 1800 s cut-off. The percentage is reported as measured and
  is not explained away, but it should be read next to the insertion counts.

The honest one-line summary: **the DQN made vehicles wait substantially less
without moving noticeably more of them**, and the effect is much larger when
the junction is not already saturated.

### Was training stable?

Reasonably, for this budget.

* Returns improved monotonically in trend but noisily episode to episode; the
  raw curves show that and are not smoothed.
* No NaN or diverging loss in any run; a test asserts the losses are finite.
* The loss does not fall to zero and is not expected to — the target moves
  every 500 gradient steps by design.
* The moderate scenario converged visibly by roughly episode 40 and then
  plateaued. The heavy scenario was still noisy at episode 100.

### Are more episodes needed?

**For heavy traffic, yes.** Its return was still fluctuating by ±10 % between
late episodes and its per-episode waiting time had not settled, so 100
episodes is a floor rather than a converged result. For moderate traffic the
curve plateaued, and more episodes would most likely change little.

More generally, 18,000 decisions is a small budget by the standards of
published traffic-RL work. It was chosen so the run finishes on a student
laptop in about 20 minutes, which was an explicit project requirement.

### Do multiple seeds agree?

* **Evaluation (3 seeds, one model):** yes. Standard deviations are small
  relative to the gap — moderate waiting 5.71 ± 0.10 against 9.91 ± 0.20; the
  distributions do not overlap. Heavy waiting 47.91 ± 2.00 against 52.60 ±
  1.46 — the means differ by more than either deviation, but the overlap is
  not negligible, so the heavy result is weaker evidence than the moderate
  one.
* **Training (2 seeds per scenario):** the two moderate runs finished within
  4 % of each other. Two seeds is enough to show the result is not a fluke and
  not enough to characterise the spread; three or more would be better.

### What remains for future work

1. More training episodes, especially for the heavy scenario.
2. A systematic hyperparameter search — the current values are sensible
   defaults, not tuned ones.
3. More actions: variable phase duration, more phases, protected left turns.
4. A comparison against PPO or A2C, and against a properly tuned
   vehicle-actuated controller, which is the fairer engineering baseline than
   a pure fixed-time plan.
5. Several coordinated intersections.
6. Time-varying demand, incidents, public-transport priority.
7. Sensor realism: noise, latency, missing detectors — the agent currently
   reads exact queue lengths instantly.
8. Real detector or camera input and hardware in the loop. **Not attempted
   here, and explicitly out of scope.**

### What this result is not

It is not evidence that DQN is the best algorithm for traffic control, that
these percentages would transfer to a real junction, or that the policy is
converged. It is evidence that, on this one simulated four-way intersection,
with this demand and this fixed-time plan as the baseline, a small DQN trained
for 100 episodes reduced mean vehicle waiting time by about 42 % under
moderate demand and about 9 % under heavy demand, and that the improvement
came from the learning rather than from the faster decision interval.
