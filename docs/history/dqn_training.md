> **Historical.** This describes the keep-or-switch Deep Q-Network and how it was
> trained, evaluated and compared, which the project built in
> Stages 1-3 (Oct-Nov 2026) and still ships as working code and
> reproducible evidence. It is **not** how IntelliSignal works today.
> The current system is one engine - an explicit list of vehicles, an
> Indian urban fleet, 15-30 s greens - described in `README.md` and
> `docs/dashboard.md`. Numbers in this file were measured before the
> Indian fleet existed and are not comparable with current results.

# IntelliSignal — Stage 3: the Deep Q-Network

This document explains the learning part of IntelliSignal: what a DQN is, why
this project uses one, exactly how it is wired to the SUMO simulation, how it
is trained and evaluated, and what the results do and do not show.

It is written to be read start to finish by someone who has not seen the code,
and to be defensible in a project presentation.

Stage 3 reports are historical. The active configuration uses a 5-second
decision interval, 5-second minimum green, gamma 0.98, and scenario-specific
observation scales.

For exact reproduction of the verified Windows environment, use
`requirements-win-py312.lock` with Python 3.12.13. `requirements.txt` remains
the portable minimum-version list.

---

## 1. The one-paragraph summary

A conventional traffic signal runs a fixed timetable: 30 seconds of green for
North–South, then 30 seconds for East–West, forever, whether a direction is
empty or jammed. IntelliSignal replaces that timetable with a small neural
network that looks at the live traffic every 5 seconds and answers one
question — *hold this green, or change it?* The network is trained by trial and
error inside the SUMO simulator: it tries decisions, is rewarded for less
congestion, and gradually learns which decision is better in which situation.
That training method is **Deep Q-learning**, and the network is the
**Deep Q-Network (DQN)**.

---

## 2. What a DQN actually is

### 2.1 The Q-value

Reinforcement learning describes a controller as an **agent** acting in an
**environment**. At each decision point the agent sees a **state**, picks an
**action**, and receives a **reward**.

A **Q-value**, written `Q(state, action)`, is the total future reward the agent
expects if it takes that action in that state and then keeps behaving as it
currently does. In this project the reward is a negative congestion cost, so a
Q-value is best read as *"how much waiting lies ahead"*, and a value closer to
zero is better.

If an agent knew every Q-value, controlling the signal would be trivial: look
at both actions, take the one with the higher Q-value. Learning the Q-values
*is* the problem.

### 2.2 Why "deep"

Classical Q-learning stores Q-values in a table, one row per state. That is
impossible here: the state contains four queue lengths and four waiting times
as continuous numbers, so there is no finite list of states to tabulate.

A DQN replaces the table with a neural network. The network takes the state as
input and outputs one Q-value per action. It **generalises**: a traffic
situation it has never seen is handled by similarity to ones it has, which is
exactly what a lookup table cannot do.

### 2.3 How it learns: the temporal-difference rule

The network is trained on its own experience. After taking action `a` in state
`s`, receiving reward `r` and landing in state `s'`:

```
what the network currently says   :  Q(s, a)
what the experience suggests      :  r + γ · max Q(s', a')
                                          a'
```

The second line is the **target**: the reward actually received, plus the
discounted value of the best thing available next. The difference between the
two lines is the error, and the network is nudged to reduce it. Repeated over
thousands of transitions, the estimates become consistent with the rewards the
simulation really gives — this is **temporal-difference learning**, and it is
the single idea at the centre of DQN.

---

## 3. Why DQN for this project

The choice follows from the environment that Stage 2 built, not from fashion:

| Property of IntelliSignal | What it implies |
|---|---|
| The action space is **discrete** and tiny: `keep` or `switch` | DQN outputs one Q-value per action, which needs a finite action list. Two actions is the easiest possible case. |
| The state is **continuous** (queues, waiting times) | A Q-table is impossible; a function approximator is required. |
| The reward is **dense** — every 5-second decision produces one | Value-based learning has a usable signal at every step, instead of one reward at the end of an episode. |
| Experience is **expensive** (every episode is a 1800-second SUMO run) | A replay buffer lets each simulated decision train the network many times, which matters on a student laptop. |
| The project must be **explainable** in a report and a viva | The whole update rule fits in the five lines above, and the two Q-values behind any decision can be shown on screen. |

**What this is not a claim about.** DQN is not universally better than PPO,
A2C, SAC or Rainbow. Policy-gradient methods such as PPO are often preferred
for traffic control, particularly for multi-intersection coordination or
continuous phase-duration control, and sumo-rl is commonly used with them. DQN
was chosen because it is the natural fit for a two-action discrete problem and
the easiest to justify line by line. **PPO is listed as future scope, not as
part of Stage 3**, and no comparison against PPO has been run here.

---

## 4. The pipeline, end to end

```
                    SUMO four-way intersection          (Stage 1, unchanged)
                              |
                            TraCI
                              |
                    IntelliSignalEnv                    (Stage 2, unchanged)
                              |
                 10-value observation + reward
                              |
            +-----------------+------------------+
            |                                    |
     policy network  ----- Q(keep), Q(switch)    |      (Stage 3: agents/)
            |                                    |
   epsilon-greedy (training) / greedy (eval)     |
            |                                    |
      action: 0 = keep, 1 = switch               |
            |                                    |
      ActionController safety rules              |      (Stage 2, unchanged)
            |                                    |
      traffic signal in SUMO                     |
            |                                    |
         reward  --------->  replay buffer  -----+
                                  |
                           random mini-batch
                                  |
                      TD loss vs the target network
                                  |
                        gradient step (Adam)
                                  |
                     every 500 steps: target <- policy
```

Two things are worth noticing about this diagram:

1. **Everything above the dashed middle already existed.** Stage 3 added the
   network, the buffer, the target network and the update. The road, the
   demand, the observation, the reward and the safety rules are Stage 1's and
   Stage 2's, used unchanged.
2. **The agent never touches the traffic light.** It outputs 0 or 1. The Stage
   2 `ActionController` decides what that means in signal terms.

---

## 5. The state: 10 numbers

The current 10-number observation is defined in `rl/state.py`:

| Index | Value | Normalised by |
|---|---|---|
| 0–3 | halting vehicles on the North, East, South, West approach | moderate 3; heavy 16 vehicles |
| 4–7 | summed current waiting time on each approach | moderate 14; heavy 142 vehicle-seconds |
| 8 | current phase: 0.0 = NS green, 1.0 = EW green | — |
| 9 | how long the current green has been running | scale stored with each model; clipped at 1.0 |

Queue and waiting scales are scenario-specific and come from measured rollout
distributions. A checkpoint trained for one scenario is not directly
interchangeable with the other. Standalone `IntelliSignalEnv` instances retain
the legacy 20-second phase-age scale. New DQN training defaults to
`max_green=60` seconds so the policy can distinguish phase ages across the
full legal range; pass `--phase-elapsed-scale 20` to reproduce the legacy
training scale. The dashboard's validated Stage 4 incumbent remains on scale
20 because that is the scale used to train its weights. Stage 5's 20-vs-60
study is still incomplete, so scale 60 is the new-training default, not an
experiment result or an automatic incumbent replacement. Rollouts use the
checkpoint's recorded scale.

The network's input size is read from `env.observation_space.shape`, never
typed in. If an eleventh value were added to the observation, the network
would follow automatically; a test asserts that.

---

## 6. The actions: keep or switch

Defined in `rl/action.py` (Stage 2), also unchanged:

```
0 = KEEP    hold the direction that currently has green
1 = SWITCH  end this green and give green to the other direction
```

### Why the agent cannot cause an unsafe signal

This is the part most worth stressing in a presentation. The DQN is not
allowed to drive the signal directly. Its 0-or-1 answer goes to the Stage 2
controller, which enforces:

* **Yellow clearance.** A switch always runs the Stage 1 yellow state (4 s)
  before the other direction turns green. Green can never turn directly into
  the conflicting direction's green.
* **`min_green` = 5 s.** A switch requested too early is refused and recorded
  as refused — it is not hidden.
* **`max_green` = 60 s.** A `keep` after 60 s of green is overridden into a
  switch. No approach can be starved, whatever the network wants.

So the learned policy is a *request*, filtered by a rule-based safety layer.
A test drives the DQN with pure exploration and asserts that no green → green
transition appears anywhere in the recorded signal states.

---

## 7. The reward

Stage 2's reward, used **unchanged** — this matters, because changing the
reward between the baseline and the learned controller would make the
comparison meaningless:

```
reward = -0.01 × (total current waiting time + total queue length)
```

averaged over every simulated second of the 5-second decision interval.

* It is always ≤ 0: congestion is a cost and zero is perfection.
* Maximising it means reducing waiting and queues — the project's actual goal.
* The `0.01` scale only keeps the numbers small; multiplying every reward by a
  positive constant cannot change which action is best.
* Averaging over the interval instead of sampling the last second removes a
  lot of noise for free.
* `fairness_weight` exists in the configuration and defaults to **0.0**.
  Fairness is guaranteed structurally by `max_green`, not bought with a reward
  term, which keeps the reward to one sentence.

No reward shaping, bonus or penalty was added for Stage 3. The reward the DQN
maximises is the congestion measure the project set out to reduce.

---

## 8. The Q-network

`agents/dqn_network.py`:

```
input  10  (from env.observation_space.shape)
        |
   Linear(10, 64) -> ReLU
        |
   Linear(64, 64) -> ReLU
        |
   Linear(64, 2)
        |
output  2  (from env.action_space.n)  ->  [Q(keep), Q(switch)]
```

About 5,000 parameters. Deliberately small: the input is 10 numbers and the
decision is binary, so a larger network would only be slower to train without
learning more. The hidden layers are configurable
(`network.hidden_layers` in `config/dqn/training.yaml`); the input and output
sizes are not configurable on purpose, because they belong to the environment.

---

## 9. The replay buffer

`agents/replay_buffer.py`. A fixed-size ring buffer of up to 50,000
transitions, each one `(state, action, reward, next_state, done, next action mask)`.

**Why it is necessary.** Consecutive traffic observations are nearly
identical — the queue 5 seconds from now looks much like the queue now.
Training a network on that stream in order makes successive gradient steps
strongly correlated and the learning unstable. The buffer fixes this two ways:

1. A **random** mini-batch mixes decisions from different moments and different
   episodes, so one update is not dominated by one traffic situation.
2. Each transition can be sampled **more than once**, so a single expensive
   SUMO run teaches the network many times.

Training does not begin until `learning_starts` = 500 transitions have
accumulated, so the first gradient step is not computed from a handful of
near-identical samples.

---

## 10. The target network

Two networks with identical architecture:

* **`policy_net`** — chooses actions and receives every gradient step.
* **`target_net`** — used *only* to compute the maximum Q-value over actions
  allowed by the next state's min/max-green timing rules.

`target_net` starts as an exact copy of `policy_net` and is refreshed from it
every 500 gradient steps.

**Why.** The target `r + γ·max Q(s', a')` is itself produced by a network. If
the same network that is being updated also produces the target, the target
moves with every step and the estimate chases a goal that runs away from it —
the classic divergence of naive Q-learning with a function approximator.
Freezing the target for 500 steps gives the policy network a stationary thing
to fit. A test asserts that the policy network has moved while the target has
not, and that the refresh happens on schedule.

---

## 11. Gamma — the discount factor

```
gamma = 0.98    (the active value in config/dqn/training.yaml)
```

Gamma says how much the agent values future reward compared with immediate
reward. At `gamma = 0`, the agent is myopic: it would only care about the next
5 seconds and would happily create a jam just after. At `gamma → 1`, rewards
far in the future count almost as much as the present.

At 0.98, one decision interval later is worth 0.98, ten intervals (50 simulated
seconds) later about 0.82, and thirty intervals about 0.55. Future reward now
decays more slowly than it did under the historical gamma 0.9 configuration.

---

## 12. Epsilon-greedy exploration

A network that always takes the action it currently believes is best never
discovers that the other one was better. **Epsilon-greedy** is the standard
fix:

```
with probability epsilon : a random action      (explore)
otherwise                : argmax Q(state)      (exploit)
```

Schedule used here:

```
epsilon_start : 1.0     at the beginning every action is random
epsilon_end   : 0.05    at the end, 1 decision in 20 is still random
epsilon_decay : linear over exploration_fraction (0.5) × total training steps
```

The decay length is **derived from the number of episodes requested**, not
fixed. A 100-episode run is 18,000 decisions, so epsilon falls to 0.05 over
the first 9,000; a 20-episode demonstration run is 3,600 decisions and decays
over the first 1,800. This is deliberate: a hard-coded schedule tuned for 100
episodes would make a 20-episode run stop exploring after two episodes and
learn nothing.

Epsilon is logged for every episode, saved in the training history CSV, and
plotted.

**During evaluation, epsilon is 0.** Every evaluation action is `argmax Q`.
The evaluation script asserts afterwards that the agent took no gradient step
and that its replay buffer did not change, so "we did not train during
evaluation" is checked, not promised.

---

## 13. The training loop

`training/train_dqn.py`:

```
for each episode:
    observation = env.reset(seed = base_seed + episode)
    repeat until the episode ends (360 decisions = 1800 simulated seconds):
        epsilon = schedule(global_step)
        action  = epsilon-greedy over legal actions for the current green age
        observation', reward, terminated, truncated = env.step(action)
        buffer.push(observation, executed_action, reward, observation', terminated)
        every train_frequency steps : one gradient step on a random batch
        every 500 gradient steps    : target_net <- policy_net
    record the episode's metrics, save checkpoints
```

Three details that matter:

* **`terminated` vs `truncated`.** Only `terminated` (no vehicle left and none
  expected) is stored as a real end of the world. `truncated` is just the
  1800-second time limit, and that transition keeps its future term —
  otherwise the agent would learn that time stops at 1800 s.
* **The SUMO seed advances per episode** (`vary_episode_seed: true`), so the
  agent sees different arrival patterns and the policy is less tied to one
  traffic sample.
* **`best.pt` follows a moving average**, not a single episode. It is saved
  when the mean return of the last 5 episodes improves. One lucky episode is
  not a better policy.

### Per-episode metrics recorded

The learning numbers come from the loop; the traffic numbers come from
`utils/metrics.py` — the same collector that produced the Stage 1 baseline,
attached to the live simulation through the environment's per-second hook. A
training episode and a baseline run are therefore measured identically:

```
episode, total_reward, average_reward, episode_length, epsilon, loss,
average_queue_length, average_waiting_time, throughput, max_queue_length,
vehicles, vehicles_still_in_network, teleports, switch_count,
forced_switches, blocked_switches, replay_size, gradient_steps,
target_updates, sumo_seed, wall_clock_seconds
```

Written to `results/training/dqn_<scenario>_seed<seed>_training.csv` after
**every** episode, so an interrupted run still leaves a usable record.

---

## 14. Hyperparameters

All in `config/dqn/training.yaml`, all overridable on the command line, all
written into the checkpoint metadata so a result can be traced to the settings
that produced it.

| Setting | Value | Why |
|---|---|---|
| `episodes` | 100 | 18,000 decisions; finishes on a laptop |
| `episode_duration` | 1800 s (scenario default) | identical to the Stage 1 baseline run |
| `hidden_layers` | [64, 64] | small enough to train fast, large enough for a 10-value input |
| `batch_size` | 64 | standard; stable with a 50k buffer |
| `learning_rate` | 0.001 | Adam's usual starting point |
| `gamma` | 0.9 | the project's planned value |
| `loss` | Huber (`smooth_l1`) | less sensitive than MSE to the occasional large TD error in heavy traffic |
| `gradient_clip` | 10.0 | a second guard against an exploding update |
| `replay_capacity` | 50,000 | holds a whole 100-episode run |
| `learning_starts` | 500 | don't learn from a near-empty buffer |
| `train_frequency` | 1 | one gradient step per decision |
| `target_update_frequency` | 500 | stationary target for ~3 episodes at a time |
| `epsilon_start` / `end` | 1.0 → 0.05 | always keeps a little exploration |
| `seed` | 42 | the project's seed |

---

## 15. Reproducibility and seeds

`--seed` sets Python's `random`, NumPy, PyTorch and the SUMO seed.

```bash
python training/train_dqn.py --scenario moderate --episodes 100 --seed 42
python training/train_dqn.py --scenario moderate --episodes 100 --seed 43
python training/train_dqn.py --scenario moderate --episodes 100 --seed 44
```

A run with the project seed (42) owns `models/dqn_<scenario>/`; any other seed
gets `models/dqn_<scenario>_seed<seed>/`, so two training runs never overwrite
each other's checkpoints.

**What a seed does not guarantee.** Results can still differ between SUMO
versions, between operating systems and between PyTorch builds. Reproducibility
here means "the same machine and the same versions give the same numbers",
which the tests check, not "these exact numbers will appear everywhere".

---

## 16. Checkpoints

`models/dqn_<scenario>/` contains:

```
best.pt                 lowest mean waiting on the configured validation seeds
final.pt                the last episode
checkpoint_ep010.pt …   every 10 episodes
metadata.json           human-readable record of the run
```

**Dashboard default:** `models/active_models.json` pins the validated Stage 4
incumbent checkpoints under `output/dqn_retraining_20261120/models/`. The
registry records each file's SHA-256 and observation/timing contract. The
original Stage 3 files remain in `models/dqn_<scenario>/` for provenance, but
their 10-second timing and shared 50/1500 queue/wait scales do not match the
current scenario settings. They have no verified observation contract and
the loader rejects them when a scenario is specified.

Each `.pt` file stores the policy network, the target network, the optimiser
state, epsilon, the episode counter, the full configuration, the architecture
string, the observation and action sizes, the seed, the save time, the PyTorch
version, and an observation contract containing the scenario queue/wait scales,
phase-age scale, signal timings, and selection validation seeds. `metadata.json`
repeats the contract in plain text. The loader compares it with the active
scenario config before inference and rejects known mismatches. The frozen Stage
4 checkpoints use the checked-in registry as their audited contract; their
source metadata remains untouched.

---

## 17. Evaluation

```bash
python evaluation/evaluate_dqn.py --scenario moderate --seeds 42 43 44
python evaluation/evaluate_dqn.py --scenario heavy --seeds 42 43 44
```

* epsilon = 0 — pure `argmax Q`
* no `remember()`, no `learn()`; asserted afterwards
* checkpoint-selection seeds are rejected by default in both evaluation and
  controller comparison; use fresh seeds for independent confirmation
* the same network, demand and duration as the fixed-time baseline
* refuses outright if no trained model exists; never evaluates random weights

Output: `results/evaluation/dqn_<scenario>_evaluation.csv` (one row per seed,
raw) and `..._evaluation_summary.csv` (mean / std / min / max per metric).

---

## 18. The Fixed-Time vs DQN comparison

```bash
python evaluation/compare_controllers.py --scenario moderate --seeds 42 43 44
python evaluation/compare_controllers.py --scenario heavy --seeds 42 43 44
```

### What makes it fair

For each seed, **one** set of SUMO input files is built by
`rl.environment.episode_scenario_config()` and both controllers run on it:

| Held identical | How |
|---|---|
| road network | `sumo/network/intersection.net.xml`, untouched since Stage 1 |
| traffic demand | the same generated route file, same seed, same vehicle arrival pattern |
| duration | the same number of simulated seconds, computed once |
| measurement | `utils.metrics.MetricsCollector` — the class that produced the Stage 1 baseline |

The only difference is who decides when the signal changes: SUMO's fixed-time
program, or the trained network. Both paths live in
`simulation/agent_simulation.py`, so there is one code path, not two that
could drift apart.

A test asserts the strongest form of this: the fixed-time comparison run at
seed 42 reproduces the **recorded Stage 1 baseline exactly** — same vehicle
count, same average waiting time, same queue, same throughput. If it did not,
every improvement percentage would be measured against the wrong reference.

### Metrics compared

```
average waiting time        seconds per vehicle (accumulated, Stage 1 definition)
average queue length        halting vehicles on the 8 incoming lanes, per step
maximum queue length        the worst single step
throughput                  vehicles that completed their route
vehicles still in network    unfinished at the end of the run
teleports                   SUMO's gridlock escape hatch; 0 is good
```

### Improvement percentages

```
lower-is-better metric :  (fixed_time − dqn) / |fixed_time| × 100
throughput             :  (dqn − fixed_time) / |fixed_time| × 100
```

A **positive** percentage always means the DQN did better and a **negative**
one always means it did worse, for every metric. A zero denominator returns
`None` and is displayed as "n/a" rather than as an invented number, because a
percentage of zero has no meaning. Nothing is hidden and no sign is flipped.

### The untrained control

```bash
python evaluation/compare_controllers.py --scenario moderate --seeds 42 43 44 --include-untrained
```

The DQN decides every 5 seconds and may switch as often as `min_green`
allows, while the fixed-time plan holds 30 s per direction. **Part of any
improvement could therefore come from the shorter decision interval rather
than from anything the network learned.** `--include-untrained` runs the same
architecture with random weights on the same traffic as a control. If the
trained network is no better than the untrained one, the learning did not do
the work — and that is worth knowing before claiming it did. The measured
answer for this project is in the [Stage 3 report](STAGE3_REPORT.md).

### Output files

```
results/evaluation/comparison_<scenario>_raw.csv            every run, nothing dropped
results/evaluation/comparison_<scenario>_aggregated.csv     mean / std per controller
results/evaluation/comparison_<scenario>_improvement.csv    improvement % per metric
results/evaluation/comparison_<scenario>_<controller>_seed<seed>_timeseries.csv
results/plots/comparison/<scenario>_*.png                   the charts
```

`results/csv/` — the recorded Stage 1 baseline — is read for reference and
**never written**. Two tests assert its bytes are unchanged.

---

## 19. The charts

Training curves (`training/plot_training.py` → `results/plots/training/`):
return, loss, epsilon, queue and waiting time per episode, plus an overview
sheet.

Comparison charts (`evaluation/plot_comparison.py` →
`results/plots/comparison/`): waiting time, queue length, throughput and max
queue as bars with standard-deviation error bars across seeds, plus queue
length over time for both controllers on the same seed.

Two rules both scripts follow:

* Every point is a value read from a CSV that a SUMO run wrote.
* A moving average is drawn **on top of** the raw curve, never instead of it,
  and the legend says how long the window is. Traffic RL training is noisy and
  the noise is part of the result. With one seed there is no standard
  deviation and no error bar is drawn — an absent error bar is honest, a
  fabricated one is not.

---

## 20. Limitations

Stated plainly, because a project report is more useful with them than
without:

1. **One intersection.** No coordination, no green wave, no network effects.
   Multi-intersection control is a different and much harder problem.
2. **Two actions.** The agent cannot choose a phase duration, a phase order,
   or a protected left-turn stage. It can only hold or change.
3. **A small training budget.** 100 episodes is about 18,000 decisions, chosen
   so the run finishes on a student laptop. Published traffic-RL work often
   uses one to two orders of magnitude more. The policy here is not converged
   in any strong sense.
4. **Simulation only.** Everything the agent learned, it learned from SUMO's
   car-following and demand model. No real detector data, no camera, no
   hardware. Transfer to a real junction is unproven and outside this project.
5. **Perfect, free observation.** The agent reads exact queue lengths and
   waiting times instantly from the simulator. A real deployment would have
   sensor noise, latency and failures.
6. **Stationary demand.** Each scenario has a constant arrival rate for the
   whole episode. There is no rush-hour ramp, no incident, no weather.
7. **The decision interval is a confound.** See the untrained control above.
8. **No algorithm comparison.** DQN was not benchmarked against PPO, A2C or
   tuned actuated control. Claims are limited to DQN versus *this* fixed-time
   plan.
9. **Noisy results.** Returns vary substantially between episodes and between
   seeds. Conclusions are drawn from means over three seeds with standard
   deviations reported, not from a single best run.

---

## 21. Future scope (explicitly not implemented)

* PPO or another policy-gradient method, as an alternative algorithm.
* More actions: variable phase durations, more phases, protected turns.
* Several coordinated intersections.
* Real detector or camera input (CCTV / YOLO) and hardware in the loop.
* Time-varying demand, incidents, public-transport priority.
* Longer training and systematic hyperparameter search.

None of these exist in the repository. Anything labelled future scope here is
future scope in the code too.

---

## 22. Command reference

```bash
# train
python training/train_dqn.py --scenario moderate --episodes 20  --seed 42   # demo
python training/train_dqn.py --scenario moderate --episodes 100 --seed 42   # default
python training/train_dqn.py --scenario heavy    --episodes 100 --seed 42
python training/train_dqn.py --scenario moderate --episodes 100 --seed 43   # second seed

# curves from a recorded history
python training/plot_training.py --scenario moderate --seed 42

# evaluate (greedy, no learning)
python evaluation/evaluate_dqn.py --scenario moderate --seeds 42 43 44

# compare against the fixed-time baseline
python evaluation/compare_controllers.py --scenario moderate --seeds 42 43 44
python evaluation/compare_controllers.py --scenario heavy --seeds 42 43 44 --include-untrained
python evaluation/plot_comparison.py --scenario moderate

# dashboard, and the tests
streamlit run dashboard.py
python -m pytest -q
```

---

## 23. Where everything lives

| File | Contents |
|---|---|
| `agents/dqn_network.py` | the Q-network |
| `agents/replay_buffer.py` | the replay buffer |
| `agents/dqn_agent.py` | the agent: epsilon-greedy, learning, target network, checkpoints |
| `agents/dqn_config.py` | hyperparameter loading and validation |
| `config/dqn/training.yaml` | every hyperparameter |
| `training/train_dqn.py` | the training loop |
| `training/plot_training.py` | training curves |
| `simulation/agent_simulation.py` | one episode under one controller, measured identically |
| `evaluation/evaluate_dqn.py` | greedy evaluation, mean/std over seeds |
| `evaluation/compare_controllers.py` | Fixed-Time vs DQN, improvement percentages |
| `evaluation/plot_comparison.py` | comparison charts |
| `tests/test_dqn.py` | the Stage 3 tests |
| `docs/history/dqn_training.md` | this document |

Unchanged from earlier stages, and used as the contract: `sumo/`, `rl/`,
`utils/scenario_builder.py`, `utils/metrics.py` (one additive argument),
`simulation/run_simulation.py`, `results/csv/`.
