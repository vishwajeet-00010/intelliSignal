# IntelliSignal — DQN plateau: diagnosis and implementation brief

**Audience:** a coding agent (Codex) implementing the fixes.
**Status of the findings below:** all measured on this machine on 2026-11-20 against this
repository at commit `c7707b6`. Numbers are reproducible with the scripts in Appendix A.
**Do not re-derive the diagnosis.** It is settled. Implement the fixes.

---

## 0. Ground rules — read before touching anything

### 0.1 The problem in one sentence

The DQN is **not under-trained**. It converges to a degenerate constant policy
("always request SWITCH") and then sits against a **16-second structural green floor**
that it cannot cross. 300 episodes, 300 models and multi-checkpoint training cannot
move a policy that is already saturated against an action-space limit.

### 0.2 Files you will modify are integrity-protected

`experiments/controlled_20261119/protected_source_before.json` records SHA-256 hashes of
41 source files, including every file this brief asks you to change:

```
rl/action.py        rl/config.py      rl/environment.py   rl/state.py
agents/dqn_agent.py training/train_dqn.py
```

That snapshot exists so a *past* experiment could prove it changed nothing. It does **not**
forbid deliberate change. But:

- **Do not** edit `models/**` — those are the incumbent checkpoints and must survive intact.
- **Do not** edit or delete anything under `experiments/**` — historical evidence.
- **Do not** silently change the hashes. Any new experiment must take a **new** integrity
  snapshot and declare in its manifest exactly which source files changed and why.
- **Do not** change the `signal:` section of `config/heavy.yaml` / `config/moderate.yaml`
  in place. That section defines the Stage-1 fixed-time baseline that every historical
  number is measured against. See FIX 7 for the correct way to introduce a tuned baseline.

### 0.3 Python environment is broken — fix this first

`.venv-stage3/pyvenv.cfg` points at `C:\Users\Vishwajeet\AppData\Roaming\uv\python\...`,
an interpreter that does not exist on this machine. The venv's `site-packages` is intact
but its `numpy` is compiled for cp312 while the system Python is 3.11.6.

Working recipe used for all measurements in this document:

```bash
# For scripts that do NOT import numpy/torch (pure traci + sumolib):
PYTHONPATH="$PWD/.venv-stage3/Lib/site-packages" python script.py

# For scripts that DO import numpy: append the venv path at runtime instead,
# so the SYSTEM numpy wins and only traci/sumolib come from the venv:
#   sys.path.append(os.path.join(ROOT, ".venv-stage3", "Lib", "site-packages"))
```

The SUMO binary resolves to
`.venv-stage3/Lib/site-packages/sumo/bin/sumo.exe` via `sumolib.checkBinary("sumo")`.

**Before running any training**, rebuild a real venv with Python 3.12 and
`torch==2.14.1 numpy==2.5.4 gymnasium==1.4.0 eclipse-sumo==1.28.0 traci==1.28.0 sumolib==1.28.0`
(versions from `experiments/controlled_20261119/execution_environment.json`).

---

## 1. The four defects

### DEFECT 1 — The 16-second green floor (root cause)

**Where:** `config/rl/environment.yaml:29,33` + `rl/action.py:151-160`

Three values interact to produce a floor nobody chose:

| setting | file:line | value |
|---|---|---|
| `decision_interval` | `config/rl/environment.yaml:29` | 10 |
| `min_green` | `config/rl/environment.yaml:33` | 10 |
| `yellow` | `config/heavy.yaml` / `config/moderate.yaml` `signal.yellow` | 4 |

`ActionController.commit()` at `rl/action.py:151-160`:

```python
def commit(self, effective_action: int) -> None:
    if effective_action == ACTION_SWITCH:
        self.phase = self.other_phase()
        self.switch_count += 1
        # Only the green part of the interval counted as green time.
        self.green_elapsed = (self.decision_interval
                              - self.yellow_time - self.all_red_time)   # = 10 - 4 - 0 = 6
    else:
        self.green_elapsed += self.decision_interval
```

and `ActionController.resolve()` at `rl/action.py:119-121`:

```python
if action == ACTION_SWITCH:
    if self.green_elapsed < self.min_green:      # 6 < 10  -> ALWAYS true after a switch
        info["switch_blocked"] = True
        return ACTION_KEEP, info
```

**Consequence.** After every switch `green_elapsed == 6`, which is below `min_green == 10`,
so the **next decision is always blocked**. Traced state machine:

```
dec0 elapsed= 0 -> SWITCH BLOCKED, forced KEEP
dec1 elapsed=10 -> SWITCH allowed, green completed at 10 s
dec2 elapsed= 6 -> SWITCH BLOCKED, forced KEEP
dec3 elapsed=16 -> SWITCH allowed, green completed at 16 s   <-- steady state
dec4 elapsed= 6 -> SWITCH BLOCKED, forced KEEP
dec5 elapsed=16 -> SWITCH allowed, green completed at 16 s
```

- **Shortest achievable green = 16 s. Shortest achievable cycle = 40 s.**
- Maximum effective switches per 180-decision episode = **90**.

Confirmed in `experiments/controlled_20261119/diagnostics.json` — every trained policy, both
scenarios, all checkpoints, reports `completed_green_duration_counts` dominated by `"16.0"`
(heavy incumbent_best: 880 of 890 greens are exactly 16.0 s).

### DEFECT 2 — Replay buffer stores the requested action, not the executed one

**Where:** `training/train_dqn.py:296-304`

```python
action = agent.act(observation, training=True)
next_observation, reward, terminated, truncated, info = env.step(action)
...
agent.remember(observation, action, reward, next_observation, terminated)
#                            ^^^^^^ REQUESTED action
```

`rl/environment.py:243` resolves the request through the safety layer and
`rl/environment.py:376` already exposes the truth as `info["effective_action"]` — it is
simply not used.

**Consequence.** In the heavy incumbent diagnostics, **811 of 1701** SWITCH requests (48%)
are silently converted to KEEP. `Q(s, SWITCH)` is therefore trained on a mixture of
real-switch and keep dynamics. This makes "always request SWITCH" a **self-reinforcing
attractor**: in states where switching would be costly the request tends to be blocked
anyway, so requesting it costs nothing.

### DEFECT 3 — Observation normalisation calibrated on a random policy

**Where:** `config/rl/environment.yaml:69-70`

```yaml
max_queue_per_approach: 50        # halting vehicles on one approach
max_waiting_per_approach: 1500    # summed current waiting time, vehicle-seconds
```

The file's own comment says these were measured "under a random policy and under a
keep-only policy". Under a *trained* policy the inputs collapse. From
`diagnostics.json` → `raw_normalized_max_by_dim` (1.0 = full input range):

| policy | 4 queue inputs | 4 waiting inputs | phase_elapsed |
|---|---|---|---|
| heavy, untrained | 0.32 – 0.80 | 0.45 – **1.21** (clips) | 1.10 |
| heavy, incumbent_best | 0.16 – 0.60 | **0.07 – 0.18** | 0.33 |
| moderate, untrained | 0.16 – 0.18 | 0.18 – 0.21 | 1.10 |
| moderate, incumbent_best | **0.06 – 0.08** | **0.023 – 0.031** | 0.43 |

**Consequence.** For moderate, 8 of the 10 observation dimensions live in the bottom 2–8 %
of their range. The only dimension carrying real signal is the categorical `current_phase`
bit (0/1). The network effectively cannot distinguish traffic states, which is precisely
why the learned policy is periodic rather than adaptive. `phase_elapsed` is squashed the
same way: `max_green` is 60 but the policy never holds a green past ~20 s.

A normaliser calibrated on a bad controller is useless for a good one.

### DEFECT 4 — Discount horizon far too short

**Where:** `config/dqn/training.yaml:50` — `gamma: 0.9`

With a 10-second decision interval, `1/(1-0.9) = 10` decisions ≈ **100 simulated seconds**.
Clearing an oversaturated heavy approach takes several minutes. The agent is structurally
myopic. (This is a contributing factor, not the root cause — see §2.3.)

---

## 2. Measured evidence

All runs: 1800 s episodes, SUMO seeds **3001–3005** (the frozen final-test seeds), the
project's own `MetricsCollector`, identical network and demand.

### 2.1 Training stopped at episode ~175

From `experiments/dqn_300_20261117/heavy/training/dqn_heavy_seed42_training.csv`,
25-episode blocks:

| ep | epsilon | switches/ep | return | wait |
|---|---|---|---|---|
| 25 | 0.918 | 63.1 | −1224.8 | 57.57 |
| 125 | 0.284 | 82.6 | −702.6 | 49.56 |
| **175** | **0.050** | **86.9** | **−612.9** | **48.44** |
| 300 | 0.050 | 86.5 | **−613.4** | 48.35 |

Return moves by **0.5 in 125 episodes / ~22,500 gradient steps**. The entire visible
"learning curve" tracks the epsilon schedule and stops the moment epsilon hits its 0.05
floor. `switch_count` converges to **86.5 against a theoretical maximum of 90** (96 %).
Moderate is identical in shape.

### 2.2 The trained DQN is behaviourally a fixed-time controller

| | DQN (from the controlled-experiment PDF) | plain fixed-time, green = 16 s |
|---|---|---|
| moderate mean waiting | 5.719 | **5.72** |
| heavy mean waiting | 46.974 | 48.04 |
| heavy throughput | 1345.0 | **1345.6** |

### 2.3 The reward function is CORRECT — the agent just cannot reach the optimum

Measured under fixed-time control at each green duration: the DQN's own reward
(`rl/reward.py` applied to `rl/state.py`'s raw readings, scale 0.01), the reported KPI,
and throughput. Heavy, 2 seeds:

| green | cycle | **DQN reward** | **reported wait** | **throughput** |
|---|---|---|---|---|
| 5 | 18 s | −0.850 | 22.42 | 1354 |
| 6 | 20 s | −0.950 | 21.61 | 1398 |
| **7** | **22 s** | **−0.595** ← max | **12.11** ← min | **1445** ← max |
| 8 | 24 s | −1.211 | 24.13 | 1400 |
| 10 | 28 s | −1.763 | 33.52 | 1356 |
| **16 (DQN floor)** | 40 s | −3.322 | 48.34 | 1348 |
| 30 (baseline) | 68 s | −5.733 | 51.43 | 1352 |

**reward argmax == KPI argmin == throughput argmax == green 7.**

This is the decisive result. Do **not** "fix" `rl/reward.py` — it ranks the optimum
correctly. The DQN implementation in `agents/dqn_agent.py` is also textbook-correct
(verified: target network, Huber loss, gradient clipping, correct terminal handling).
The agent follows the reward faithfully and hits a wall at 16 s.

### 2.4 How much performance is on the table

**HEAVY** — 19 green durations × 5 seeds:

| green | cycle | wait | sd | mean queue | throughput | left in net |
|---|---|---|---|---|---|---|
| 5 | 18 s | 22.49 | 0.44 | 18.13 | 1355 | 110 |
| 6 | 20 s | 21.40 | 1.24 | 17.64 | 1400 | 102 |
| **7** | **22 s** | **12.46** | **0.46** | **10.18** | **1443** | **61** |
| 8 | 24 s | 24.13 | 0.44 | 19.84 | 1399 | 105 |
| 9 | 26 s | 27.32 | 1.14 | 22.18 | 1388 | 106 |
| 10 | 28 s | 33.79 | 0.78 | 26.96 | 1358 | 112 |
| 12 | 32 s | 41.26 | 0.93 | 32.53 | 1342 | 115 |
| 14 | 36 s | 45.00 | 0.94 | 35.43 | 1344 | 112 |
| **16 ← DQN floor** | 40 s | **48.04** | 1.05 | 37.92 | 1346 | 116 |
| 20 | 48 s | 50.89 | 2.53 | 40.26 | 1353 | 113 |
| **30 ← baseline** | 68 s | **51.68** | 0.79 | 40.25 | 1352 | 95 |
| 50 | 108 s | 59.85 | 1.44 | 46.58 | 1344 | 98 |
| 70 | 148 s | 63.28 | 0.87 | 49.46 | 1339 | 104 |

vs the DQN's 46.97: **−73 % waiting, +98 vehicles throughput.**

Note the `left in net` column: at green 7 only 61 vehicles are stranded versus 95–116
everywhere else. That is the mechanism — queues drop from ~38 to ~10, approach spillback
stops, and the intersection reaches its real capacity. This is why a *shorter* cycle
raises throughput here, contrary to classical Webster reasoning.

**MODERATE** — 15 green durations × 5 seeds:

| green | cycle | wait | sd | mean queue |
|---|---|---|---|---|
| 3 | 14 s | 3.23 | 0.13 | 1.21 |
| 4 | 16 s | 3.25 | 0.07 | 1.20 |
| **5** | **18 s** | **3.03** | **0.10** | **1.09** |
| 6 | 20 s | 3.37 | 0.12 | 1.23 |
| 7 | 22 s | 3.29 | 0.08 | 1.19 |
| 8 | 24 s | 3.93 | 0.13 | 1.35 |
| 10 | 28 s | 4.45 | 0.18 | 1.60 |
| **16 ← DQN floor** | 40 s | **5.72** | 0.09 | 2.07 |
| **30 ← baseline** | 68 s | **9.81** | 0.14 | 3.62 |
| 50 | 108 s | 16.17 | 0.04 | 6.04 |

vs the DQN's 5.72: **−47 % waiting.**

### 2.5 The 7-second optimum is a knife-edge — this is the case FOR RL

6 s → 21.40, **7 s → 12.46**, 8 s → 24.13. A ±1 s error destroys ~75 % of the gain.
A hand-tuned fixed-time plan is fragile to any demand change; a controller that reacts to
observed queues should hold this regime robustly. **That is the project's actual value
proposition, and it has never been tested, because the action space cannot act at that
timescale.** Keep this framing in the final report.

---

## 3. Fixes, in order

### FIX 1 (P0) — Break the 16 s floor

**File:** `config/rl/environment.yaml`

```yaml
# line 29
decision_interval: 10     ->  decision_interval: 5
# line 33
min_green: 10             ->  min_green: 5
```

**Why this works.** With `decision_interval=5`, `yellow=4`: after a switch
`green_elapsed = 5 - 4 = 1`, which is `< 5` so the next decision is blocked, then
`1 + 5 = 6 >= 5` and a switch is allowed. Achievable greens become
**{6, 11, 16, 21, …}** instead of {16, 26, 36, …}.

An always-switch policy then produces 6 s green + 4 s yellow per phase = a 20 s cycle,
which is **exactly** the fixed-time `green=6` row in §2.4.

**Predicted effect (read directly off the measured tables):**

| scenario | now (16 s floor) | after FIX 1 (6 s floor) | change |
|---|---|---|---|
| heavy | 46.97 | **21.40** | −54 % |
| moderate | 5.72 | **3.37** | −41 % |

**Validate the constraint.** Two places enforce
`decision_interval > yellow_time + all_red_time`: `rl/config.py:125-130` and the
`ActionController.__init__` guard at `rl/action.py:88-93`. With `5 > 4 + 0` both pass.
Do not go below 5 without doing FIX 6 first, or `plan()` will produce a zero-length green.

**Also check** `rl/config.py:133`: `max_green` must exceed `min_green` — 60 > 5, fine.
Leave `max_green: 60` alone; it is the fairness guarantee.

**Episode length changes.** 1800 s / 5 s = **360 decisions per episode**, double the
current 180. `config/dqn/training.yaml` derives the epsilon schedule from
`exploration_fraction × episodes × decisions_per_episode`, so it rescales automatically.
Replay capacity 50 000 still holds a 100-episode run (36 000 transitions).

---

### FIX 2 (P0) — Store the action that was actually executed

**File:** `training/train_dqn.py:304`

```python
# BEFORE
agent.remember(observation, action, reward, next_observation, terminated)

# AFTER
executed = int(info.get("effective_action", action))
agent.remember(observation, executed, reward, next_observation, terminated)
```

`info["effective_action"]` is already populated at `rl/environment.py:376`. No change is
needed in `rl/environment.py`.

**Better, do this as well — mask unavailable actions at selection time.**

`agents/dqn_agent.py:364-374`, `act()` currently is:

```python
def act(self, observation, training: bool = False) -> int:
    if training and self._random.random() < self.epsilon:
        return int(self._random.randrange(self.num_actions))
    return int(np.argmax(self.q_values(observation)))
```

Add an optional mask so a blocked SWITCH can never be chosen or explored:

```python
def act(self, observation, training: bool = False,
        valid_actions: Optional[Sequence[int]] = None) -> int:
    """valid_actions: the actions the environment will actually honour.
    None means all actions are available (preserves the old behaviour)."""
    if valid_actions is None:
        valid_actions = range(self.num_actions)
    valid = [int(a) for a in valid_actions]
    if not valid:
        raise ValueError("act() needs at least one valid action")
    if training and self._random.random() < self.epsilon:
        return int(self._random.choice(valid))
    values = self.q_values(observation)
    return int(max(valid, key=lambda a: values[a]))
```

and at the call site, `training/train_dqn.py:296`:

```python
switch_available = env.controller.green_elapsed >= env.controller.min_green
valid = [ACTION_KEEP, ACTION_SWITCH] if switch_available else [ACTION_KEEP]
action = agent.act(observation, training=True, valid_actions=valid)
```

**Keep `act()` backward-compatible** (`valid_actions=None` ⇒ old behaviour). The dashboard,
`evaluation/evaluate_dqn.py` and `agents/dqn_agent.py::select_action` all call `act()` and
must keep working unchanged. Apply the same mask in `evaluation/evaluate_dqn.py` so greedy
evaluation matches training.

---

### FIX 3 (P1) — Recalibrate the observation normalisation, per scenario

**Do this AFTER FIX 1**, because the fix changes the queue distribution (heavy mean queue
drops from ~38 to ~10–18).

`rl/config.py:87` already supports a per-scenario override — a scenario file may carry its
own `rl:` section and it is deep-merged over `config/rl/environment.yaml`. **No code change
is required.** Add to `config/heavy.yaml` and `config/moderate.yaml`:

```yaml
# config/heavy.yaml — append; do NOT touch the existing `signal:` block
rl:
  observation:
    max_queue_per_approach: <p99 of per-approach queue from new rollouts>
    max_waiting_per_approach: <p99 of per-approach waiting from new rollouts>
```

**Procedure (do not guess the constants):**

1. Apply FIX 1 + FIX 2.
2. Run 10 greedy rollouts per scenario with the current incumbent, logging the **raw,
   un-normalised** per-approach queue and waiting from `StateExtractor.raw()`.
3. Set each divisor to roughly the **99th percentile** of the observed values, so a good
   policy uses most of [0, 1] and only genuine outliers clip.
4. Re-run and confirm `raw_normalized_max_by_dim` now lands in **0.6–1.0**, not 0.02–0.18.

For reference, the current trained policy's actual maxima are: heavy waiting ≈ 264
(divisor 1500), moderate queue ≈ 4 (divisor 50), moderate waiting ≈ 47 (divisor 1500).

**Also fix `phase_elapsed`.** `rl/state.py:109` divides `green_elapsed` by `max_green`
(60), but no green ever exceeds ~20 s, so the input never leaves [0, 0.33]. Normalise by a
realistic upper bound instead — e.g. `min(max_green, 4 * min_green)` — or leave `max_green`
as the divisor and accept it, but say so explicitly in the report.

**Integrity note:** editing `config/heavy.yaml` / `config/moderate.yaml` changes their
hashes. Adding an `rl:` section does **not** change the `signal:` block, so the Stage-1
fixed-time baseline behaviour is untouched — but you must declare the file change in the
new experiment's manifest.

---

### FIX 4 (P1) — Lengthen the discount horizon

**File:** `config/dqn/training.yaml:50`

```yaml
gamma: 0.9    ->   gamma: 0.98
```

With `decision_interval=5` (FIX 1), `gamma=0.9` gives a 50-second horizon; `0.98` gives
`1/(1-0.98) = 50` decisions ≈ 250 s, which is the right order for queue clearing.

If training becomes unstable at 0.98, step down to 0.95 rather than reverting to 0.9, and
record which value was used.

---

### FIX 5 (P1) — Select checkpoints on the real objective

**File:** `training/train_dqn.py:168-177`

`best.pt` is currently chosen by the **5-episode mean training return**. That signal is
contaminated by epsilon-greedy exploration noise and measures the reward proxy rather than
the reported KPI. `experiments/controlled_20261119/diagnostics.json` records
`rank_reversal_count: 4` — four checkpoint/seed pairs where reward and waiting disagree.

Replace with: every `checkpoint_every` episodes, run **greedy** evaluation on a held-out
validation seed set and keep the checkpoint with the lowest mean waiting.

**Critical caveat already documented by the project's own reviewer**
(`experiments/controlled_20261119/REVIEW.md`): evaluation helpers reseed global RNG, and
constructing a network consumes Torch RNG. Mid-training evaluation **must** either run in a
separate process or fully capture and restore Python/NumPy/Torch RNG state around the
evaluation, or it will perturb the training run it is supposed to measure.

---

### FIX 6 (P2) — The real fix: replace the action space with green-duration selection

FIX 1 lowers the floor from 16 s to 6 s but the reachable set is still coarse
({6, 11, 16, 21, …}) and **misses the measured heavy optimum of 7 s** — the difference
between 21.40 s and 12.46 s of waiting.

The underlying design flaw is that one `decision_interval` is simultaneously:
the decision cadence, the green-duration quantum, and the yellow-overhead denominator
(`plan()` serves the yellow *inside* the interval, `rl/action.py:131-148`).

**Target design.** At the start of each green, the agent chooses its duration from a
discrete menu:

```python
GREEN_DURATIONS = [6, 7, 8, 10, 12, 16, 20, 26, 34]   # seconds
NUM_ACTIONS = len(GREEN_DURATIONS)
```

The environment then runs that green, followed by yellow + all-red, and asks again.

**Why this is better:**
- Every useful duration, including 7 s, is reachable.
- `min_green` / `decision_interval` coupling disappears entirely.
- No blocked actions — today 48 % of heavy decisions are discarded.
- Each decision is meaningful, so credit assignment is far cleaner.
- It is the standard formulation in the traffic-signal RL literature.

**Files to change:** `rl/action.py` (`NUM_ACTIONS`, `resolve`, `plan`, `commit`),
`rl/environment.py` (`action_space`, `step` timing loop, episode truncation accounting),
`rl/state.py` (`phase_elapsed` semantics). `agents/dqn_network.py` takes its output size
from `env.action_space.n`, so the network adapts automatically.

**`max_green` must still be enforced** — cap the menu so no green exceeds it.

Treat FIX 6 as a separate, reviewed change. Land FIX 1–5 first and measure.

---

### FIX 7 (P2) — Re-tune the fixed-time baseline before claiming any improvement

`IntelliSignal_Controlled_Comparison_20261119.pdf` reports the challenger as
"44.97 % below fixed-time" (moderate) and "9.24 % below fixed-time" (heavy). Both
baselines use `ns_green: 30, ew_green: 30`. §2.4 shows a **properly tuned fixed-time
controller beats the DQN by 47 % (moderate) and 73 % (heavy)**.

The promotion decision itself (retain the incumbent) was correct. The *improvement claim*
is not defensible against an untuned baseline.

**Do not overwrite `config/heavy.yaml:signal` or `config/moderate.yaml:signal`** — every
historical number depends on them. Instead:

1. Add new scenario files `config/heavy_tuned.yaml` and `config/moderate_tuned.yaml`,
   identical except `ns_green`/`ew_green` set from a committed sweep.
   Note `utils/config_loader.available_scenarios()` lists every `.yaml` directly inside
   `config/`, so these will appear as additional selectable scenarios — intended here.
2. Commit the sweep script (Appendix A) as `evaluation/fixed_time_sweep.py` so the tuned
   baseline is reproducible rather than asserted.
3. Report **three** controllers: untuned fixed-time (historical), **tuned fixed-time**,
   and the DQN. The honest claim is whatever survives against the tuned baseline.

Consider asymmetric greens too: demand is N/S = 850 and E/W = 650 veh/h but the plan is
a symmetric 30/30. A proportional split is an obvious additional baseline.

---

## 4. Acceptance criteria

Run on seeds **3001–3005**, 1800 s, greedy actions, the project's `MetricsCollector`.

**Gate A — FIX 1 + FIX 2 landed (expect from a short 100-episode run):**

| metric | must beat | source |
|---|---|---|
| heavy mean waiting | **< 25.0** s (vs 46.97 now) | §2.4, green 6 = 21.40 |
| moderate mean waiting | **< 4.0** s (vs 5.72 now) | §2.4, green 6 = 3.37 |
| heavy throughput | **≥ 1390** (vs 1345 now) | §2.4 |
| heavy `completed_green_duration_counts` | must contain values **other than** a single dominant bucket | adaptivity check |

**Gate B — the agent is no longer degenerate.** Re-run the diagnostics that produced
`experiments/controlled_20261119/diagnostics.json` and assert:

- `effective_switches / max_possible_switches` is **below 0.90** (currently 0.96–0.99).
  If it is still pinned at the ceiling, the agent has merely found a new floor and
  FIX 6 is required.
- `raw_normalized_max_by_dim` for the 8 traffic dimensions is in **0.6–1.0**
  (currently 0.02–0.18) — confirms FIX 3.
- `blocked_switches` is **0** if action masking was added.

**Gate C — training actually learns.** Plot return against episode and confirm it keeps
improving **after** epsilon reaches `epsilon_end`. If the curve flattens exactly when
epsilon flattens (as it does today at episode ~175), the policy is still degenerate and
no amount of additional training will help.

**Gate D — honest comparison.** The DQN must be compared against the **tuned** fixed-time
baseline from FIX 7, not the 30 s one.

---

## 5. Suggested sequencing

1. Rebuild the Python 3.12 environment (§0.3).
2. Commit the sweep script and the tuned baselines (FIX 7 steps 1–2). This delivers the
   −73 % / −47 % win immediately and independently of any RL work.
3. Land FIX 1 + FIX 2. Run **100 episodes**, not 300 — if the floor is the problem, 100
   episodes will show it, and today's evidence proves episodes 175–300 contribute nothing.
4. Check Gate A, B, C.
5. Land FIX 3 + FIX 4 + FIX 5. Re-run.
6. Only then consider FIX 6 and a longer run.
7. New experiment directory with a fresh integrity snapshot declaring every changed source
   file.

**Do not run 300 episodes or train 300 models again until Gate C passes.** The existing
evidence shows that budget buys nothing while the policy is saturated.

---

## Appendix A — Reproduction scripts

Both scripts write only throwaway SUMO inputs named `<scenario>_<tag>` and delete them on
exit. Neither modifies any project file. Recommended home: `evaluation/fixed_time_sweep.py`.

### A.1 Fixed-time cycle-length sweep (produced §2.4)

```python
"""Fixed-time green-duration sweep. Writes throwaway SUMO inputs named
'<scenario>_cyclesweep' and removes them at the end."""
import os, sys, csv, glob, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from simulation.run_simulation import prepare_sumo_imports, sumo_binary
from utils.config_loader import load_config
from utils.scenario_builder import build_scenario
from utils.metrics import MetricsCollector

prepare_sumo_imports()
import traci


def run_one(scenario, green, seed, duration=1800.0):
    cfg = load_config(scenario)
    cfg["scenario"]["name"] = scenario + "_cyclesweep"
    cfg["simulation"]["duration"] = float(duration)
    cfg["simulation"]["seed"] = int(seed)
    cfg["signal"]["ns_green"] = float(green)
    cfg["signal"]["ew_green"] = float(green)
    files = build_scenario(cfg)

    step_length = float(cfg["simulation"]["step_length"])
    collector = MetricsCollector(step_length=step_length)
    traci.start([sumo_binary(False), "-c", files["sumocfg"],
                 "--no-warnings", "--no-step-log"],
                label="sweep_{}_{}_{}".format(scenario, green, seed))
    try:
        for _ in range(int(round(duration / step_length))):
            traci.simulationStep()
            collector.collect_step()
    finally:
        try:
            traci.close()
        except Exception:
            pass
    s = collector.summary(scenario)
    s["green"] = green
    s["seed"] = seed
    s["cycle"] = 2 * (green + float(cfg["signal"]["yellow"]))
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--greens", default="5,6,7,8,9,10,12,16,20,30,50")
    ap.add_argument("--seeds", default="3001,3002,3003,3004,3005")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = []
    for g in [float(x) for x in a.greens.split(",")]:
        for sd in [int(x) for x in a.seeds.split(",")]:
            r = run_one(a.scenario, g, sd)
            rows.append(r)
            print("{} green={:>4.0f} cycle={:>5.0f} seed={} wait={:>8.2f} "
                  "queue={:>6.2f} tp={:>5d} left={:>4d}".format(
                      a.scenario, g, r["cycle"], sd, r["average_waiting_time"],
                      r["average_queue_length"], r["throughput"],
                      r["vehicles_still_in_network"]), flush=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for pat in ("sumo/routes/*_cyclesweep*", "sumo/config/*_cyclesweep*",
                "sumo/additional/*_cyclesweep*"):
        for p in glob.glob(os.path.join(ROOT, pat)):
            os.remove(p)
```

Run:

```bash
PYTHONPATH="$PWD/.venv-stage3/Lib/site-packages" python evaluation/fixed_time_sweep.py \
    --scenario heavy --greens 5,6,7,8,9,10,12,16,20,30,50 --out sweep_heavy.csv
```

### A.2 Reward-proxy probe (produced §2.3)

Identical to A.1 plus a per-second accumulation of the exact quantities `rl/reward.py`
consumes. Insert inside the stepping loop:

```python
# inlined verbatim from rl/state.py so the probe measures the identical lanes
APPROACH_ORDER = ("north", "east", "south", "west")
APPROACH_LANES = {"north": ["N2C_0", "N2C_1"], "east": ["E2C_0", "E2C_1"],
                  "south": ["S2C_0", "S2C_1"], "west": ["W2C_0", "W2C_1"]}
SCALE = 0.01   # config/rl/environment.yaml -> reward.scale

wait_sum = queue_sum = 0.0
n = 0
# ... inside the per-second loop, after traci.simulationStep():
w = q = 0.0
for ap in APPROACH_ORDER:
    for lane in APPROACH_LANES[ap]:
        w += traci.lane.getWaitingTime(lane)
        q += traci.lane.getLastStepHaltingNumber(lane)
wait_sum += w
queue_sum += q
n += 1
# ... after the loop:
dqn_reward_per_decision = -SCALE * (wait_sum / n + queue_sum / n)
```

This script imports nothing from `rl/`, so under the broken venv append the venv path at
runtime (§0.3) rather than putting it on `PYTHONPATH`, or the cp312 NumPy will shadow the
system one.

---

## Appendix B — What is NOT broken (do not "fix" these)

Verified during the investigation; changing them will waste time or cause regressions.

- **`rl/reward.py`** — ranks the true optimum correctly (§2.3). Leave it.
- **`agents/dqn_agent.py::learn()`** — textbook DQN. Target network used for the max,
  Huber loss, gradient clipping at 10.0, and `terminated` (not `truncated`) as the done
  flag so the 1800 s time limit correctly keeps its bootstrap term. Correct.
- **`utils/metrics.py`** — reproduces the published baseline exactly: fixed-time green 30,
  seed 3001 measured 51.99 here against 52.47 in the PDF (different seed set). Correct.
- **The experiment governance** — `freeze.json`, protected hashes, seed separation,
  independent review. This is genuinely strong work and should be preserved. The only gap
  is the untuned baseline (FIX 7).
- **Episode count / model count** — 300 episodes and 300 models are not the problem and
  increasing them further will not help.
