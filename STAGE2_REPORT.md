# IntelliSignal — Stage 2 Final Report

Reinforcement-learning environment foundation.
Only verified facts appear in this report.

> **No DQN has been implemented and no agent has been trained.**
> There is no neural network, no replay buffer, no target network and no
> training loop anywhere in this repository. Stage 2 delivers the environment
> that a DQN will plug into; the DQN is Stage 3.

---

## 1. Objective

Turn the Stage 1 SUMO simulation into a Gymnasium-compatible reinforcement-learning
environment:

```
SUMO -> TraCI -> SUMO-RL / Gymnasium -> State -> Action -> Reward
     -> reset() / step() -> an RL-ready interface
```

and validate it, without training anything.

---

## 2. Existing Stage 1 foundation (reused, not rebuilt)

Stage 1 was inspected before any change was made. Nothing in it was rewritten.
The RL environment reuses it directly:

| Stage 1 component | How Stage 2 uses it |
|---|---|
| `sumo/network/intersection.net.xml` | The RL environment drives this exact network. No second network was created. |
| `config/moderate.yaml`, `config/heavy.yaml` | Read unmodified by `utils.config_loader.load_config()`. Both files are byte-for-byte unchanged. |
| `utils/scenario_builder.py` | Generates the RL environment's SUMO inputs. The 16-character signal state strings (`NS_GREEN`, `NS_YELLOW`, `EW_GREEN`, `EW_YELLOW`, `ALL_RED`) are imported, so the RL signal uses the same phase definitions as the fixed-time controller. |
| `signal.yellow` / `signal.all_red` | Inherited as the RL environment's clearance timings, so both controllers clear the junction identically. |
| `simulation/run_simulation.py` | `prepare_sumo_imports()` and `sumo_binary()` are reused to locate SUMO. |
| `utils/metrics.py` | Untouched. It remains the Stage 1 reporting metric and will provide the Fixed-Time vs DQN comparison in a later stage. |

---

## 3. Changes made

### Files created

| File | Purpose |
|---|---|
| `rl/__init__.py` | Package exports; registers `IntelliSignal-v0` with Gymnasium |
| `rl/environment.py` | The Gymnasium environment (`IntelliSignalEnv`) |
| `rl/state.py` | The 10-number observation and its normalisation |
| `rl/action.py` | Keep / switch actions and the signal safety rules |
| `rl/reward.py` | The congestion-penalty reward |
| `rl/config.py` | Loads and validates the RL settings |
| `rl/sumo_rl_adapter.py` | SUMO-RL compatibility layer |
| `rl/run_environment.py` | Manual demonstration script (no training) |
| `config/rl/environment.yaml` | All RL settings, in one YAML file |
| `tests/test_rl_environment.py` | 47 Stage 2 tests |
| `docs/rl_environment.md` | Full explanation of the environment |
| `STAGE2_REPORT.md` | This file |

### Files modified

| File | Change |
|---|---|
| `README.md` | Stage 2 sections added: status, technology, architecture, project structure, run commands, tests, limitations, next stage |
| `docs/architecture.md` | The SUMO-RL / Gymnasium layer moved from "future" to "implemented"; the DQN layer remains marked as not built |
| `requirements.txt` | `gymnasium>=1.0` and `sumo-rl>=1.4` added, with a note that Stable-Baselines3 and PyTorch are deliberately **not** installed |

### Files NOT modified

`config/moderate.yaml`, `config/heavy.yaml`, every file in `sumo/network/`,
`utils/config_loader.py`, `utils/scenario_builder.py`, `utils/metrics.py`,
`simulation/run_simulation.py`, `simulation/run_scenario.py`,
`simulation/plot_results.py`, `tests/test_simulation.py`, and everything in
`results/` and `presentation/`.

### Why the RL settings live in `config/rl/environment.yaml`

`available_scenarios()` lists every `.yaml` file directly inside `config/`. A file
named `config/rl.yaml` would therefore have appeared as a third traffic scenario
called "rl" and broken `python simulation/run_scenario.py all`. Putting it in the
`config/rl/` subdirectory keeps "moderate" and "heavy" as the only two scenarios,
and a test asserts exactly that (`test_rl_config_file_does_not_add_a_scenario`).

---

## 4. RL environment architecture

```
                    +-------------------------------+
                    |            SUMO               |
                    |  4-way intersection, vehicles |
                    +---------------+---------------+
                                    |
                                  TraCI
                                    |
                    +---------------v---------------+
                    |   SUMO-RL / Gymnasium layer   |
                    |      rl/environment.py        |
                    +---------------+---------------+
                                    |
                  +-----------------+-----------------+
                  |                                   |
        Observation / State                        Reward
        (rl/state.py)                            (rl/reward.py)
                  |                                   ^
                  v                                   |
             [ the agent ]  <-- STAGE 3, NOT BUILT    |
                  |                                   |
                Action                                |
            (rl/action.py)                            |
                  |                                   |
                  v                                   |
           Traffic Signal                             |
                  |                                   |
                  v                                   |
                SUMO  ----------------------------->--+
                  |
                  v
          Next Observation
```

Every environment instance owns its own labelled TraCI connection
(`traci.start(..., label=...)` + `traci.getConnection(label)`), so several
environments can live in one Python process without interfering.

---

## 5. State representation

`observation_space = Box(low=0.0, high=1.0, shape=(10,), dtype=float32)`

| Index | Name | Source |
|---|---|---|
| 0–3 | `north_queue`, `east_queue`, `south_queue`, `west_queue` | `traci.lane.getLastStepHaltingNumber()` summed over each approach's 2 lanes |
| 4–7 | `north_waiting`, `east_waiting`, `south_waiting`, `west_waiting` | `traci.lane.getWaitingTime()` summed over each approach's 2 lanes, in vehicle-seconds |
| 8 | `current_phase` | 0.0 = North-South green, 1.0 = East-West green |
| 9 | `phase_elapsed` | seconds the current green has run, divided by `max_green` |

Every entry is divided by a constant and clipped into `[0, 1]`, so the
observation can never leave the declared space.

**The normalisation constants were measured, not guessed.** Both scenarios were
run for a full 1800-second episode under a random policy and under a keep-only
policy, recording the per-approach values at every decision:

| Scenario / policy | max queue / approach | p99 queue | max waiting / approach | p99 waiting |
|---|---|---|---|---|
| moderate, random | 8 | 6.8 | 319 | 189 |
| moderate, keep-only | 9 | 8.0 | 299 | 264 |
| heavy, random | **50** | 50.0 | **1848** | 1392 |
| heavy, keep-only | 40 | 38.8 | 1715 | 1629 |

Chosen: `max_queue_per_approach: 50`, `max_waiting_per_approach: 1500`.
50 vehicles is close to the physical capacity of one approach (2 lanes × 200 m).
The same constants are used for both scenarios so that one trained agent can
later be evaluated on both.

**Current waiting time is used, not accumulated waiting time.** Accumulated
waiting time only grows, so it would drift upwards through an episode and tell
the agent more about elapsed time than about traffic. Current waiting time
resets when a vehicle moves, stays bounded, and describes the situation now.
Stage 1's reporting metric still uses accumulated waiting time and is unchanged.

---

## 6. Action representation

`action_space = Discrete(2)`

| Action | Name | Effect on the traffic light |
|---|---|---|
| `0` | KEEP | The current green is held for the whole decision interval: `[(current green, 10 s)]` |
| `1` | SWITCH | Yellow first, then the other direction's green: `[(current yellow, 4 s), (other green, 6 s)]` |

With `all_red` set above zero the switch plan becomes
`[(yellow, 4 s), (all red, n s), (other green, 6 − n s)]`.
The segments always add up to exactly one decision interval.

### Safety rules (all three are tested)

1. **Yellow is mandatory.** A switch always passes through the yellow state
   taken from the Stage 1 signal design. One direction's green can never become
   the other's green directly — the test `test_green_never_turns_directly_into_the_other_green`
   asserts that no switch plan begins with a green state.
2. **`min_green` = 10 s.** A switch requested sooner is refused and executed as
   KEEP, reported in `info["switch_blocked"]`.
3. **`max_green` = 60 s.** A KEEP requested after 60 s of green is overridden
   into a SWITCH, reported in `info["forced_switch"]`. This is the fairness
   guarantee: no approach can be held at red indefinitely.

Both refusals and overrides are reported in `info` rather than hidden.

---

## 7. Reward design

```
reward = -scale * (  waiting_time_weight * total_waiting_time
                   + queue_weight        * total_queue_length
                   + fairness_weight     * worst_approach_waiting )
```

Defaults: `scale = 0.01`, `waiting_time_weight = 1.0`, `queue_weight = 1.0`,
`fairness_weight = 0.0` — that is, simply

```
reward = -0.01 * (total waiting time + total queue length)
```

* Always negative or zero; zero is an empty intersection. Maximising the reward
  means reducing congestion.
* `scale` only keeps the numbers small; it multiplies every reward by the same
  positive constant and cannot change which action is best.
* The queue and waiting values are **averaged over every simulated second of
  the decision interval**, not sampled once at the end. A snapshot is noisy —
  a queue can empty in the last second of a green — and a noisy reward makes
  learning harder.
* `info["reward_components"]` reports the waiting, queue and fairness parts
  separately, so any reward can be taken apart and checked. A test asserts the
  three components sum to the reward.

**Fairness.** The default reward has no fairness term because fairness is
already guaranteed structurally by `max_green`. A hard rule cannot be traded
away by a policy that finds a high-reward but unfair behaviour, and it keeps
the reward explainable in one line. A `fairness_weight` is implemented and
tested and can be switched on for experiments later.

---

## 8. Decision interval

`decision_interval: 10` seconds, configurable in `config/rl/environment.yaml`.

Reasons it exists:

1. **Safety and realism.** SUMO steps once per simulated second. A signal free
   to change every step could flicker between directions.
2. **Learning difficulty.** At one decision per second an 1800 s episode is 1800
   decisions, most of which change nothing; at 10 s it is 180 meaningful ones.
3. **Credit assignment.** The effect of a green on queues takes several seconds
   to appear, so the interval has to be long enough to contain it.

**Why 10 s and not 5 s.** The Stage 1 signal uses a 4-second yellow, which the
RL environment inherits. With a 5-second interval a switch would spend 4 s on
yellow and leave 1 s of green — almost no vehicles would get through. At 10 s a
switch still leaves 6 s of real green. `rl/config.py` enforces
`decision_interval > yellow_time + all_red_time` and refuses to start otherwise
(tested by `test_decision_interval_must_exceed_the_clearance_time`).

Between decisions SUMO keeps simulating normally, one second at a time.

---

## 9. Gymnasium interface

Modern Gymnasium API, verified by the official checker:

```python
observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step(action)
env.close()
```

* `terminated` — `True` only when SUMO reports no vehicles left and none
  expected.
* `truncated` — `True` once the episode has used its full length (1800 s by
  default, the same length as the Stage 1 runs).
* `info` — `scenario`, `simulation_time`, `decision`, `phase`, `phase_name`,
  `green_elapsed`, `queue_per_approach`, `waiting_per_approach`, `total_queue`,
  `total_waiting_time`, `vehicles_in_network`, `switch_count`,
  `requested_action`, `effective_action`, `switch_blocked`, `forced_switch`,
  `reward_components`, `interval_mean_queue`, `interval_mean_waiting`, and on
  the final step `episode_return` and `episode_decisions`.

`gymnasium.utils.env_checker.check_env()` passes with **no warnings**. That is
the same contract Stable-Baselines3 requires, so Stage 3 will not need changes
in `rl/`.

The environment is also registered as `IntelliSignal-v0`, so
`gym.make("IntelliSignal-v0", scenario="moderate")` works.

---

## 10. SUMO-RL role

SUMO-RL is installed (version 1.4.5) and is genuinely exercised, but
`IntelliSignalEnv` is not a subclass of it. The reason is concrete: SUMO-RL's
single-light action space is `Discrete(number of green phases)` — "name the
phase you want next" — while this project specifies a two-action keep/switch
space together with a specific 10-number observation and a specific reward.
Expressing those through SUMO-RL would mean overriding almost all of it.

What is done instead:

* `rl/environment.py` uses SUMO-RL's timing vocabulary and semantics.
  `decision_interval` is SUMO-RL's `delta_time`; `yellow_time`, `min_green` and
  `max_green` behave identically. The two are conceptually interchangeable.
* `rl/sumo_rl_adapter.py` builds a real `sumo_rl.SumoEnvironment` on **the same
  network and route files** the Stage 1 baseline used, with the same timing
  settings from the same YAML file.
* The test `test_sumo_rl_environment_runs_on_the_stage1_network` creates that
  environment, resets it, steps it and checks the reward is finite. **It passes.**
  This verifies the IntelliSignal intersection is a valid SUMO-RL environment,
  so SUMO-RL's multi-agent tooling remains available to a later stage.

`rl/sumo_rl_adapter.ensure_sumo_home()` also handles a practical detail:
`sumo_rl` raises `ImportError` at import time unless `SUMO_HOME` is set, which
is not guaranteed when SUMO came from the `eclipse-sumo` pip package. The helper
derives it from the installed package when it is missing.

---

## 11. Test results

Command: `python -m pytest -q`

```
71 passed in 54.91s
```

| Suite | Tests | Result |
|---|---|---|
| `tests/test_simulation.py` (Stage 1) | 24 | **24 passed** — unchanged |
| `tests/test_rl_environment.py` (Stage 2) | 47 | **47 passed** |
| **Total** | **71** | **71 passed, 0 failed** |

Stage 1 alone was also run separately: `24 passed in 6.62s`.

What the 47 Stage 2 tests cover:

| Area | Checks |
|---|---|
| RL configuration | loads for both scenarios; inherits the Stage 1 yellow and all-red; rejects an interval shorter than the clearance; does not add a third scenario |
| Action controller (no SUMO needed) | KEEP fills the interval; SWITCH inserts yellow then the other green; all-red is inserted when configured; a green never follows a green directly; `min_green` refuses a switch; `max_green` forces one; `commit()` alternates the phase and resets the timer; an impossible interval is rejected |
| Reward (no SUMO needed) | negative under congestion; zero when empty; less congestion scores higher; the components sum to the reward |
| Spaces | observation is `Box(0,1,(10,),float32)`; action space contains 0 and 1 and rejects 2; four approaches of two lanes each |
| `reset()` | returns `(observation, info)`; observation is inside the space; starts on North-South green with the right signal state; can be called twice; the same seed reproduces the same rewards |
| `step()` | returns the 5-tuple; KEEP and SWITCH both work; SWITCH changes the phase; reward is finite and ≤ 0; `terminated`/`truncated` are booleans and never both true; `info` carries every documented key; simulation time advances by exactly one decision interval; 10 random steps run; the episode truncates at the configured length; an invalid action raises; `step()` before `reset()` raises; the signal state persists across steps (SUMO's `online` program) |
| Shutdown | closes cleanly; `close()` is safe twice; the environment can be reset and reused after closing |
| Gymnasium compliance | `gymnasium.utils.env_checker.check_env` passes; `gym.make("IntelliSignal-v0")` works |
| SUMO-RL | importable; a real `SumoEnvironment` resets and steps on the Stage 1 network |
| Stage 1 protection | the RL environment modifies no Stage 1 SUMO file; the RL environment modifies no Stage 1 result file |

---

## 12. Manual environment demonstration

`python rl/run_environment.py --scenario heavy --policy alternate --steps 12 --seed 42`

```
  Observation space : Box(0.0, 1.0, (10,), float32)
  Action space      : Discrete(2)   0 = keep phase, 1 = switch phase
  Decision interval : 10 s per step
  Min / max green   : 10 s / 60 s
  Yellow / all-red  : 4 s / 0 s
  Reward            : reward = -0.01 x (1.0 x waiting + 1.0 x queue + 0.0 x worst_approach)

Step  Action   Applied   Sim time  Phase     Reward   Queue    Waiting  Queue N/E/S/W
   1    KEEP      KEEP       10s      0     -0.000       0        0.0  0/0/0/0
   2    KEEP      KEEP       20s      0     -0.031       5       11.0  0/2/0/3
   3  SWITCH    SWITCH       30s      1     -0.160       2        7.0  1/0/1/0
   4    KEEP      KEEP       40s      1     -0.296       7       50.0  4/0/3/0
   5    KEEP      KEEP       50s      1     -1.125      12      150.0  6/0/6/0
   6  SWITCH    SWITCH       60s      0     -1.017       2        2.0  0/1/0/1
   7    KEEP      KEEP       70s      0     -0.330       8       47.0  1/3/1/3
   8    KEEP      KEEP       80s      0     -0.906      11      124.0  0/5/1/5
   9  SWITCH    SWITCH       90s      1     -0.892       6       33.0  3/0/3/0
  10    KEEP      KEEP      100s      1     -0.823       9      111.0  4/0/5/0
  11    KEEP      KEEP      110s      1     -1.946      18      250.0  8/1/9/0
  12  SWITCH    SWITCH      120s      0     -1.736       5       14.0  2/2/1/0

  Total reward       : -9.262
  Mean reward / step : -0.772
  Phase switches     : 4
  Vehicles in network: 45
```

The same command on the moderate scenario:

```
  Total reward       : -3.727
  Mean reward / step : -0.311
  Phase switches     : 4
  Vehicles in network: 19
```

**Sanity check.** The mean reward per step is −0.311 for moderate traffic and
−0.772 for heavy traffic — about 2.5× worse. That is the same direction and
roughly the same magnitude of difference the Stage 1 metrics showed
(waiting 10.04 s → 51.90 s, queue 3.70 → 40.39). The reward signal therefore
responds to congestion the way it is supposed to.

A separate run with `--policy random` was also executed on both scenarios and
behaved correctly, including `min_green` refusing three switch requests, each
one reported in the output.

**No agent was trained in any of these runs.** The actions come from a fixed
rule chosen on the command line.

---

## 13. Verification that Stage 1 is intact

| Check | Result |
|---|---|
| `config/moderate.yaml`, `config/heavy.yaml` | MD5 unchanged |
| `results/csv/moderate_results.csv` | MD5 unchanged |
| `results/csv/heavy_results.csv` | MD5 unchanged |
| `results/csv/*_timeseries.csv` | MD5 unchanged |
| `results/plots/*.png` | **pixel-identical** (see note below) |
| `python simulation/run_simulation.py --scenario moderate` | ran, reproduced 702 / 10.04 / 3.70 / 693 exactly |
| `python simulation/run_simulation.py --scenario heavy` | ran, reproduced 1445 / 51.90 / 40.39 / 1351 exactly |
| `tests/test_simulation.py` | 24 passed |

Measured Stage 1 results after all Stage 2 work, read from the CSV files:

```
moderate,702,10.04,3.7,693,1800.0,10,702,9,0,moderate.yaml
heavy,1445,51.9,40.39,1351,1800.0,72,1445,94,0,heavy.yaml
```

Identical to the Stage 1 report.

**Note on the plot files.** Their MD5 byte hashes do change, because Stage 1's
own test `test_plots_are_generated_from_the_csv_files` regenerates them and
matplotlib stamps a creation date into the PNG metadata. The *pixels* were
compared directly and are identical across regenerations, so the graphs
themselves are unchanged. This behaviour existed in Stage 1 and is not caused
by Stage 2.

**File separation.** The RL environment writes its SUMO inputs under a separate
name, so the Stage 1 generated files are never overwritten:

```
sumo/routes/     moderate.rou.xml   moderate_rl.rou.xml   moderate_sumorl.rou.xml
                 heavy.rou.xml      heavy_rl.rou.xml
sumo/config/     moderate.sumocfg   moderate_rl.sumocfg   moderate_sumorl.sumocfg
                 heavy.sumocfg      heavy_rl.sumocfg
sumo/additional/ moderate_tls.add.xml  moderate_rl_tls.add.xml  moderate_sumorl_tls.add.xml
                 heavy_tls.add.xml     heavy_rl_tls.add.xml
```

All of these are regenerated on every run and are covered by the existing
`.gitignore` rules.

---

## 14. Environment status

Verified in the workspace where Stage 2 was built and executed:

| Component | Version |
|---|---|
| Python | 3.10.12 |
| SUMO | 1.27.1 |
| TraCI / sumolib | 1.27.1 |
| gymnasium | 1.3.0 |
| sumo-rl | 1.4.5 |
| pettingzoo (sumo-rl dependency) | 1.27.0 |
| NumPy | 2.2.6 |
| Pandas | 2.3.3 |
| Matplotlib | 3.10.9 |
| PyYAML | 6.0.3 |
| pytest | 9.1.1 |
| Stable-Baselines3 / PyTorch | **not installed, on purpose** |

**Check this on your Windows machine before relying on it.** Stage 2 was built
and executed in the Linux workspace attached to this session, not in your
Windows Python 3.14 installation. Install the new dependencies there and verify:

```
pip install -r requirements.txt
python -c "import gymnasium; print(gymnasium.__version__)"
python -c "import sumo_rl; print(sumo_rl.__version__)"
python -m pytest -q
python rl/run_environment.py --scenario moderate
```

Two things worth knowing for that check:

* `sumo_rl` raises `ImportError: Please declare the environment variable 'SUMO_HOME'`
  at import time if `SUMO_HOME` is not set. Stage 1 did not need it (it uses
  `sumolib.checkBinary`), so this may be the first time it matters.
  `rl/sumo_rl_adapter.ensure_sumo_home()` sets it automatically from the
  installed `eclipse-sumo` package when it is missing, but setting it properly
  on Windows is cleaner.
* The declared Python support is `gymnasium >= 3.10` and `pettingzoo >= 3.10, < 3.15`,
  so Python 3.14 is inside both ranges. Whether pre-built wheels exist for
  3.14 was not verified here — if `pip install` tries to build from source,
  that is the reason.

---

## 15. Known limitations

- **No agent exists.** Stage 2 delivers the environment only. The environment
  has never been driven by a learned policy, because no learned policy exists.
- The only controller run end to end over the full 1800-second scenarios is the
  Stage 1 fixed-time signal. The RL environment has been run for shorter
  episodes (60–1800 s) under fixed rules and random actions.
- The environment controls a single traffic light at a single intersection;
  it is single-agent only.
- The reward weights and the normalisation constants are reasonable, measured
  defaults, not tuned values. Tuning belongs to the training stage.
- `min_green = 10 s` combined with `decision_interval = 10 s` means a switch
  cannot be followed by another switch at the very next decision. This is
  intended, and it is visible in the demonstration output.
- Observation normalisation uses one shared scale for both scenarios, so
  moderate-traffic waiting values occupy only the lower part of the range.
  This is deliberate — a shared scale is what allows one agent to be evaluated
  on both scenarios — but it is a real property of the representation.
- Stage 2 has not been executed on the Windows Python 3.14 environment.

---

## 16. Next development stage (Stage 3 — not started)

Will not be started without explicit instruction.

```
Done (Stage 1):  SUMO -> Fixed-Time Controller -> Metrics -> CSV -> Graphs
Done (Stage 2):  SUMO -> TraCI -> Gymnasium/SUMO-RL -> State/Action/Reward -> reset()/step()

Next (Stage 3):  DQN agent -> Training -> Evaluation -> Fixed-Time vs DQN -> Dashboard
```

Planned work:

1. Install `stable-baselines3` and `torch`.
2. Attach a DQN to `IntelliSignalEnv`: 10 inputs, 2 Q-value outputs.
3. Train with a replay buffer and a target network.
4. Evaluate the trained policy on the same network and the same two scenarios.
5. Compare Fixed-Time vs DQN using the Stage 1 metrics from `utils/metrics.py`
   — average waiting time, average queue length, throughput.

Nothing in `rl/` has to change for any of that, which was the purpose of
Stage 2.
