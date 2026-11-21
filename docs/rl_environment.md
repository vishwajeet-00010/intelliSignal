# IntelliSignal — The Reinforcement-Learning Environment (Stage 2)

> This is the environment reference. The DQN was added in Stage 3 and its
> training and evaluation are described in `docs/dqn_training.md`. The active
> timing and normalization values are read from the current YAML configuration;
> the older Stage 2 reports remain historical snapshots.

---

## 1. Why an RL environment is required

Stage 1 produced a fixed-time traffic signal: North-South gets 30 s of green,
East-West gets 30 s of green, forever, no matter what the traffic is doing.
The measured Stage 1 results show what that costs — average waiting time rises
from 10.04 s to 51.90 s per vehicle when demand roughly doubles.

To do better, the signal has to *decide*. Reinforcement learning is a way of
learning decisions from experience, but a learning algorithm cannot talk to
SUMO directly. It needs the problem presented in a specific shape:

| The agent needs | Which means |
|---|---|
| something to look at | an **observation** — numbers describing the traffic right now |
| something to do | an **action** — a choice that changes the signal |
| something to aim for | a **reward** — a number saying how good the result was |
| a way to take a turn | **`step()`** — apply the action, let time pass, report back |
| a way to start over | **`reset()`** — begin a fresh episode |

Packaging SUMO into exactly that shape is what Stage 2 does. Once the shape is
right, *any* RL algorithm can be attached without touching the traffic code.

---

## 2. What Gymnasium does

[Gymnasium](https://gymnasium.farama.org/) is the standard interface for RL
environments in Python — the successor to OpenAI Gym. It does not contain any
learning algorithm. It defines a contract:

```python
observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step(action)
env.close()
```

plus `env.observation_space` and `env.action_space`, which describe the shape
and the valid range of what goes in and out.

Following that contract is what makes the project future-proof. Stable-Baselines3,
which the Stage 3 DQN uses, accepts any environment that satisfies it.
`IntelliSignalEnv` is checked against the contract automatically by
`gymnasium.utils.env_checker.check_env` in the test suite.

## 3. What SUMO-RL does

[SUMO-RL](https://github.com/LucasAlegre/sumo-rl) is a library that wraps SUMO
as a Gymnasium / PettingZoo environment. It manages the SUMO process, the
decision interval, the yellow clearance, and supplies a default observation and
reward for one or many traffic lights.

**IntelliSignal implements its own environment rather than subclassing SUMO-RL,
and it is worth being clear about why.** SUMO-RL's action space for one traffic
light is `Discrete(number of green phases)`: the agent names the green phase it
wants next. The IntelliSignal specification calls for a two-action space —
*keep* or *switch* — together with a specific 10-number observation and a
specific reward. Building those on top of SUMO-RL would mean overriding almost
all of it, which is more code and harder to explain, not less.

So `rl/environment.py` talks to TraCI directly, but deliberately borrows
SUMO-RL's vocabulary and semantics:

| IntelliSignal | SUMO-RL | Meaning |
|---|---|---|
| `decision_interval` | `delta_time` | simulated seconds per agent decision |
| `yellow_time` | `yellow_time` | clearance inserted on a phase change |
| `min_green` | `min_green` | shortest allowed green |
| `max_green` | `max_green` | longest allowed green |

SUMO-RL is still a real part of the project, not a name in a list.
`rl/sumo_rl_adapter.py` builds a genuine `sumo_rl.SumoEnvironment` on **the same
network and route files** the Stage 1 baseline used, and the test suite runs it
(`test_sumo_rl_environment_runs_on_the_stage1_network`). That proves the
IntelliSignal intersection is a valid SUMO-RL environment, so if a later stage
wants SUMO-RL's multi-agent tooling or its built-in reward functions, the entry
point already exists and is tested.

---

## 4. Architecture

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
             [ the agent ]  <-- Stage 3 DQN |
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

One turn of that loop is one call to `env.step(action)`.

### Where each piece lives

| File | Responsibility |
|---|---|
| `rl/environment.py` | The Gymnasium `Env`: owns the SUMO process, runs the loop |
| `rl/state.py` | What the agent sees — builds and normalises the observation |
| `rl/action.py` | What the agent can do — turns an action into signal states |
| `rl/reward.py` | What the agent is rewarded for |
| `rl/config.py` | Loads the settings from `config/rl/environment.yaml` |
| `rl/sumo_rl_adapter.py` | SUMO-RL compatibility layer |
| `rl/run_environment.py` | Manual demonstration — no training |

Everything about the road, the demand and the vehicles is reused from Stage 1
unchanged: `utils/config_loader.py`, `utils/scenario_builder.py` and the
existing `sumo/network/intersection.net.xml`.

---

## 5. The observation (state)

`Box(low=0.0, high=1.0, shape=(10,), dtype=float32)`

| Index | Name | Meaning |
|---|---|---|
| 0 | `north_queue` | halting vehicles on the North approach |
| 1 | `east_queue` | halting vehicles on the East approach |
| 2 | `south_queue` | halting vehicles on the South approach |
| 3 | `west_queue` | halting vehicles on the West approach |
| 4 | `north_waiting` | summed current waiting time, North approach |
| 5 | `east_waiting` | summed current waiting time, East approach |
| 6 | `south_waiting` | summed current waiting time, South approach |
| 7 | `west_waiting` | summed current waiting time, West approach |
| 8 | `current_phase` | 0.0 = North-South green, 1.0 = East-West green |
| 9 | `phase_elapsed` | how long the current green has been running |

**Where the numbers come from.** Each approach is two incoming lanes
(`N2C_0`/`N2C_1` and so on). Queue is `traci.lane.getLastStepHaltingNumber()`
summed over the two lanes — "halting" is SUMO's own definition, a vehicle
moving slower than 0.1 m/s. Waiting is `traci.lane.getWaitingTime()` summed
over the two lanes, measured in vehicle-seconds.

**Why *current* waiting time and not *accumulated* waiting time.** Stage 1's
reporting metric uses accumulated waiting time, which only ever grows for a
given vehicle. That is the right thing for a final report but the wrong thing
for a state vector: it would drift upwards through the episode and tell the
agent more about how long the episode has run than about the traffic. Current
waiting time resets when a vehicle moves again, so it stays in a bounded range
and actually describes the situation at the junction.

**Normalisation.** Every entry is divided by a scale and clipped into
`[0, 1]`. The current scenario-specific queue and waiting scales were measured
from ten greedy incumbent rollouts on seeds 2001–2010 after the 5-second
decision cadence and action mask were introduced:

| Scenario | queue scale / approach | waiting scale / approach |
|---|---:|---:|
| moderate | 3 vehicles | 14 vehicle-seconds |
| heavy | 16 vehicles | 142 vehicle-seconds |

These are divisors, not hard limits. Values above the scale saturate at 1.0.
Because the scales differ by scenario, the trained checkpoint is scenario
specific; a single model should not be moved across scenarios without
recalibration and evaluation. The fallback values in
`config/rl/environment.yaml` apply only when a scenario has no override.

Standalone environment instances retain the legacy 20-second phase-age scale
for compatibility. New DQN training defaults to the full 60-second
`max_green` range and writes that scale into the checkpoint contract. A loaded
policy supplies its recorded scale to evaluation; changing the observation
scale of an existing model would change its inputs and requires retraining.

---

## 6. The action space

`Discrete(2)`

| Action | Name | Effect |
|---|---|---|
| `0` | KEEP | the direction that currently has green keeps it |
| `1` | SWITCH | the current green ends and the other direction gets green |

There are only two green phases at this intersection:

```
phase 0 : North-South green   state GGGgrrrrGGGgrrrr
phase 1 : East-West   green   state rrrrGGGgrrrrGGGg
```

`G` is a protected green, `g` a permissive green (the left turn gives way),
`y` yellow, `r` red — the same 16-character state strings the Stage 1
fixed-time controller uses, imported from `utils/scenario_builder.py`.

### Exactly what happens to the traffic light

Each decision covers `decision_interval` seconds. The controller turns the
action into a list of `(signal state, seconds)` segments that fill that
interval exactly:

```
action = 0 (KEEP)
    [ (current green, 5 s) ]

action = 1 (SWITCH), with yellow = 4 s and all-red = 0 s
    [ (current yellow, 4 s), (other green, 1 s) ]

action = 1 (SWITCH), with a hypothetical 7 s interval and 2 s all-red
    [ (current yellow, 4 s), (all red, 2 s), (other green, 1 s) ]
```

### Safety rules — a switch is never instantaneous and never rapid

1. **Yellow is mandatory.** A switch always passes through the yellow (and,
   if configured, all-red) state taken from the Stage 1 signal design. One
   direction's green can never become the other direction's green directly.
   This is enforced structurally: the switch plan is built from the yellow
   state first, and a test asserts no plan starts with a green.
2. **`min_green`** — a switch requested before the current green has run for
   `min_green` seconds is refused and treated as KEEP. The refusal is reported
   in `info["switch_blocked"]`; it is never hidden from the agent or the log.
3. **`max_green`** — a KEEP requested after the current green has already run
   for `max_green` seconds is overridden into a SWITCH, reported in
   `info["forced_switch"]`. This is the fairness guarantee: no approach can be
   held at red indefinitely.

Because `decision_interval` must be longer than `yellow + all_red` (validated
when the configuration loads), a switch always leaves some real green time in
the interval.

---

## 7. The reward

```
reward = -scale * (  waiting_time_weight * total_waiting_time
                   + queue_weight        * total_queue_length
                   + fairness_weight     * worst_approach_waiting )
```

With the defaults (`scale = 0.01`, both weights `1.0`, fairness `0.0`) this is
simply:

```
reward = -0.01 * (total waiting time + total queue length)
```

* The reward is always **negative or zero**. Congestion is a cost; the best the
  agent can do is push it towards zero. Maximising reward therefore means
  reducing congestion — which is the actual goal of the project.
* `scale` only keeps the numbers small (a raw sum runs into the hundreds). It
  multiplies every reward by the same positive constant, so it cannot change
  which action is best.
* The values are **averaged over every simulated second of the decision
  interval**, not read once at the end. A single snapshot is noisy — a queue
  can empty in the last second of a green — and a noisy reward makes learning
  harder. Averaging is nearly free and gives a far more stable signal.

`info["reward_components"]` reports the waiting, queue and fairness parts
separately, so the reward can always be taken apart and checked.

### Fairness

The default reward has no fairness term, because fairness is already guaranteed
*structurally* by `max_green`: after 60 seconds the environment forces a switch
whatever the agent asks for. Buying fairness with a hard rule rather than a
reward term keeps the reward simple enough to explain in one line, and it
cannot be traded away by a policy that finds a high-reward but unfair
behaviour.

A `fairness_weight` is nevertheless implemented and tested. Setting it above
zero adds a penalty proportional to the waiting time of the single worst
approach, which discourages ignoring a quiet direction even within the
`max_green` window. It is available for experiments in a later stage.

---

## 8. The decision interval

`decision_interval: 5` seconds (configurable in `config/rl/environment.yaml`).

**Why it exists.**

1. *Safety and realism.* SUMO steps once per simulated second. A signal allowed
   to change every step could flicker between directions — physically
   meaningless and dangerous.
2. *Learning difficulty.* At one decision per second an 1800 s episode is 1800
   decisions, most of which change nothing. At 5 s it is 360 decisions, each
   of which actually matters. Fewer, more meaningful decisions make the
   learning problem dramatically easier.
3. *Credit assignment.* The effect of a green phase on queues takes several
   seconds to appear. A decision interval long enough to contain that effect
   means the reward genuinely reflects the action that caused it.

The 5-second setting is an explicit trade-off: a decision that switches spends
4 seconds in yellow and only 1 second in the new green during that interval.
`min_green: 5` then prevents another early switch. This increases control
frequency and changes the reward horizon compared with the historical 10-second
setup; comparisons must use the same current environment. The configuration
loader enforces `decision_interval > yellow_time + all_red_time` and refuses to
start otherwise.

Between decisions SUMO keeps simulating normally, one second at a time, and the
environment samples the queues and waiting times on every one of those seconds
for the reward average.

---

## 9. How `reset()` works

```python
observation, info = env.reset(seed=42)
```

1. Any SUMO process still running is closed, so an old simulation is never left
   behind.
2. The Stage 1 scenario file is read and the SUMO inputs are generated with the
   Stage 1 builder — but under the name `<scenario>_rl`, so
   `sumo/routes/moderate_rl.rou.xml` is written and the Stage 1
   `moderate.rou.xml` is never touched.
3. SUMO starts with the given seed, so the same seed gives the same episode.
4. The signal is set to the North-South green and the phase timers are cleared.
5. The first observation is read and returned together with `info`.

## 10. How `step(action)` works

```python
observation, reward, terminated, truncated, info = env.step(action)
```

1. `ActionController.resolve()` decides what will *actually* happen — the
   requested action, unless `min_green` blocks a switch or `max_green` forces
   one.
2. `ActionController.plan()` turns that into `(signal state, seconds)` segments
   filling the decision interval.
3. For each segment the state is written to the traffic light with
   `traci.trafficlight.setRedYellowGreenState()`, then SUMO is stepped second by
   second. Queues and waiting times are sampled on every step.
   *(Setting the state this way moves the traffic light to SUMO's `online`
   program, where the state is held until it is changed again. The test suite
   asserts this rather than assuming it.)*
4. The samples are averaged and passed to the reward function.
5. The new observation is built from the final sample plus the phase and its
   elapsed time.
6. `terminated` is `True` only if SUMO reports no vehicles left and none
   expected. `truncated` is `True` once the episode has used its full length
   (1800 s by default — the same length as the Stage 1 runs).
7. `info` is assembled and everything is returned.

### What `info` contains

| Key | Meaning |
|---|---|
| `scenario` | `"moderate"` or `"heavy"` |
| `simulation_time` | SUMO's current simulated time, in seconds |
| `decision` | how many decisions have been taken this episode |
| `phase`, `phase_name` | current green phase, as an index and as text |
| `green_elapsed` | seconds the current green has been running |
| `queue_per_approach` | halting vehicles, per approach |
| `waiting_per_approach` | current waiting time, per approach |
| `total_queue`, `total_waiting_time` | the same, summed |
| `vehicles_in_network` | vehicles currently being simulated |
| `switch_count` | phase switches so far this episode |
| `requested_action` / `effective_action` | what was asked for, what was done |
| `switch_blocked` / `forced_switch` | why they differ, when they do |
| `reward_components` | the waiting, queue and fairness parts of the reward |
| `interval_mean_queue` / `interval_mean_waiting` | the averages the reward used |
| `episode_return`, `episode_decisions` | added on the final step only |

---

## 11. How the environment connects to SUMO

```
IntelliSignalEnv
      |  traci.start([...], label="intellisignal_rl_<scenario>_<id>")
      |  traci.getConnection(label)  ->  a private connection object
      v
  SUMO process (one per environment instance)
```

Every environment instance owns a **labelled TraCI connection**, so several
environments can exist in the same Python process without interfering — which
is what lets the test suite create and destroy environments freely, and what
will allow parallel training later.

`close()` shuts the connection down and is safe to call twice. `reset()` calls
it first, and it is also called from `__del__`, so a forgotten `close()` does
not leave a SUMO process running.

The SUMO inputs are generated by the **Stage 1** builder from the **Stage 1**
scenario files. The RL environment adds nothing to the road and changes nothing
about the demand — it only takes over the traffic light.

---

## 12. How the DQN connects to this environment (Stage 3 — built)

Nothing in `rl/` had to change. Stage 3 reads this environment through the
public API it already offered:

```python
from agents.dqn_agent import build_agent_for_env
from rl import IntelliSignalEnv

env = IntelliSignalEnv(scenario="moderate")

# The network is SIZED FROM THE ENVIRONMENT: 10 inputs from
# env.observation_space.shape, 2 outputs from env.action_space.n.
agent = build_agent_for_env(env)

observation, info = env.reset(seed=42)
action = agent.act(observation, training=True)         # 0 = keep, 1 = switch
next_observation, reward, terminated, truncated, info = env.step(action)
agent.remember(observation, action, reward, next_observation, terminated)
agent.learn()                                          # one gradient step
```

Stable-Baselines3 is deliberately not used: the DQN, the replay buffer and the
target network are written out in full in `agents/`, because a 5,000-parameter
network with two actions does not need a framework and the explicit version is
what can be explained line by line in a report.

### The three additive seams Stage 3 opened here

These integration seams are optional and preserve the original environment
contract. Later releases separately changed timing and observation scales;
those active settings are described above and are not historical Stage 2
defaults.

| Addition | Purpose |
|---|---|
| `env.step_callback` | Called after every single simulated second. The Stage 3 evaluation attaches the **Stage 1** `MetricsCollector` to it, so a DQN run is measured by exactly the code that measured the fixed-time baseline. |
| `env.connection` | Exposes the live TraCI connection so that collector can read the same running simulation. |
| `env.extra_sumo_args` | Extra SUMO command-line options, used only to silence SUMO's end-of-run report during a 100-episode training run. |
| `episode_scenario_config()` | The scenario-to-episode transformation, lifted out of `_start_sumo` to module level so the fixed-time comparison run can build byte-identical SUMO inputs. |

### What the agent is still not allowed to do

The DQN returns 0 or 1. The training policy masks switches before `min_green`,
and `rl/action.py` remains the final safety guard: a switch always runs the
yellow clearance, an early switch request is refused and recorded, and a
`keep` after `max_green` is overridden into a switch. A Stage 3 test drives the network with pure
exploration for 300 simulated seconds and asserts that no green → green
transition appears anywhere in the recorded signal states.

Full details: [`dqn_training.md`](dqn_training.md).

---

## 13. Trying it out

```bash
# manual demonstration, no training - the actions come from a fixed rule
python rl/run_environment.py --scenario moderate
python rl/run_environment.py --scenario heavy --policy alternate --steps 12
python rl/run_environment.py --scenario heavy --gui        # watch it in sumo-gui

# the same environment, driven by the trained DQN instead (Stage 3)
python evaluation/evaluate_dqn.py --scenario moderate --seeds 42 43 44

# the tests
python -m pytest tests/test_rl_environment.py -q
python -m pytest -q                                        # every stage
```
