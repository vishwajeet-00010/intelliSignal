# IntelliSignal — Project Architecture

This document separates **what is implemented now** from **what is planned for the
later AI stages**. Nothing in the "future" section exists in the code today.

Stage 1 (the simulation and the fixed-time baseline) and Stage 2 (the reinforcement-
learning environment) are both implemented. The DQN agent is not.

---

## 1. Stage 1 architecture — implemented

```
        Traffic Demand
              |
              v
         SUMO Network
              |
              v
    Fixed-Time Traffic Signal
              |
              v
      Vehicle Simulation
              |
              v
            TraCI
              |
              v
      Metric Collection
              |
              v
         CSV Results
              |
              v
         Matplotlib
              |
              v
            Graphs
```

### What each block is, in this repository

| Block | Where it lives | What it does |
|---|---|---|
| Traffic Demand | `config/moderate.yaml`, `config/heavy.yaml` | Vehicles per hour on each approach, turning split, simulation length, signal timings, vehicle properties. Loaded by `utils/config_loader.py`, which also validates it. |
| SUMO Network | `sumo/network/intersection.*.xml` | A four-way intersection: four 200 m approaches, two lanes in each direction, one signalised junction `C`. Built from nodes / edges / connections by `netconvert` (`sumo/network/build_network.py`). |
| Fixed-Time Traffic Signal | `utils/scenario_builder.py` → `sumo/additional/<scenario>_tls.add.xml` | Writes the four-phase signal program from the YAML timings and activates it at time 0. |
| Vehicle Simulation | SUMO | Moves the vehicles, applies car-following and the signal. Nothing about vehicle behaviour is written by this project. |
| TraCI | `simulation/run_simulation.py` | Starts SUMO as a sub-process, advances it one step at a time, reads the state, and always closes it again. |
| Metric Collection | `utils/metrics.py` | Reads vehicles, waiting times, halting vehicles per lane and completed trips out of the live simulation. |
| CSV Results | `results/csv/` | `<scenario>_results.csv` (one summary row) and `<scenario>_timeseries.csv` (one row per simulated second). |
| Matplotlib / Graphs | `simulation/plot_results.py` → `results/plots/` | Reads only the CSV files. If a CSV is missing, its graph is not drawn. |

### How the four-way intersection is modelled

```
                          N
                          |
                    (N2C) | (C2N)
                          |
            (C2W)         |         (E2C)
    W ----------------- [ C ] ----------------- E
            (W2C)         |         (C2E)
                          |
                    (C2S) | (S2C)
                          |
                          S
```

* `C` is the signalised junction, at coordinate (0, 0).
* `N`, `S`, `E`, `W` are the network boundaries, 200 m away in each direction.
* Each approach has **two incoming lanes**:
  * lane 0 (kerb side) — right turn and straight ahead
  * lane 1 (centre side) — left turn
* The junction has **16 signal-controlled links**, four per approach, in the order
  North (0–3), East (4–7), South (8–11), West (12–15). A signal state is therefore a
  16-character string made of four 4-character blocks.

### The fixed-time signal program

```
  North-South green   (GGGg rrrr GGGg rrrr)     configurable, default 30 s
          |
          v
  North-South yellow  (yyyy rrrr yyyy rrrr)     configurable, default  4 s
          |
          v
  East-West green     (rrrr GGGg rrrr GGGg)     configurable, default 30 s
          |
          v
  East-West yellow    (rrrr yyyy rrrr yyyy)     configurable, default  4 s
          |
          +------------> repeat
```

`G` is a protected green, `g` is a permissive green (the left turn must give way to
oncoming traffic), `y` is yellow and `r` is red. An optional all-red clearance phase can
be switched on by setting `signal.all_red` in the YAML file.

The plan never changes during a run and never looks at the traffic. That is the point:
it is the **baseline**.

### Data flow of one run

```
1.  python simulation/run_simulation.py --scenario heavy
2.  config_loader      reads and validates config/heavy.yaml
3.  scenario_builder   writes  sumo/routes/heavy.rou.xml
                               sumo/additional/heavy_tls.add.xml
                               sumo/config/heavy.sumocfg
4.  run_simulation     starts SUMO through TraCI
5.  loop, once per simulated second:
        traci.simulationStep()
        MetricsCollector.collect_step()
6.  traci.close()      in a finally block, so SUMO is never left running
7.  results written to results/csv/heavy_results.csv
                        results/csv/heavy_timeseries.csv
8.  plot_results       redraws results/plots/*.png from the CSV files
```

---

## 2. Stage 2 architecture — implemented

The reinforcement-learning environment wraps the Stage 1 simulation. It replaces
exactly one block of the diagram above — the fixed-time traffic signal — with an
interface an agent can act through. Everything else is reused unchanged.

```
        SUMO  (the same network, the same two scenarios)
               |
             TraCI
               |
     SUMO-RL / Gymnasium layer        rl/environment.py
               |
   +-----------+-----------+
   |                       |
 Observation             Reward
 (rl/state.py)        (rl/reward.py)
   |                       ^
   v                       |
[ the agent ] <-- Stage 3, NOT BUILT
   |                       |
 Action                    |
 (rl/action.py)            |
   |                       |
   v                       |
Traffic Signal             |
   |                       |
   v                       |
 SUMO  --------------------+
   |
   v
Next Observation
```

| Block | Where it lives | What it does |
|---|---|---|
| Gymnasium environment | `rl/environment.py` | Owns the SUMO process and implements `reset()` / `step()` / `close()` |
| Observation (state) | `rl/state.py` | 10 numbers: queue and waiting time per approach, current phase, phase elapsed - all normalised into [0, 1] |
| Action | `rl/action.py` | `Discrete(2)`: 0 = keep the current green, 1 = switch. Enforces mandatory yellow, `min_green` and `max_green` |
| Reward | `rl/reward.py` | `-scale x (waiting + queue)`, averaged over the decision interval |
| Settings | `config/rl/environment.yaml` | Decision interval, min/max green, normalisation constants, reward weights |
| SUMO-RL adapter | `rl/sumo_rl_adapter.py` | Builds a real `sumo_rl.SumoEnvironment` on the same network, proving compatibility |

A full explanation is in [`rl_environment.md`](rl_environment.md).

---

## 3. Stage 3 architecture — implemented

```
                    IntelliSignal
                         |
          +--------------+--------------+
          |                             |
     Fixed-Time                     DQN Agent
     (rule-based)                   (learned)
          |                             |
          |                    +--------+--------+
          |                    |                 |
          |              Policy Network    Target Network
          |               10->64->64->2     (frozen copy,
          |                    |             refreshed every
          |                    |             500 gradient steps)
          |                    +--------+--------+
          |                             |
          |                       Replay Buffer
          |                      (50,000 transitions)
          |                             |
          +--------------+--------------+
                         |
                  Signal Control        rl/action.py - yellow clearance,
                         |              min_green, max_green
                       SUMO <-> TraCI
                         |
              +----------+----------+
              |                     |
          Observation            Reward
          (10 values)        (-congestion)
              |                     |
              +----------+----------+
                         |
                   Training / Evaluation
                         |
                 Fixed-Time vs DQN
                         |
                  Streamlit Dashboard
```

| Block | Where it lives | What it does |
|---|---|---|
| Q-network | `agents/dqn_network.py` | 10 inputs (from `env.observation_space.shape`) → two Q-values (from `env.action_space.n`). About 5,000 parameters. |
| Replay buffer | `agents/replay_buffer.py` | Ring buffer of `(state, action, reward, next_state, done)`; uniform random mini-batches break the correlation between consecutive traffic observations. |
| DQN agent | `agents/dqn_agent.py` | Epsilon-greedy action selection, the temporal-difference update, the target network, and self-describing checkpoints. |
| Hyperparameters | `config/dqn/training.yaml` | Every number the agent and the loop use, overridable on the command line and recorded in each checkpoint. |
| Training loop | `training/train_dqn.py` | Collects experience from `IntelliSignalEnv`, learns, saves checkpoints and a per-episode history CSV. |
| Episode runner | `simulation/agent_simulation.py` | Runs one episode under one controller. Both controllers get byte-identical SUMO inputs and the same metrics collector. |
| Evaluation | `evaluation/evaluate_dqn.py` | Greedy (epsilon = 0), no learning, several seeds, mean and standard deviation. |
| Comparison | `evaluation/compare_controllers.py` | Fixed-Time vs DQN on identical traffic, improvement percentages, raw and aggregated CSVs. |

### What Stage 3 did NOT change

This is the part worth stressing. Stage 3 is an addition on top of Stages 1
and 2, not a rewrite of them:

* `sumo/` — the road network, untouched.
* `config/moderate.yaml`, `config/heavy.yaml` — the two scenarios, untouched.
* `rl/state.py`, `rl/action.py`, `rl/reward.py` — the observation, the action
  meaning, the safety rules and the reward, all untouched. Changing the reward
  between the baseline and the learned controller would have made the
  comparison meaningless.
* `results/csv/` — the recorded fixed-time baseline, never written by any
  Stage 3 code. Two tests assert its bytes are unchanged.
* `simulation/run_simulation.py` — the Stage 1 command line, untouched.

Three small **additive** seams were opened, each backwards compatible and each
covered by a test:

| Change | Why |
|---|---|
| `utils/metrics.py` takes an optional `conn=` | Stage 1 uses the module-level `traci`; the RL environment uses a labelled connection. One collector now measures both, which is what makes the comparison fair. |
| `rl/environment.py` gained `step_callback`, a `connection` property and `extra_sumo_args` | Lets the Stage 1 collector measure an agent-driven run second by second, and quiets SUMO's end-of-run report during a 100-episode training run. |
| `rl/environment.episode_scenario_config()` lifted to module level | Both controllers build their SUMO inputs by calling the *same* function, so identical demand and seed are guaranteed rather than hoped for. |
| `TrafficState.observation()` added | A learned agent needs the per-approach observation; the NS/EW summary cannot be fed to it. The vector travels in `extra` and rule-based agents ignore it. |

### Why Stages 1 and 2 made this possible

* The network and both scenarios already existed and were reproducible.
* `utils/metrics.py` already produced exactly the numbers the comparison
  needs, independently of who controls the signal.
* The signal program is generated rather than hand-written, so replacing the
  fixed plan with an agent's decisions needed no change to the network.
* A measured fixed-time baseline existed. Without it, an adaptive controller
  could not be shown to be better or worse than anything.
* The environment already satisfied the Gymnasium API contract, so attaching a
  network was an addition, not a rewrite.

---

## 4. Future scope — NOT implemented

```
            FUTURE
              |
   PPO / A2C alternatives
              |
   CCTV / YOLO detection
              |
   Hardware in the loop
              |
   Multi-intersection coordination
```

None of these exist in the code. Stage 3 is one intersection, two actions, a
simulated world and a DQN. The limitations section of
[`dqn_training.md`](dqn_training.md) states exactly what the result does and
does not show.
