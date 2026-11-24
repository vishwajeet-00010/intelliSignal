# IntelliSignal

**AI-Based Adaptive Traffic Signal Management Using SUMO and Deep Reinforcement Learning**

> **Current status (22 Nov 2026): the DQN beats the best fixed-time plan on
> both scenarios.** Measured over confirmatory seeds 6101–6120, 1800 s per
> run, with the project's own `MetricsCollector`:
>
> | scenario | best fixed plan | duration DQN | change |
> |---|---|---|---|
> | heavy | 5.754 s (ns 7 / ew 6) | **5.151 s** | **−10.5 %** |
> | moderate | 2.761 s (ns 6 / ew 4) | **2.237 s** | **−19.0 %** |
>
> Throughput is higher in both cases (1468.5 against 1466.5; 691.8 against
> 690.8), so the lower waiting is not bought by stranding vehicles. Heavy
> wins on 17 of 20 paired seeds, moderate on 20 of 20.
>
> This needed a second action space — **one decision per green, choosing that
> green's duration** — plus a semi-Markov correction to the learning. See
> [`docs/phase_duration_control.md`](docs/phase_duration_control.md) and
> [`output/dqn_phase_20261122/FINDINGS.md`](output/dqn_phase_20261122/FINDINGS.md).
>
> **Two earlier claims in this README are corrected by that work.** The
> fixed-time baselines it had been compared against were not the best fixed
> plans: demand is asymmetric, and so is the optimum. Against the properly
> tuned plan, the deployed Stage 4 models *lose* — heavy 16.469 s against
> 5.754 s, moderate 2.918 s against 2.761 s. Section 11b below is historical
> Stage 3 verification against a 30/30 plan and should be read as such.
>
> **24 Nov 2026 — operational timing profiles.** Greens are now bounded by
> a selectable **timing profile** enforced everywhere a controller runs:
> `responsive` (4–20 s, what the earlier models were trained with) and
> `standard` (15–30 s, conventional practice). Under `standard`, on an
> untouched test split of six varied workloads, the new DQN reaches
> **15.89 s total delay per requested vehicle against the tuned fixed
> plan's 22.51 s (−29.4 %), winning on 6 of 6 workloads** while completing
> more vehicles. The repeated four-second greens were traced to
> **observation saturation**, not to timing: on custom traffic the inputs
> pinned at their clipping limit in up to 93 % of decisions. See
> [`output/audit_20261123/FINDINGS.md`](output/audit_20261123/FINDINGS.md).
>
> The 15–30 s profile costs about **7.4×** on heavy traffic compared with
> unconstrained short cycles. That is the price of the operating policy,
> and the two profiles are never compared with each other.
>
> **23 Nov 2026 — built for the demonstration.** Three additions:
> **Custom Traffic** (describe demand lane by lane, see the exact preview,
> run precisely that), a **Hybrid** controller that gives the DQN a fixed
> amount of wall-clock time to prove itself and falls back to the best
> measured fixed plan if it does not, and a simplified dashboard. See
> [`docs/demo_walkthrough.md`](docs/demo_walkthrough.md) for a ten-minute
> run-through and [`docs/custom_traffic_and_hybrid.md`](docs/custom_traffic_and_hybrid.md)
> for how they work.
>
> The Stage 4 incumbents remain registered in `models/active_models.json` and
> still drive the Moderate and Heavy pages. The new models are registered separately under
> `phase_duration_models`; dashboard integration is not done.

The existing Windows environment is `.venv-stage3` (Python 3.12.13, installed
SUMO/TraCI/sumolib packages 1.28.0, PyTorch 2.14.1). Comparison CSVs are in
[`verification/windows_20261116/`](verification/windows_20261116/), with
fixed-time, trained DQN, and untrained controls evaluated at seeds 42, 43, and 44
for 1800 seconds. Waiting time decreased by **42.3% (moderate)** and **8.9%
(heavy)**; throughput decreased by **0.3%** and **0.6%**, respectively.
Use [`requirements-win-py312.lock`](requirements-win-py312.lock) to recreate
that exact Windows Python environment. `requirements.txt` remains the portable
minimum-version list.
The 16 Nov comparisons below are historical Stage 3 verification. For the
latest measured DQN results and current project status, see
[`HANDOFF.md`](HANDOFF.md) and the Stage 4 experiment directory.

---

## 1. Problem

Traditional traffic signals at road intersections are **fixed-time** controllers: each
direction receives a green phase of a pre-defined length, and that length does not
change no matter how many vehicles are actually waiting. When the traffic demand
changes — peak hours, one direction busier than another, an unexpected surge — a fixed
schedule keeps giving green time to an empty approach while a full approach waits.
The result is unnecessary waiting time, longer queues and lower throughput.

## 2. Proposed concept

Treat the traffic signal as a decision-making agent. At every decision point the agent
observes the current traffic state (queue lengths, waiting times, current phase) and
chooses which phase to run next. A **Deep Q-Network (DQN)** learns this decision policy
from experience by maximising a reward that penalises waiting time and queues.

The adaptive controller can only be judged against something, so the project first
builds a **fixed-time baseline** in a realistic simulator. That baseline is Stage 1.

## 3. Current stages

**Stage 1** implements the initial SUMO traffic simulation environment, a four-way
intersection, a fixed-time traffic-signal controller, two traffic-demand scenarios,
traffic-metric collection through TraCI, and the initial result visualisations.

**Stage 2** wraps that same simulation as a Gymnasium-compatible reinforcement-learning
environment: a 10-number observation, a two-action space (keep / switch phase), a
congestion-based reward, and a `reset()` / `step()` interface a DQN can plug into.
Stage 2 builds and validates the environment **only** — it trains nothing. See
[`docs/rl_environment.md`](docs/rl_environment.md).

**Stage 3** adds the Deep Q-Network: a PyTorch Q-network over that same environment, a
replay buffer, a separate target network, a training loop, greedy evaluation, and a
multi-seed Fixed-Time vs DQN comparison on identical traffic. The agent only ever
answers *keep* or *switch*; the Stage 2 safety controller still decides what that means
for the signal. See [`docs/dqn_training.md`](docs/dqn_training.md).

**The dashboard** is a Streamlit web interface over the same pipeline: pick a scenario
and a controller, press Run, and a real SUMO simulation produces the metrics and charts
shown. It never recalculates a metric or invents a number. The DQN mode is enabled only
when a trained checkpoint actually exists for the selected scenario, and DQN results are
kept in their own files so they can never be confused with the fixed-time baseline. See
[`docs/dashboard.md`](docs/dashboard.md).

```bash
streamlit run dashboard.py
```

### What works today

| Component | Status |
|---|---|
| Four-way intersection road network (SUMO) | Implemented and verified |
| Fixed-time traffic-signal controller (NS green → yellow → EW green → yellow) | Implemented, timings configurable in YAML |
| Traffic generation from real SUMO flow/route files | Implemented |
| Scenario 1 — moderate traffic | Implemented and executed |
| Scenario 2 — heavy traffic | Implemented and executed |
| TraCI-based metric collection | Implemented |
| CSV result files | Generated from real runs |
| Matplotlib graphs | Generated automatically from the CSV files |
| Gymnasium RL environment (`rl/`) | Implemented — `reset()`, `step()`, spaces, reward |
| Observation / action / reward definition | Implemented and documented |
| SUMO-RL compatibility on the same network | Implemented and tested |
| Manual RL environment demonstration | `python rl/run_environment.py` |
| Streamlit dashboard (`dashboard.py`) | Implemented — runs the real pipeline |
| Agent interface + fixed-time controller (`agents/`) | Implemented |
| Simulation controller (`simulation/controller.py`) | Implemented |
| **DQN Q-network (PyTorch, 10 → 64 → 64 → 2)** | Implemented (`agents/dqn_network.py`) |
| **Replay buffer (50,000 transitions, random mini-batches)** | Implemented (`agents/replay_buffer.py`) |
| **Target network with periodic refresh** | Implemented (`agents/dqn_agent.py`) |
| **Epsilon-greedy exploration with a schedule that follows the episode count** | Implemented |
| **Training loop, checkpoints, training history CSV, training curves** | Implemented (`training/`) |
| **Greedy evaluation over multiple seeds** | Implemented (`evaluation/evaluate_dqn.py`) |
| **Fixed-Time vs DQN comparison + improvement percentages + charts** | Implemented (`evaluation/compare_controllers.py`) |
| **DQN mode in the dashboard, enabled only when a model exists** | Implemented |
| Automated tests | **224 passed, 1 skipped** (verified 20 Nov 2026); historical 16 Nov user-reported run: 221 passed, 1 skipped |

### What is NOT implemented

Real-world traffic data, camera/CCTV or YOLO vehicle detection, hardware
integration (Raspberry Pi, Arduino, physical signals), cloud deployment,
multi-intersection coordination, variable phase durations, and alternative RL
algorithms such as PPO or A2C.

These are future scope, and nothing in the repository pretends otherwise.
See the limitations section of [`docs/dqn_training.md`](docs/dqn_training.md)
for exactly what the DQN result does and does not show.

## 4. Technology used so far

Stage 1:

```
Python
SUMO
TraCI
NumPy
Pandas
Matplotlib
PyYAML
```

Stage 2 adds:

```
Gymnasium      (the standard RL environment interface)
SUMO-RL        (SUMO <-> Gymnasium integration)
```

The dashboard adds:

```
Streamlit      (the web interface)
Altair         (the interactive charts; ships with Streamlit)
```

Stage 3 adds:

```
PyTorch        (the Q-network, autograd and the Adam optimiser)
```

Stable-Baselines3 is deliberately **not** used. A 5,000-parameter network with
a two-action space does not need a training framework, and writing the DQN
update out in full — replay buffer, target network, temporal-difference loss —
is what makes it explainable line by line in a project report.

### Future scope (not implemented)

```
PPO / A2C      alternative RL algorithms
YOLO / CCTV    camera-based vehicle detection
Raspberry Pi   hardware in the loop
Multi-junction coordination
```

---

## 5. Stage 1 architecture

```
        Traffic Demand   (config/moderate.yaml, config/heavy.yaml)
               |
               v
         SUMO Network    (sumo/network/intersection.net.xml)
               |
               v
    Fixed-Time Traffic Signal   (generated sumo/additional/<scenario>_tls.add.xml)
               |
               v
       Vehicle Simulation       (SUMO)
               |
               v
             TraCI              (Python <-> SUMO, step by step)
               |
               v
       Metric Collection        (utils/metrics.py)
               |
               v
          CSV Results           (results/csv/)
               |
               v
          Matplotlib            (simulation/plot_results.py)
               |
               v
             Graphs             (results/plots/)
```

## 5b. Stage 2 architecture — the RL environment

```
        SUMO  (the same 4-way intersection and the same two scenarios)
               |
             TraCI
               |
     SUMO-RL / Gymnasium layer      rl/environment.py
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

One turn of that loop is one call to `env.step(action)`.
The full explanation is in [`docs/rl_environment.md`](docs/rl_environment.md).

## 5c. Stage 3 architecture — the DQN

```
                    IntelliSignal
                         |
          +--------------+--------------+
          |                             |
     Fixed-Time                     DQN Agent
          |                             |
          |                    +--------+--------+
          |                    |                 |
          |              Policy Network    Target Network
          |                    |                 |
          |                    +--------+--------+
          |                             |
          |                       Replay Buffer
          |                             |
          +--------------+--------------+
                         |
                  Signal Control          rl/action.py safety rules
                         |
                       SUMO  <->  TraCI
                         |
              +----------+----------+
              |                     |
          Observation            Reward
              |                     |
              +----------+----------+
                         |
                   Training / Evaluation
                         |
                 Fixed-Time vs DQN
                         |
                  Streamlit Dashboard
```

The agent outputs only `0 = keep` or `1 = switch`. The Stage 2 action
controller turns that into a legal signal change with the Stage 1 yellow
clearance, honours `min_green`, and forces a switch at `max_green` — so a
learned policy cannot produce an unsafe or unfair signal.

**There is a second action space**, added on 22 Nov 2026, which is the one
that finally beats fixed-time: instead of deciding keep-or-switch every few
seconds, the agent makes **one decision per green and chooses how long that
green lasts**, from a menu bounded by `min_green` and `max_green`. Safety is
then structural rather than enforced — an unsafe green is not expressible,
every action ends with the yellow clearance, and every action hands the road
to the other direction next. The two action spaces live side by side; nothing
about the Stage 2 environment or the deployed Stage 4 models changed.

A longer explanation is in [`docs/phase_duration_control.md`](docs/phase_duration_control.md),
[`docs/dqn_training.md`](docs/dqn_training.md) and
[`docs/architecture.md`](docs/architecture.md).

---

## 6. Project structure

```
intelliSignal/
├── README.md
├── requirements.txt
├── requirements-win-py312.lock  exact verified Windows/Python 3.12 environment
├── .gitignore
│
├── config/                      scenario definitions (the only place numbers live)
│   ├── moderate.yaml
│   ├── heavy.yaml
│   ├── rl/
│   │   └── environment.yaml     Stage 2: RL environment settings
│   └── dqn/
│       └── training.yaml        Stage 3: every DQN hyperparameter
│
├── sumo/
│   ├── network/                 the four-way intersection
│   │   ├── intersection.nod.xml     nodes
│   │   ├── intersection.edg.xml     edges (roads)
│   │   ├── intersection.con.xml     turning movements
│   │   ├── intersection.net.xml     built network (netconvert output)
│   │   └── build_network.py         rebuilds the network
│   ├── routes/                  generated traffic demand (.rou.xml)
│   ├── additional/              generated fixed-time signal program
│   └── config/                  generated .sumocfg files
│
├── simulation/
│   ├── run_simulation.py        run one scenario, collect metrics, save CSV
│   ├── run_scenario.py          one-command demonstration wrapper
│   ├── controller.py            the layer the dashboard talks to
│   ├── agent_simulation.py      Stage 3: one episode under one controller,
│   │                            fixed-time and DQN measured identically
│   └── plot_results.py          build the graphs from the CSV files
│
├── utils/
│   ├── config_loader.py         load and validate a scenario YAML file
│   ├── scenario_builder.py      YAML -> SUMO route / signal / config files
│   └── metrics.py               TraCI metric collection
│
├── dashboard.py                 the Streamlit entry point
├── dashboard/                   the presentation layer
│   ├── theme.py                 palette, CSS, formatting
│   ├── state.py                 session state; a run needs a button press
│   ├── controls.py              the sidebar
│   ├── components.py            status lights, KPI cards, intersection, signal
│   └── charts.py                the Altair charts
│
├── agents/                      the decision-making layer
│   ├── base_agent.py            TrafficAgent interface, TrafficState, actions
│   ├── fixed_time_agent.py      the Stage 1 controller behind that interface
│   ├── dqn_agent.py             Stage 3: the DQN agent (epsilon-greedy,
│   │                            learning, target network, checkpoints)
│   ├── dqn_network.py           Stage 3: the Q-network (10 -> 64 -> 64 -> 2)
│   ├── replay_buffer.py         Stage 3: the replay buffer
│   └── dqn_config.py            Stage 3: hyperparameters and validation
│
├── rl/                          Stage 2: the RL environment (no agent, no training)
│   ├── environment.py           the Gymnasium environment
│   ├── state.py                 the 10-number observation
│   ├── action.py                keep / switch, and the signal safety rules
│   ├── reward.py                the congestion penalty
│   ├── config.py                loads config/rl/environment.yaml
│   ├── sumo_rl_adapter.py       SUMO-RL compatibility layer
│   └── run_environment.py       manual demonstration
│
├── training/                    Stage 3: learning
│   ├── train_dqn.py             the training loop (CLI)
│   └── plot_training.py         reward / loss / epsilon / queue curves
│
├── evaluation/                  Stage 3: measuring and comparing
│   ├── evaluate_dqn.py          greedy evaluation over several seeds
│   ├── compare_controllers.py   Fixed-Time vs DQN on identical traffic
│   └── plot_comparison.py       the comparison charts
│
├── models/                      active-model registry and historical checkpoints
│   ├── active_models.json       SHA-pinned Stage 4 dashboard incumbents
│   ├── dqn_moderate/            original Stage 3 checkpoints
│   └── dqn_heavy/               original Stage 3 checkpoints
│
├── results/
│   ├── csv/                     the recorded Stage 1 baseline (NEVER overwritten)
│   ├── data/                    results from dashboard runs (fixed-time and DQN
│   │                            in separate files)
│   ├── training/                Stage 3: per-episode training history CSVs
│   ├── evaluation/              Stage 3: raw, aggregated and improvement CSVs
│   └── plots/                   generated graphs
│       ├── training/            Stage 3: training curves
│       └── comparison/          Stage 3: Fixed-Time vs DQN charts
│
├── tests/
│   ├── test_simulation.py       24 Stage 1 tests
│   ├── test_rl_environment.py   47 Stage 2 tests
│   ├── test_dashboard.py        51 dashboard / agent / controller tests
│   └── test_dqn.py              Stage 3: network, buffer, agent, training,
│                                evaluation, comparison, dashboard
│
├── docs/
│   ├── architecture.md
│   ├── rl_environment.md        Stage 2: the RL environment explained
│   ├── dqn_training.md          Stage 3: the DQN explained end to end
│   └── dashboard.md             the dashboard explained
│
└── presentation/
    ├── stage1_progress.md       slide content for the 2nd progress presentation
    ├── stage1_speaking_notes.md speaking notes + expected questions
    ├── capture_screenshots.py   captures real screenshots from a running SUMO
    └── screenshots/
```

Nothing in `sumo/routes/`, `sumo/additional/` or `sumo/config/` is written by hand —
those files are generated from the YAML configuration every time a scenario runs.

---

## 7. Installation

1. **Install SUMO.** Either download it from <https://sumo.dlr.de/docs/Downloads.php>
   and set the `SUMO_HOME` environment variable, or install the packaged version
   together with the Python requirements below (it ships the SUMO binaries).

2. **Install the Python requirements:**

   ```bash
   pip install -r requirements.txt
   ```

   For the verified Windows/Python 3.12.13 environment, install the exact
   snapshot with `pip install -r requirements-win-py312.lock` instead.

3. **Check that SUMO is reachable:**

   ```bash
   python -c "import sumolib; print(sumolib.checkBinary('sumo'))"
   ```

---

## 8. How to run

### The dashboard

```bash
streamlit run dashboard.py
```

It opens in a browser. Then:

1. **Select a scenario** in the sidebar — Moderate or Heavy traffic.
2. **Select a control mode** — Fixed-Time, or DQN Adaptive. The DQN mode is
   enabled only when a trained checkpoint exists for the selected scenario;
   with no model it is disabled and says *"DQN unavailable — train a model
   first"*, with the command that trains one.
3. **Set the simulation duration** — 1800 s matches the recorded Stage 1
   results; a shorter run is quicker to show live.
4. **Press Run Simulation.** A real SUMO simulation runs. Nothing runs on a
   page refresh.
5. **View the metrics** — average waiting time, average queue length,
   throughput and simulation time, read from the CSV that run produced.
6. **View the charts** on the Results page, and step through the recorded
   signal states with the slider on the Simulation page. For a DQN run the
   slider also shows the two Q-values the network produced at that second.
7. **Compare scenarios** in the comparison table and its charts.
8. **The DQN Model page** shows what has actually been trained — the
   checkpoint, the architecture, the hyperparameters, the training date and
   the recorded training curves.
9. **The Comparison page** shows Fixed-Time vs DQN with improvement
   percentages, from the multi-seed experiment on disk and from runs you
   started in the dashboard.

Results from the dashboard are written to `results/data/`, with fixed-time and
DQN runs in separate files. The recorded Stage 1 baseline in `results/csv/` is
never overwritten.

### The command line

Run the moderate-traffic scenario:

```bash
python simulation/run_simulation.py --scenario moderate
```

Run the heavy-traffic scenario:

```bash
python simulation/run_simulation.py --scenario heavy
```

Watch the simulation in the SUMO GUI (this is the live demonstration):

```bash
python simulation/run_simulation.py --scenario moderate --gui
python simulation/run_simulation.py --scenario heavy --gui
```

Run a scenario and regenerate the graphs in one step:

```bash
python simulation/run_simulation.py --scenario moderate --plot
```

Run both scenarios and build every graph with a single command:

```bash
python simulation/run_scenario.py all
```

Other useful commands:

```bash
python simulation/run_scenario.py moderate     # one scenario, end to end
python simulation/plot_results.py              # rebuild graphs from existing CSVs
python sumo/network/build_network.py           # rebuild the intersection network
python presentation/capture_screenshots.py     # capture real SUMO screenshots
pytest -v tests/                               # run the test suite
```

A shorter run is handy when demonstrating:

```bash
python simulation/run_simulation.py --scenario heavy --gui --duration 300
```

### Stage 2 — the RL environment

Watch the environment run. **This trains nothing**; the actions come from a
fixed rule, not from a learned agent:

```bash
python rl/run_environment.py --scenario moderate
python rl/run_environment.py --scenario heavy --policy alternate --steps 12
python rl/run_environment.py --scenario heavy --gui          # see it in sumo-gui
```

Use it from Python:

```python
from rl import IntelliSignalEnv

env = IntelliSignalEnv(scenario="heavy")
observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step(1)   # 1 = switch phase
env.close()
```

Or through Gymnasium:

```python
import gymnasium as gym
import rl                                    # registers the environment id

env = gym.make("IntelliSignal-v0", scenario="moderate")
```

### Stage 3 — train, evaluate and compare the DQN

Train. A 20-episode run is a quick demonstration; 100 episodes is the default.
Neither guarantees a good policy — the comparison below is what decides that.

```bash
python training/train_dqn.py --scenario moderate --episodes 20  --seed 42   # demo
python training/train_dqn.py --scenario moderate --episodes 100 --seed 42
python training/train_dqn.py --scenario heavy    --episodes 100 --seed 42
python training/train_dqn.py --scenario moderate --episodes 100 --seed 43   # 2nd seed
```

Evaluate the trained policy greedily (epsilon = 0, no learning):

```bash
python evaluation/evaluate_dqn.py --scenario moderate --seeds 42 43 44
```

Compare it against the fixed-time baseline on identical traffic:

```bash
python evaluation/compare_controllers.py --scenario moderate --seeds 42 43 44
python evaluation/compare_controllers.py --scenario heavy --seeds 42 43 44

# add an untrained network as a control, to check how much the LEARNING did
python evaluation/compare_controllers.py --scenario moderate --seeds 42 43 44 --include-untrained
```

Redraw the charts from the recorded CSVs:

```bash
python training/plot_training.py --scenario moderate --seed 42
python evaluation/plot_comparison.py --scenario moderate
```

Use the agent from Python:

```python
from agents import DQNAgent

agent = DQNAgent.for_scenario("moderate")        # loads the registered incumbent
action = agent.act(observation, training=False)  # 0 = keep, 1 = switch
print(agent.q_values(observation))               # [Q(keep), Q(switch)]
```

Training and evaluation write to `results/training/`, `results/evaluation/`,
`results/plots/` and `models/`. The Stage 1 baseline in `results/csv/` is never
written by any of them, and two tests assert that.

---

## 9. What the simulation measures

All four metrics are read from the running simulation through TraCI. None of them is
estimated or hard-coded.

| Metric | Definition | Source |
|---|---|---|
| `vehicles` | vehicles that entered the simulation | `traci.simulation.getDepartedNumber()` accumulated |
| `average_waiting_time` | mean accumulated waiting time per vehicle, in seconds | `traci.vehicle.getAccumulatedWaitingTime()` |
| `average_queue_length` | mean number of halting vehicles on the 8 incoming lanes, averaged over every simulation step | `traci.lane.getLastStepHaltingNumber()` |
| `throughput` | vehicles that completed their route, i.e. drove through the intersection and left the network | `traci.simulation.getArrivedNumber()` accumulated |

Two extra values are recorded because they make the results easier to trust:
`max_queue_length`, `vehicles_still_in_network` and `teleports` (SUMO removes a vehicle
that has been blocked for a very long time; a teleport count of 0 means the simulation
never gridlocked artificially).

> **Note on the waiting-time figure.** This project averages the waiting time over
> **every** vehicle that entered the network, including the ones still queueing when the
> simulation window ends. SUMO's own end-of-run statistic averages only over vehicles
> that *finished* their trip, so it reports a lower number in the heavy scenario. Both
> are correct; they answer slightly different questions. The project metric is the
> honest one for a congested scenario, because it does not silently drop the vehicles
> that are still waiting.

---

## 10. Scenarios

Both scenarios use the **same road network and the same signal plan**. Only the traffic
demand differs, so the two runs can be compared directly.

| | Moderate | Heavy |
|---|---|---|
| North approach | 400 veh/h | 850 veh/h |
| South approach | 400 veh/h | 850 veh/h |
| East approach | 300 veh/h | 650 veh/h |
| West approach | 300 veh/h | 650 veh/h |
| **Total demand** | **1400 veh/h** | **3000 veh/h** |
| Simulated time | 1800 s | 1800 s |
| Signal cycle | 68 s (30 s NS green, 4 s yellow, 30 s EW green, 4 s yellow) | identical |

Turning split per approach: 25 % right, 55 % straight, 20 % left.

---

## 11. Stage 1 results

Measured with SUMO 1.27.1, seed 42, 1800 s of simulated time per scenario.
Source files: `results/csv/moderate_results.csv`, `results/csv/heavy_results.csv`.

| Metric | Moderate traffic | Heavy traffic |
|---|---|---|
| Vehicles in simulation | 702 | 1445 |
| Average waiting time | 10.04 s / vehicle | 51.90 s / vehicle |
| Average queue length | 3.70 vehicles | 40.39 vehicles |
| Maximum queue length | 10 vehicles | 72 vehicles |
| Throughput (completed trips) | 693 vehicles | 1351 vehicles |
| Still in network at the end | 9 | 94 |
| Teleports (gridlock indicator) | 0 | 0 |

Reading of the result: the same fixed-time plan that keeps waiting time near 10 seconds
under moderate demand lets it grow to roughly 52 seconds when demand is a little over
twice as high, and the average queue grows by about a factor of eleven. Throughput rises
(more vehicles are served in absolute terms) but no longer keeps up with demand — 94
vehicles are still waiting when the window closes. **This is exactly the situation the
adaptive controller is meant to improve, and it is the baseline the DQN is measured
against in section 11b.**

Graphs (all generated from the CSV files above):

- `results/plots/01_average_waiting_time.png`
- `results/plots/02_average_queue_length.png`
- `results/plots/03_throughput.png`
- `results/plots/04_queue_length_over_time.png`
- `results/plots/05_summary_comparison.png`

---


## 11b. Stage 3 results — Fixed-Time vs DQN (historical)

> **Superseded.** The "Fixed-Time" column here is the 30/30 plan from
> `config/*.yaml`, which is not the best fixed plan for either scenario — the
> demand is asymmetric and so is the optimum. Measured over seeds 6101–6120
> the best plans are ns 7 / ew 6 at **5.754 s** (heavy) and ns 6 / ew 4 at
> **2.761 s** (moderate), and against those the Stage 3/4 DQN loses. For the
> current comparison against properly tuned baselines see
> [`output/dqn_phase_20261122/FINDINGS.md`](output/dqn_phase_20261122/FINDINGS.md).
> This section is kept because it is what was measured on 16 Nov 2026 and the
> numbers are reproducible; it is not the project's current result.

Every number below comes from a SUMO run performed by
`evaluation/compare_controllers.py`. For each seed, **one** set of SUMO input
files is built and both controllers run on it, so the road, the demand, the
arrival pattern, the duration and the metric definitions are identical. The
only difference is who decides when the signal changes.

Mean ± sample standard deviation over seeds 42, 43 and 44, 1800 s per run:

### Moderate traffic (1400 veh/h)

| Metric | Fixed-Time | DQN | DQN improvement |
|---|---|---|---|
| Average waiting time (s/vehicle) | 9.91 ± 0.20 | **5.71 ± 0.10** | **+42.3 %** |
| Average queue length (vehicles) | 3.66 ± 0.07 | **2.08 ± 0.05** | **+43.0 %** |
| Maximum queue length (vehicles) | 9.7 ± 0.6 | **7.0 ± 0.0** | **+27.6 %** |
| Throughput (vehicles completed) | 693 ± 0 | 691 ± 2 | −0.3 % |
| Vehicles still in network | 9 ± 0 | 11 ± 2 | −22.2 % |
| Teleports | 0 | 0 | n/a |

### Heavy traffic (3000 veh/h)

| Metric | Fixed-Time | DQN | DQN improvement |
|---|---|---|---|
| Average waiting time (s/vehicle) | 52.60 ± 1.46 | **47.91 ± 2.00** | **+8.9 %** |
| Average queue length (vehicles) | 40.98 ± 1.19 | **37.75 ± 1.48** | **+7.9 %** |
| Maximum queue length (vehicles) | 71.3 ± 3.1 | **69.7 ± 2.3** | **+2.3 %** |
| Throughput (vehicles completed) | 1351 ± 4 | 1344 ± 6 | −0.6 % |
| Vehicles still in network | 96 ± 4 | 116 ± 2 | −20.9 % |
| Teleports | 0 | 0 | n/a |

A positive percentage means the DQN did better; a negative one means it did
worse. Both are shown.

### Reading the result honestly

**The improvement is real and it is larger under moderate demand.** Cutting
mean waiting time from 9.9 s to 5.7 s is a genuine 42 % reduction, consistent
across all three seeds (the worst DQN seed is still better than the best
fixed-time seed). Under heavy demand the gain shrinks to about 9 %: when every
approach is oversaturated there is simply less for a smarter signal to win,
because green time is the scarce resource and the DQN cannot create more of
it.

**Throughput is 0.3–0.6 % lower, not higher.** That is within a handful of
vehicles and is not a win for the DQN. Throughput under this demand is bounded
by the demand itself (693 of 702 vehicles complete in the moderate scenario),
so there is almost no headroom; the DQN trades a couple of completed trips for
a much shorter wait for everyone.

**"Vehicles still in network" is worse, and the raw numbers explain why.**
Under moderate demand it is 11 instead of 9 — two vehicles. Under heavy demand
the DQN also *admitted more traffic*: 1462 vehicles entered the network under
the DQN against 1445 under fixed-time at seed 42, because a less congested
approach lets SUMO insert vehicles that were previously held back. More
vehicles in means more unfinished at the 1800 s cut-off. The percentage is
reported as measured and is not explained away, but it should be read together
with the insertion counts in
`results/evaluation/comparison_<scenario>_raw.csv`.

### The control that makes the result meaningful

The DQN decides every 10 s and may switch as often as `min_green` allows,
while the fixed-time plan holds 30 s per direction. Some of the improvement
could therefore come from the shorter decision interval rather than from
anything the network learned. Running the **same architecture with random
weights** on the same traffic settles it:

| Average waiting time (s) | Fixed-Time | **DQN (trained)** | DQN (untrained control) |
|---|---|---|---|
| Moderate | 9.91 | **5.71** | 10.91 |
| Heavy | 52.60 | **47.91** | 88.21 |

The untrained network is **worse than fixed-time** in both scenarios — much
worse under heavy demand, where it also fails to admit traffic at all (about
1200 vehicles enter instead of 1445). So the shorter decision interval is not
what produced the improvement; the learning is.

### Training

| Run | Episodes | Mean return, first 5 | Mean return, last 5 | Wall clock |
|---|---|---|---|---|
| moderate, seed 42 | 100 | −98.9 | **−32.4** | 1098 s |
| moderate, seed 43 | 100 | −100.6 | **−31.3** | 748 s |
| heavy, seed 42 | 100 | −1316.0 | **−621.9** | 1492 s |

Two training seeds on the moderate scenario ended within 4 % of each other
(−32.4 and −31.3), so the learning is not a one-seed accident. Returns are
noisy from episode to episode and the curves in
`results/plots/training/` are drawn raw, with any moving average clearly
labelled as such.

Full discussion, including every limitation, is in
[`docs/dqn_training.md`](docs/dqn_training.md) and
[`STAGE3_REPORT.md`](STAGE3_REPORT.md).

---

## 12. Tests

Latest verified Windows run (22 Nov 2026): **317 passed, 1 skipped**, using
`.venv-stage3`.

The 20 Nov figure of "224 passed, 1 skipped" recorded in earlier reports was
not reproducible: `test_training_replay_uses_executed_action_and_masks_unavailable_switch`
failed on `main` and also at `8b8fa34`, before any of this work. It is fixed
(commit `43cd8dd`).

```bash
python -m pytest -q                               # entire current suite
python -m pytest tests/test_simulation.py -q      # Stage 1: 24 tests
python -m pytest tests/test_rl_environment.py -q  # Stage 2: 47 tests
python -m pytest tests/test_dashboard.py -q       # dashboard: 55 tests
python -m pytest tests/test_dqn.py -q             # Stage 3: 96 tests
python -m pytest tests/test_phase_dqn.py -q       # duration action space: 33
python -m pytest tests/test_phase_environment.py -q  # its environment: 13
python -m pytest tests/test_phase_registry.py -q  # promotion + loading: 11
```

```
221 passed, 1 skipped in 166.72s (0:02:46) — historical run, 16 Nov 2026
```

The result above was reported by the user on Windows on 16 Nov 2026 using
`.venv-stage3`. The earlier Linux run recorded 222 passed in 157 s.
The skipped test's identity was not included in the supplied output; the
process-cleanup test uses `pgrep` and skips when that utility is unavailable.

**Stage 1 (24 tests)** covers configuration loading and validation, rejection of invalid
configuration, existence and contents of the SUMO network, generation of the route file
and the signal program from the YAML values, availability of SUMO and TraCI, opening and
closing a TraCI connection, a full short simulation that produces metrics and terminates,
the CSV columns, and the graph generation.

**Stage 2 (47 tests)** covers the RL settings and their validation, the action
controller's timing and safety rules without needing SUMO, the reward arithmetic,
environment construction for both scenarios, the observation and action spaces,
`reset()` and `step()` under the modern Gymnasium API, reward finiteness, the contents of
`info`, episode truncation, clean shutdown and reuse, the official
`gymnasium.utils.env_checker` compliance check, a real SUMO-RL environment built on the
same network, and two regression tests asserting that the RL environment does not modify
any Stage 1 file or result.

Neither of those two files was touched by Stage 3. All 71 of their tests pass against the
Stage 3 code — which is the evidence that the additive changes to `utils/metrics.py` and
`rl/environment.py` changed no existing behaviour.

**Dashboard (55 tests)** covers the agent interface and the fixed-time controller
(including that it ignores the traffic state), the simulation controller's environment
detection, scenario reading, control modes, running, result loading and every error path,
the signal-state decoder and the charts, and the Streamlit app itself driven through
`streamlit.testing.v1.AppTest`: startup, all six pages, scenario selection, the DQN mode
enabled or disabled according to whether a model exists, a real simulation executed
through the Run button, and a regression test asserting the dashboard never overwrites
the Stage 1 results.

**Stage 3 (96 tests)** covers the Q-network (shapes, finite Q-values, sizing from the
environment's own spaces, rejection of impossible sizes), the replay buffer (insertion,
eviction, batch dimensions and dtypes, refusal to over-sample, reproducibility), the
agent (greedy determinism, epsilon = 1 explores both actions, epsilon = 0 never explores,
learning waiting for `learning_starts`, the policy network actually moving, the target
network starting identical and being refreshed on schedule, no gradient reaching the
target network, both loss functions, both action conventions, Q-values in `explain()`,
a checkpoint round-trip reproducing identical decisions, and the refusal to act without
trained weights), fairness (the fixed-time comparison run reproducing the recorded
Stage 1 baseline **exactly**, both controllers seeing the same number of simulated
seconds, no unsafe green → green transition under pure exploration, and the refusal to
write into `results/csv/`), a real short training run end to end (checkpoints, metadata,
history columns, no NaN loss, curves generated, no leaked SUMO process), evaluation
(greedy, **no gradient step and no replay-buffer change**, multiple seeds, no fabricated
standard deviation for a single seed, reproducibility, raw and aggregated results kept
separate), the comparison (identical traffic for both controllers, every run kept,
improvement signs, a zero denominator handled, the baseline left byte-identical, the
untrained control, the charts), and the dashboard integration (the mode gated per
scenario on a real checkpoint, a clear refusal without one, honest model status, a DQN
run really running the network and writing its own files, no fallback to the baseline
under a DQN label, and the Comparison page reading real evaluation CSVs).

Six dashboard tests that asserted "the DQN does not exist" were rewritten to assert the
rule that replaced them, and four were added. No test was deleted. The before/after
table is in [`STAGE3_REPORT.md`](STAGE3_REPORT.md).

---

## 13. Current limitations

- The intersection is a **single**, simplified four-way junction with two lanes per
  approach. There is no coordination between junctions and no green wave.
- The agent has only **two actions** — hold the green or change it. It cannot choose a
  phase duration, a phase order or a protected left-turn stage.
- **100 training episodes** is about 18,000 decisions, chosen so a run finishes on a
  student laptop. Published traffic-RL work often uses one to two orders of magnitude
  more. The policy is not converged in any strong sense.
- The traffic is **simulated, not measured** from a real road. No detector data, no
  camera, no hardware. Transfer to a real junction is unproven.
- The agent reads **exact** queue lengths and waiting times instantly from the
  simulator. A real deployment would have sensor noise, latency and failures.
- Demand is **stationary** within an episode: no rush-hour ramp, no incident, no weather.
- Only **two demand scenarios** are studied.
- The DQN was **not benchmarked against PPO, A2C or tuned actuated control**. The claims
  here are limited to DQN versus *this* fixed-time plan.
- Returns vary substantially between episodes and seeds, so conclusions are drawn from
  means over three evaluation seeds with standard deviations reported — never from one
  good run.
- The dashboard's intersection diagram is a schematic, not a SUMO rendering. Its signal
  colours are real when recorded data is supplied; the vehicle marks are indicative.
- The dashboard runs a simulation to completion and then shows the result; it does not
  stream a live simulation frame by frame.
- There is no pedestrian, public-transport or emergency-vehicle modelling.

These are deliberate boundaries of the project as scoped, not defects. The longer
version is in the limitations section of [`docs/dqn_training.md`](docs/dqn_training.md).

---

## 14. What is built, and what comes next

```
Done (Stage 1):
SUMO -> Fixed-Time Controller -> Traffic Metrics -> CSV -> Graphs

Done (Stage 2):
SUMO -> TraCI -> SUMO-RL / Gymnasium -> State + Action + Reward -> reset() / step()

Done (dashboard):
Streamlit -> Simulation Controller -> Fixed-Time / DQN -> SUMO -> Metrics -> Charts

Done (Stage 3):
10-value state -> Q-network -> Q(keep), Q(switch) -> epsilon-greedy
   -> safety controller -> SUMO -> reward -> replay buffer -> target network
   -> training -> saved model -> greedy evaluation -> Fixed-Time vs DQN

Future scope - NOT BUILT:
PPO / A2C  |  CCTV + YOLO detection  |  hardware in the loop
multi-intersection coordination  |  variable phase durations
real traffic datasets  |  time-varying demand and incidents
```

Nothing in `rl/` had to change for Stage 3 — which was the point of Stage 2. Nothing in
`sumo/`, `config/moderate.yaml`, `config/heavy.yaml` or `results/csv/` changed either:
the road, the demand, the observation, the action, the reward, the safety rules and the
recorded baseline are all exactly as the earlier stages left them. Stage 3 added the
network, the buffer, the target network, the training loop, the evaluation and the
comparison on top.

The future-scope items above are listed as future scope in the code as well. Nothing in
the repository pretends to implement them.
