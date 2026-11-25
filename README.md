# IntelliSignal

**AI-Based Adaptive Traffic Signal Management Using SUMO and Deep Reinforcement Learning**

> **Current status (25 Nov 2026).** IntelliSignal is one engine: an
> explicit list of vehicles drawn from an Indian urban fleet, a signal
> held to 15-30 s greens with a 3 s amber and a 2 s all-red, and a
> trained controller that chooses how long each green runs. Every
> controller the project compares - fixed-time, actuated, max-pressure
> and the model - is run by the same code on the same vehicles and
> measured the same two ways, and every run is audited a second time
> from the signal SUMO actually displayed.
>
> The current measured results are in
> [`output/standard_20261125/FINDINGS.md`](output/standard_20261125/FINDINGS.md),
> which also records how the model was chosen: twenty independently
> trained runs and four averaged ensembles, re-scored on validation
> traffic, against a baseline that is the best of every fixed plan on the
> menu rather than the end of one search.
>
> **Earlier figures in this file are historical.** Sections 11 and 11b
> were measured on a single generic car with no startup lost time, under
> different timing rules, and are not comparable with anything above.
> Stages 1-3 still ship as working code and reproducible evidence; their
> documentation is in [`docs/history/`](docs/history/).

The existing Windows environment is `.venv-stage3` (Python 3.12.13, installed
SUMO/TraCI/sumolib packages 1.28.0, PyTorch 2.14.1). Comparison CSVs are in
[`verification/windows_20261116/`](verification/windows_20261116/), with
fixed-time, trained DQN, and untrained controls evaluated at seeds 42, 43, and 44
for 1800 seconds. Waiting time decreased by **42.3% (moderate)** and **8.9%
(heavy)**; throughput decreased by **0.3%** and **0.6%**, respectively.
Use [`requirements-win-py312.lock`](requirements-win-py312.lock) to recreate
that exact Windows Python environment. `requirements.txt` remains the portable
minimum-version list.
The 16 Nov comparisons above are historical Stage 3 verification. The
current results are in
[`output/standard_20261125/FINDINGS.md`](output/standard_20261125/FINDINGS.md);
[`HANDOFF.md`](HANDOFF.md) is a snapshot of where the project stood on
21 Nov 2026 and is kept as a record of that point.

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

Stages 1-3 built the simulation, the environment and the first DQN. They
still run and are still covered by tests, and their documents are in
[`docs/history/`](docs/history/), marked historical. **They are not how
the project works today**; the current system is described after them.

**Stage 1** implements the initial SUMO traffic simulation environment, a four-way
intersection, a fixed-time traffic-signal controller, two traffic-demand scenarios,
traffic-metric collection through TraCI, and the initial result visualisations.

**Stage 2** wraps that same simulation as a Gymnasium-compatible reinforcement-learning
environment: a 10-number observation, a two-action space (keep / switch phase), a
congestion-based reward, and a `reset()` / `step()` interface a DQN can plug into.
Stage 2 builds and validates the environment **only** — it trains nothing. See
[`docs/history/rl_environment.md`](docs/history/rl_environment.md).

**Stage 3** adds the Deep Q-Network: a PyTorch Q-network over that same environment, a
replay buffer, a separate target network, a training loop, greedy evaluation, and a
multi-seed Fixed-Time vs DQN comparison on identical traffic. The agent only ever
answers *keep* or *switch*; the Stage 2 safety controller still decides what that means
for the signal. See [`docs/history/dqn_training.md`](docs/history/dqn_training.md).

**The current system** replaces the keep-or-switch formulation with one
that chooses how long each green runs, and the generic car with a real
fleet.

- **One operating timing profile**, `standard`: greens of 15-30 s, a 3 s
  amber, a 2 s all-red, and a maximum red of four times the preceding
  green, enforced on every controller in `rl/timing_profile.py`. A
  `responsive` 4-20 s profile exists only so the historical models can
  still be loaded and explained.
- **An Indian urban fleet** (`custom_traffic/fleet.py`): 62 %
  motorcycles, 22 % cars, 11 % auto-rickshaws, 5 % buses, each with its
  own length, acceleration and startup delay. Demand is an explicit
  `<vehicle>` element per vehicle, so two controllers meet identical
  traffic and a comparison is a measurement rather than an estimate.
- **Two model families**, both trained by
  `training/train_profile_dqn.py`: candidate **A** chooses a whole green
  from a menu of durations, candidate **B** serves the minimum green and
  then extends or ends it. Which environment a checkpoint runs in is read
  from the checkpoint by `rl/policy_rollout.py`.
- **Four controllers offered**: the trained DQN, fixed-time, actuated and
  max-pressure, plus Hybrid on the Compare page, which tunes a plan,
  tests the model against it and keeps whichever wins.
- **One headline measurement**: total delay per requested vehicle, from
  SUMO's own trip records, counting time lost on the road plus time spent
  unable to enter, including vehicles still on the network when the run
  ends. Mean waiting time is reported beside it as the historical
  measure.
- **Every run audited twice**: once by the controller's own account of
  itself, and once from the signal SUMO actually displayed
  (`simulation/recorder.py`), which shares no code with any controller.

**The dashboard** is the product. Five pages - Control room, Traffic,
Compare, The model, Method - all of them going through
`simulation/engine.py`, which runs every controller through
`evaluation/harness.py`. The numbers on screen and the numbers in the
reports are therefore the same numbers. See
[`docs/dashboard.md`](docs/dashboard.md).

```bash
streamlit run dashboard.py
```

The current measured figures are in
[`output/standard_20261125/FINDINGS.md`](output/standard_20261125/FINDINGS.md).

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

The table above is the Stage 1-3 position and is kept as a record of it.
Since then the dashboard has been rebuilt around the five pages described
above, variable phase durations have been implemented, and the fleet and
the timing rules have changed.

### What is NOT implemented

Real-world traffic data, camera/CCTV or YOLO vehicle detection, hardware
integration (Raspberry Pi, Arduino, physical signals), cloud deployment,
multi-intersection coordination, and alternative RL algorithms such as
PPO or A2C.

These are future scope, and nothing in the repository pretends otherwise.
The Method page of the dashboard states what the current result does and
does not show.

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
The full explanation is in [`docs/history/rl_environment.md`](docs/history/rl_environment.md).

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
[`docs/history/dqn_training.md`](docs/history/dqn_training.md) and
[`docs/history/architecture.md`](docs/history/architecture.md).

---

## 6. Project structure

Grouped by what each part is for, not by the stage that built it. Three
groups: the **product** a visitor touches, the **current pipeline** that
produces every number the product shows, and the **historical evidence**
from Stages 1-3, which still runs and is still covered by tests but is no
longer how the project works.

### The product

```
dashboard.py                  the Streamlit entry point: pages, sidebar
dashboard/
  ui.py                       stylesheet, colours, tiles, badges, verdicts
  state.py                    every session key and its default
  jobs.py                     runs on worker threads, keyed so they are
                              never started twice
  junction.py                 the junction replay, drawn from the recording
  charts.py                   the Altair charts
  custom_form.py              describing traffic lane by lane
  hybrid_panel.py             Hybrid's progress and its decision
  views/
    control_room.py           choose a controller, run it, watch it
    traffic.py                what is on the road
    compare.py                every controller on the same vehicles, Hybrid
    model.py                  which model is in charge, and how it was chosen
    method.py                 how this is modelled, measured and limited
    common.py                 the pieces more than one page shows
.streamlit/config.toml        the theme
```

### The current pipeline

```
simulation/
  engine.py                   the one service the dashboard talks to:
                              presets, controllers, a run, a comparison
  recorder.py                 the per-second recording, and the independent
                              audit of the signal SUMO actually displayed
  hybrid.py                   tune a plan, test the model, keep the winner
  hybrid_job.py               that search, on a thread, behind the browser
  custom_runs.py              running an explicit workload outside the UI

evaluation/
  harness.py                  every controller runs through here
  select_profile_model.py     score every trained run on the validation
                              traffic, try ensembles, sweep the fixed plans
  promote_profile_model.py    write the chosen model into the registry
  compare_profiles.py         candidate A against candidate B
  fixed_time_sweep.py         measure every plan on the menu
  gates.py                    the promotion conditions, as a pure function

rl/
  timing_profile.py           the timing rules every controller obeys
  phase_environment.py        candidate A: one decision per green, chosen
                              from a menu of durations
  phase_extend_env.py         candidate B: serve the minimum green, then
                              extend or end
  phase_state.py              the ten-number observation and its scales
  phase_action.py             the action space and the clearance rules
  policy_rollout.py           reads a checkpoint and runs it in the
                              environment its action space belongs to

agents/
  phase_dqn_agent.py          the duration/extend agent and the registry
  phase_ensemble.py           several networks averaged into one policy

custom_traffic/
  spec.py                     what the user typed; None means "not said"
  resolver.py                 every value decided, each tagged user /
                              derived / auto
  builder.py                  one explicit <vehicle> element per vehicle
  workload.py                 the resolved workload and its identifier
  fleet.py                    the Indian urban fleet and its parameters
  network.py                  checks a lane can actually serve its route

training/
  train_profile_dqn.py        trains candidate A or candidate B
  workload_library.py         the training, validation and test splits

utils/
  delay_accounting.py         total delay per requested vehicle, from
                              SUMO's own trip records
  sumo_launch.py              starting and cleaning up a SUMO process
  metrics.py                  the live TraCI metric collection
  scenario_builder.py         YAML -> SUMO route / signal / config files
  config_loader.py            load and validate a scenario file

tools/clean_workspace.py      report, and with --apply remove, what the
                              runs left behind

config/indian_arterial.yaml   the junction the product runs on
models/active_models.json     the model registry (below)
sumo/network/                 the four-arm junction, and its builder
results/runs/                 every run the dashboard has measured
output/standard_20261125/     the current training and evaluation run
```

### Historical evidence (Stages 1-3)

Kept because the published numbers must reproduce. All of it still runs
and all of it is still covered by the test suite.

```
rl/environment.py, state.py, action.py, reward.py, config.py,
    sumo_rl_adapter.py, run_environment.py
                              the Gymnasium keep/switch environment
agents/dqn_agent.py, dqn_network.py, replay_buffer.py, dqn_config.py,
    base_agent.py, fixed_time_agent.py
training/train_dqn.py, train_phase_dqn.py, plot_training.py
evaluation/evaluate_dqn.py, compare_controllers.py, plot_comparison.py,
    evaluate_phase_dqn.py, promote_phase_model.py,
    calibrate_phase_observation.py, collect_observation_stats.py
simulation/run_simulation.py, run_scenario.py, controller.py,
    agent_simulation.py, plot_results.py
                              the earlier dashboard; the current one does
                              not import them
config/moderate.yaml, heavy.yaml, config/rl/, config/dqn/
results/csv/                  the recorded Stage 1 baseline, never
                              overwritten
results/training/, evaluation/, plots/
experiments/, verification/   the controlled experiments and their reports
docs/history/                 architecture.md, rl_environment.md,
                              dqn_training.md
```

### Tests and documentation

```
tests/
  test_engine.py              the one service behind the dashboard
  test_recorder.py            the recording and the independent audit
  test_policy_rollout.py      a checkpoint runs in its own action space
  test_sumo_launch.py         SUMO starts and is cleaned up
  test_timing_profile.py      the timing rules and the red-limit arithmetic
  test_custom_traffic.py      the spec, the resolver and the builder
  test_delay_accounting.py    the headline measurement
  test_harness.py             every controller through one harness
  test_hybrid.py              the decision rule, without SUMO
  test_gates.py               the promotion conditions
  test_phase_dqn.py, test_phase_environment.py, test_phase_registry.py
  test_clearance_outside_interval.py
  test_simulation.py, test_rl_environment.py, test_dqn.py,
      test_dashboard.py       Stages 1-3

docs/
  dashboard.md                the product, page by page
  custom_traffic_and_hybrid.md  describing traffic, and the Hybrid rule
  demo_walkthrough.md         a ten-minute demonstration
  phase_duration_control.md   why a controller chooses a duration
  history/                    Stages 1-3, marked historical
```

### The model registry

`models/active_models.json` holds three separate sections, because they
answer three different questions and a checkpoint from one is not usable
by another.

| section | what is in it | read by |
|---|---|---|
| `active_models` | the Stage 3/4 keep-or-switch incumbents, by scenario | `agents/dqn_agent.py` |
| `phase_duration_models` | the November duration-menu models, by scenario | `agents/phase_dqn_agent.py`, as evidence |
| `profile_models` | the operating model, keyed by timing profile | `agents/phase_dqn_agent.py` |

**The dashboard reads `profile_models`**, through
`load_registered_profile_model()`. Each record pins the file's SHA-256,
the timing profile it was trained for, the split it was selected on, the
split it was reported on, and the selection report it came from; the
loader refuses a checkpoint that no longer matches. The other two sections
exist so the earlier results reproduce, and the Model page lists them as
evidence rather than offering them as controllers.

Nothing in `sumo/routes/`, `sumo/additional/` or `sumo/config/` is written
by hand. Those files are generated from the workload or the scenario
configuration every time something runs, and `tools/clean_workspace.py`
removes what is left behind.

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

Everything below is run from the repository root, with `.venv-stage3`
active on Windows.

### The dashboard

```bash
streamlit run dashboard.py
```

That is the whole product. It opens in a browser on the **Control room**
page; the sidebar chooses the page and the traffic. Nothing has to be
trained or generated first. If SUMO is missing the sidebar says which
check failed, and if no validated model is registered for the operating
timing profile the DQN option says so and Hybrid decides instead.

Five pages: Control room, Traffic, Compare, The model, Method.
`?page=Compare` in the URL opens one of them directly. Runs the dashboard
measures are written to `results/runs/`.
[`docs/dashboard.md`](docs/dashboard.md) describes all of it.

### Train a model

Two model families are trained by the same script. Candidate **A** picks a
whole green from the profile's menu of durations
(`rl/phase_environment.py`); candidate **B** serves the minimum green and
then repeatedly chooses between extending and ending it
(`rl/phase_extend_env.py`). Which environment a checkpoint later runs in
is read from the checkpoint itself by `rl/policy_rollout.py`, never
assumed by the caller.

```bash
python training/train_profile_dqn.py --profile standard --candidate B \
    --episodes 320 --validate-every 20 --seed 42 --out output/run
```

One run per seed, into its own `--out` directory. `--pilot` runs a short
sanity check instead of a full training run.

### Choose a model, and promote it

Selection scores every trained run on the validation traffic, tries
ensembles of the leading runs, sweeps every fixed plan on the profile's
menu for the baseline, and reads the test split once at the end.

```bash
python evaluation/select_profile_model.py --profile standard \
    --runs output/standard_20261125/runs --out output/selection \
    --seeds 3 --workers 14
```

`--runs` is the directory holding the training runs, `--seeds` is how many
SUMO seeds each workload is measured at, and `--workers` is how many runs
go in parallel. It writes `selection.json`, which is the file the Model
page reads.

Promotion writes the winner into `profile_models` in
`models/active_models.json`, with the file's SHA-256, the split it was
selected on and the split it was reported on:

```bash
python evaluation/promote_profile_model.py --profile standard \
    --selection output/selection/selection.json --selected-on validation
```

The dashboard picks up the new record without being restarted.

### Compare two candidates directly

```bash
python evaluation/compare_profiles.py --profile standard \
    --candidate-a <a.pt> --candidate-b <b.pt>
```

Both checkpoints and a tuned fixed plan on the same workloads. `--no-tune`
with `--ns-green` and `--ew-green` fixes the baseline plan instead of
sweeping for it.

### Housekeeping and tests

```bash
python tools/clean_workspace.py            # report what the runs left behind
python tools/clean_workspace.py --apply    # and remove it
python -m pytest tests/ -q                 # the whole suite
```

`clean_workspace.py` reports before it removes anything, leaves files
younger than `--min-age-minutes` alone, and refuses to sweep at all while
something is simulating unless it is given `--force`.

### Historical (Stages 1-3)

These still run and are still covered by the test suite. They use the
`moderate` and `heavy` scenarios, which are a single generic car with no
startup delay under the older timing rules, so their numbers are not
comparable with anything measured on the Indian fleet.

```bash
# Stage 1: the simulation and the fixed-time baseline
python simulation/run_simulation.py --scenario moderate
python simulation/run_simulation.py --scenario heavy --gui --duration 300
python simulation/run_scenario.py all
python simulation/plot_results.py
python sumo/network/build_network.py

# Stage 2: the Gymnasium environment, which trains nothing
python rl/run_environment.py --scenario moderate
python rl/run_environment.py --scenario heavy --policy alternate --steps 12

# Stage 3: the keep-or-switch DQN
python training/train_dqn.py --scenario moderate --episodes 100 --seed 42
python evaluation/evaluate_dqn.py --scenario moderate --seeds 42 43 44
python evaluation/compare_controllers.py --scenario moderate --seeds 42 43 44
python evaluation/compare_controllers.py --scenario moderate --seeds 42 43 44 \
    --include-untrained
python training/plot_training.py --scenario moderate --seed 42
python evaluation/plot_comparison.py --scenario moderate
```

Those write to `results/training/`, `results/evaluation/`,
`results/plots/` and `models/`. The Stage 1 baseline in `results/csv/` is
never written by any of them, and two tests assert that.

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

## 11. Stage 1 results (historical)

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
[`docs/history/dqn_training.md`](docs/history/dqn_training.md) and
[`STAGE3_REPORT.md`](STAGE3_REPORT.md).

---

## 12. Tests

```bash
python -m pytest -q                               # the whole suite
python -m pytest tests/test_engine.py -q          # the one engine behind the dashboard
python -m pytest tests/test_recorder.py -q        # the independent signal audit
python -m pytest tests/test_policy_rollout.py -q  # which environment a checkpoint runs in
python -m pytest tests/test_sumo_launch.py -q     # starting SUMO safely in parallel
python -m pytest tests/test_dashboard.py -q       # the five pages, really executed
python -m pytest tests/test_hybrid.py -q          # the decision rule and its seeds
python -m pytest tests/test_phase_registry.py -q  # promotion, fingerprints, ensembles
```

The run count and timing for the current suite are recorded with the
results in
[`output/standard_20261125/FINDINGS.md`](output/standard_20261125/FINDINGS.md).

`tests/conftest.py` does three things for every test: it puts the
repository on the import path, it cancels any background simulation a
test left running, and it fails any test that rewrites the committed
figures in `results/plots`. There is also an end-to-end check outside
pytest, which runs the real simulator against the real registry:

```bash
python tools/verify_product.py            # all checks
python tools/verify_product.py --quick    # the ones that do not simulate
```

Roughly half the suite covers Stages 1-3. Those tests are kept because
they are the evidence that the historical results still reproduce, not
because the current product uses that code.

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

## 13. Limitations

These are the boundaries of the project as it stands today. The dashboard
states them on its Method page too, because a result whose limits are
hidden is worth less than one whose limits are stated.

**The junction**

- A **single** four-way junction, two lanes each way. No coordination with
  a neighbour, no green wave, no bus stops, parking or pedestrians.
- The roads leaving the junction are free, so a queue can never back up
  into it from downstream.
- There is no public-transport priority and no emergency-vehicle
  pre-emption.

**The traffic**

- The fleet shares are **representative, not surveyed**. 62 % two-wheelers,
  22 % cars, 11 % auto-rickshaws and 5 % buses is typical of an Indian
  urban arterial, not a count taken at a particular junction.
- **Saturation headway was not validated against field data.** It emerges
  from the headway, minimum gap and startup delay given to each vehicle
  class and lands in a plausible range, which is not the same as having
  been measured.
- Demand is **deterministic**: every vehicle is written out individually,
  which is what makes two controllers comparable, but it is not the same
  as stochastic arrivals. The controller is therefore not tested against
  the variability a real junction sees.
- The traffic is **simulated, not measured** from a real road. No detector
  data, no camera, no hardware; transfer to a real junction is unproven.

**What the controller can see and do**

- It reads **exact** queue lengths and waiting times from the simulator.
  A real deployment would have sensor noise, latency and failures.
- It chooses the length of each green. It does not choose the phase
  **order**, and there is no protected left-turn stage.
- Training is 320 episodes per run. Published traffic-RL work often uses
  an order of magnitude more.

**What the comparison does and does not show**

- The claims are limited to this junction, this fleet and these timing
  rules, against a tuned fixed plan, actuated control and max-pressure.
  The DQN has **not** been compared with PPO or A2C.
- Six test traffic patterns at three SUMO seeds each is a small sample,
  and the two candidate action spaces are close enough that the
  difference between them is within noise.
- `moderate` and `heavy` still use a single generic car with no startup
  delay. Their numbers are preserved so the earlier stages reproduce, and
  nothing puts them in a table with anything measured on the Indian fleet.

The Stage 1-3 limitations, which described a controller that could only
hold or change the green, are in
[`docs/history/dqn_training.md`](docs/history/dqn_training.md).

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

Done (current):
Indian urban fleet -> explicit vehicle list -> one engine
   -> fixed-time / actuated / max-pressure / trained DQN, all under the
      standard 15-30 s timing profile
   -> SUMO trip records -> total delay per requested vehicle
   -> independent audit of the signal that was actually displayed
   -> train both candidates -> select on validation -> promote with a
      pinned sha256 -> report once on test -> the dashboard

Future scope - NOT BUILT:
PPO / A2C  |  CCTV + YOLO detection  |  hardware in the loop
multi-intersection coordination  |  real traffic datasets
stochastic arrivals  |  pedestrians, bus stops and parking
```

Nothing in `rl/` had to change for Stage 3 — which was the point of Stage 2. Nothing in
`sumo/`, `config/moderate.yaml`, `config/heavy.yaml` or `results/csv/` changed either:
the road, the demand, the observation, the action, the reward, the safety rules and the
recorded baseline are all exactly as the earlier stages left them. Stage 3 added the
network, the buffer, the target network, the training loop, the evaluation and the
comparison on top.

The current system was added the same way. It did not modify the Stage 1-3
code: the keep-or-switch environment, the Stage 3 agent and the recorded
baseline are untouched and still tested. What it added is a second action
space, a fleet, a timing profile, an honest delay measurement, an
independent audit of the signal, and one engine that every one of them
runs through.

The future-scope items above are listed as future scope in the code as well. Nothing in
the repository pretends to implement them.
