# IntelliSignal — 2nd Progress Presentation (Stage 1)

> Slide content for the second progress review.
> Every number in this deck comes from an executed simulation
> (`results/csv/`) — nothing is estimated.
> Fill the `[ ... ]` placeholders with your own details before presenting.

---

## Slide 1 — Title

# IntelliSignal
### AI-Based Adaptive Traffic Signal Management Using SUMO and Deep Reinforcement Learning

**BTech — Computer Science and Engineering**
**Second Progress Presentation — Stage 1**

| | |
|---|---|
| Student / Team | [ Your name(s) ] |
| Roll number(s) | [ ... ] |
| Guide / Supervisor | [ ... ] |
| Department | [ Department of Computer Science and Engineering ] |
| Institution | [ Your college / university ] |
| Date | [ ... ] |

---

## Slide 2 — Problem Statement

**Traffic demand changes; a fixed-time signal does not.**

- Traffic density at an intersection varies with the time of day, the direction of
  travel and unpredictable events.
- A conventional traffic signal is **fixed-time**: each direction gets a pre-defined
  green duration, decided in advance.
- Because the schedule is fixed, a busy approach can be held at red while an almost
  empty approach keeps its full green time.
- The consequences are measurable: longer waiting time per vehicle, longer queues, and
  throughput that no longer keeps up with demand.

**This project investigates an adaptive traffic-signal approach based on Deep
Reinforcement Learning, in which the signal chooses its phases from the traffic state
instead of from a fixed schedule.**

---

## Slide 3 — Project Objective

1. Simulate traffic realistically using **SUMO**.
2. Model a **four-way signalised intersection**.
3. Create **different traffic conditions** (moderate and heavy demand).
4. Establish a **fixed-time baseline controller**.
5. Collect **traffic metrics** from the simulation: waiting time, queue length, throughput.
6. *(later stage)* Develop a **DQN-based adaptive controller**.
7. *(later stage)* **Compare Fixed-Time vs DQN** on the same network and scenarios.

> Objectives 1–5 are complete and demonstrated today.
> **Objectives 6 and 7 are the next development stage — the DQN is not implemented yet.**

---

## Slide 4 — Technology Stack

**Used in Stage 1 (implemented):**

```
Python
SUMO
TraCI
NumPy
Pandas
Matplotlib
PyYAML
```

**Planned in later stages (not implemented yet):**

```
SUMO-RL
Gymnasium
Stable-Baselines3
PyTorch
DQN
Streamlit
```

| Technology | Why it is used |
|---|---|
| SUMO | Open-source microscopic traffic simulator — every vehicle is modelled individually |
| TraCI | Lets a Python program control and read a running SUMO simulation |
| PyYAML | Keeps all tunable values (demand, signal timings, duration) in configuration files |
| Pandas | Stores the measured results as CSV |
| Matplotlib | Produces the result graphs from those CSV files |

---

## Slide 5 — Current Architecture

```
        Traffic Demand
              |
              v
             SUMO
              |
              v
      4-Way Intersection
              |
              v
       Fixed-Time Signal
              |
              v
      Vehicle Simulation
              |
              v
            TraCI
              |
              v
           Metrics
              |
              v
             CSV
              |
              v
            Graphs
```

**This is the foundation for the later RL system.** The RL stage replaces exactly one
block — the fixed-time signal — with a learning agent. The network, the scenarios and
the measurement layer stay the same, which is what makes a fair Fixed-Time vs DQN
comparison possible later.

---

## Slide 6 — Current Implementation

Completed and verified in Stage 1:

- Project structure created (`config/`, `sumo/`, `simulation/`, `utils/`, `results/`, `tests/`, `docs/`, `presentation/`)
- SUMO environment configured and verified (SUMO 1.27.1, TraCI connection working)
- Four-way intersection network created — 4 approaches × 200 m, 2 lanes per direction, 16 signal-controlled links
- Fixed-time signal implemented — NS green → yellow → EW green → yellow, timings configurable in YAML
- Vehicles generated from real SUMO flow/route files
- **Moderate traffic scenario** created and executed (1400 veh/h)
- **Heavy traffic scenario** created and executed (3000 veh/h)
- TraCI metric collection implemented (vehicles, waiting time, queue length, throughput)
- Result CSV files generated from the actual runs
- Graphs generated automatically from those CSV files
- 24 automated tests written and passing

**Not implemented in this stage:** DQN, neural network, training loop, Gymnasium
environment, SUMO-RL wrapper, adaptive control, dashboard.

---

## Slide 7 — Live Simulation / Screenshots

Screenshots captured from the running simulation (`presentation/screenshots/`):

| # | File | Shows |
|---|---|---|
| 1 | `01_intersection_network.png` | The four-way intersection network in SUMO |
| 2 | `02_vehicles_approaching.png` | Vehicles approaching the intersection |
| 3 | `03_vehicles_waiting_red.png` | A queue held at a red signal |
| 4 | `04_vehicles_moving_green.png` | The same approach discharging on green |
| 5 | `05_moderate_traffic.png` | Moderate traffic scenario |
| 6 | `06_heavy_traffic.png` | Heavy traffic scenario |

**Live demonstration during the presentation:**

```bash
python simulation/run_simulation.py --scenario moderate --gui
python simulation/run_simulation.py --scenario heavy --gui
```

---

## Slide 8 — Initial Results

Measured with SUMO 1.27.1, fixed seed 42, 1800 s of simulated time per scenario.
Same network, same signal plan — only the demand differs.

| Metric | Moderate traffic | Heavy traffic |
|---|---|---|
| Total demand | 1400 veh/h | 3000 veh/h |
| Vehicles in simulation | **702** | **1445** |
| Average waiting time | **10.04 s / vehicle** | **51.90 s / vehicle** |
| Average queue length | **3.70 vehicles** | **40.39 vehicles** |
| Maximum queue length | 10 vehicles | 72 vehicles |
| Throughput (completed trips) | **693 vehicles** | **1351 vehicles** |
| Still waiting at the end | 9 | 94 |
| Teleports (gridlock indicator) | 0 | 0 |

**Reading of the result**

- Demand roughly doubles (1400 → 3000 veh/h).
- Average waiting time grows about **5×** (10.0 s → 51.9 s).
- Average queue grows about **11×** (3.7 → 40.4 vehicles).
- Throughput rises in absolute terms but no longer keeps up with demand: 94 vehicles are
  still waiting when the window closes, against 9 in the moderate case.

The fixed-time plan is adequate at moderate demand and clearly inadequate at heavy
demand. **This is the gap the adaptive controller is meant to close, and this table is
the baseline the DQN will be measured against.**

Graphs: `results/plots/01_average_waiting_time.png`,
`02_average_queue_length.png`, `03_throughput.png`,
`04_queue_length_over_time.png`, `05_summary_comparison.png`.

---

## Slide 9 — Current Limitations

**Stage 1 currently uses:**

- Fixed-time signal control only
- Simulated traffic (SUMO), not field-measured traffic
- A simplified single four-way intersection
- A limited number of scenarios (two demand levels)

**Stage 1 does not yet include:**

- Adaptive AI control
- A DQN or any learned policy
- Real-world traffic data
- Physical traffic-signal or hardware integration

**This is a planned progression, not a shortfall.** An adaptive controller can only be
evaluated against a measured baseline, and building that baseline correctly was the
objective of this stage.

---

## Slide 10 — Next Stage

```
Current:
    SUMO
      |
      v
Fixed-Time Controller
      |
      v
  Traffic Metrics


Next:
    SUMO-RL
      |
      v
   Gymnasium
      |
      v
State + Action + Reward
      |
      v
      DQN
      |
      v
Adaptive Signal Control
      |
      v
 Fixed-Time vs DQN
```

**Planned work for the next stage**

1. Wrap the existing scenario as a Gymnasium / SUMO-RL environment.
2. Define the state (queue length and waiting time per approach, current phase and its
   elapsed time).
3. Define the action set (keep the current phase, or switch to the next one).
4. Define the reward (a penalty proportional to accumulated waiting time / queue length).
5. Train a DQN agent with experience replay and a target network.
6. Evaluate the trained agent on the **same** network and the **same** two scenarios.
7. Compare Fixed-Time vs DQN using the **same** metrics shown on Slide 8.
