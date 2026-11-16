# IntelliSignal — The Streamlit Dashboard

> The dashboard is the **presentation layer**. It shows what the project has
> actually built and measured. It does not contain a traffic model, a metric
> calculation or a controller of its own, and it does not contain a DQN,
> because no DQN exists yet.

```bash
streamlit run dashboard.py
```

---

## 1. Why a dashboard, and why it is not just a UI

A command-line project is hard to show to a panel. The dashboard makes the
same pipeline visible and interactive: pick a scenario, press a button, watch
a real SUMO simulation run, and read the metrics that simulation produced.

What keeps it honest is that it reuses the existing code rather than
reimplementing it:

| The dashboard needs | It calls | It does **not** |
|---|---|---|
| scenario settings | `utils.config_loader.load_config()` | hard-code any timing or demand |
| a simulation | `simulation.run_simulation.run_scenario()` | run SUMO its own way |
| metrics | `utils.metrics.MetricsCollector` (via the runner) | recompute waiting time or queues |
| results | the CSV files written by that run | invent or cache numbers |
| SUMO health | `sumolib` + a real `sumo --version` call | assume SUMO is installed |

Every number on every page came out of a CSV that a SUMO run wrote.

---

## 2. Architecture

```
                USER
                 |
                 v
          STREAMLIT DASHBOARD          dashboard.py + dashboard/
                 |
                 v
        SIMULATION CONTROLLER          simulation/controller.py
                 |
        +--------+--------+
        |                 |
        v                 v
 Fixed-Time          Future DQN        agents/
 Controller            Agent
        |                 |
        +--------+--------+
                 v
                TraCI
                 v
                SUMO                   sumo/
                 v
         Traffic Simulation
                 v
             Metrics                   utils/metrics.py
                 v
              Results                  results/
                 v
          Streamlit Charts             dashboard/charts.py
```

**The one rule that makes the DQN stage easy:** `dashboard.py` and everything
in `dashboard/` never import SUMO, TraCI, `rl/` or an agent. They talk to
`SimulationController` and render what it returns. Adding a DQN therefore
changes the back end, not the interface.

### Files

| File | Responsibility |
|---|---|
| `dashboard.py` | Entry point; the four pages |
| `dashboard/theme.py` | Palette, CSS, formatting helpers |
| `dashboard/state.py` | Session state; guarantees a run happens only on a button press |
| `dashboard/controls.py` | The sidebar: navigation, scenario, control mode, duration, Run |
| `dashboard/components.py` | Status lights, KPI cards, intersection diagram, signal readout |
| `dashboard/charts.py` | The Altair charts |
| `simulation/controller.py` | The only thing the UI talks to |
| `agents/base_agent.py` | `TrafficAgent` interface, `TrafficState`, the action space |
| `agents/fixed_time_agent.py` | The Stage 1 controller behind that interface |
| `agents/dqn_agent.py` | Declared, **not implemented** |

---

## 3. The pages

### Dashboard
Headline KPIs for the selected scenario, the intersection diagram, the scenario
configuration read from YAML, and the scenario comparison.

### Simulation
Where a run is started. After a run it shows the measured result and a
**signal-state scrubber**: a slider over the recorded time series, showing the
signal colours TraCI actually recorded at that second, the queue at that
second, and the fixed-time controller's reasoning for that moment.

### Results
The three required bar charts (waiting time, queue length, throughput), queue
length over time, vehicles over time, the comparison table, and the real SUMO
screenshots from `presentation/screenshots/`.

### DQN Model

Everything that has actually been trained for the selected scenario, plus the
recorded training curves. With no model: the reason and the command to fix it.

### Comparison

Fixed-Time vs DQN with improvement percentages — from the multi-seed
experiment on disk, and from runs started in the dashboard.

### About Project
Implementation status, the architecture diagram, the fixed-time baseline with
its cycle drawn as a timeline, the planned DQN stage, the detected environment,
and the commands to run everything.

---

## 4. Where the results live

| Directory | Written by | Used for |
|---|---|---|
| `results/csv/` | the command line (`run_simulation.py`, `run_scenario.py`) | the recorded Stage 1 baseline |
| `results/data/` | the dashboard's Run button | dashboard runs |
| `results/plots/` | `simulation/plot_results.py` | the static Matplotlib graphs |

**The dashboard never writes to `results/csv/`.** That folder holds the Stage 1
numbers quoted in the reports, and overwriting them silently would be a way to
lose a result you cannot get back. A test asserts this
(`test_dashboard_never_writes_to_the_stage1_results`).

When a scenario has no dashboard run yet, the dashboard falls back to the
Stage 1 file and labels the KPI cards "Stage 1 baseline", so the pages have
real numbers before anything has been run.

### A comparison must be like-for-like

If one scenario has been run for 300 s and the other for 1800 s, the
comparison table shows a warning. The numbers are all real, but they are not
comparable, and the dashboard says so rather than letting the bars imply
otherwise.

---

## 5. What is real, and what is a diagram

Being able to answer this in a viva matters more than the pixels.

| Element | Status |
|---|---|
| KPI values | **Real** — read from the CSV a SUMO run wrote |
| Charts | **Real** — built from those CSV files |
| Status lights | **Real** — `sumo --version` is executed, `traci` is imported, the network file is checked |
| Signal colours on the Simulation page | **Real** — decoded from the 16-character SUMO signal states TraCI recorded each second |
| Queue and vehicle counts on the slider | **Real** — from the recorded time series |
| Scenario settings shown | **Real** — read from the YAML files |
| SUMO screenshots on the Results page | **Real** — captured by sumo-gui |
| The intersection diagram | **A schematic.** The road layout matches the modelled network and the signal colours are real when recorded data is supplied, but the vehicle marks are indicative. Every place it appears says so. |
| DQN results | **None exist.** The mode is listed, disabled and explained. |

---

## 6. How the signal state is decoded

SUMO describes a traffic light as one character per controlled link. The
IntelliSignal junction has 16, four per approach:

```
index   0  1  2  3   4  5  6  7   8  9 10 11  12 13 14 15
        \----------/  \----------/  \---------/  \---------/
           NORTH          EAST         SOUTH        WEST

state   G  G  G  g   r  r  r  r   G  G  G  g   r  r  r  r
        = North-South green, East-West red
```

`G` protected green, `g` permissive green (the left turn gives way), `y`
yellow, `r` red. `dashboard/components.decode_signal_state()` turns that string
into a colour per approach, and `direction_states()` collapses it to NS and EW.
This is tested against all three real states the controller produces.

---

## 7. Error handling

The dashboard is built to explain problems rather than show a traceback.
`simulation/controller.py` raises typed errors, each carrying a short hint, and
the dashboard renders both:

| Situation | What the user sees |
|---|---|
| SUMO not installed / `SUMO_HOME` not set | "SUMO is not available. Please verify that SUMO is installed and SUMO_HOME is configured correctly." plus install guidance |
| Network file missing | A message naming `python sumo/network/build_network.py` |
| Unknown or missing scenario | The list of scenarios that do exist |
| DQN mode selected | The mode is disabled, and the reason is stated |
| No results yet | "Run a scenario from the Simulation page" |
| Result file unreadable or empty | A message naming the file |
| SUMO fails to start (port or process conflict) | "SUMO could not be started" plus a suggestion to check for a running SUMO process |
| Anything unexpected | A plain sentence and a pointer to the terminal — never a raw traceback |

---

## 8. Performance

- A simulation runs **only** when the Run button is pressed. Changing pages,
  moving the slider or switching scenario never starts one. A test asserts this
  (`test_no_simulation_runs_until_the_button_is_pressed`).
- The environment probe (which shells out to `sumo --version`) is cached for
  30 seconds, so it does not run on every click but still notices a fixed
  installation.
- The controller object is created once per session with `@st.cache_resource`.
- A full 1800 s run takes roughly 4 s (moderate) to 10 s (heavy) of real time,
  so a live demonstration does not stall.

---

## 9. The DQN in the dashboard (Stage 3)

The prediction made when this dashboard was written held: the DQN was a
back-end change. `dashboard/` needed no new widget framework and the four
original pages still work the same way. What was added:

### How the DQN mode becomes available

It is **not** a boolean somebody flipped. Every time the sidebar is drawn,
`SimulationController.control_modes(scenario)` asks two questions:

1. Is PyTorch installed?
2. Does a trained checkpoint exist in `models/dqn_<scenario>/`?

Only if both are true is the mode selectable. Otherwise it is listed, disabled
and explained:

```
DQN unavailable — train a model first.

The DQN is implemented, but no trained checkpoint exists for the heavy
scenario. A network with random weights is not a controller, so this mode
stays disabled rather than producing meaningless decisions.

Train one with:
    python training/train_dqn.py --scenario heavy --episodes 20 --seed 42
```

Because the question is asked per scenario, switching the scenario selector
from Moderate to Heavy can disable the mode on its own — which is correct, and
is covered by a test.

### What a DQN run actually does

`SimulationController.run(mode="dqn_adaptive")` loads the selected checkpoint,
sets epsilon to 0, and runs a real SUMO simulation through
`simulation/agent_simulation.py`, which drives the Stage 2 environment with
the network and measures it with the Stage 1 metrics collector. There is no
replay of a stored result and no fall back to the fixed-time runner.

### Separate files, so a label can never lie

| Run | Written to |
|---|---|
| Fixed-time, from the dashboard | `results/data/<scenario>_results.csv` |
| DQN, from the dashboard | `results/data/<scenario>_dqn_results.csv` |
| DQN decision log (per-decision Q-values) | `results/data/<scenario>_dqn_decisions.csv` |
| The recorded Stage 1 baseline | `results/csv/` — read only, never written |

`load_results(scenario, mode)` has a baseline fallback for fixed-time, which
it labels "Stage 1 baseline". For the DQN there is **deliberately no
fallback**: with no DQN run it returns nothing and the page says "No DQN run
for this scenario yet". A dashboard that showed fixed-time numbers when asked
for DQN numbers would be lying, and a test asserts it does not.

### New sidebar controls, shown only in DQN mode

* **Trained Model** — which checkpoint to run (`best.pt`, `final.pt`, or any
  `checkpoint_ep*.pt`), listed from what is actually on disk.
* **Random Seed** — the SUMO seed. The same seed gives the same traffic, which
  is how a DQN run and a fixed-time run can be compared fairly.

Neither appears in fixed-time mode, where the seed comes from the scenario
YAML and there is no model to choose.

### Two new pages

**DQN Model.** Model status, checkpoint path, scenario, architecture,
parameter count, episodes trained, the hyperparameters actually used, the
training date, the list of checkpoints found, a diagram of how one decision is
made, and the recorded training curves (return, loss, epsilon, queue, waiting
time, throughput per episode) read from `results/training/`. With no model it
shows the reason and the command that trains one, and invents nothing.

**Comparison.** Fixed-Time vs DQN with improvement percentages, from two
sources kept apart:

* the multi-seed command-line experiment in `results/evaluation/` — mean ±
  sample standard deviation, bar charts with error bars, queue length over
  time for both controllers on the same seed, and an expander listing every
  individual run so no seed is hidden;
* the last fixed-time and DQN runs started in the dashboard itself, which
  refuses to show a percentage when the two runs covered different simulated
  times.

A positive percentage always means the DQN did better and a negative one
always means it did worse. `None` is rendered as "n/a" rather than as an
invented number.

### Q-values on the Simulation page

In DQN mode the time slider shows the **recorded** Q-values at that second,
read from the decision log. They are not recomputed from a reconstructed
state: the 10-value observation cannot be rebuilt from the saved time series,
so rather than invent one, the page shows what was recorded or says it has
nothing to show.

### What did not change

`dashboard/theme.py`, the status lights, the KPI cards, the intersection
schematic and the signal decoder are all untouched. `agent_decision_panel()`
gained a Q-value block; `charts.py` gained three Stage 3 chart builders that
follow the same rules as the existing ones.

---

## 10. Design notes

The two scenario colours (`#3b6fb6` moderate, `#d98324` heavy) are fixed per
scenario and assigned by identity, never by rank, so a scenario keeps its
colour on every chart and page. The pair was checked for colour-vision
separation (worst case ΔE 25.0 for protanopia, 30.6 for normal vision against
white), and every bar carries a printed value, so no chart depends on colour
alone. The theme is pinned to light in `.streamlit/config.toml` so the
dashboard looks the same on any machine during a demonstration.
