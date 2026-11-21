# IntelliSignal — Streamlit Dashboard: Final Report

This is a historical dashboard report describing the dashboard before Stage 3 DQN integration. Its original claims below are preserved as a record of that implementation point; they do not describe the current repository. The DQN was implemented and integrated in Stage 3, and subsequent evaluation is documented in `STAGE3_REPORT.md`, `HANDOFF.md`, and `output/dqn_stage4_20261120/stage4_validation_report.pdf`.

> **Historical status at the time of this report:** the DQN had not yet been
> implemented. This status was superseded by Stage 3 and later retraining.

---

## 1. Files created

### Presentation layer

| File | Why |
|---|---|
| `dashboard.py` | The Streamlit entry point and the four pages. It talks only to `SimulationController`, so the DQN stage will not touch it. |
| `dashboard/__init__.py` | Package marker and an overview of the layer. |
| `dashboard/theme.py` | Palette, CSS and formatting helpers, so styling lives in one place. |
| `dashboard/state.py` | Session state. Holds the rule that a simulation runs only on a button press. |
| `dashboard/controls.py` | The sidebar. Builds the control-mode selector from the controller, so a new mode needs no edit here. |
| `dashboard/components.py` | Status lights, KPI cards, the intersection diagram, the signal readout, the agent-decision panel, error rendering. |
| `dashboard/charts.py` | The Altair charts, built only from the result CSV files. |
| `.streamlit/config.toml` | Pins the theme so the dashboard looks identical on any machine during a demonstration. |

### Decision layer

| File | Why |
|---|---|
| `agents/base_agent.py` | The `TrafficAgent` interface, the `TrafficState` object and the action space. This is the seam a DQN plugs into. |
| `agents/fixed_time_agent.py` | The Stage 1 fixed-time plan expressed through that interface — a real implementation, proving the interface works. |
| `agents/dqn_agent.py` | A declared, explicitly **not implemented** stub. Every method raises with an explanation. |
| `agents/__init__.py` | The agent registry and `create_agent()`. |

### Integration layer

| File | Why |
|---|---|
| `simulation/controller.py` | The only thing the user interface talks to: environment health, scenarios, control modes, running a simulation, loading results, and typed errors with hints. |

### Tests and documentation

| File | Why |
|---|---|
| `tests/test_dashboard.py` | 51 tests for the three layers above, including the Streamlit app driven through `AppTest`. |
| `docs/dashboard.md` | How the dashboard works, what is real versus schematic, and the exact DQN extension points. |
| `DASHBOARD_REPORT.md` | This file. |
| `presentation/dashboard_screenshots/` | Six screenshots of the dashboard actually running. |
| `results/data/` | Where dashboard runs write, so `results/csv/` is never overwritten. |

## 2. Files modified

| File | Change | Why |
|---|---|---|
| `README.md` | Added the dashboard to the status banner, the stage description, the technology list, the project structure, a "How to run" section with the seven demonstration steps, the test counts, the limitations and the next stage. | The README is the first thing a panel reads. |
| `requirements.txt` | Added `streamlit>=1.49`. | The dashboard uses the `width="stretch"` API that replaced the deprecated `use_container_width`. |

## 3. Files deliberately NOT modified

`config/moderate.yaml`, `config/heavy.yaml`, every file in `sumo/network/`,
`utils/config_loader.py`, `utils/scenario_builder.py`, `utils/metrics.py`,
`simulation/run_simulation.py`, `simulation/run_scenario.py`,
`simulation/plot_results.py`, everything in `rl/`, `tests/test_simulation.py`,
`tests/test_rl_environment.py`, and everything in `results/csv/`.

**No Stage 1 or Stage 2 code was changed to build the dashboard.** The
controller calls the existing `run_scenario()` exactly as the command line
does. This was possible because a full 1800 s run already takes only a few
seconds, so no progress-callback plumbing was needed.

---

## 4. Architecture

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

The rule that makes the next stage cheap: **nothing in `dashboard/` imports
SUMO, TraCI, `rl/` or an agent.** The UI asks the controller and renders the
answer.

### What the dashboard reuses rather than reimplements

| It needs | It calls |
|---|---|
| scenario settings | `utils.config_loader.load_config()` |
| a simulation | `simulation.run_simulation.run_scenario()` |
| metrics | `utils.metrics.MetricsCollector`, through that runner |
| results | the CSV files that run wrote |
| SUMO health | `sumolib` plus a real `sumo --version` call |

No metric is recalculated in the dashboard and no timing or demand value is
typed into the interface.

---

## 5. Commands

```bash
streamlit run dashboard.py                        # the dashboard

python simulation/run_scenario.py all              # both scenarios + graphs
python simulation/run_simulation.py --scenario heavy --gui
python rl/run_environment.py --scenario moderate   # the RL environment demo
python -m pytest -q                                # 122 tests
```

---

## 6. Testing results

`python -m pytest -q` → **122 passed, 0 failed** (115 s).

| Suite | Tests | Result |
|---|---|---|
| `tests/test_simulation.py` (Stage 1) | 24 | **24 passed** — unchanged |
| `tests/test_rl_environment.py` (Stage 2) | 47 | **47 passed** — unchanged |
| `tests/test_dashboard.py` (new) | 51 | **51 passed** |

What the 51 new tests cover, against the requested list:

| Requested | Covered by |
|---|---|
| dashboard startup | `test_dashboard_starts_without_an_exception`, `test_dashboard_shows_the_project_title` |
| scenario selection | `test_scenario_selection_switches_scenario` |
| simulation execution | `test_running_a_simulation_from_the_dashboard_produces_real_metrics`, `test_simulation_runs_and_returns_real_metrics` |
| result loading | `test_results_are_loaded_back_from_disk`, `test_results_fall_back_to_the_stage1_baseline` |
| missing files | `test_missing_results_return_none_rather_than_raising` |
| invalid configuration | `test_unknown_scenario_raises_a_friendly_error`, `test_running_an_unknown_scenario_is_refused` |
| SUMO unavailable | `test_sumo_unavailable_produces_an_explained_error`, `test_every_controller_error_carries_a_hint` |

Plus: every page renders; the DQN mode is listed, disabled and explained; no
simulation starts on a page refresh; the signal-state decoder is tested against
all three real SUMO states; every chart specification is validated; and a
regression test asserts the dashboard never writes to `results/csv/`.

### Stage 1 integrity, verified after all the work

| Check | Result |
|---|---|
| `results/csv/moderate_results.csv` | MD5 unchanged |
| `results/csv/heavy_results.csv` | MD5 unchanged |
| `results/csv/*_timeseries.csv` | MD5 unchanged |
| `config/moderate.yaml`, `config/heavy.yaml` | MD5 unchanged |
| `python simulation/run_simulation.py --scenario moderate` | reproduced 702 / 10.04 / 3.70 / 693 |
| `python simulation/run_simulation.py --scenario heavy` | reproduced 1445 / 51.90 / 40.39 / 1351 |
| `python rl/run_environment.py` | still runs |

---

## 7. End-to-end verification

The dashboard was not only unit-tested — it was **run and driven in a real
browser**. A Chromium session loaded `http://127.0.0.1:8599`, navigated every
page, set the duration to 300 s, pressed **Run Simulation** and read the
result back off the page:

```
run button enabled: True
RUN RESULT: Finished in 1.6 s of real time — average waiting time 8.17 s,
            average queue 3.15 vehicles, throughput 105 vehicles.
DQN warning present: True
run button disabled for DQN: True
```

Those numbers came from a SUMO process the dashboard started. The six
screenshots in `presentation/dashboard_screenshots/` are from that session.

| Screenshot | Shows |
|---|---|
| `01_dashboard.png` | KPI cards, status lights, intersection, scenario configuration |
| `02_simulation_page.png` | The Simulation page before a run |
| `03_simulation_after_run.png` | The measured result and the recorded signal-state scrubber |
| `04_results.png` | The three bar charts, queue over time, vehicles over time |
| `05_about_project.png` | Implementation status, architecture, fixed-time baseline |
| `06_dqn_mode_disabled.png` | The DQN mode selected, disabled and explained |

---

## 8. How to demonstrate it to the panel

A ten-minute run-through that follows the pipeline:

1. **Open it.** `streamlit run dashboard.py`. Point at the three status
   lights — "SUMO 1.27.1 is really installed, TraCI really imports, two
   scenarios really exist. The dashboard checked; it did not assume."
2. **Project title and problem.** The header states what the project is.
3. **The intersection.** The diagram on the Dashboard page: four approaches,
   two lanes each way, one signalised junction. Say plainly that this is a
   schematic and the real SUMO screenshots are on the Results page.
4. **Scenario configuration.** Point out that the demand and signal timings on
   screen are read from `config/moderate.yaml`, not typed into the dashboard.
   Open the YAML if challenged.
5. **Run a simulation.** Sidebar → Moderate → Run Simulation. It finishes in a
   few seconds. "That was a real SUMO process, driven through TraCI."
6. **Traffic metrics.** The four KPI cards. Every one came from the CSV that
   run just wrote — the file path is printed underneath.
7. **The signal.** Simulation page → drag the time slider. The NS/EW colours
   and the raw 16-character SUMO state are what TraCI recorded at that second.
   This is the best moment to explain how TraCI communicates with SUMO.
8. **Graphs.** Results page: waiting time, queue length, throughput, and queue
   over time. Point at the saw-tooth — "that is the signal cycle; queues build
   on red and discharge on green."
9. **Scenario comparison.** Switch to Heavy and compare. Roughly double the
   demand gives about five times the waiting time and eleven times the queue.
   "That is the gap the adaptive controller is meant to close."
10. **The fixed-time baseline and the future DQN.** About Project page: the
    cycle timeline, then select **DQN Adaptive** in the sidebar. It is
    disabled and says it is not implemented. Say so out loud — the baseline is
    built and measured, and the DQN is the next stage.

Two things worth having ready:

- If asked *"are these numbers real?"* — press Run again with the same
  duration and seed and show that the numbers reproduce, then open
  `results/data/moderate_results.csv`.
- If asked *"where is the AI?"* — the honest answer is that Stage 2 built the
  reinforcement-learning environment a DQN will use (`rl/`), and Stage 3
  trains the agent. Show `agents/dqn_agent.py` — it raises
  `AgentNotImplementedError` by design.

---

## 9. Exact extension points for the DQN stage

Four changes, none of them in `dashboard/`:

1. **`agents/dqn_agent.py`** — implement `__init__` (load trained weights),
   `select_action` (run the network, return the higher-valued action) and
   `explain` (report the Q-values). Set `DQN_IMPLEMENTED = True`.
2. **`agents/__init__.py`** — nothing. `DQNAgent` is already registered.
3. **`simulation/controller.py`** — `control_modes()` already reads
   `DQNAgent.available`, so the mode becomes selectable by itself. Add a branch
   in `run()` that drives `rl.IntelliSignalEnv` with the agent instead of
   calling the fixed-time runner.
4. **`dashboard/`** — nothing. The sidebar lists the mode already, the Run
   button enables itself, and `agent_decision_panel()` already renders whatever
   `explain()` returns, including Q-values, in the "DQN Decision" panel on the
   Simulation page.

The bridge between the two action conventions is already written and tested:
`agents/base_agent.phase_action_to_env_action()` converts "give phase X green"
(how an agent reasons) into the RL environment's keep/switch action (how the
signal is safely changed).

---

## 10. Problems encountered, and how they were resolved

| Problem | Cause | Fix |
|---|---|---|
| The intersection diagram rendered as an empty grey box | Streamlit sanitises inline SVG — both `st.markdown(unsafe_allow_html=True)` and `st.html()` strip it | Render it through `st.iframe`, falling back to `st.components.v1.html` on older versions. Verified in a real browser. |
| `'function' object has no attribute 'clear'` after a run | `load_comparison.clear()` was called on a function that is not cached | Removed the call; the comparison is cheap and always read fresh |
| Deprecation warnings on every chart and table | `use_container_width` is deprecated in current Streamlit | Switched to `width="stretch"` and pinned `streamlit>=1.49` |
| `st.components.v1.html` deprecation warning | Superseded by `st.iframe` | Prefer `st.iframe`, keep the old call as a fallback |
| Raw `**not**` appearing in a caption | Markdown syntax inside a raw HTML block is not processed | Used `<strong>` in the HTML caption |
| Comparing a 300 s run against an 1800 s run looked like a real difference | The dashboard falls back to the Stage 1 baseline per scenario, so the two can have different lengths | The comparison now warns when the simulated times differ |
| Direction labels sat on the dark road and were unreadable | First SVG layout | Labels moved onto the grass, stop lines and per-approach signal heads added |

---

## 11. Assumptions made

1. **The dashboard runs on the same machine as the project.** It reads the
   project's own config and result files directly rather than through an API.
2. **`results/csv/` is the protected Stage 1 baseline.** Dashboard runs go to
   `results/data/`. This was a judgement call: the brief said not to overwrite
   important results without warning, and those files are quoted in the Stage 1
   and Stage 2 reports.
3. **When a scenario has no dashboard run, the Stage 1 file is shown and
   labelled "Stage 1 baseline".** This means the dashboard has real numbers on
   first open rather than an empty screen.
4. **sumo-gui is offered as a command, not launched from the web server.**
   Starting a GUI process from Streamlit is unreliable, and a failed launch
   must never take the dashboard down. The exact command is shown for copying.
5. **The theme is pinned to light.** A demonstration should look the same on
   any machine rather than following the operating system's dark mode.
6. **The intersection diagram is a schematic.** The road layout matches the
   modelled network and the signal colours are real when recorded data is
   supplied, but the vehicle marks are indicative, and the caption says so
   everywhere it appears.
7. **`dashboard.py` and the `dashboard/` package coexist.** Python resolves the
   package first, and Streamlit executes the script directly, so both work.
   This was verified rather than assumed.

---

## 12. Known limitations

- The dashboard runs a simulation to completion and then displays it. It does
  not stream a live simulation frame by frame into the browser.
- The signal-state scrubber replays recorded data; it is not a live feed.
- Only the fixed-time control mode can be executed, because it is the only one
  implemented.
- The dashboard was built and verified in the Linux workspace attached to this
  session and in a headless Chromium browser. **It has not been run on the
  Windows Python 3.14 installation**, where `pip install -r requirements.txt`
  is needed to add Streamlit.
- `results/data/` is delivered empty, so the dashboard opens on the recorded
  Stage 1 baseline.

---

## 13. Status summary

| Component | Status |
|---|---|
| Stage 1 — SUMO, fixed-time signal, scenarios, metrics, CSV, graphs | Implemented, unchanged, verified |
| Stage 2 — Gymnasium RL environment | Implemented, unchanged, verified |
| Agent interface + fixed-time controller | Implemented |
| Simulation controller | Implemented |
| Streamlit dashboard | Implemented and verified in a browser |
| **DQN agent, training, Fixed-Time vs DQN comparison** | Implemented in Stage 3; see the current Stage 3 and Stage 4 reports |
