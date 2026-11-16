# IntelliSignal — Handoff

Handoff updated: 16 Nov 2026. Historical evidence: the four status notes written at the end of Stage 1 (19 Sep), Stage 2 (20 Sep), the dashboard (21 Sep) and Stage 3 (11 Oct 2026). Current Windows evidence: the user-reported test output, comparison CSVs under `verification/windows_20261116/`, and the user's dashboard walkthrough confirmation.

**Project location:** `C:\Users\Vishwajeet\OneDrive\Desktop\project\intelliSignal` (Windows).

> **Read this first.** Stages 1–3 are implemented and have now been verified on Windows. The existing `.venv-stage3` interpreter runs Python 3.12.13; installed package metadata shows SUMO/TraCI/sumolib 1.28.0 and PyTorch 2.14.1. The user reported **221 passed, 1 skipped in 166.72 s**, reproduced both scenario comparisons, and confirmed the dashboard checks all work. Earlier stage reports describe Linux runs with SUMO 1.27.1; preserve them as historical records.

### Windows verification — 16 Nov 2026

- **Tests:** user-reported result above. The skipped test was not identified in the supplied summary. `tests/test_dqn.py` has a process-cleanup test that skips without `pgrep`; this is a likely explanation, not a confirmed skip reason.
- **Comparisons:** existing `best.pt` models, seeds 42/43/44, 1800 seconds, including the untrained control. CSVs: `verification/windows_20261116/moderate/` and `verification/windows_20261116/heavy/`. Both seed-42 fixed-time baselines and the aggregate comparison results match the reference figures below.
- **Dashboard:** the user confirmed all checks worked after opening all six pages, running fixed-time and DQN for both scenarios, and checking playback/results/comparisons. This was a manual user check, not an automated browser run by the assistant.
- **Models:** no retraining was needed. The selected moderate and heavy best checkpoints are from episodes 72 and 95 of the original 100-episode training runs.
- **Output locations:** `--results-dir` isolates comparison CSVs; comparison plots still write to `results/plots/comparison/`.
- **Remaining:** clean screenshots, presentation updates, and optional longer heavy-scenario training.

---

## 1. Purpose

IntelliSignal is a traffic-signal control project (academic/presentation context: it has slides, speaking notes with Q&A, and a "viva" explainability goal). It compares:

- a **fixed-time** traffic signal (baseline), against
- a **DQN (Deep Q-Network) reinforcement-learning agent**,

on the same simulated four-way intersection, under two demand scenarios (moderate and heavy), measured with identical metrics. A Streamlit dashboard presents the simulation, the trained model, and the comparison.

## 2. Technologies

| Area | Technology |
|---|---|
| Traffic simulator | SUMO 1.27.1, driven through TraCI (`eclipse-sumo` pip package) |
| RL environment | Gymnasium (custom `gym.Env`, registered as `IntelliSignal-v0`) |
| SUMO-RL | `sumo_rl` used only in an adapter + one integration test (not as the base of the env) |
| Deep learning | PyTorch (`torch>=2.0`) — hand-written DQN, **no Stable-Baselines3** |
| Dashboard | Streamlit (`>=1.49`), Altair charts, inline SVG via `st.iframe` |
| Plots | Matplotlib |
| Config | YAML |
| Tests | pytest |

## 3. Folder structure

Reconstructed from the status notes; **verify against the real tree** (`tree /F` on Windows) — some minor files may be missing here.

```
intelliSignal/
├── config/
│   ├── moderate.yaml, heavy.yaml        # scenario configs (available_scenarios() lists config/*.yaml)
│   ├── rl/environment.yaml              # RL env settings (in a subdir ON PURPOSE, see §6)
│   └── dqn/training.yaml                # every DQN hyperparameter
├── sumo/
│   ├── network/intersection.*.xml       # hand-authored network (committed)
│   ├── routes/ additional/ config/      # GENERATED each run, git-ignored
├── utils/        config_loader.py, scenario_builder.py, metrics.py
├── simulation/   run_simulation.py, run_scenario.py, plot_results.py,
│                 controller.py (only thing the UI talks to), agent_simulation.py (fairness guarantee)
├── rl/           environment.py, state.py, action.py, reward.py, config.py,
│                 sumo_rl_adapter.py, run_environment.py
├── agents/       base_agent.py, fixed_time_agent.py, dqn_agent.py, dqn_network.py,
│                 replay_buffer.py, dqn_config.py, registry.py
├── training/     train_dqn.py, plot_training.py
├── evaluation/   evaluate_dqn.py, compare_controllers.py, plot_comparison.py
├── dashboard.py + dashboard/   theme, state, controls, components, charts
├── models/       dqn_<scenario>/*.pt (seed 42), dqn_<scenario>_seed<N>/ for other seeds
├── results/      csv/ (PROTECTED Stage 1 baseline), data/ (dashboard run output), graphs
├── tests/        test_simulation.py, test_rl_environment.py, test_dashboard.py, test_dqn.py
├── docs/         architecture.md, rl_environment.md, dashboard.md, dqn_training.md
├── presentation/ slides, capture_screenshots.py, dashboard_screenshots/ (01–11)
├── .streamlit/config.toml               # pinned light theme
├── README.md, requirements.txt, .gitignore
└── STAGE1_REPORT.md, STAGE2_REPORT.md, DASHBOARD_REPORT.md, STAGE3_REPORT.md
```

## 4. Working features (all verified in real SUMO runs, in the Linux container)

**Stage 1 — simulation and baseline**
- Four-way intersection: 4 × 200 m approaches, 2 lanes per direction, 16 signal-controlled links at junction `C`.
- Fixed-time signal (NS green → yellow → EW green → yellow), timings in YAML.
- Two scenarios: moderate (1400 veh/h) and heavy (3000 veh/h), 1800 s each, seed 42.
- TraCI metric collection, CSV results, Matplotlib graphs.
- Baseline (reproducible, identical on every rerun):

| Metric | Moderate | Heavy |
|---|---|---|
| Vehicles | 702 | 1445 |
| Avg waiting time | 10.04 s | 51.90 s |
| Avg queue | 3.70 | 40.39 |
| Max queue | 10 | 72 |
| Throughput | 693 | 1351 |
| Still in network | 9 | 94 |

**Stage 2 — RL environment** (`IntelliSignal-v0`, passes `check_env` with no warnings)
- Observation `Box(0,1,(10,),float32)` = [N,E,S,W queue][N,E,S,W waiting][phase][phase elapsed].
- Action `Discrete(2)`: 0 = keep, 1 = switch, with safety rules.
- Reward `-0.01 * (total_waiting + total_queue)`, averaged over the decision interval.
- `decision_interval=10 s`, `min_green=10 s`, `max_green=60 s`, `yellow=4 s`.

**Dashboard** (6 pages incl. DQN Model and Comparison)
- UI only talks to `SimulationController`; real signal colours decoded from recorded SUMO states with a time slider.
- Control modes: fixed-time and DQN (DQN enabled only when a model file and torch exist for the scenario).
- Driven end-to-end in real Chromium, including a real DQN run via the Run button.

**Stage 3 — DQN**
- Network `10 → 64 → 64 → 2` (4,994 params, matching the implementation and model metadata; the historical Stage 3 report says 5,058); gamma 0.9, lr 0.001, batch 64, Huber loss, replay 50k, learning_starts 500, target update every 500 gradient steps, epsilon 1.0 → 0.05 linearly over 50 % of the run.
- Training, evaluation, fixed-time-vs-DQN comparison, and comparison plots.
- Results (3 eval seeds; DQN vs fixed-time):

| | Moderate | Heavy |
|---|---|---|
| Avg waiting | 9.91 → 5.71 s (+42.3 %) | 52.60 → 47.91 s (+8.9 %) |
| Avg queue | 3.66 → 2.08 (+43.0 %) | 40.98 → 37.75 (+7.9 %) |
| Max queue | 9.7 → 7.0 (+27.6 %) | 71.3 → 69.7 (+2.3 %) |
| Throughput | 693 → 691 (−0.3 %) | 1351 → 1344 (−0.6 %) |

- Untrained control (random weights): moderate 10.91 s, heavy 88.21 s — worse than fixed-time, so the gain comes from learning, not from the 10 s decision interval.
- Fixed-time comparison at seed 42 reproduces the Stage 1 baseline exactly (asserted by a test).

**Tests:** latest Windows run, reported by the user: **221 passed, 1 skipped in 166.72 s**. Historical Linux run: **222 passed** (24 Stage 1 + 47 Stage 2 + 55 dashboard + 96 Stage 3). See the Windows verification note above for the unconfirmed skip reason.

## 5. Unfinished / planned features (NOT built)

- **Windows runtime verification is complete** for tests, comparisons, and the user-confirmed dashboard walkthrough. Fresh screenshot capture remains outstanding.
- **Heavy-scenario convergence:** at 100 episodes the heavy agent is still improving; moderate plateaued around episode 40. A longer run is untested.
- **Benchmarks against PPO/A2C or a tuned actuated controller** — listed as future scope only.
- **Clean screenshots:** Stage 1 screenshots were captured headlessly and the wide views crop the top of the northern approach. Re-run `python presentation/capture_screenshots.py` on Windows.
- Anything beyond a single intersection (multi-intersection, real-world data, pedestrians, emergency vehicles) is not planned in the notes; treat as out of scope unless Adesh says otherwise.

## 6. Known bugs, caveats and gotchas

No open functional bugs are recorded. The issues below are the real ones:

- **Use the verified `.venv-stage3` environment:** Python 3.12.13 with installed PyTorch 2.14.1. The earlier Python 3.14 compatibility warning is historical; this verification did not test Python 3.14.
- **`sumo_rl` raises ImportError unless `SUMO_HOME` is set.** `rl/sumo_rl_adapter.ensure_sumo_home()` derives it from the eclipse-sumo package, but setting it properly on Windows is cleaner. Stage 1 doesn't need it.
- **Dependency versions are not pinned** in `requirements.txt`. The historical Linux and current Windows environments differ; use the verified environment for reproduction. CPU execution is sufficient for this small network.
- **Do not run training and evaluation for the same scenario in parallel** — both rebuild `sumo/routes/<scenario>_rl.rou.xml`. Different scenarios are safe in parallel.
- **"Still in network" is misleading under heavy demand.** The DQN admits more vehicles (1462 vs 1445) because clearer approaches let SUMO insert held-back vehicles. Always read it next to the insertion count.
- **Waiting-time definitions differ:** the project averages over all vehicles that entered (heavy: 51.90 s); SUMO's own end-of-run statistic averages completed trips only (heavy: 35.55 s). Both correct; the project's is the honest one for a congested run.
- **Streamlit gotchas:** inline SVG is sanitised by both `st.markdown(unsafe_allow_html=True)` and `st.html()` → use `st.iframe(html, height=...)`. Use `width="stretch"` not `use_container_width` (needs streamlit>=1.49). `st.components.v1.html` is deprecated. AppTest reports Altair charts as `UnknownElement`, so don't assert on chart type.
- **Shell gotcha (Linux):** `pkill -f "streamlit run dashboard.py"` kills your own shell; use `fuser -k <port>/tcp`.
- SUMO prints a statistics block per episode; training silences it with `--duration-log.statistics false --no-step-log true` via `extra_sumo_args`.

## 7. Important design decisions (do not undo without reason)

1. **Own Gymnasium env, not a sumo-rl subclass.** sumo-rl's action space is "name the next phase"; the spec needs keep/switch plus a custom 10-dim observation and reward. A real `sumo_rl.SumoEnvironment` is built in `rl/sumo_rl_adapter.py` and exercised by a test as the integration proof.
2. **Decision interval is 10 s, not 5.** Yellow is 4 s; `rl/config.py` enforces `decision_interval > yellow + all_red`.
3. **State uses current waiting time, not accumulated** (accumulated would leak episode time). Reporting metric still uses accumulated.
4. **Normalisation divisors are measured, not guessed:** 50 (queue) and 1500 (waiting), shared across scenarios so one agent can be evaluated on both.
5. **Fairness comes from `max_green` (structural), not a reward term.** `fairness_weight` exists, is tested, defaults to 0.0.
6. **Generated vs committed files:** `sumo/routes|additional|config/` are generated from YAML on every run. The RL env writes under `<scenario>_rl`, the adapter under `<scenario>_sumorl`, so Stage 1 files are never overwritten (regression tests enforce this).
7. **`config/rl/environment.yaml` lives in a subdirectory** so `available_scenarios()` (which lists `config/*.yaml`) doesn't treat it as a third scenario.
8. **netconvert's default TLS program has `programID="0"`**, so the project's program is `programID="fixed"`, activated at t=0 by a WAUT switch in the additional file.
9. **Link layout at junction `C`:** 0–3 North, 4–7 East, 8–11 South, 12–15 West, each (right, through, through, left); signal state strings are 16 chars.
10. **`setRedYellowGreenState` moves the TL to SUMO's `online` program** and persists across steps (asserted by test).
11. **No Stable-Baselines3.** The DQN update is written out in full so it can be explained line by line. Only `torch>=2.0` was added.
12. **`select_action` is polymorphic:** a 10-value observation returns the env action (keep/switch); a `TrafficState` returns the phase (0 = NS, 1 = EW). The observation travels in `TrafficState.extra["observation"]`. `phase_action_to_env_action()` bridges the two conventions.
13. **DQN availability is read from the filesystem** (`control_modes()` checks `models/dqn_<scenario>/*.pt` + torch), never a boolean flag.
14. **No DQN fallback to fixed-time results.** `load_results(scenario, "dqn_adaptive")` returns None if absent. Separate files: `<scenario>_dqn_results.csv`, `..._decisions.csv`.
15. **`best.pt` follows a 5-episode moving average**, not one lucky episode. Seed 42 owns `models/dqn_<scenario>/`; other seeds go to `_seed<N>` directories.
16. **Keep using `--include-untrained`** (random-weight control); it is what makes the improvement claim defensible.
17. **`results/csv/` is protected.** Dashboard runs write to `results/data/`; the dashboard falls back to the Stage 1 baseline and labels it. A test asserts the baseline is never overwritten.
18. **Fairness guarantee:** `simulation/agent_simulation.py` runs both controllers on byte-identical SUMO inputs, measured by the Stage 1 `MetricsCollector`.
19. **UI never imports SUMO/TraCI/rl/agents** — it only calls `SimulationController`.
20. Stage 1/2 code was touched only through additive seams: `MetricsCollector(conn=)`, `env.step_callback`, `env.connection`, `env.extra_sumo_args`, module-level `episode_scenario_config()`, `TrafficState.observation()`.

## 8. Required services and environment variables

**External services:** none. No databases, APIs, cloud accounts or network calls at runtime. Everything runs locally.

**Required local software:**
- SUMO: current Windows packages are 1.28.0; the historical Linux setup used 1.27.1. Both reproduced the recorded comparisons.
- Python: current Windows environment is **3.12.13** in `.venv-stage3`.
- Chromium/browser only if re-capturing screenshots

**Environment variables:**
- `SUMO_HOME` — path to the SUMO installation. Required by `sumo_rl`; auto-derived from eclipse-sumo if unset, but set it explicitly on Windows.

No secrets or API keys are used by the project.

## 9. Setup and run commands

Use the existing verified Windows environment from the project root:

```powershell
.\.venv-stage3\Scripts\python.exe -m pytest -q
# Latest user-reported result: 221 passed, 1 skipped in 166.72 s
.\.venv-stage3\Scripts\python.exe -m streamlit run dashboard.py

# Completed verification commands (rerunning replaces these CSVs and shared plots):
.\.venv-stage3\Scripts\python.exe evaluation/compare_controllers.py --scenario moderate --seeds 42 43 44 --include-untrained --results-dir verification/windows_20261116/moderate
.\.venv-stage3\Scripts\python.exe evaluation/compare_controllers.py --scenario heavy --seeds 42 43 44 --include-untrained --results-dir verification/windows_20261116/heavy
```

The following are historical setup/reference commands; a new environment is not needed on this machine. Commands marked ✔ appear in the stage notes. Confirm other flags with `--help` before use.

```bash
# --- setup (Windows, from the project root) ---
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
set SUMO_HOME=<path to SUMO or eclipse-sumo package dir>   # PowerShell: $env:SUMO_HOME="..."

# --- verify ---
python -m pytest -q                                   # historical Linux: 222 passed

# --- Stage 1: baseline ---
python simulation/run_simulation.py --help            # confirm scenario flag names
python simulation/plot_results.py

# --- dashboard ---
streamlit run dashboard.py                            # ✔ entry point

# --- DQN (confirm exact invocation and flags with --help) ---
python training/train_dqn.py --help                   # ✔ has --episodes
python evaluation/evaluate_dqn.py --help
python evaluation/compare_controllers.py --help       # supports --include-untrained (✔ flag exists)
python evaluation/plot_comparison.py

# --- screenshots ---
python presentation/capture_screenshots.py            # ✔
```

Reference training run used for the reported results: 100 episodes per scenario, seeds 42 and 43, about 750–1500 s each on the Linux container.

## 10. Recommended next steps (in order)

Windows tests, baseline/comparison reproduction, model loading, and the manual dashboard walkthrough are complete; no retraining is needed to demonstrate the current project.

1. **Presentation prep:** capture clean screenshots and update slides/speaking notes with Stage 3 and Windows verification results. Report waiting/queue improvements alongside the small throughput decreases. The existing deck covers Stage 1 only.
2. **Improve the heavy-scenario result (optional experiment):** train more episodes into separate model/results directories, inspect convergence, and compare against the preserved current model using the same seeds and `--include-untrained`.
3. **Optional extensions:** benchmark against a tuned actuated controller and PPO/A2C; test generalisation (train on one scenario, evaluate on the other).

## 11. Working conventions inherited from earlier stages

- Never overwrite `results/csv/` or the scenario YAMLs; tests will fail if you do.
- Add features through new modules or additive seams rather than editing Stage 1/2 internals.
- Keep every claim traceable to a real SUMO run; avoid quoting numbers from memory.
- Stage reports (`STAGE1_REPORT.md` to `STAGE3_REPORT.md`, `DASHBOARD_REPORT.md`) hold the detailed evidence behind each claim above.
