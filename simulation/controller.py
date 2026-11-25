"""
IntelliSignal - Stages 1-3
simulation/controller.py

The service layer of the Stage 1-3 dashboard. **Historical.**

> The dashboard was rebuilt on 25 Nov 2026 around `simulation/engine.py`
> and no longer imports anything in this file. It is kept because the
> Stage 1-3 tests run against it, and those tests are the evidence that
> the published Stage 1-3 numbers still reproduce. Nothing here is on
> the product's code path; new work belongs in `simulation/engine.py`.

What follows describes the arrangement as it was.

The layer between the user interface and the simulation.

                USER
                 |
                 v
          STREAMLIT DASHBOARD          dashboard/
                 |
                 v
        SIMULATION CONTROLLER          <-- THIS FILE
                 |
        +--------+--------+
        |                 |
        v                 v
 Fixed-Time           DQN Agent
 Controller          (Stage 3)          agents/
        |                 |
        +--------+--------+
                 v
                TraCI
                 v
                SUMO
                 v
             Metrics                    utils/metrics.py
                 v
              Results                   results/
                 v
          Streamlit Charts

Why this file exists
--------------------
The dashboard must never import SUMO, TraCI or an agent directly. It asks this
controller for scenarios, control modes, environment health, a simulation run
and the results, and it renders whatever comes back. That is what made the
DQN a drop-in: Stage 3 added a branch in `run()` and a model check in
`control_modes()`, and the user interface code for the existing pages did not
change at all.

Nothing in Stage 1 or Stage 2 was modified to build this. The controller calls
the existing `simulation.run_simulation.run_scenario()` for fixed-time and the
Stage 3 `simulation.agent_simulation.run_agent_episode()` for the DQN, which
itself drives the existing Stage 2 environment.

Results policy
--------------
The Stage 1 command line writes to `results/csv/`. Those files are the
recorded Stage 1 baseline and the dashboard must not quietly overwrite them,
so simulations started from the dashboard write to `results/data/` instead.
When a scenario has no dashboard result yet, the controller falls back to the
Stage 1 file and says so, which means the dashboard has real numbers to show
before anything has been run.

DQN runs are kept in separate files again - `<scenario>_dqn_results.csv` -
for the same reason: a DQN run must never be mistaken for the fixed-time
baseline, in either direction. There is no fallback for DQN results. If no
DQN run exists, the dashboard is told so; it is never shown fixed-time
numbers under a DQN heading.

Honesty rules enforced here, not in the user interface
------------------------------------------------------
  * The DQN control mode is available only when PyTorch is installed AND a
    trained checkpoint exists for the selected scenario. It is not a boolean
    somebody can flip.
  * `run(mode="dqn_adaptive")` loads real weights and runs a real SUMO
    simulation. If the model is missing it raises; it never silently runs the
    fixed-time controller and labels the result "DQN".
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.config_loader import (ConfigError, available_scenarios,  # noqa: E402
                                 cycle_time, load_config, total_demand)

#: Where the Stage 1 command line writes. Read-only as far as the dashboard
#: is concerned.
BASELINE_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "csv")

#: Where simulations started from the dashboard write.
DASHBOARD_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "data")

PLOTS_DIR = os.path.join(PROJECT_ROOT, "results", "plots")
SCREENSHOT_DIR = os.path.join(PROJECT_ROOT, "presentation", "screenshots")

#: Stage 3 output, written by training/ and evaluation/ on the command line.
TRAINING_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "training")
EVALUATION_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "evaluation")

#: Control-mode keys. The dashboard only ever uses these constants.
MODE_FIXED_TIME = "fixed_time"
MODE_DQN = "dqn_adaptive"
MODE_HYBRID = "hybrid"

#: The pseudo-scenario for user-defined traffic. It has no YAML file: the
#: demand comes from a resolved workload instead, and everything else is
#: inherited from the workload's base scenario.
SCENARIO_CUSTOM = "custom"

#: mode key -> the controller label written into every result file
RESULT_CONTROLLER = {MODE_FIXED_TIME: "fixed_time", MODE_DQN: "dqn",
                     MODE_HYBRID: "hybrid"}


# ===========================================================================
# Errors - every one of these is caught by the dashboard and shown as text
# ===========================================================================
class SimulationError(RuntimeError):
    """Base class for every problem the dashboard has to explain to a user."""

    #: Short, non-technical guidance shown under the message.
    hint: str = ""

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        if hint:
            self.hint = hint


class SumoUnavailableError(SimulationError):
    hint = ("Install SUMO from https://sumo.dlr.de/docs/Downloads.php and set "
            "the SUMO_HOME environment variable, or run "
            "`pip install -r requirements.txt`, which installs the packaged "
            "SUMO binaries.")


class ScenarioNotFoundError(SimulationError):
    hint = "Pick one of the scenarios listed in the sidebar."


class ControlModeUnavailableError(SimulationError):
    hint = "Switch the control mode back to Fixed-Time."


class ResultsMissingError(SimulationError):
    hint = "Run the scenario once from the Simulation page to produce results."


class ModelMissingError(SimulationError):
    """No trained DQN checkpoint for the requested scenario."""

    hint = ("Train one on the command line, for example:  "
            "python training/train_dqn.py --scenario moderate --episodes 20 "
            "--seed 42")


# ===========================================================================
# Value objects
# ===========================================================================
@dataclass(frozen=True)
class ControlMode:
    """One way of driving the traffic signal."""

    key: str
    label: str
    description: str
    available: bool
    agent_name: Optional[str] = None
    unavailable_reason: str = ""
    badge: str = ""


@dataclass
class EnvironmentStatus:
    """Detected health of everything the simulation needs.

    Every field is measured, never assumed. The dashboard shows these as the
    status indicators, so they must be honest.
    """

    python_version: str = ""
    sumo_available: bool = False
    sumo_version: str = ""
    sumo_binary: str = ""
    sumo_home: str = ""
    traci_available: bool = False
    traci_version: str = ""
    network_file_present: bool = False
    scenarios: List[str] = field(default_factory=list)
    gymnasium_available: bool = False
    sumo_rl_available: bool = False
    torch_available: bool = False
    torch_version: str = ""
    trained_models: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def simulation_ready(self) -> bool:
        return (self.sumo_available and self.traci_available
                and self.network_file_present and bool(self.scenarios))

    def indicators(self) -> List[Dict[str, Any]]:
        """The three status lights the dashboard shows at the top."""
        return [
            {"label": "SUMO Connected", "ok": self.sumo_available,
             "detail": self.sumo_version or "not found"},
            {"label": "TraCI Available", "ok": self.traci_available,
             "detail": self.traci_version or "not found"},
            {"label": "Simulation Ready", "ok": self.simulation_ready,
             "detail": "{} scenarios".format(len(self.scenarios))
                       if self.scenarios else "no scenarios"},
        ]


@dataclass
class SimulationRun:
    """The outcome of one simulation, or one set of loaded results."""

    scenario: str
    mode: str
    metrics: Dict[str, Any]
    source: str                      # "dashboard run" | "Stage 1 baseline"
    csv_path: str = ""
    timeseries_path: str = ""
    finished_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    def metric(self, key: str, default: Any = None) -> Any:
        return self.metrics.get(key, default)


# ===========================================================================
# The controller
# ===========================================================================
class SimulationController:
    """Everything the dashboard is allowed to ask the simulation to do."""

    def __init__(self, results_dir: str = DASHBOARD_RESULTS_DIR):
        self.results_dir = results_dir

    # ------------------------------------------------------------------
    # Environment health
    # ------------------------------------------------------------------
    def environment_status(self) -> EnvironmentStatus:
        """Detect what is actually installed. Nothing here is hard-coded."""
        status = EnvironmentStatus()
        status.python_version = "{}.{}.{}".format(*sys.version_info[:3])
        status.sumo_home = os.environ.get("SUMO_HOME", "")

        # --- SUMO -----------------------------------------------------
        try:
            import sumolib
            binary = sumolib.checkBinary("sumo")
            status.sumo_binary = binary or ""
            if binary:
                import subprocess
                completed = subprocess.run([binary, "--version"],
                                           capture_output=True, text=True,
                                           timeout=20)
                first_line = (completed.stdout or completed.stderr).strip().split("\n")[0]
                status.sumo_version = first_line.replace("Eclipse SUMO ", "").strip()
                status.sumo_available = completed.returncode == 0
        except Exception as exc:
            status.errors.append("SUMO: {}".format(exc))

        # --- TraCI ----------------------------------------------------
        try:
            import traci
            status.traci_available = True
            try:
                from importlib import metadata
                status.traci_version = metadata.version("traci")
            except Exception:
                status.traci_version = "installed"
        except Exception as exc:
            status.errors.append("TraCI: {}".format(exc))

        # --- project files --------------------------------------------
        network = os.path.join(PROJECT_ROOT, "sumo", "network",
                               "intersection.net.xml")
        status.network_file_present = os.path.isfile(network)
        if not status.network_file_present:
            status.errors.append(
                "Network file missing: sumo/network/intersection.net.xml")

        try:
            status.scenarios = available_scenarios()
        except Exception as exc:
            status.errors.append("Scenarios: {}".format(exc))

        # --- optional RL stack (Stage 2) ------------------------------
        try:
            import gymnasium  # noqa: F401
            status.gymnasium_available = True
        except Exception:
            pass
        try:
            from rl.sumo_rl_adapter import sumo_rl_available
            status.sumo_rl_available = bool(sumo_rl_available())
        except Exception:
            pass

        # --- Stage 3 learning stack -----------------------------------
        try:
            import torch                                    # noqa: F401
            status.torch_available = True
            status.torch_version = torch.__version__
        except Exception:
            pass

        try:
            status.trained_models = [
                scenario for scenario in status.scenarios
                if self.has_trained_model(scenario)]
        except Exception as exc:
            status.errors.append("Models: {}".format(exc))

        return status

    # ------------------------------------------------------------------
    # Scenarios
    # ------------------------------------------------------------------
    def available_scenarios(self) -> List[str]:
        """Scenario names, lightest demand first."""
        preferred = ["moderate", "heavy"]
        names = available_scenarios()
        return sorted(names, key=lambda n: (preferred.index(n)
                                            if n in preferred else 99, n))

    def scenario_config(self, scenario: str) -> Dict[str, Any]:
        try:
            return load_config(scenario)
        except ConfigError as exc:
            raise ScenarioNotFoundError(str(exc)) from exc

    def scenario_summary(self, scenario: str) -> Dict[str, Any]:
        """Everything the dashboard shows about a scenario, read from YAML.

        No timing or demand number is hard-coded in the user interface.
        """
        config = self.scenario_config(scenario)
        signal = config["signal"]
        simulation = config["simulation"]
        demand = config["demand"]
        return {
            "name": config["scenario"]["name"],
            "description": config["scenario"].get("description", ""),
            "duration": float(simulation["duration"]),
            "step_length": float(simulation["step_length"]),
            "seed": int(simulation["seed"]),
            "ns_green": float(signal["ns_green"]),
            "ew_green": float(signal["ew_green"]),
            "yellow": float(signal["yellow"]),
            "all_red": float(signal["all_red"]),
            "cycle_time": cycle_time(config),
            "total_demand": total_demand(config),
            "demand_per_approach": {k: demand[k]
                                    for k in ("north", "south", "east", "west")},
            "turn_ratios": dict(demand["turn_ratios"]),
            "config_path": config.get("_path", ""),
        }

    # ------------------------------------------------------------------
    # Control modes
    # ------------------------------------------------------------------
    def control_modes(self, scenario: Optional[str] = None) -> List[ControlMode]:
        """Every way of driving the signal, and whether it actually works.

        The DQN entry is available only when PyTorch is installed AND a
        trained checkpoint exists for `scenario`. Both conditions are
        measured here, every time the sidebar is drawn - there is no flag to
        flip and no cached "yes". Without a scenario the question is asked for
        any scenario that has a model.
        """
        from agents import AGENT_REGISTRY

        fixed = AGENT_REGISTRY["fixed_time"]
        dqn = AGENT_REGISTRY["dqn"]

        torch_ok = self.torch_available()
        model_path = self.model_path(scenario) if scenario else None
        has_model = (bool(model_path) if scenario
                     else bool(self.scenarios_with_models()))
        available = bool(torch_ok and has_model)

        if not torch_ok:
            reason = (
                "PyTorch is not installed, so the DQN controller cannot run.\n\n"
                "Install the project requirements:\n"
                "    pip install -r requirements.txt")
        elif not has_model:
            reason = (
                "DQN unavailable — train a model first.\n\n"
                "The DQN is implemented, but no trained checkpoint exists{}. "
                "A network with random weights is not a controller, so this "
                "mode stays disabled rather than producing meaningless "
                "decisions.\n\n"
                "Train one with:\n"
                "    python training/train_dqn.py --scenario {} "
                "--episodes 20 --seed 42".format(
                    " for the {} scenario".format(scenario) if scenario else "",
                    scenario or "moderate"))
        else:
            reason = ""

        hybrid_reason = "" if torch_ok else (
            "PyTorch is not installed, so there is no DQN for Hybrid to "
            "try.\n\nInstall the project requirements:\n"
            "    pip install -r requirements.txt")

        return [
            ControlMode(
                key=MODE_FIXED_TIME,
                label="Fixed-Time",
                description=fixed.description,
                available=True,
                agent_name="fixed_time",
                badge="IMPLEMENTED",
            ),
            ControlMode(
                key=MODE_DQN,
                label="DQN Adaptive",
                description=dqn.description,
                available=available,
                agent_name="dqn",
                unavailable_reason=reason,
                badge="TRAINED MODEL" if available else "NO MODEL",
            ),
            ControlMode(
                key=MODE_HYBRID,
                label="Hybrid",
                description=(
                    "Starts the traffic under a valid fixed-time plan, then "
                    "spends a fixed amount of wall-clock time tuning that "
                    "plan and testing a DQN against it. The DQN takes over "
                    "only if it measurably wins; otherwise the best measured "
                    "fixed plan keeps the signal."),
                available=bool(torch_ok),
                agent_name="hybrid",
                unavailable_reason=hybrid_reason,
                badge="TIME-LIMITED" if torch_ok else "NO PYTORCH",
            ),
        ]

    def control_mode(self, key: str, scenario: Optional[str] = None) -> ControlMode:
        for mode in self.control_modes(scenario):
            if mode.key == key:
                return mode
        raise ControlModeUnavailableError("Unknown control mode '{}'.".format(key))

    def agent_for_mode(self, mode_key: str, scenario: str,
                       model_path: Optional[str] = None):
        """The decision-making object behind a control mode.

        For fixed-time this is the Stage 1 plan read from the scenario YAML.
        For the DQN it is a real network loaded from a real checkpoint - or an
        error, never an untrained network.
        """
        from agents import create_agent

        mode = self.control_mode(mode_key, scenario)
        if not mode.available:
            raise ControlModeUnavailableError(mode.unavailable_reason)

        if mode.key == MODE_DQN:
            return self.load_dqn_agent(scenario, model_path)
        return create_agent(mode.agent_name,
                            scenario_config=self.scenario_config(scenario))

    # ------------------------------------------------------------------
    # Stage 3: trained models
    # ------------------------------------------------------------------
    @staticmethod
    def torch_available() -> bool:
        try:
            import torch                                    # noqa: F401
            return True
        except Exception:
            return False

    def available_models(self, scenario: str) -> List[Dict[str, Any]]:
        """Every checkpoint found for a scenario, best first. Possibly empty."""
        if not self.torch_available():
            return []
        try:
            from agents.dqn_agent import find_models
            return find_models(scenario)
        except Exception:
            return []

    def model_path(self, scenario: str,
                   name: Optional[str] = None) -> Optional[str]:
        """The checkpoint to use: the named one, or the preferred one, or None."""
        models = self.available_models(scenario)
        if not models:
            return None
        if name:
            for model in models:
                if model["name"] == name or model["path"] == name:
                    return model["path"]
            return None
        return models[0]["path"]

    def has_trained_model(self, scenario: str) -> bool:
        return self.model_path(scenario) is not None

    def scenarios_with_models(self) -> List[str]:
        return [s for s in self.available_scenarios() if self.has_trained_model(s)]

    def load_dqn_agent(self, scenario: str, model_path: Optional[str] = None):
        """Load the trained DQN for a scenario, or explain why it cannot be."""
        if not self.torch_available():
            raise ControlModeUnavailableError(
                "PyTorch is not installed, so the DQN controller cannot run.",
                hint="Run: pip install -r requirements.txt")

        path = model_path or self.model_path(scenario)
        if not path:
            raise ModelMissingError(
                "DQN unavailable — train a model first.\n\n"
                "No trained checkpoint was found for the '{}' scenario."
                .format(scenario))
        try:
            from agents.dqn_agent import DQNAgent
            agent = DQNAgent.load(path, scenario=scenario)
            agent.set_epsilon(0.0)              # the dashboard never explores
            return agent
        except SimulationError:
            raise
        except Exception as exc:
            raise ModelMissingError(
                "The DQN checkpoint could not be loaded: {}".format(exc))

    def model_status(self, scenario: str) -> Dict[str, Any]:
        """What the dashboard's model page shows. Every value is measured.

        When no model exists this returns `available=False` and a reason -
        it never invents an architecture or a training date.
        """
        status: Dict[str, Any] = {
            "scenario": scenario,
            "torch_available": self.torch_available(),
            "available": False,
            "reason": "",
            "checkpoints": [],
            "model_path": "",
            "metadata": {},
            "summary": {},
        }
        if not status["torch_available"]:
            status["reason"] = ("PyTorch is not installed, so no model can be "
                                "loaded.")
            return status

        models = self.available_models(scenario)
        status["checkpoints"] = [
            {"name": m["name"],
             "path": os.path.relpath(m["path"], PROJECT_ROOT),
             "size_kb": round(os.path.getsize(m["path"]) / 1024.0, 1)}
            for m in models]
        if not models:
            status["reason"] = (
                "DQN unavailable — train a model first. No checkpoint exists "
                "in models/dqn_{}/.".format(scenario))
            return status

        try:
            agent = self.load_dqn_agent(scenario)
        except SimulationError as exc:
            status["reason"] = str(exc)
            return status

        status["available"] = True
        status["model_path"] = os.path.relpath(agent.model_path, PROJECT_ROOT)
        status["summary"] = agent.summary()
        status["metadata"] = agent.metadata
        return status

    # ------------------------------------------------------------------
    def training_history(self, scenario: str):
        """The training-history CSVs for a scenario, as one DataFrame or None.

        Read from results/training/. A `seed` column distinguishes runs, so
        several seeds can be drawn on one chart without being averaged
        together.
        """
        import glob

        import pandas as pd

        pattern = os.path.join(TRAINING_RESULTS_DIR,
                               "dqn_{}_seed*_training.csv".format(scenario))
        frames = []
        for path in sorted(glob.glob(pattern)):
            try:
                frame = pd.read_csv(path)
            except Exception:
                continue
            if frame.empty:
                continue
            name = os.path.basename(path)
            seed = name.split("_seed")[-1].split("_")[0]
            frame["seed"] = seed
            frame["source"] = name
            frames.append(frame)
        if not frames:
            return None
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    def evaluation_results(self, scenario: str) -> Dict[str, Any]:
        """The Fixed-Time vs DQN comparison produced on the command line.

        Returns the raw per-seed runs, the aggregated mean/std table and the
        improvement table, or `available=False` with a reason. Nothing is
        computed from thin air: if the CSVs are not there, the dashboard says
        so and offers the command that produces them.
        """
        import pandas as pd

        out: Dict[str, Any] = {
            "scenario": scenario, "available": False, "reason": "",
            "raw": None, "aggregated": None, "improvement": None,
            "seeds": [], "command": (
                "python evaluation/compare_controllers.py --scenario {} "
                "--seeds 42 43 44".format(scenario)),
        }
        paths = {
            "raw": os.path.join(EVALUATION_RESULTS_DIR,
                                "comparison_{}_raw.csv".format(scenario)),
            "aggregated": os.path.join(
                EVALUATION_RESULTS_DIR,
                "comparison_{}_aggregated.csv".format(scenario)),
            "improvement": os.path.join(
                EVALUATION_RESULTS_DIR,
                "comparison_{}_improvement.csv".format(scenario)),
        }
        if not os.path.isfile(paths["aggregated"]):
            out["reason"] = (
                "No Fixed-Time vs DQN evaluation has been run for the '{}' "
                "scenario yet.".format(scenario))
            return out

        for key, path in paths.items():
            if os.path.isfile(path):
                try:
                    frame = pd.read_csv(path)
                    out[key] = frame if not frame.empty else None
                except Exception as exc:
                    out["reason"] = "Could not read {}: {}".format(
                        os.path.basename(path), exc)
                    return out

        if out["aggregated"] is None:
            out["reason"] = "The aggregated evaluation file is empty."
            return out

        out["available"] = True
        if out["raw"] is not None and "seed" in out["raw"].columns:
            out["seeds"] = sorted(int(s) for s in out["raw"]["seed"].unique())
        out["paths"] = {k: os.path.relpath(v, PROJECT_ROOT)
                        for k, v in paths.items() if os.path.isfile(v)}
        return out

    # ------------------------------------------------------------------
    def evaluation_timeseries(self, scenario: str, seed: int = None):
        """Per-second queue length for both controllers from one comparison seed."""
        import pandas as pd

        evaluation = self.evaluation_results(scenario)
        seeds = evaluation.get("seeds") or []
        if seed is None:
            seed = seeds[0] if seeds else 42

        series = {}
        for controller in ("fixed_time", "dqn"):
            path = os.path.join(
                EVALUATION_RESULTS_DIR,
                "comparison_{}_{}_seed{}_timeseries.csv".format(
                    scenario, controller, seed))
            if not os.path.isfile(path):
                continue
            try:
                frame = pd.read_csv(path)
            except Exception:
                continue
            if not frame.empty:
                series[controller] = frame
        return series or None

    # ------------------------------------------------------------------
    # Running a simulation
    # ------------------------------------------------------------------
    def run(self, scenario: str, mode: str = MODE_FIXED_TIME,
            duration: Optional[float] = None,
            use_gui: bool = False,
            seed: Optional[int] = None,
            model_path: Optional[str] = None) -> SimulationRun:
        """Run one scenario under one controller and return its real metrics.

        Fixed-time reuses `simulation.run_simulation.run_scenario()` unchanged
        - the same function the command line calls - so the dashboard cannot
        drift away from the command-line results.

        The DQN branch runs `simulation.agent_simulation.run_agent_episode()`,
        which drives the Stage 2 environment with the trained network and
        measures it with the same Stage 1 metrics collector. It is a real
        SUMO simulation, not a replay of a stored result.
        """
        control_mode = self.control_mode(mode, scenario)
        if not control_mode.available:
            raise ControlModeUnavailableError(control_mode.unavailable_reason)

        if scenario not in self.available_scenarios():
            raise ScenarioNotFoundError(
                "Scenario '{}' does not exist. Available: {}.".format(
                    scenario, ", ".join(self.available_scenarios())))

        status = self.environment_status()
        if not status.sumo_available or not status.traci_available:
            raise SumoUnavailableError(
                "SUMO is not available.\n\n"
                "Please verify that SUMO is installed and SUMO_HOME is "
                "configured correctly.")
        if not status.network_file_present:
            raise SimulationError(
                "The SUMO network file is missing.",
                hint="Rebuild it with: python sumo/network/build_network.py")

        os.makedirs(self.results_dir, exist_ok=True)
        started = datetime.now()

        if mode == MODE_DQN:
            return self._run_dqn(scenario, duration, use_gui, seed,
                                 model_path, started)

        try:
            from simulation.run_simulation import run_scenario
            summary = run_scenario(scenario, use_gui=use_gui, quiet=True,
                                   duration=duration,
                                   results_dir=self.results_dir)
        except SystemExit as exc:
            raise SimulationError(
                "SUMO could not be started.\n\n{}".format(exc),
                hint="Check that no other SUMO process is running, then retry."
            ) from exc
        except Exception as exc:
            raise SimulationError(
                "The simulation failed: {}".format(exc),
                hint="Check the terminal running Streamlit for the full error."
            ) from exc

        summary["controller"] = RESULT_CONTROLLER[MODE_FIXED_TIME]
        elapsed = (datetime.now() - started).total_seconds()
        return SimulationRun(
            scenario=scenario,
            mode=mode,
            metrics=summary,
            source="dashboard run",
            csv_path=os.path.join(self.results_dir,
                                  "{}_results.csv".format(scenario)),
            timeseries_path=os.path.join(self.results_dir,
                                         "{}_timeseries.csv".format(scenario)),
            finished_at=datetime.now(),
            duration_seconds=elapsed,
        )

    # ------------------------------------------------------------------
    def _run_dqn(self, scenario: str, duration: Optional[float],
                 use_gui: bool, seed: Optional[int],
                 model_path: Optional[str],
                 started: datetime) -> SimulationRun:
        """One real SUMO run driven by the trained network."""
        agent = self.load_dqn_agent(scenario, model_path)
        stem = "{}_dqn".format(scenario)

        try:
            from simulation.agent_simulation import run_agent_episode, save_run
            summary, timeseries = run_agent_episode(
                scenario, agent, seed=int(seed) if seed is not None else 42,
                duration=duration, use_gui=use_gui, record_decisions=True)
            paths = save_run(summary, timeseries, self.results_dir, stem)
            self._save_decisions(summary.pop("decision_log", None), stem)
        except SystemExit as exc:
            raise SimulationError(
                "SUMO could not be started.\n\n{}".format(exc),
                hint="Check that no other SUMO process is running, then retry."
            ) from exc
        except Exception as exc:
            raise SimulationError(
                "The DQN simulation failed: {}".format(exc),
                hint="Check the terminal running Streamlit for the full error."
            ) from exc

        summary["model"] = os.path.relpath(agent.model_path, PROJECT_ROOT)
        summary["episodes_trained"] = agent.episode
        summary["architecture"] = agent.architecture()
        elapsed = (datetime.now() - started).total_seconds()
        return SimulationRun(
            scenario=scenario,
            mode=MODE_DQN,
            metrics=summary,
            source="dashboard run · DQN {}".format(
                os.path.basename(agent.model_path)),
            csv_path=paths["results"],
            timeseries_path=paths["timeseries"],
            finished_at=datetime.now(),
            duration_seconds=elapsed,
        )

    # ------------------------------------------------------------------
    # Custom workloads
    # ------------------------------------------------------------------
    def run_custom(self, workload, mode: str = MODE_FIXED_TIME,
                   plan=None, agent=None, seed: Optional[int] = None,
                   use_gui: bool = False,
                   profile: Optional[str] = None) -> SimulationRun:
        """Run a user-defined workload under one controller.

        The workload carries its own explicit vehicle list, so `scenario` is
        not consulted for demand at all - only for the signal plan and the
        vehicle type, both of which the workload copied in when it was
        resolved. That is what lets two controllers be compared on exactly
        the same traffic rather than on the same random seed.
        """
        from simulation.custom_runs import (run_dqn, run_fixed,
                                            save_custom_run)

        started = datetime.now()
        status = self.environment_status()
        if not status.sumo_available or not status.traci_available:
            raise SumoUnavailableError("SUMO is not available.")

        decisions = None
        if mode == MODE_DQN:
            if agent is None:
                agent = self.load_phase_agent(workload.base_scenario)
            summary, timeseries, decisions = run_dqn(workload, agent,
                                                     seed=seed,
                                                     profile=profile)
        else:
            summary, timeseries = run_fixed(workload, plan=plan, seed=seed,
                                            use_gui=use_gui)
        summary["timing_profile"] = profile or "responsive"

        paths = save_custom_run(workload, summary, timeseries, decisions)
        summary["workload_id"] = workload.workload_id
        return SimulationRun(
            scenario=SCENARIO_CUSTOM,
            mode=mode,
            metrics=summary,
            source="custom workload {}".format(workload.workload_id[:8]),
            csv_path=paths["results"],
            timeseries_path=paths["timeseries"],
            finished_at=datetime.now(),
            duration_seconds=(datetime.now() - started).total_seconds(),
        )

    # ------------------------------------------------------------------
    def load_phase_agent(self, scenario: str, profile: Optional[str] = None):
        """The promoted duration-choosing model to use.

        When a timing profile is in force, the model registered for THAT
        PROFILE is used: it was trained under those green limits, on the
        varied workload library. The per-scenario models are used only
        when no profile is given, or when no model exists for the profile.

        Separate from `load_dqn_agent`, which loads the Stage 3 KEEP/SWITCH
        agent the Moderate and Heavy pages use. The two are different action
        spaces and different checkpoints; mixing them up would not be a
        fallback, it would be a crash.
        """
        from agents.phase_dqn_agent import (PhaseModelError,
                                            load_registered_phase_model,
                                            load_registered_profile_model,
                                            registered_profile_model)

        if profile and registered_profile_model(profile):
            try:
                return load_registered_profile_model(profile)
            except PhaseModelError as exc:
                raise ModelMissingError(
                    "The model registered for the '{}' timing profile "
                    "cannot be used.\n\n{}".format(profile, exc)) from exc
        try:
            return load_registered_phase_model(scenario)
        except PhaseModelError as exc:
            raise ModelMissingError(
                "No duration-choosing model is registered for '{}'{}.\n\n{}"
                .format(scenario,
                        " or for the '{}' profile".format(profile)
                        if profile else "", exc)) from exc

    def qualified_phase_model(self, scenario: str,
                              profile: Optional[str] = None):
        """The model to use, or the reason there is none.

        Returns `(agent, reason)`. `agent` is None when the DQN cannot
        honestly drive the signal under this profile, and `reason` says
        why in words a non-specialist can read. The caller is expected to
        fall back to Hybrid rather than to refuse the run: the point of
        Hybrid is that it measures both and picks, so "we have no
        validated network here" is a reason to let it choose, not a
        reason to stop.

        A model qualifies only if it is registered for this profile, its
        green menu fits the profile, and its promotion record shows it
        actually beating the tuned fixed plan it was measured against.
        """
        from agents.phase_dqn_agent import (PhaseModelError,
                                            registered_phase_model,
                                            registered_profile_model)
        from rl.timing_profile import compatible, get_profile

        timing = get_profile(profile) if profile else None
        record = (registered_profile_model(profile) if profile else None)
        if record is None and profile:
            scenario_record = registered_phase_model(scenario)
            if scenario_record is None:
                return None, (
                    "No DQN has been trained and validated for the '{}' "
                    "timing profile yet.".format(profile))
            if timing is not None and not compatible(
                    timing, scenario_record.get("durations") or []):
                return None, (
                    "The only trained model here uses {:g}-{:g} s greens, "
                    "which the '{}' profile ({:g}-{:g} s) does not allow. "
                    "It is kept as historical evidence, not used."
                    .format(min(scenario_record["durations"]),
                            max(scenario_record["durations"]), profile,
                            timing.min_green, timing.max_green))

        measurement = (record or {}).get("measurement") or {}
        learned = measurement.get("mean_total_delay")
        baseline = measurement.get("baseline_total_delay")
        if record is not None and learned is not None and baseline is not None:
            if learned >= baseline:
                return None, (
                    "The DQN registered for this profile did not beat the "
                    "tuned fixed plan when it was measured ({:.2f} s against "
                    "{:.2f} s).".format(learned, baseline))

        try:
            return self.load_phase_agent(scenario, profile=profile), ""
        except Exception as exc:
            return None, str(exc)

    def has_phase_model(self, scenario: str,
                        profile: Optional[str] = None) -> bool:
        from agents.phase_dqn_agent import (registered_phase_model,
                                            registered_profile_model)
        if profile and registered_profile_model(profile):
            return True
        return bool(registered_phase_model(scenario))

    # ------------------------------------------------------------------
    def _save_decisions(self, decisions, stem: str) -> str:
        """Record what the network actually decided, second by second.

        One row per decision with the two Q-values the network produced. The
        dashboard reads this back, so the Q-values it shows for a moment in
        the run are the ones the network really output at that moment - not a
        re-evaluation of a reconstructed state.
        """
        if not decisions:
            return ""
        import pandas as pd

        path = os.path.join(self.results_dir, "{}_decisions.csv".format(stem))
        pd.DataFrame(decisions).to_csv(path, index=False)
        return path

    def load_decisions(self, scenario: str):
        """The recorded DQN decision log for a scenario, or None."""
        import pandas as pd

        path = os.path.join(self.results_dir,
                            "{}_dqn_decisions.csv".format(scenario))
        if not os.path.isfile(path):
            return None
        try:
            frame = pd.read_csv(path)
        except Exception:
            return None
        return frame if not frame.empty else None

    # ------------------------------------------------------------------
    # Reading results
    # ------------------------------------------------------------------
    def _result_paths(self, scenario: str, mode: str = MODE_FIXED_TIME):
        """Which result files to read, and what to call their source.

        Fixed-time: the dashboard's own run if there is one, otherwise the
        recorded Stage 1 baseline, which is labelled as such.

        DQN: only a DQN run counts. There is deliberately NO fallback - a
        dashboard that showed fixed-time numbers when asked for DQN numbers
        would be lying.
        """
        if mode == MODE_DQN:
            dqn = os.path.join(self.results_dir,
                               "{}_dqn_results.csv".format(scenario))
            if os.path.isfile(dqn):
                return dqn, self.results_dir, "dashboard run · DQN"
            return None, None, None

        dashboard = os.path.join(self.results_dir,
                                 "{}_results.csv".format(scenario))
        baseline = os.path.join(BASELINE_RESULTS_DIR,
                                "{}_results.csv".format(scenario))
        if os.path.isfile(dashboard):
            return dashboard, self.results_dir, "dashboard run"
        if os.path.isfile(baseline):
            return baseline, BASELINE_RESULTS_DIR, "Stage 1 baseline"
        return None, None, None

    def load_results(self, scenario: str,
                     mode: str = MODE_FIXED_TIME) -> Optional[SimulationRun]:
        """Load the most relevant results for a scenario and mode, or None."""
        import pandas as pd

        path, directory, source = self._result_paths(scenario, mode)
        if path is None:
            return None
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            raise ResultsMissingError(
                "The result file for '{}' could not be read: {}".format(scenario, exc)
            ) from exc
        if frame.empty:
            raise ResultsMissingError(
                "The result file for '{}' is empty.".format(scenario))

        stem = "{}_dqn".format(scenario) if mode == MODE_DQN else scenario
        timeseries = os.path.join(directory, "{}_timeseries.csv".format(stem))

        metrics = frame.iloc[0].to_dict()
        # The Stage 1 CSV layout has no "controller" column - it predates the
        # DQN. Which file the row came from is what identifies the controller,
        # and `_result_paths` has just decided that, so the label is filled in
        # here rather than left blank for the user interface to guess.
        metrics.setdefault("controller", RESULT_CONTROLLER[mode])

        return SimulationRun(
            scenario=scenario,
            mode=mode,
            metrics=metrics,
            source=source,
            csv_path=path,
            timeseries_path=timeseries if os.path.isfile(timeseries) else "",
            finished_at=datetime.fromtimestamp(os.path.getmtime(path)),
        )

    def load_timeseries(self, scenario: str, mode: str = MODE_FIXED_TIME):
        """Per-second recorded data for a scenario, or None."""
        import pandas as pd

        run = self.load_results(scenario, mode)
        if run is None or not run.timeseries_path:
            return None
        try:
            frame = pd.read_csv(run.timeseries_path)
        except Exception:
            return None
        return frame if not frame.empty else None

    def comparison_table(self, scenarios: Optional[List[str]] = None):
        """One row per scenario, for the comparison page."""
        import pandas as pd

        scenarios = scenarios or self.available_scenarios()
        rows = []
        for scenario in scenarios:
            try:
                run = self.load_results(scenario)
            except ResultsMissingError:
                run = None
            if run is None:
                continue
            metrics = run.metrics
            rows.append({
                "scenario": scenario,
                "source": run.source,
                "vehicles": metrics.get("vehicles"),
                "average_waiting_time": metrics.get("average_waiting_time"),
                "average_queue_length": metrics.get("average_queue_length"),
                "max_queue_length": metrics.get("max_queue_length"),
                "throughput": metrics.get("throughput"),
                "simulation_time": metrics.get("simulation_time"),
                "teleports": metrics.get("teleports"),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def controller_comparison(self, scenario: str) -> Dict[str, Any]:
        """Fixed-Time vs DQN from runs started in the dashboard itself.

        This is the "I just ran both, show me the difference" view. It is
        separate from `evaluation_results()`, which reads the multi-seed
        command-line experiment.

        Returns `available=False` with a reason unless BOTH runs exist, and
        sets `comparable=False` when they covered different simulated times -
        two runs of different lengths are not a comparison, and the dashboard
        says so instead of printing a percentage.
        """
        from simulation.agent_simulation import LOWER_IS_BETTER, improvement

        metrics = ["average_waiting_time", "average_queue_length",
                   "max_queue_length", "throughput",
                   "vehicles_still_in_network", "teleports"]

        out: Dict[str, Any] = {
            "scenario": scenario, "available": False, "reason": "",
            "comparable": True, "fixed": None, "dqn": None, "rows": [],
        }
        try:
            fixed = self.load_results(scenario, MODE_FIXED_TIME)
            dqn = self.load_results(scenario, MODE_DQN)
        except ResultsMissingError as exc:
            out["reason"] = str(exc)
            return out

        if dqn is None:
            out["reason"] = ("No DQN run for this scenario yet. Select the DQN "
                             "Adaptive mode in the sidebar and press Run "
                             "Simulation.")
            return out
        if fixed is None:
            out["reason"] = ("No fixed-time run for this scenario yet. Run it "
                             "once so there is something to compare against.")
            return out

        out["available"] = True
        out["fixed"] = fixed
        out["dqn"] = dqn

        fixed_time_s = fixed.metric("simulation_time")
        dqn_time_s = dqn.metric("simulation_time")
        if fixed_time_s and dqn_time_s and abs(
                float(fixed_time_s) - float(dqn_time_s)) > 0.5:
            out["comparable"] = False
            out["reason"] = (
                "These two runs cover different simulated times ({:.0f} s and "
                "{:.0f} s), so the numbers are not comparable. Re-run both "
                "with the same duration.".format(float(fixed_time_s),
                                                 float(dqn_time_s)))

        for metric in metrics:
            fixed_value = fixed.metric(metric)
            dqn_value = dqn.metric(metric)
            out["rows"].append({
                "metric": metric,
                "lower_is_better": bool(LOWER_IS_BETTER.get(metric, True)),
                "fixed_time": fixed_value,
                "dqn": dqn_value,
                "improvement_percent": (improvement(fixed_value, dqn_value, metric)
                                        if out["comparable"] else None),
            })
        return out

    # ------------------------------------------------------------------
    # Assets
    # ------------------------------------------------------------------
    def network_screenshot(self) -> Optional[str]:
        """A real SUMO screenshot of the intersection, if one was captured."""
        for name in ("01_intersection_network.png", "05_moderate_traffic.png"):
            path = os.path.join(SCREENSHOT_DIR, name)
            if os.path.isfile(path):
                return path
        return None

    def screenshots(self) -> List[Dict[str, str]]:
        """Every captured SUMO screenshot, with a caption."""
        captions = {
            "01_intersection_network.png": "The four-way intersection network in SUMO",
            "02_vehicles_approaching.png": "Vehicles approaching the intersection",
            "03_vehicles_waiting_red.png": "A queue held at a red signal",
            "04_vehicles_moving_green.png": "The same approach discharging on green",
            "05_moderate_traffic.png": "Moderate traffic scenario",
            "06_heavy_traffic.png": "Heavy traffic scenario",
        }
        found = []
        for name, caption in captions.items():
            path = os.path.join(SCREENSHOT_DIR, name)
            if os.path.isfile(path):
                found.append({"path": path, "caption": caption, "name": name})
        return found

    def gui_command(self, scenario: str, duration: Optional[float] = None) -> str:
        """The exact command to open this scenario in sumo-gui.

        Shown rather than executed: launching a GUI process from a web server
        is unreliable, and a failed launch must never take the dashboard down.
        """
        command = "python simulation/run_simulation.py --scenario {} --gui".format(scenario)
        if duration:
            command += " --duration {:.0f}".format(duration)
        return command

    def sumo_gui_available(self) -> bool:
        try:
            import sumolib
            return bool(sumolib.checkBinary("sumo-gui"))
        except Exception:
            return False

    # ------------------------------------------------------------------
    def disk_usage(self) -> Dict[str, Any]:
        """Free space where results are written - used for a friendly warning."""
        try:
            usage = shutil.disk_usage(PROJECT_ROOT)
            return {"free_mb": usage.free / (1024 * 1024)}
        except Exception:
            return {"free_mb": None}
