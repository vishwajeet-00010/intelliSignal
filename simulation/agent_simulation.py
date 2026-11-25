"""
IntelliSignal - Stage 3
simulation/agent_simulation.py

Runs ONE episode of the intersection under ONE controller, and measures it
with the Stage 1 metrics collector.

                    config/<scenario>.yaml
                             |
                rl.environment.episode_scenario_config()
                             |
                 the SAME generated SUMO input files
                 (network + routes + demand + seed)
                             |
              +--------------+--------------+
              |                             |
       SUMO's own fixed-time         IntelliSignalEnv
       program (Stage 1 plan)        driven by a TrafficAgent
              |                             |
              +--------------+--------------+
                             |
               utils.metrics.MetricsCollector
                             |
                    the same eight metrics

Why both controllers live in one file
-------------------------------------
This is the fairness guarantee of the whole comparison. Both functions below
build their SUMO inputs with the *same* call, run the *same* number of
simulated seconds, and are measured by the *same* collector class reading the
*same* eight incoming lanes. The only difference between a fixed-time run and
a DQN run is which code decides when the signal changes.

What is NOT here
----------------
No metric is computed in this file. `utils/metrics.py` does that, exactly as
it did for the Stage 1 baseline. No reward, observation, action or safety rule
is defined here either - the agent-driven run goes through
`rl.IntelliSignalEnv`, so a switch still passes through the Stage 1 yellow
clearance and still honours min_green / max_green.

Results policy
--------------
Nothing in this file writes to results/csv/. That directory holds the
recorded Stage 1 baseline. Callers pass their own directory
(results/evaluation/, results/data/, ...) to `save_run()`.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.base_agent import TrafficState                      # noqa: E402
from rl.config import load_rl_config                            # noqa: E402
from rl.environment import (IntelliSignalEnv,                    # noqa: E402
                            episode_scenario_config)
from simulation.run_simulation import (prepare_sumo_imports,     # noqa: E402
                                       sumo_binary)
from utils.config_loader import load_config                      # noqa: E402
from utils.metrics import MetricsCollector                       # noqa: E402
from utils.scenario_builder import (TLS_ID, TLS_PROGRAM_ID,      # noqa: E402
                                    build_scenario)
from utils.sumo_launch import start_sumo  # noqa: E402

#: Controller identifiers used in every result CSV and chart.
FIXED_TIME = "fixed_time"
DQN = "dqn"

CONTROLLER_LABELS = {FIXED_TIME: "Fixed-Time", DQN: "DQN"}

#: SUMO prints a statistics block at the end of every run. During a 100-episode
#: training run that is thousands of lines of noise, so it is switched off.
#: It is reporting only - it changes nothing about the simulation.
QUIET_SUMO_ARGS = ["--duration-log.statistics", "false", "--no-step-log", "true"]

#: The columns every Stage 3 result CSV has, in this order. The first eight
#: after the labels are the Stage 1 metrics, unchanged.
RESULT_COLUMNS = [
    "scenario", "controller", "seed",
    "vehicles", "average_waiting_time", "average_queue_length",
    "max_queue_length", "throughput", "vehicles_still_in_network",
    "teleports", "simulation_time", "vehicles_departed",
]


# ===========================================================================
# helpers
# ===========================================================================
def episode_plan(scenario: str, duration: Optional[float] = None
                 ) -> Dict[str, Any]:
    """How long one episode is, in the units both controllers need.

    Returns the scenario configuration, the RL settings, the episode duration
    in simulated seconds, the number of agent decisions, the warm-up steps and
    the total number of 1-second simulation steps that will be measured.
    """
    scenario_config = load_config(scenario)
    rl_config = load_rl_config(scenario_config)
    env_cfg = rl_config["environment"]

    step_length = float(scenario_config["simulation"]["step_length"])
    decision_interval = float(env_cfg["decision_interval"])
    episode_duration = float(duration if duration is not None
                             else env_cfg["episode_duration"])
    decisions = int(episode_duration // decision_interval)
    if decisions < 1:
        raise ValueError(
            "An episode of {:.0f} s is shorter than one decision interval "
            "({:.0f} s).".format(episode_duration, decision_interval))

    return {
        "scenario_config": scenario_config,
        "rl_config": rl_config,
        "step_length": step_length,
        "decision_interval": decision_interval,
        "episode_duration": episode_duration,
        "decisions": decisions,
        "warmup_steps": int(round(float(env_cfg.get("warmup_time", 0.0))
                                  / step_length)),
        "measured_steps": int(round(decisions * decision_interval / step_length)),
    }


class _LateCollector:
    """A per-second hook that starts collecting once SUMO is up.

    `IntelliSignalEnv.reset()` is what starts SUMO, so the collector cannot
    exist before reset() is called - but the hook has to be installed before
    it. This object is installed first and armed immediately afterwards. Any
    warm-up second that passes before it is armed is not measured, and the
    fixed-time run skips exactly the same number of seconds, so the two runs
    stay comparable. (`warmup_time` is 0 by default, so normally no second is
    skipped at all.)
    """

    def __init__(self):
        self.collector: Optional[MetricsCollector] = None

    def arm(self, collector: MetricsCollector) -> MetricsCollector:
        self.collector = collector
        return collector

    def __call__(self) -> None:
        if self.collector is not None:
            self.collector.collect_step()


def _finish(collector: MetricsCollector, scenario: str, controller: str,
            seed: int, extra: Dict[str, Any] = None) -> Dict[str, Any]:
    """Turn a finished collector into the result row."""
    summary = collector.summary(scenario)
    summary["controller"] = controller
    summary["seed"] = int(seed)
    if extra:
        summary.update(extra)
    return summary


# ===========================================================================
# Fixed-time: SUMO runs the Stage 1 signal program itself
# ===========================================================================
def run_fixed_time_episode(scenario: str, seed: int = 42,
                           duration: Optional[float] = None,
                           quiet: bool = True,
                           use_gui: bool = False
                           ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run one fixed-time episode on the comparison's SUMO inputs.

    No Python code touches the traffic light here: the generated additional
    file contains the Stage 1 fixed-time program and SUMO activates it at
    t = 0, which is checked rather than assumed. This is the Stage 1 baseline
    controller, re-run at a chosen seed so it can be compared against the DQN
    on identical traffic.
    """
    prepare_sumo_imports()
    import traci

    plan = episode_plan(scenario, duration)
    config = episode_scenario_config(
        plan["scenario_config"], scenario, plan["episode_duration"],
        plan["decision_interval"], seed)
    files = build_scenario(config)

    cmd = [
        sumo_binary(use_gui),
        "-c", files["sumocfg"],
        "--seed", str(int(seed)),
        "--start", "--quit-on-end",
        "--no-warnings", "true",
    ]
    if quiet:
        cmd += QUIET_SUMO_ARGS

    label = "intellisignal_fixed_{}_{}".format(scenario, seed)
    conn = start_sumo(cmd, label=label)
    try:
        program = conn.trafficlight.getProgram(TLS_ID)
        if program != TLS_PROGRAM_ID:
            raise RuntimeError(
                "Traffic light '{}' is running program '{}' but the fixed-time "
                "baseline needs '{}'. The comparison would not be measuring "
                "the Stage 1 controller.".format(TLS_ID, program, TLS_PROGRAM_ID))

        for _ in range(plan["warmup_steps"]):
            conn.simulationStep()

        collector = MetricsCollector(step_length=plan["step_length"], conn=conn)
        for _ in range(plan["measured_steps"]):
            conn.simulationStep()
            collector.collect_step()
    finally:
        try:
            conn.close()
        except Exception:
            pass

    summary = _finish(collector, scenario, FIXED_TIME, seed, {
        "episode_duration": plan["episode_duration"],
        "signal_program": TLS_PROGRAM_ID,
        "switch_count": _fixed_time_switches(plan),
        "decisions": plan["decisions"],
    })
    return summary, collector.timeseries


def _fixed_time_switches(plan: Dict[str, Any]) -> int:
    """How many times the fixed-time plan changes direction in one episode.

    Read from the scenario's own signal timings, not assumed: one cycle
    contains two direction changes.
    """
    signal = plan["scenario_config"]["signal"]
    cycle = (float(signal["ns_green"]) + float(signal["ew_green"])
             + 2.0 * float(signal["yellow"]) + 2.0 * float(signal["all_red"]))
    return int(plan["episode_duration"] // cycle) * 2


# ===========================================================================
# Agent-driven: the Stage 2 environment, decided by a TrafficAgent
# ===========================================================================
def run_agent_episode(scenario: str, agent, seed: int = 42,
                      duration: Optional[float] = None,
                      controller: str = DQN,
                      quiet: bool = True,
                      use_gui: bool = False,
                      record_decisions: bool = False,
                      phase_elapsed_scale: Optional[float] = None
                      ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run one episode with a `TrafficAgent` choosing keep/switch.

    The agent is used in its decision mode only: no exploration, no learning,
    no replay buffer writes. The environment still owns every safety rule.

    `record_decisions=True` additionally returns, in the summary, one row per
    decision (phase, action, Q-values when the agent provides them), which the
    dashboard uses to show what the network actually did.
    """
    plan = episode_plan(scenario, duration)
    if phase_elapsed_scale is None:
        phase_elapsed_scale = getattr(agent, "phase_elapsed_scale", None)

    hook = _LateCollector()
    env = IntelliSignalEnv(
        scenario=scenario,
        episode_duration=plan["episode_duration"],
        seed=int(seed),
        render_mode="human" if use_gui else None,
        extra_sumo_args=QUIET_SUMO_ARGS if quiet else None,
        phase_elapsed_scale=phase_elapsed_scale,
    )
    env.step_callback = hook

    decisions: List[Dict[str, Any]] = []
    total_reward = 0.0
    steps = 0
    blocked_switches = 0
    green_durations = []

    try:
        observation, info = env.reset(seed=int(seed))
        collector = hook.arm(MetricsCollector(step_length=plan["step_length"],
                                              conn=env.connection))

        while True:
            state = TrafficState.from_env_info(info, observation=observation)
            action = int(agent.env_action(state))

            if record_decisions:
                row = {
                    "decision": steps,
                    "simulation_time": info.get("simulation_time", 0.0),
                    "phase": info.get("phase"),
                    "env_action": action,
                    "total_queue": info.get("total_queue"),
                    "total_waiting_time": info.get("total_waiting_time"),
                    "green_elapsed": info.get("green_elapsed"),
                }
                values = getattr(agent, "q_values", None)
                if callable(values):
                    try:
                        q = values(observation)
                        row["q_keep"] = float(q[0])
                        row["q_switch"] = float(q[1])
                    except Exception:
                        pass
                decisions.append(row)

            observation, reward, terminated, truncated, info = env.step(action)
            blocked_switches += int(bool(info.get("switch_blocked", False)))
            if info.get("effective_action") == 1:
                green_durations.append(float(state.time_since_phase_change))
            total_reward += float(reward)
            steps += 1
            if terminated or truncated:
                break
    finally:
        env.close()

    summary = _finish(collector, scenario, controller, seed, {
        "episode_duration": plan["episode_duration"],
        "episode_return": round(total_reward, 4),
        "decisions": steps,
        "switch_count": int(info.get("switch_count", 0)),
        "blocked_switches": blocked_switches,
        "completed_green_duration_counts": {
            str(float(duration)): int(green_durations.count(duration))
            for duration in sorted(set(green_durations))
        },
        "terminated_early": bool(steps < plan["decisions"]),
    })
    if record_decisions:
        summary["decision_log"] = decisions
    return summary, collector.timeseries


# ===========================================================================
# Saving
# ===========================================================================
def save_run(summary: Dict[str, Any], timeseries: List[Dict[str, Any]],
             directory: str, stem: str,
             write_timeseries: bool = True) -> Dict[str, str]:
    """Write `<stem>_results.csv` and `<stem>_timeseries.csv`.

    Returns the paths. Refuses to write into results/csv/, which holds the
    recorded Stage 1 baseline.
    """
    import pandas as pd

    baseline = os.path.join(PROJECT_ROOT, "results", "csv")
    if os.path.abspath(directory) == os.path.abspath(baseline):
        raise ValueError(
            "results/csv/ holds the recorded Stage 1 baseline and must never "
            "be written by Stage 3. Use results/evaluation/ or results/data/.")

    os.makedirs(directory, exist_ok=True)
    row = {k: summary.get(k) for k in RESULT_COLUMNS}
    row["episode_return"] = summary.get("episode_return")
    row["switch_count"] = summary.get("switch_count")

    results_path = os.path.join(directory, "{}_results.csv".format(stem))
    pd.DataFrame([row]).to_csv(results_path, index=False)

    paths = {"results": results_path, "timeseries": ""}
    if write_timeseries and timeseries:
        ts_path = os.path.join(directory, "{}_timeseries.csv".format(stem))
        pd.DataFrame(timeseries).to_csv(ts_path, index=False)
        paths["timeseries"] = ts_path
    return paths


# ===========================================================================
# Improvement percentages
# ===========================================================================
#: True when a SMALLER value is a better result.
LOWER_IS_BETTER = {
    "average_waiting_time": True,
    "average_queue_length": True,
    "max_queue_length": True,
    "vehicles_still_in_network": True,
    "teleports": True,
    "throughput": False,
    "vehicles": False,
}


def improvement(baseline_value: Optional[float], dqn_value: Optional[float],
                metric: str) -> Optional[float]:
    """Percentage improvement of the DQN over the fixed-time baseline.

        lower-is-better metric :  (fixed - dqn) / fixed * 100
        throughput             :  (dqn - fixed) / fixed * 100

    A POSITIVE number always means the DQN did better and a NEGATIVE number
    always means it did worse - the sign is never flipped to flatter the
    agent. Returns None when the baseline is zero or a value is missing,
    because a percentage of zero has no meaning; callers show "n/a" rather
    than a made-up number.
    """
    if baseline_value is None or dqn_value is None:
        return None
    try:
        base = float(baseline_value)
        value = float(dqn_value)
    except (TypeError, ValueError):
        return None
    if base == 0.0:
        return None
    lower_better = LOWER_IS_BETTER.get(metric, True)
    delta = (base - value) if lower_better else (value - base)
    return float(delta / abs(base) * 100.0)
