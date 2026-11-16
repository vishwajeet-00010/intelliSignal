"""
IntelliSignal - Stage 2
rl/environment.py

The Gymnasium-compatible reinforcement-learning environment.

    SUMO -> TraCI -> Gymnasium environment -> observation / reward
                                   ^                      |
                                   +------ action --------+

It drives the SAME four-way intersection, the SAME two traffic scenarios and
the SAME network as Stage 1. Nothing about the road, the demand or the Stage 1
results is changed: the scenario files are read with the Stage 1 loader and
the SUMO inputs are generated with the Stage 1 builder.

**There is no agent in this file.** No neural network, no replay buffer, no
training. Stage 2 produces the environment a DQN will plug into later; the
DQN itself is a later stage.

Episode structure
-----------------
    reset()   starts a fresh SUMO process and returns the first observation
    step(a)   advances SUMO by `decision_interval` seconds under action `a`
              and returns (observation, reward, terminated, truncated, info)
    close()   shuts SUMO down

An episode is truncated after `episode_duration` simulated seconds (1800 s by
default, the same length as the Stage 1 runs). It terminates early only if
SUMO reports that no vehicle is left and none is expected.

Why the Stage 1 files are not overwritten
-----------------------------------------
The environment builds its SUMO inputs under the scenario name
"<scenario>_rl", so it writes sumo/routes/moderate_rl.rou.xml rather than
sumo/routes/moderate.rou.xml. The Stage 1 scenario files, results and plots
are never touched.

Stage 3 additions (all optional, nothing changed for Stage 2 callers)
---------------------------------------------------------------------
`step_callback`     called once after every single simulated second, while
                    the decision interval is being run. Stage 3 uses it to
                    let the Stage 1 `MetricsCollector` measure a DQN run with
                    exactly the code that measured the fixed-time baseline.
`connection`        the live TraCI connection object, so that collector can
                    read the same simulation.
`extra_sumo_args`   extra command-line options for SUMO, used only to quieten
                    its end-of-run report during long training runs.
`episode_scenario_config()`
                    the scenario-to-episode transformation, lifted out of
                    `_start_sumo` so the fixed-time comparison run can build
                    byte-identical SUMO inputs. The observation, the action,
                    the reward and the safety rules are untouched.
"""

from __future__ import annotations

import copy
import os
import sys
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl.action import (ACTION_KEEP, ACTION_NAMES, ACTION_SWITCH,  # noqa: E402
                       NUM_ACTIONS, PHASE_NAMES, PHASE_NS, ActionController)
from rl.config import load_rl_config                              # noqa: E402
from rl.reward import RewardFunction                              # noqa: E402
from rl.state import (APPROACH_ORDER, OBSERVATION_SIZE,           # noqa: E402
                      StateExtractor)
from simulation.run_simulation import (prepare_sumo_imports,      # noqa: E402
                                       sumo_binary)
from utils.config_loader import load_config                       # noqa: E402
from utils.scenario_builder import TLS_ID, build_scenario         # noqa: E402

# Scenario files built for the RL environment get this suffix, so the Stage 1
# generated files are never overwritten.
RL_SUFFIX = "_rl"


# ---------------------------------------------------------------------------
def episode_scenario_config(scenario_config: Dict[str, Any], scenario: str,
                            episode_duration: float, decision_interval: float,
                            seed: int) -> Dict[str, Any]:
    """The Stage 1 scenario configuration, adjusted for one RL episode.

    Three changes, and nothing else:

      * the scenario is renamed "<scenario>_rl", so Stage 1's generated files
        are never overwritten
      * the SUMO duration is the episode length plus one decision interval,
        so the simulation never runs out before the last decision is served
      * the SUMO seed is the episode's seed

    This lives at module level because the Fixed-Time vs DQN comparison has to
    build the *same* SUMO inputs for both controllers. Calling one function
    from both places is what guarantees the two runs see an identical network,
    an identical demand pattern and an identical seed - rather than two
    similar-looking code paths that could drift apart.
    """
    config = copy.deepcopy(scenario_config)
    config["scenario"]["name"] = scenario + RL_SUFFIX
    config["simulation"]["duration"] = float(episode_duration) + float(decision_interval)
    config["simulation"]["seed"] = int(seed)
    return config


class IntelliSignalEnv(gym.Env):
    """Adaptive traffic-signal control at the IntelliSignal intersection.

    Observation : Box(0.0, 1.0, shape=(10,), float32)   - see rl/state.py
    Action      : Discrete(2)  0 = keep phase, 1 = switch phase - see rl/action.py
    Reward      : negative congestion cost              - see rl/reward.py
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    # ------------------------------------------------------------------
    def __init__(self, scenario: str = "moderate",
                 episode_duration: Optional[float] = None,
                 render_mode: Optional[str] = None,
                 seed: Optional[int] = None,
                 sumo_warnings: bool = False,
                 extra_sumo_args: Optional[list] = None):
        super().__init__()

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError("Unsupported render_mode: {}".format(render_mode))

        prepare_sumo_imports()

        self.scenario = scenario
        self.render_mode = render_mode
        self.sumo_warnings = sumo_warnings
        self.extra_sumo_args = list(extra_sumo_args or [])

        #: Optional hook called after every simulated second (Stage 3).
        #: Used by the evaluation runner to let the Stage 1 MetricsCollector
        #: measure an agent-driven run with the baseline's own code.
        self.step_callback = None

        # Stage 1 scenario definition, read unmodified.
        self.scenario_config = load_config(scenario)
        self.rl_config = load_rl_config(self.scenario_config)

        env_cfg = self.rl_config["environment"]
        obs_cfg = self.rl_config["observation"]
        reward_cfg = self.rl_config["reward"]

        self.step_length = float(self.scenario_config["simulation"]["step_length"])
        self.decision_interval = float(env_cfg["decision_interval"])
        self.warmup_time = float(env_cfg.get("warmup_time", 0.0))
        self.episode_duration = float(
            episode_duration if episode_duration is not None
            else env_cfg["episode_duration"])
        self.max_decisions = int(self.episode_duration // self.decision_interval)
        if self.max_decisions < 1:
            raise ValueError(
                "episode_duration ({:.0f} s) is shorter than one decision "
                "interval ({:.0f} s).".format(self.episode_duration,
                                              self.decision_interval))

        self.controller = ActionController(
            decision_interval=self.decision_interval,
            yellow_time=env_cfg["yellow_time"],
            all_red_time=env_cfg["all_red_time"],
            min_green=env_cfg["min_green"],
            max_green=env_cfg["max_green"],
            step_length=self.step_length,
        )
        self.state_extractor = StateExtractor(
            max_queue_per_approach=obs_cfg["max_queue_per_approach"],
            max_waiting_per_approach=obs_cfg["max_waiting_per_approach"],
            max_green=env_cfg["max_green"],
        )
        self.reward_fn = RewardFunction(**reward_cfg)

        # --- Gymnasium spaces -----------------------------------------
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBSERVATION_SIZE,), dtype=np.float32)
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # --- episode state --------------------------------------------
        self._conn = None
        self._label = "intellisignal_rl_{}_{}".format(scenario, id(self))
        self._decision = 0
        self._sumo_seed = seed if seed is not None else int(
            self.scenario_config["simulation"]["seed"])
        self._files = None
        self._closed_cleanly = True
        self._cumulative_reward = 0.0

    # ==================================================================
    # Gymnasium API
    # ==================================================================
    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None
              ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Start a fresh episode and return the first observation."""
        super().reset(seed=seed)
        if seed is not None:
            self._sumo_seed = int(seed)

        self.close()                       # never leave an old SUMO running
        self._start_sumo()

        self.controller.reset(PHASE_NS)
        self._decision = 0
        self._cumulative_reward = 0.0

        # Show the starting green before the first step, then warm up.
        self._conn.trafficlight.setRedYellowGreenState(
            TLS_ID, self.controller.current_state())
        warmup_steps = int(round(self.warmup_time / self.step_length))
        for _ in range(warmup_steps):
            self._conn.simulationStep()
            self._after_simulation_step()

        raw = self.state_extractor.raw(self._conn)
        observation = self.state_extractor.observe(
            raw, self.controller.phase, self.controller.green_elapsed)
        info = self._build_info(raw, reward_parts=None, action_info=None)
        return observation, info

    # ------------------------------------------------------------------
    def step(self, action: int
             ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Apply one decision and advance SUMO by one decision interval."""
        if self._conn is None:
            raise RuntimeError("step() called before reset(); the environment "
                               "has no running SUMO simulation.")
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(
                "Invalid action {}. Valid actions are 0 (keep) and 1 (switch)."
                .format(action))

        effective_action, action_info = self.controller.resolve(action)
        plan = self.controller.plan(effective_action)

        # --- run the interval, sampling every simulated second ---------
        samples = []
        for state, steps in plan:
            self._conn.trafficlight.setRedYellowGreenState(TLS_ID, state)
            for _ in range(steps):
                self._conn.simulationStep()
                self._after_simulation_step()
                samples.append(self.state_extractor.raw(self._conn))

        self.controller.commit(effective_action)
        self._decision += 1

        raw = samples[-1]
        averaged = self._average(samples)
        reward_parts = self.reward_fn(averaged)
        reward = reward_parts["reward"]
        self._cumulative_reward += reward

        observation = self.state_extractor.observe(
            raw, self.controller.phase, self.controller.green_elapsed)

        # --- episode end ----------------------------------------------
        no_traffic_left = self._conn.simulation.getMinExpectedNumber() <= 0
        terminated = bool(no_traffic_left)
        truncated = bool(self._decision >= self.max_decisions)

        action_info["effective_action"] = effective_action
        info = self._build_info(raw, reward_parts, action_info, averaged)

        if terminated or truncated:
            info["episode_return"] = self._cumulative_reward
            info["episode_decisions"] = self._decision

        return observation, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Shut SUMO down. Safe to call more than once."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                self._closed_cleanly = False
            finally:
                self._conn = None

    # ------------------------------------------------------------------
    def render(self):
        """Rendering is done by sumo-gui itself.

        Pass render_mode="human" to the constructor and the simulation runs
        in sumo-gui, where the intersection and the signal are visible.
        """
        return None

    # ==================================================================
    # internals
    # ==================================================================
    @property
    def connection(self):
        """The live TraCI connection, or None when SUMO is not running.

        Exposed so the Stage 1 `MetricsCollector` can read the same running
        simulation the agent is controlling.
        """
        return self._conn

    # ------------------------------------------------------------------
    def _after_simulation_step(self) -> None:
        """Run the per-second hook, if one was installed."""
        if self.step_callback is not None:
            self.step_callback()

    # ------------------------------------------------------------------
    def _start_sumo(self) -> None:
        import traci

        # Build the SUMO inputs under a separate name so that the Stage 1
        # generated files stay exactly as they are.
        config = episode_scenario_config(
            self.scenario_config, self.scenario, self.episode_duration,
            self.decision_interval, self._sumo_seed)
        self._files = build_scenario(config)

        cmd = [
            sumo_binary(self.render_mode == "human"),
            "-c", self._files["sumocfg"],
            "--seed", str(self._sumo_seed),
            "--start", "--quit-on-end",
        ]
        if not self.sumo_warnings:
            cmd += ["--no-warnings", "true"]
        cmd += [str(arg) for arg in self.extra_sumo_args]

        traci.start(cmd, label=self._label)
        self._conn = traci.getConnection(self._label)
        self._closed_cleanly = True

    # ------------------------------------------------------------------
    @staticmethod
    def _average(samples) -> Dict[str, Dict[str, float]]:
        """Mean queue and waiting time per approach over a decision interval."""
        count = float(len(samples))
        queues, waiting = {}, {}
        for approach in APPROACH_ORDER:
            queues[approach] = sum(s["queues"][approach] for s in samples) / count
            waiting[approach] = sum(s["waiting"][approach] for s in samples) / count
        return {"queues": queues, "waiting": waiting}

    # ------------------------------------------------------------------
    def _build_info(self, raw, reward_parts, action_info,
                    averaged=None) -> Dict[str, Any]:
        """The debugging / evaluation information returned with every step."""
        info: Dict[str, Any] = {
            "scenario": self.scenario,
            "simulation_time": float(self._conn.simulation.getTime()),
            "decision": self._decision,
            "phase": int(self.controller.phase),
            "phase_name": PHASE_NAMES[self.controller.phase],
            "green_elapsed": float(self.controller.green_elapsed),
            "queue_per_approach": dict(raw["queues"]),
            "waiting_per_approach": dict(raw["waiting"]),
            "total_queue": float(sum(raw["queues"].values())),
            "total_waiting_time": float(sum(raw["waiting"].values())),
            "vehicles_in_network": int(self._conn.vehicle.getIDCount()),
            "switch_count": int(self.controller.switch_count),
        }
        if action_info is not None:
            info["requested_action"] = action_info["requested_action"]
            info["requested_action_name"] = ACTION_NAMES[action_info["requested_action"]]
            info["effective_action"] = action_info["effective_action"]
            info["effective_action_name"] = ACTION_NAMES[action_info["effective_action"]]
            info["switch_blocked"] = action_info["switch_blocked"]
            info["forced_switch"] = action_info["forced_switch"]
        if reward_parts is not None:
            info["reward_components"] = {
                "waiting_penalty": reward_parts["waiting_penalty"],
                "queue_penalty": reward_parts["queue_penalty"],
                "fairness_penalty": reward_parts["fairness_penalty"],
            }
            info["interval_mean_queue"] = reward_parts["total_queue_length"]
            info["interval_mean_waiting"] = reward_parts["total_waiting_time"]
        return info

    # ------------------------------------------------------------------
    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
