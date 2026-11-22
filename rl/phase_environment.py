"""
IntelliSignal
rl/phase_environment.py

The Gymnasium environment for the duration action space.

One `step()` is one whole green plus its clearance:

    step(i)  ->  run GREEN for durations[i] seconds
                 run YELLOW for yellow_time seconds
                 run ALL-RED for all_red_time seconds   (skipped when 0)
                 the phase changes; return the next observation

It drives the same intersection, the same demand and the same SUMO inputs as
every other part of the project: the scenario is built with the Stage 1
`build_scenario`, so the road, the flows and the vehicle type are identical
to the fixed-time baseline it is measured against.

The reward is an INTEGRAL, not a mean
-------------------------------------
Decisions here cover different amounts of simulated time - a 4 s green plus
clearance is 8 s, a 20 s green plus clearance is 24 s. The Stage 2 reward is
the mean congestion over the interval, which would charge those two decisions
the same amount for three times the traffic. This reward sums the congestion
over every second the decision occupies:

    reward = -scale * SUM over the decision's seconds of
                 ( waiting_weight * total current waiting time
                 + queue_weight   * halting vehicles )

so a decision pays for exactly the congestion it presides over. Together with
`DQNConfig.gamma_per_second`, which discounts by `gamma ** seconds`, that is
the standard semi-Markov treatment of a variable-duration transition. Getting
only one of the two right is worse than getting neither: the cost would be
charged over the real time span while the future was discounted as though no
time had passed.

Episode length
--------------
An episode is exactly `episode_duration` simulated seconds - 1800 s by
default, the same as the Stage 1 baseline and the same as
`evaluation/fixed_time_sweep.py`, which runs that many steps and no more.
The last decision of an episode is cut short when it would run past the end,
rather than being allowed to overrun, so two controllers compared on the same
seed see exactly the same 1800 seconds of traffic.
"""

from __future__ import annotations

import copy
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl.phase_action import (DEFAULT_DURATIONS, PHASE_NAMES,       # noqa: E402
                             PHASE_NS, PhaseDurationController)
from rl.phase_state import (OBSERVATION_SIZE,                      # noqa: E402
                            PhaseStateExtractor)
from simulation.run_simulation import (prepare_sumo_imports,       # noqa: E402
                                       sumo_binary)
from utils.config_loader import load_config                        # noqa: E402
from utils.scenario_builder import TLS_ID, build_scenario          # noqa: E402

#: Generated SUMO inputs get this suffix, so no Stage 1 or Stage 2 file is
#: ever overwritten.
PHASE_SUFFIX = "_phase_rl"


class PhaseDurationEnv(gym.Env):
    """Choose each green's duration at the moment that green starts.

    Observation : Box(0, 1, shape=(10,), float32)   - see rl/phase_state.py
    Action      : Discrete(len(durations))          - see rl/phase_action.py
    Reward      : negative integrated congestion    - see the module docstring
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, scenario: str = "heavy",
                 durations=DEFAULT_DURATIONS,
                 episode_duration: Optional[float] = None,
                 seed: Optional[int] = None,
                 render_mode: Optional[str] = None,
                 waiting_weight: float = 1.0,
                 queue_weight: float = 1.0,
                 reward_scale: float = 0.01,
                 gamma_per_second: float = 1.0,
                 state_extractor: Optional[PhaseStateExtractor] = None,
                 extra_sumo_args: Optional[list] = None,
                 tag: str = ""):
        super().__init__()

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError("Unsupported render_mode: {}".format(render_mode))

        prepare_sumo_imports()

        self.scenario = scenario
        self.render_mode = render_mode
        self.tag = str(tag)
        self.extra_sumo_args = list(extra_sumo_args or [])

        #: Optional hook called after every simulated second, so the Stage 1
        #: MetricsCollector can measure an agent-driven run with the code
        #: that measured the fixed-time baseline.
        self.step_callback = None

        self.scenario_config = load_config(scenario)
        signal = self.scenario_config["signal"]
        simulation = self.scenario_config["simulation"]

        self.step_length = float(simulation["step_length"])
        self.episode_duration = float(
            episode_duration if episode_duration is not None
            else simulation["duration"])
        self.total_steps = int(round(self.episode_duration / self.step_length))

        self.controller = PhaseDurationController(
            durations=durations,
            yellow_time=float(signal["yellow"]),
            all_red_time=float(signal["all_red"]),
            min_green=float(min(durations)),
            max_green=float(max(durations)),
            step_length=self.step_length,
        )
        self.state_extractor = state_extractor or PhaseStateExtractor()

        self.waiting_weight = float(waiting_weight)
        self.queue_weight = float(queue_weight)
        self.reward_scale = float(reward_scale)
        if self.reward_scale <= 0:
            raise ValueError("reward_scale must be positive.")
        # The congestion inside one decision is discounted by how far into
        # the decision it happens, with the same per-second rate the agent
        # bootstraps with. Without it, a 24 s decision is charged the last
        # of its seconds at full weight while the agent values the state
        # that follows at gamma**24 - which quietly overcharges long greens
        # and pushes the policy towards the shortest action on the menu.
        self.gamma_per_second = float(gamma_per_second)
        if not 0.0 < self.gamma_per_second <= 1.0:
            raise ValueError("gamma_per_second must be in (0, 1].")

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBSERVATION_SIZE,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.controller.num_actions)

        self._conn = None
        self._label = "intellisignal_phase_{}_{}".format(scenario, id(self))
        self._sumo_seed = seed if seed is not None else int(simulation["seed"])
        self._files = None
        self._steps_done = 0
        self._decision = 0
        self._cumulative_reward = 0.0
        self._episode_queue_cost = 0.0
        self._episode_waiting_cost = 0.0
        self._greens: List[float] = []

    # ==================================================================
    # Gymnasium API
    # ==================================================================
    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None
              ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._sumo_seed = int(seed)

        self.close()
        self._start_sumo()

        self.controller.reset(PHASE_NS)
        self._steps_done = 0
        self._decision = 0
        self._cumulative_reward = 0.0
        self._episode_queue_cost = 0.0
        self._episode_waiting_cost = 0.0
        self._greens = []

        self._conn.trafficlight.setRedYellowGreenState(
            TLS_ID, self.controller.current_state())

        raw = self.state_extractor.raw(self._conn)
        observation = self._observe(raw)
        return observation, self._build_info(raw, None)

    # ------------------------------------------------------------------
    def step(self, action: int
             ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._conn is None:
            raise RuntimeError("step() called before reset(); the environment "
                               "has no running SUMO simulation.")
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(
                "Invalid action {}. Valid actions are 0..{}."
                .format(action, self.controller.num_actions - 1))

        requested_green = self.controller.duration_for(action)
        plan = self.controller.plan(action)

        waiting_cost = 0.0
        queue_cost = 0.0
        raw_queue = 0.0
        raw_waiting = 0.0
        seconds = 0.0
        green_served = 0.0
        last_raw = None
        hit_end = False

        for state, steps in plan:
            if hit_end:
                break
            is_green = state == self.controller.current_state()
            self._conn.trafficlight.setRedYellowGreenState(TLS_ID, state)
            for _ in range(steps):
                if self._steps_done >= self.total_steps:
                    hit_end = True
                    break
                self._conn.simulationStep()
                self._steps_done += 1
                if self.step_callback is not None:
                    self.step_callback()
                last_raw = self.state_extractor.raw(self._conn)
                within = self.gamma_per_second ** seconds
                waiting_cost += within * sum(last_raw["waiting"].values())
                queue_cost += within * sum(last_raw["queues"].values())
                raw_queue += sum(last_raw["queues"].values())
                raw_waiting += sum(last_raw["waiting"].values())
                seconds += self.step_length
                if is_green:
                    green_served += self.step_length

        if last_raw is None:                 # the episode ended on the boundary
            last_raw = self.state_extractor.raw(self._conn)

        # The green that was actually served, which is what the controller
        # must record - a decision cut short at 1800 s did not get its full
        # duration and the observation must not claim it did.
        self.controller.commit(action)
        self.controller.last_green[1 - self.controller.phase] = green_served
        self._greens.append(green_served)
        self._decision += 1

        self._episode_queue_cost += raw_queue
        self._episode_waiting_cost += raw_waiting
        reward = -self.reward_scale * (self.waiting_weight * waiting_cost
                                       + self.queue_weight * queue_cost)
        self._cumulative_reward += reward

        observation = self._observe(last_raw)

        no_traffic_left = self._conn.simulation.getMinExpectedNumber() <= 0
        terminated = bool(no_traffic_left)
        truncated = bool(self._steps_done >= self.total_steps)

        info = self._build_info(last_raw, action)
        info.update({
            "requested_green": float(requested_green),
            "green_served": float(green_served),
            # `duration` is what the semi-Markov discount uses. It is the
            # simulated seconds this decision actually occupied, clearance
            # included, which is not the same as the green it asked for.
            "duration": float(max(seconds, self.step_length)),
            "interval_waiting_cost": float(waiting_cost),
            "interval_queue_cost": float(queue_cost),
        })
        if terminated or truncated:
            info["episode_return"] = self._cumulative_reward
            info["episode_decisions"] = self._decision
            info["episode_greens"] = list(self._greens)
            # Cheap, collector-free episode statistics. The mean queue is
            # the integral of halting vehicles divided by the steps run,
            # which is exactly what MetricsCollector reports - so a training
            # episode can be logged without paying for the per-vehicle
            # accumulated-waiting scan the collector does every step.
            steps = max(self._steps_done, 1)
            info["episode_mean_queue"] = self._episode_queue_cost / steps
            info["episode_mean_waiting_sum"] = self._episode_waiting_cost / steps

        return observation, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            finally:
                self._conn = None

    def render(self):
        return None

    @property
    def connection(self):
        return self._conn

    # ==================================================================
    # internals
    # ==================================================================
    def _observe(self, raw) -> np.ndarray:
        phase = self.controller.phase
        return self.state_extractor.observe(
            raw, phase,
            last_served_green=self.controller.last_green[phase],
            last_other_green=self.controller.last_green[1 - phase])

    # ------------------------------------------------------------------
    def episode_scenario_config(self) -> Dict[str, Any]:
        """The Stage 1 scenario, adjusted for one episode of this environment.

        The duration is the episode length exactly - not the episode length
        plus a margin - because `evaluation/fixed_time_sweep.py` builds its
        baseline the same way. Two controllers compared on one seed must be
        given byte-identical SUMO inputs, and the flow definitions carry the
        duration, so a margin here would quietly change the demand file.
        """
        config = copy.deepcopy(self.scenario_config)
        config["scenario"]["name"] = self.scenario + PHASE_SUFFIX + self.tag
        config["simulation"]["duration"] = float(self.episode_duration)
        config["simulation"]["seed"] = int(self._sumo_seed)
        return config

    # ------------------------------------------------------------------
    def _start_sumo(self) -> None:
        import traci

        self._files = build_scenario(self.episode_scenario_config())
        cmd = [
            sumo_binary(self.render_mode == "human"),
            "-c", self._files["sumocfg"],
            "--seed", str(self._sumo_seed),
            "--start", "--quit-on-end",
            "--no-warnings", "true",
        ]
        cmd += [str(arg) for arg in self.extra_sumo_args]
        traci.start(cmd, label=self._label)
        self._conn = traci.getConnection(self._label)
        # Batch the per-second lane readings into one round-trip. TraCI
        # latency, not SUMO, is what an episode spends its time on.
        self.state_extractor.subscribe(self._conn)

    # ------------------------------------------------------------------
    def _build_info(self, raw, action: Optional[int]) -> Dict[str, Any]:
        phase = self.controller.phase
        other = 1 - phase
        info: Dict[str, Any] = {
            "scenario": self.scenario,
            "simulation_time": float(self._conn.simulation.getTime()),
            "decision": self._decision,
            "steps_done": self._steps_done,
            "phase": int(phase),
            "phase_name": PHASE_NAMES[phase],
            "served_queue": raw["queues"][phase],
            "other_queue": raw["queues"][other],
            "total_queue": float(sum(raw["queues"].values())),
            "total_waiting_time": float(sum(raw["waiting"].values())),
            "vehicles_in_network": int(self._conn.vehicle.getIDCount()),
            "switch_count": int(self.controller.switch_count),
        }
        if action is not None:
            info["action"] = int(action)
        return info

    # ------------------------------------------------------------------
    def cleanup_generated_inputs(self) -> None:
        """Remove this environment's throwaway SUMO inputs."""
        import glob
        stem = self.scenario + PHASE_SUFFIX + self.tag
        for pattern in ("sumo/routes/{}*", "sumo/config/{}*",
                        "sumo/additional/{}*"):
            for path in glob.glob(os.path.join(PROJECT_ROOT,
                                               pattern.format(stem))):
                try:
                    os.remove(path)
                except OSError:
                    pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
