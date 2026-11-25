"""
IntelliSignal
rl/phase_extend_env.py

Candidate B: a minimum-green lock, then demand-responsive extensions.

Where Candidate A chooses a whole green up front from a menu, this one
commits to the minimum green and then asks, every few seconds, whether to
keep going:

    phase starts
        run min_green           <- the lock, not a decision
        repeat:
            observe             <- AFTER the lock, so the queue the agent
            EXTEND -> run `extension` more seconds of green    sees is the
            END    -> run the clearance and hand over          one it is
        forced END at max_green                                deciding on

The order matters more than it looks. Until 25 Nov 2026 the lock was
served at the start of the step that followed it, which meant the first
EXTEND/END of every phase was decided from the junction as it looked
fifteen seconds earlier - before the green it was being asked about had
done any work. A policy cannot learn to extend on a queue it is not
shown, and the promoted model duly ended almost every green at the lock,
behaving as a fifteen-second fixed plan while reporting itself adaptive.
The lock is now served at the end of the step that starts the phase, so
every observation the agent acts on is the state at a real decision
point.

With the standard profile and a three-second extension the reachable
greens are 15, 18, 21, 24, 27 and 30 seconds - the same range Candidate A
covers, reached a different way. The comparison between them is
therefore about *when* the decision is made, not about what is reachable.

The point of the lock
---------------------
**A short decision interval must not become a short green interval.** The
agent is asked something every three seconds, but the shortest green it
can produce is fifteen, because the lock runs before the first question
is asked. That is the structural difference from the 2026-11-21 one-second
experiment, where a short decision cadence collapsed into short greens.

Action masking
--------------
At `max_green` only END is legal, and the mask is applied in the learning
target as well as in action selection - `agents/phase_dqn_agent.py` takes
`valid_actions`, and an illegal action is excluded from the bootstrap
maximum rather than merely avoided at run time.

Durations differ between steps - a lock-plus-extension step is eighteen
seconds, a plain extension is three - so the reward is integrated over
the seconds a step occupies and the bootstrap is discounted by
`gamma ** seconds`, exactly as in `rl/phase_environment.py`.
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

from rl.phase_action import (PHASE_NAMES, PHASE_NS, GREEN_STATE,
                             YELLOW_STATE, PhaseDurationController)
from rl.phase_state import OBSERVATION_SIZE, PhaseStateExtractor
from rl.timing_profile import get_profile
from simulation.run_simulation import prepare_sumo_imports, sumo_binary
from utils.config_loader import load_config
from utils.scenario_builder import ALL_RED, TLS_ID, build_scenario
from utils.sumo_launch import SumoStartError, start_sumo

EXTEND = 0
END = 1
ACTION_NAMES = {EXTEND: "EXTEND", END: "END"}
NUM_ACTIONS = 2

EXTEND_SUFFIX = "_ext_rl"


def extension_ladder(profile, extension: float) -> Tuple[float, ...]:
    """Every green the lock-and-extend scheme can actually produce."""
    greens = []
    green = float(profile.min_green)
    while green <= profile.max_green + 1e-9:
        greens.append(round(green, 3))
        green += float(extension)
    if greens[-1] < profile.max_green - 1e-9:
        greens.append(float(profile.max_green))
    return tuple(greens)


class PhaseExtendEnv(gym.Env):
    """Discrete(2): extend the current green, or end it."""

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, scenario: str = "heavy", profile: str = "standard",
                 extension: float = 3.0,
                 episode_duration: Optional[float] = None,
                 seed: Optional[int] = None,
                 waiting_weight: float = 1.0, queue_weight: float = 1.0,
                 reward_scale: float = 0.005,
                 gamma_per_second: float = 0.99,
                 state_extractor: Optional[PhaseStateExtractor] = None,
                 workload=None, tripinfo_path: Optional[str] = None,
                 tag: str = ""):
        super().__init__()
        prepare_sumo_imports()

        self.profile = get_profile(profile)
        self.extension = float(extension)
        self.workload = workload
        self.scenario = workload.base_scenario if workload is not None else scenario
        self.tag = str(tag)
        self.tripinfo_path = tripinfo_path
        self.step_callback = None

        self.scenario_config = load_config(self.scenario)
        signal = self.scenario_config["signal"]
        simulation = self.scenario_config["simulation"]

        self.step_length = float(
            workload.step_length if workload is not None
            else simulation["step_length"])
        if episode_duration is not None:
            self.episode_duration = float(episode_duration)
        elif workload is not None:
            self.episode_duration = float(workload.duration_s)
        else:
            self.episode_duration = float(simulation["duration"])
        self.total_steps = int(round(self.episode_duration / self.step_length))

        self.ladder = extension_ladder(self.profile, self.extension)
        self.controller = PhaseDurationController(
            durations=self.ladder,
            yellow_time=float(signal["yellow"]),
            all_red_time=float(signal["all_red"]),
            min_green=self.profile.min_green,
            max_green=self.profile.max_green,
            step_length=self.step_length, profile=self.profile)

        self.state_extractor = state_extractor or PhaseStateExtractor()
        self.waiting_weight = float(waiting_weight)
        self.queue_weight = float(queue_weight)
        self.reward_scale = float(reward_scale)
        self.gamma_per_second = float(gamma_per_second)

        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(OBSERVATION_SIZE,),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self._conn = None
        self._label = "intellisignal_ext_{}_{}".format(self.scenario, id(self))
        self._sumo_seed = seed if seed is not None else int(simulation["seed"])
        self._files = None
        self._steps_done = 0
        self._green_so_far = 0.0
        self._phase_open = False
        self._decision = 0
        self._greens: List[float] = []
        self._cumulative_reward = 0.0
        #: The lock of the current phase is served at the end of the step
        #: before it, so its seconds and its cost are carried here and
        #: charged to the decision they belong to.
        self._lock_cost = None
        self._lock_seconds = 0.0
        self._lock_hit_end = False
        self._green_truncated = False
        #: Called with the connection the moment SUMO is up and before
        #: a single second is simulated. A caller that measures the run
        #: attaches here, because `reset` itself advances the clock and
        #: anything attached afterwards would miss the opening seconds.
        self.on_connect = None

    # ==================================================================
    @property
    def connection(self):
        return self._conn

    def valid_actions(self) -> List[int]:
        """END only, once the green has reached the profile's maximum."""
        if self._green_so_far + self.extension > self.profile.max_green + 1e-9:
            return [END]
        return [EXTEND, END]

    # ==================================================================
    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._sumo_seed = int(seed)
        self.close()
        self._start_sumo()
        if self.on_connect is not None:
            self.on_connect(self._conn)

        self.controller.reset(PHASE_NS)
        self._steps_done = 0
        self._green_so_far = 0.0
        self._phase_open = False
        self._decision = 0
        self._greens = []
        self._cumulative_reward = 0.0
        self._lock_cost = None
        self._lock_seconds = 0.0
        self._lock_hit_end = False
        self._green_truncated = False

        # The first phase's lock runs before anything is asked, so the
        # first question is put at the same point in the phase as every
        # later one: the junction as it stands after the minimum green.
        raw = self._serve_lock()
        return self._observe(raw), self._info(raw)

    # ------------------------------------------------------------------
    def _serve_lock(self):
        """Show green and run the minimum, which is never a decision."""
        self._conn.trafficlight.setRedYellowGreenState(
            TLS_ID, GREEN_STATE[self.controller.phase])
        cost = {"waiting": 0.0, "queue": 0.0, "raw_q": 0.0, "raw_w": 0.0}
        ran, seconds, raw, hit_end = self._run(
            int(round(self.profile.min_green / self.step_length)), cost, 0.0)
        self._green_so_far += ran
        self._phase_open = True
        self._lock_cost = cost
        self._lock_seconds = seconds
        self._lock_hit_end = hit_end
        return raw if raw is not None else self.state_extractor.raw(self._conn)

    # ------------------------------------------------------------------
    def step(self, action: int):
        if self._conn is None:
            raise RuntimeError("step() called before reset().")
        action = int(action)
        if action not in (EXTEND, END):
            raise ValueError("Invalid action {}. Use 0 (EXTEND) or 1 (END)."
                             .format(action))

        phase = self.controller.phase
        # "Forced" means the agent asked to extend and the maximum green
        # refused it - not that it chose to end at the maximum itself.
        forced = (action == EXTEND and EXTEND not in self.valid_actions())
        if EXTEND not in self.valid_actions():
            action = END

        # The lock of this phase was served at the end of the previous
        # step, and its cost belongs to this decision - the agent is
        # choosing what happens from here, with those seconds already on
        # the clock.
        cost = self._lock_cost or {"waiting": 0.0, "queue": 0.0,
                                   "raw_q": 0.0, "raw_w": 0.0}
        seconds = self._lock_seconds
        green_ran = 0.0
        last_raw = None
        hit_end = self._lock_hit_end
        self._lock_cost = None
        self._lock_seconds = 0.0
        self._lock_hit_end = False

        if not hit_end and action == EXTEND:
            self._conn.trafficlight.setRedYellowGreenState(TLS_ID,
                                                           GREEN_STATE[phase])
            ran, seconds, last_raw, hit_end = self._run(
                int(round(self.extension / self.step_length)), cost, seconds)
            green_ran += ran

        self._green_so_far += green_ran
        ended = action == END or hit_end
        # Did the clock stop the GREEN, or only the clearance after it?
        green_cut_short = bool(hit_end)

        if ended and not hit_end:
            self._conn.trafficlight.setRedYellowGreenState(TLS_ID,
                                                           YELLOW_STATE[phase])
            _, seconds, last_raw, hit_end = self._run(
                int(round(self.controller.yellow_time / self.step_length)),
                cost, seconds)
            if self.controller.all_red_time > 0 and not hit_end:
                self._conn.trafficlight.setRedYellowGreenState(TLS_ID, ALL_RED)
                _, seconds, last_raw, hit_end = self._run(
                    int(round(self.controller.all_red_time / self.step_length)),
                    cost, seconds)

        if last_raw is None:
            last_raw = self.state_extractor.raw(self._conn)

        if ended:
            served = self._green_so_far
            index = min(range(len(self.ladder)),
                        key=lambda i: abs(self.ladder[i] - served))
            # Only a green the clock cut short is evidence of nothing. A
            # green that ran its full length and then had its clearance
            # cut off was still chosen by the controller.
            green_truncated = bool(hit_end) and green_cut_short
            self.controller.commit(
                index, served=served, truncated=green_truncated,
                info={"requested_action": int(action),
                      "executed_action": int(action),
                      "forced_end_at_max_green": bool(forced),
                      "extensions": int(round(
                          max(served - self.profile.min_green, 0.0)
                          / self.extension))})
            self._greens.append(served)
            self._green_so_far = 0.0
            self._phase_open = False
            self._green_truncated = green_truncated
            if not hit_end:
                # `commit` has already handed the junction to the other
                # group. Serving its lock here is what makes the next
                # observation a decision point rather than a phase start.
                last_raw = self._serve_lock()

        self._decision += 1
        reward = -self.reward_scale * (self.waiting_weight * cost["waiting"]
                                       + self.queue_weight * cost["queue"])
        self._cumulative_reward += reward

        # Every controller is given the same horizon. Stopping early
        # when the network happens to drain would average the queue over
        # fewer seconds than a controller that ran to the end, and the
        # two numbers would then not be comparable.
        terminated = False
        truncated = bool(self._steps_done >= self.total_steps)

        info = self._info(last_raw)
        info.update({
            "action": action, "action_name": ACTION_NAMES[action],
            "forced_end_at_max_green": bool(forced),
            "green_so_far": float(self._green_so_far),
            "green_served": float(self._greens[-1]) if ended and self._greens else 0.0,
            "phase_ended": bool(ended),
            # A green the end of the episode cut short was not chosen by
            # the controller, so it is no evidence about the controller.
            # A green that finished and only lost its clearance was.
            "truncated_by_episode_end": bool(ended and green_cut_short),
            "duration": float(max(seconds, self.step_length)),
            "valid_actions": self.valid_actions(),
        })
        if terminated or truncated:
            info["episode_return"] = self._cumulative_reward
            info["episode_greens"] = list(self._greens)
            info["timing_compliance"] = self.controller.compliance()
            info["decision_log"] = list(self.controller.decision_log)
        return self._observe(last_raw), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    def _run(self, steps: int, cost: Dict[str, float], seconds: float):
        """Advance `steps` seconds, accumulating the discounted cost."""
        ran = 0.0
        last_raw = None
        hit_end = False
        for _ in range(max(steps, 0)):
            if self._steps_done >= self.total_steps:
                hit_end = True
                break
            self._conn.simulationStep()
            self._steps_done += 1
            if self.step_callback is not None:
                self.step_callback()
            last_raw = self.state_extractor.raw(self._conn)
            within = self.gamma_per_second ** seconds
            waiting = sum(last_raw["waiting"].values())
            queue = sum(last_raw["queues"].values())
            cost["waiting"] += within * waiting
            cost["queue"] += within * queue
            cost["raw_w"] += waiting
            cost["raw_q"] += queue
            seconds += self.step_length
            ran += self.step_length
        return ran, seconds, last_raw, hit_end

    # ------------------------------------------------------------------
    def _observe(self, raw) -> np.ndarray:
        phase = self.controller.phase
        return self.state_extractor.observe(
            raw, phase,
            last_served_green=self.controller.last_green[phase],
            last_other_green=self.controller.last_green[1 - phase])

    def _info(self, raw) -> Dict[str, Any]:
        phase = self.controller.phase
        return {
            "scenario": self.scenario, "profile": self.profile.name,
            "simulation_time": float(self._conn.simulation.getTime()),
            "decision": self._decision, "phase": int(phase),
            "phase_name": PHASE_NAMES[phase],
            "steps_done": self._steps_done,
            "total_queue": float(sum(raw["queues"].values())),
            "total_waiting_time": float(sum(raw["waiting"].values())),
            "switch_count": int(self.controller.switch_count),
        }

    # ------------------------------------------------------------------
    def _build_inputs(self):
        if self.workload is not None:
            from custom_traffic.builder import build_workload
            return build_workload(self.workload,
                                  tag=EXTEND_SUFFIX + self.tag,
                                  duration=self.episode_duration)
        config = copy.deepcopy(self.scenario_config)
        config["scenario"]["name"] = self.scenario + EXTEND_SUFFIX + self.tag
        config["simulation"]["duration"] = float(self.episode_duration)
        config["simulation"]["seed"] = int(self._sumo_seed)
        return build_scenario(config)

    def _start_sumo(self) -> None:
        import traci
        # Two tries, rebuilding the inputs in between - see the note in
        # rl/phase_environment.py.
        for attempt in (1, 2):
            self._files = self._build_inputs()
            if self.tripinfo_path:
                from utils.delay_accounting import enable_tripinfo
                enable_tripinfo(self._files["sumocfg"], self.tripinfo_path)
            try:
                self._conn = start_sumo(
                    [sumo_binary(False), "-c", self._files["sumocfg"],
                     "--seed", str(self._sumo_seed), "--start",
                     "--quit-on-end", "--no-warnings", "true"],
                    label=self._label)
                break
            except SumoStartError:
                if attempt == 2:
                    raise
        self.state_extractor.subscribe(self._conn)

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            finally:
                self._conn = None

    def cleanup_generated_inputs(self) -> None:
        import glob
        if self.workload is not None:
            from custom_traffic.builder import cleanup_workload_inputs
            cleanup_workload_inputs(self.workload, tag=EXTEND_SUFFIX + self.tag)
            return
        stem = self.scenario + EXTEND_SUFFIX + self.tag
        from utils.scenario_builder import (ADDITIONAL_DIR, ROUTES_DIR,
                                            SUMOCFG_DIR)
        for directory, pattern in ((ROUTES_DIR, "{}*"),
                                   (ADDITIONAL_DIR, "{}*"),
                                   (SUMOCFG_DIR, "{}*")):
            for path in glob.glob(os.path.join(directory,
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
