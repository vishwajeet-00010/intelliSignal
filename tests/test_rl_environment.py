"""
IntelliSignal - Stage 2
tests/test_rl_environment.py

Tests for the reinforcement-learning environment.

**No DQN, no agent and no training is tested here, because none exists.**
These tests check that the environment itself behaves the way Gymnasium and,
later, a DQN will require:

  * it can be created for both Stage 1 scenarios
  * reset() and step() follow the modern Gymnasium API
  * the observation always belongs to the observation space
  * both actions work, and the safety rules around switching hold
  * the reward is a finite number and its components add up
  * info carries the metrics needed for debugging and evaluation
  * SUMO is started and shut down cleanly
  * the Stage 1 files and results are not touched

Episodes here are 60-120 simulated seconds, not 1800, so the suite stays fast.

Run with:
    python -m pytest -q
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl.action import (ACTION_KEEP, ACTION_SWITCH, PHASE_EW,        # noqa: E402
                       PHASE_NS, ActionController)
from rl.config import RLConfigError, load_rl_config, validate_rl_config  # noqa: E402
from rl.environment import IntelliSignalEnv                          # noqa: E402
from rl.reward import RewardFunction                                 # noqa: E402
from rl.state import (APPROACH_LANES, APPROACH_ORDER,                # noqa: E402
                      OBSERVATION_LABELS, OBSERVATION_SIZE)
from utils.config_loader import available_scenarios, load_config     # noqa: E402
from utils.scenario_builder import (ALL_RED, EW_GREEN, EW_YELLOW,    # noqa: E402
                                    NS_GREEN, NS_YELLOW, TLS_ID)

SCENARIOS = ["moderate", "heavy"]

# Short episodes keep the suite fast.
SHORT_EPISODE = 60.0     # 6 decisions at a 10 s decision interval
MEDIUM_EPISODE = 120.0   # 12 decisions


@pytest.fixture
def env():
    """A moderate-scenario environment that is always closed afterwards."""
    environment = IntelliSignalEnv(scenario="moderate",
                                   episode_duration=MEDIUM_EPISODE)
    try:
        yield environment
    finally:
        environment.close()


# ==================================================================
# configuration
# ==================================================================
def test_rl_config_loads_for_both_scenarios():
    for scenario in SCENARIOS:
        config = load_rl_config(load_config(scenario))
        env_cfg = config["environment"]
        assert env_cfg["decision_interval"] > 0
        # null placeholders must have been filled in from the scenario file
        assert env_cfg["yellow_time"] is not None
        assert env_cfg["all_red_time"] is not None
        assert env_cfg["episode_duration"] is not None


def test_rl_config_inherits_the_stage1_signal_timings():
    """The RL environment must use the same clearance as the Stage 1 signal."""
    for scenario in SCENARIOS:
        scenario_config = load_config(scenario)
        rl_config = load_rl_config(scenario_config)
        assert rl_config["environment"]["yellow_time"] == float(
            scenario_config["signal"]["yellow"])
        assert rl_config["environment"]["all_red_time"] == float(
            scenario_config["signal"]["all_red"])


def test_decision_interval_must_exceed_the_clearance_time():
    bad = load_rl_config(load_config("moderate"))
    bad["environment"]["decision_interval"] = 3      # shorter than yellow (4 s)
    with pytest.raises(RLConfigError):
        validate_rl_config(bad)


def test_rl_config_file_does_not_add_a_scenario():
    """config/rl/environment.yaml must not appear as a traffic scenario."""
    assert sorted(available_scenarios()) == sorted(SCENARIOS)


# ==================================================================
# action controller (no SUMO needed - pure logic)
# ==================================================================
def _controller(**kwargs) -> ActionController:
    defaults = dict(decision_interval=10, yellow_time=4, all_red_time=0,
                    min_green=10, max_green=60, step_length=1.0)
    defaults.update(kwargs)
    return ActionController(**defaults)


def test_keep_holds_the_current_green_for_the_whole_interval():
    controller = _controller()
    plan = controller.plan(ACTION_KEEP)
    assert plan == [(NS_GREEN, 10)]


def test_switch_passes_through_yellow_before_the_other_green():
    controller = _controller()
    controller.green_elapsed = 30
    effective, _ = controller.resolve(ACTION_SWITCH)
    assert effective == ACTION_SWITCH
    plan = controller.plan(effective)
    assert plan == [(NS_YELLOW, 4), (EW_GREEN, 6)]
    # the segments must fill exactly one decision interval
    assert sum(steps for _, steps in plan) == 10


def test_switch_inserts_all_red_when_configured():
    controller = _controller(all_red_time=2)
    controller.green_elapsed = 30
    plan = controller.plan(ACTION_SWITCH)
    assert plan == [(NS_YELLOW, 4), (ALL_RED, 2), (EW_GREEN, 4)]


def test_green_never_turns_directly_into_the_other_green():
    """Safety: a switch plan must never put two greens back to back."""
    controller = _controller()
    controller.green_elapsed = 30
    states = [state for state, _ in controller.plan(ACTION_SWITCH)]
    assert states[0] in (NS_YELLOW, EW_YELLOW)
    assert NS_GREEN not in states[:1] and EW_GREEN not in states[:1]


def test_switch_is_refused_before_min_green():
    controller = _controller()
    controller.green_elapsed = 5          # min_green is 10
    effective, info = controller.resolve(ACTION_SWITCH)
    assert effective == ACTION_KEEP
    assert info["switch_blocked"] is True
    assert info["forced_switch"] is False


def test_keep_is_overridden_into_a_switch_after_max_green():
    controller = _controller()
    controller.green_elapsed = 60         # max_green is 60
    effective, info = controller.resolve(ACTION_KEEP)
    assert effective == ACTION_SWITCH
    assert info["forced_switch"] is True


def test_commit_alternates_the_phase_and_resets_the_green_timer():
    controller = _controller()
    assert controller.phase == PHASE_NS
    controller.green_elapsed = 30
    controller.commit(ACTION_SWITCH)
    assert controller.phase == PHASE_EW
    assert controller.green_elapsed == 6          # 10 s interval - 4 s yellow
    controller.commit(ACTION_KEEP)
    assert controller.green_elapsed == 16
    assert controller.switch_count == 1


def test_controller_rejects_an_interval_shorter_than_the_clearance():
    with pytest.raises(ValueError):
        _controller(decision_interval=4, yellow_time=4)


# ==================================================================
# reward (no SUMO needed - pure arithmetic)
# ==================================================================
def _measurements(queue: float, waiting: float):
    return {"queues": {a: queue for a in APPROACH_ORDER},
            "waiting": {a: waiting for a in APPROACH_ORDER}}


def test_reward_is_negative_when_there_is_congestion():
    parts = RewardFunction()(_measurements(queue=5, waiting=50))
    assert parts["reward"] < 0


def test_reward_is_zero_on_an_empty_intersection():
    parts = RewardFunction()(_measurements(queue=0, waiting=0))
    assert parts["reward"] == 0.0


def test_less_congestion_gives_a_higher_reward():
    light = RewardFunction()(_measurements(queue=2, waiting=10))["reward"]
    heavy = RewardFunction()(_measurements(queue=20, waiting=400))["reward"]
    assert light > heavy


def test_reward_components_add_up_to_the_reward():
    reward_fn = RewardFunction(fairness_weight=0.5)
    parts = reward_fn(_measurements(queue=4, waiting=40))
    total = (parts["waiting_penalty"] + parts["queue_penalty"]
             + parts["fairness_penalty"])
    assert parts["reward"] == pytest.approx(total)


# ==================================================================
# environment: construction and spaces
# ==================================================================
def test_environment_can_be_created(env):
    assert env.scenario == "moderate"
    assert env.decision_interval > 0


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_both_scenarios_initialise(scenario):
    environment = IntelliSignalEnv(scenario=scenario,
                                   episode_duration=SHORT_EPISODE)
    try:
        observation, info = environment.reset(seed=1)
        assert environment.observation_space.contains(observation)
        assert info["scenario"] == scenario
    finally:
        environment.close()


def test_observation_space_is_the_documented_shape(env):
    assert env.observation_space.shape == (OBSERVATION_SIZE,)
    assert OBSERVATION_SIZE == 10
    assert len(OBSERVATION_LABELS) == 10
    assert env.observation_space.dtype == np.float32


def test_action_space_contains_both_actions(env):
    assert env.action_space.n == 2
    assert env.action_space.contains(ACTION_KEEP)     # action 0
    assert env.action_space.contains(ACTION_SWITCH)   # action 1
    assert not env.action_space.contains(2)


def test_every_approach_maps_to_two_incoming_lanes():
    assert len(APPROACH_LANES) == 4
    for approach in APPROACH_ORDER:
        assert len(APPROACH_LANES[approach]) == 2


# ==================================================================
# environment: reset
# ==================================================================
def test_reset_returns_observation_and_info(env):
    observation, info = env.reset(seed=42)
    assert isinstance(observation, np.ndarray)
    assert observation.shape == (OBSERVATION_SIZE,)
    assert observation.dtype == np.float32
    assert isinstance(info, dict)


def test_reset_observation_is_inside_the_observation_space(env):
    observation, _ = env.reset(seed=42)
    assert env.observation_space.contains(observation)


def test_reset_starts_on_the_north_south_green(env):
    _, info = env.reset(seed=42)
    assert info["phase"] == PHASE_NS
    assert info["phase_name"] == "North-South green"
    assert env._conn.trafficlight.getRedYellowGreenState(TLS_ID) == NS_GREEN


def test_reset_can_be_called_twice(env):
    env.reset(seed=1)
    env.step(ACTION_KEEP)
    observation, info = env.reset(seed=1)
    assert info["decision"] == 0
    assert env.observation_space.contains(observation)


def test_reset_with_the_same_seed_is_reproducible():
    results = []
    for _ in range(2):
        environment = IntelliSignalEnv(scenario="moderate",
                                       episode_duration=SHORT_EPISODE)
        try:
            environment.reset(seed=7)
            rewards = [environment.step(ACTION_KEEP)[1] for _ in range(5)]
            results.append(rewards)
        finally:
            environment.close()
    assert results[0] == pytest.approx(results[1])


# ==================================================================
# environment: step
# ==================================================================
def test_step_keep_returns_the_five_element_gymnasium_tuple(env):
    env.reset(seed=42)
    result = env.step(ACTION_KEEP)
    assert len(result) == 5
    observation, reward, terminated, truncated, info = result
    assert env.observation_space.contains(observation)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_step_switch_works_and_changes_the_phase(env):
    env.reset(seed=42)
    # hold long enough that min_green allows a switch
    env.step(ACTION_KEEP)
    env.step(ACTION_KEEP)
    phase_before = env.controller.phase
    observation, reward, terminated, truncated, info = env.step(ACTION_SWITCH)
    assert info["effective_action"] == ACTION_SWITCH
    assert info["phase"] != phase_before
    assert env.observation_space.contains(observation)


def test_reward_is_a_finite_number(env):
    env.reset(seed=42)
    for action in (ACTION_KEEP, ACTION_SWITCH, ACTION_KEEP):
        _, reward, _, _, _ = env.step(action)
        assert isinstance(reward, float)
        assert math.isfinite(reward)
        assert reward <= 0.0


def test_terminated_and_truncated_are_booleans(env):
    env.reset(seed=42)
    _, _, terminated, truncated, _ = env.step(ACTION_KEEP)
    assert terminated in (True, False)
    assert truncated in (True, False)
    assert not (terminated and truncated)


def test_info_contains_the_expected_metrics(env):
    env.reset(seed=42)
    _, _, _, _, info = env.step(ACTION_KEEP)
    for key in ("simulation_time", "decision", "phase", "phase_name",
                "queue_per_approach", "waiting_per_approach", "total_queue",
                "total_waiting_time", "vehicles_in_network",
                "requested_action", "effective_action", "switch_blocked",
                "forced_switch", "reward_components"):
        assert key in info, "info is missing '{}'".format(key)
    assert set(info["queue_per_approach"]) == set(APPROACH_ORDER)
    assert set(info["waiting_per_approach"]) == set(APPROACH_ORDER)
    for key in ("waiting_penalty", "queue_penalty", "fairness_penalty"):
        assert key in info["reward_components"]


def test_simulation_time_advances_by_one_decision_interval(env):
    env.reset(seed=42)
    _, _, _, _, first = env.step(ACTION_KEEP)
    _, _, _, _, second = env.step(ACTION_KEEP)
    assert second["simulation_time"] - first["simulation_time"] == pytest.approx(
        env.decision_interval)


def test_environment_runs_many_steps(env):
    env.reset(seed=42)
    rng = np.random.default_rng(0)
    steps = 0
    for _ in range(10):
        action = int(rng.integers(0, 2))
        observation, reward, terminated, truncated, _ = env.step(action)
        assert env.observation_space.contains(observation)
        steps += 1
        if terminated or truncated:
            break
    assert steps >= 5


def test_episode_is_truncated_at_the_configured_length():
    environment = IntelliSignalEnv(scenario="moderate",
                                   episode_duration=SHORT_EPISODE)
    try:
        environment.reset(seed=42)
        truncated = False
        for _ in range(environment.max_decisions):
            _, _, terminated, truncated, _ = environment.step(ACTION_KEEP)
            if terminated or truncated:
                break
        assert truncated is True
        assert environment._decision == environment.max_decisions
    finally:
        environment.close()


def test_an_invalid_action_is_rejected(env):
    env.reset(seed=42)
    with pytest.raises(ValueError):
        env.step(5)


def test_step_before_reset_is_rejected():
    environment = IntelliSignalEnv(scenario="moderate",
                                   episode_duration=SHORT_EPISODE)
    try:
        with pytest.raises(RuntimeError):
            environment.step(ACTION_KEEP)
    finally:
        environment.close()


def test_signal_state_is_held_for_the_whole_segment(env):
    """The state set through TraCI must persist across simulation steps."""
    env.reset(seed=42)
    env.step(ACTION_KEEP)
    state = env._conn.trafficlight.getRedYellowGreenState(TLS_ID)
    assert state == NS_GREEN
    assert env._conn.trafficlight.getProgram(TLS_ID) == "online"


# ==================================================================
# environment: shutdown
# ==================================================================
def test_environment_closes_cleanly(env):
    env.reset(seed=42)
    env.step(ACTION_KEEP)
    env.close()
    assert env._conn is None
    assert env._closed_cleanly is True


def test_close_is_safe_to_call_twice(env):
    env.reset(seed=42)
    env.close()
    env.close()
    assert env._conn is None


def test_environment_can_be_reused_after_close(env):
    env.reset(seed=42)
    env.close()
    observation, _ = env.reset(seed=42)
    assert env.observation_space.contains(observation)


# ==================================================================
# Gymnasium compliance
# ==================================================================
def test_passes_the_gymnasium_environment_checker():
    from gymnasium.utils.env_checker import check_env

    environment = IntelliSignalEnv(scenario="moderate",
                                   episode_duration=SHORT_EPISODE)
    try:
        check_env(environment, skip_render_check=True)
    finally:
        environment.close()


def test_environment_is_registered_with_gymnasium():
    import gymnasium as gym

    import rl  # noqa: F401  (importing registers the id)

    environment = gym.make("IntelliSignal-v0", scenario="moderate",
                           episode_duration=SHORT_EPISODE)
    try:
        observation, info = environment.reset(seed=3)
        assert environment.observation_space.contains(observation)
    finally:
        environment.close()


# ==================================================================
# SUMO-RL compatibility
# ==================================================================
def test_sumo_rl_is_importable():
    from rl.sumo_rl_adapter import sumo_rl_available, sumo_rl_version

    if not sumo_rl_available():
        pytest.skip("sumo-rl is not installed in this environment")
    assert sumo_rl_version() is not None


def test_sumo_rl_environment_runs_on_the_stage1_network():
    """Proves the Stage 1 intersection is a valid SUMO-RL environment."""
    from rl.sumo_rl_adapter import make_sumo_rl_env, sumo_rl_available

    if not sumo_rl_available():
        pytest.skip("sumo-rl is not installed in this environment")

    environment = make_sumo_rl_env("moderate", num_seconds=100)
    try:
        observation, info = environment.reset()
        assert environment.observation_space.contains(observation)
        observation, reward, terminated, truncated, info = environment.step(
            environment.action_space.sample())
        assert math.isfinite(float(reward))
    finally:
        environment.close()


# ==================================================================
# Stage 1 must not be disturbed
# ==================================================================
def test_rl_environment_does_not_touch_the_stage1_sumo_files():
    """The RL env writes <scenario>_rl.* and must leave the Stage 1 files alone."""
    routes = os.path.join(PROJECT_ROOT, "sumo", "routes", "moderate.rou.xml")
    sumocfg = os.path.join(PROJECT_ROOT, "sumo", "config", "moderate.sumocfg")
    if not (os.path.isfile(routes) and os.path.isfile(sumocfg)):
        pytest.skip("Stage 1 SUMO files have not been generated yet")

    before = {path: os.path.getmtime(path) for path in (routes, sumocfg)}

    environment = IntelliSignalEnv(scenario="moderate",
                                   episode_duration=SHORT_EPISODE)
    try:
        environment.reset(seed=42)
        environment.step(ACTION_KEEP)
    finally:
        environment.close()

    for path, mtime in before.items():
        assert os.path.getmtime(path) == mtime, (
            "the RL environment modified a Stage 1 file: {}".format(path))

    # ... and it did create its own separate ones
    assert os.path.isfile(os.path.join(PROJECT_ROOT, "sumo", "routes",
                                       "moderate_rl.rou.xml"))


def test_rl_environment_does_not_touch_the_stage1_results():
    results = os.path.join(PROJECT_ROOT, "results", "csv")
    if not os.path.isdir(results):
        pytest.skip("Stage 1 results have not been generated yet")

    files = [os.path.join(results, name) for name in os.listdir(results)
             if name.endswith(".csv")]
    before = {path: os.path.getmtime(path) for path in files}

    environment = IntelliSignalEnv(scenario="heavy",
                                   episode_duration=SHORT_EPISODE)
    try:
        environment.reset(seed=42)
        environment.step(ACTION_SWITCH)
    finally:
        environment.close()

    for path, mtime in before.items():
        assert os.path.getmtime(path) == mtime, (
            "the RL environment modified a Stage 1 result file: {}".format(path))
