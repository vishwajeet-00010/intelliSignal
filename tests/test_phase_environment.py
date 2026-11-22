"""Integration tests for the duration-choosing environment.

These start SUMO, so the episodes are 120-200 simulated seconds rather than
1800 and the suite stays fast. What they are here to protect:

  * an episode is exactly `episode_duration` seconds, so a learned policy
    and a fixed-time plan measured on the same seed see the same traffic
  * a decision's reward is charged over the seconds the decision occupies,
    which is the property that makes the semi-Markov update correct
  * the generated SUMO inputs are the environment's own, and no Stage 1 or
    Stage 2 file is touched
"""

from __future__ import annotations

import glob
import math
import os
import sys

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl.phase_action import PHASE_EW, PHASE_NS                  # noqa: E402
from rl.phase_environment import PHASE_SUFFIX, PhaseDurationEnv  # noqa: E402
from rl.phase_state import OBSERVATION_SIZE                     # noqa: E402

SHORT = 200.0
DURATIONS = (4, 6, 8, 12)


@pytest.fixture
def env():
    environment = PhaseDurationEnv(scenario="heavy", durations=DURATIONS,
                                   episode_duration=SHORT, tag="_test")
    try:
        yield environment
    finally:
        environment.close()
        environment.cleanup_generated_inputs()


# ---------------------------------------------------------------------------
def test_spaces_match_the_menu_and_the_observation(env):
    assert env.action_space.n == len(DURATIONS)
    assert env.observation_space.shape == (OBSERVATION_SIZE,)


def test_reset_returns_an_observation_inside_the_space(env):
    observation, info = env.reset(seed=4242)
    assert env.observation_space.contains(observation)
    assert info["phase"] == PHASE_NS
    assert info["decision"] == 0


def test_a_step_advances_green_plus_clearance(env):
    env.reset(seed=4242)
    before = env._steps_done
    _, _, _, _, info = env.step(2)                 # 8 s of green
    assert info["green_served"] == 8.0
    assert info["duration"] == 8.0 + env.controller.yellow_time
    assert env._steps_done - before == int(info["duration"])


def test_the_phase_alternates_every_decision(env):
    env.reset(seed=4242)
    phases = []
    for _ in range(4):
        phases.append(env.controller.phase)
        env.step(0)
    assert phases == [PHASE_NS, PHASE_EW, PHASE_NS, PHASE_EW]


def test_the_episode_is_exactly_the_requested_number_of_seconds(env):
    env.reset(seed=4242)
    truncated = False
    while not truncated:
        _, _, terminated, truncated, _ = env.step(1)
        if terminated:
            break
    assert env._steps_done == int(SHORT / env.step_length)


WARMUP_DECISIONS = 8


def warm_up(environment, seed=4242):
    """Reset and fill the approaches, so there is congestion to charge for.

    The first dozen seconds of an episode have no queue at all - every
    reward is exactly zero - so a comparison made there compares nothing.
    """
    environment.reset(seed=seed)
    for _ in range(WARMUP_DECISIONS):
        environment.step(1)


def test_a_longer_decision_costs_more_than_a_shorter_one(env):
    """The reward is an integral, so it must scale with the time served."""
    warm_up(env)
    _, short_reward, _, _, short_info = env.step(0)     # 4 s green
    warm_up(env)
    _, long_reward, _, _, long_info = env.step(3)       # 12 s green

    assert short_reward < 0.0, "the warm-up must leave real congestion"
    assert long_info["duration"] > short_info["duration"]
    assert long_reward < short_reward, (
        "a decision that presides over three times as much traffic must not "
        "be charged the same as the short one - that is the bug that made "
        "switching look free")


def test_the_reward_is_finite_and_never_positive(env):
    env.reset(seed=4242)
    for action in (0, 1, 2, 3):
        _, reward, _, _, _ = env.step(action)
        assert math.isfinite(reward)
        assert reward <= 0.0


def test_within_step_discounting_shrinks_the_cost_but_keeps_the_sign():
    undiscounted = PhaseDurationEnv(scenario="heavy", durations=DURATIONS,
                                    episode_duration=SHORT,
                                    gamma_per_second=1.0, tag="_test_g1")
    discounted = PhaseDurationEnv(scenario="heavy", durations=DURATIONS,
                                  episode_duration=SHORT,
                                  gamma_per_second=0.9, tag="_test_g09")
    try:
        warm_up(undiscounted)
        _, plain, _, _, _ = undiscounted.step(3)
        warm_up(discounted)
        _, discounted_reward, _, _, _ = discounted.step(3)
        assert plain < 0 and discounted_reward < 0
        assert discounted_reward > plain, (
            "seconds late in a long decision must weigh less than the first "
            "second, or the long actions are systematically overcharged")
    finally:
        for environment in (undiscounted, discounted):
            environment.close()
            environment.cleanup_generated_inputs()


def test_an_invalid_action_is_refused(env):
    env.reset(seed=4242)
    with pytest.raises(ValueError, match="Invalid action"):
        env.step(len(DURATIONS))


def test_stepping_before_reset_is_refused():
    environment = PhaseDurationEnv(scenario="heavy", durations=DURATIONS,
                                   episode_duration=SHORT, tag="_test_norst")
    with pytest.raises(RuntimeError, match="before reset"):
        environment.step(0)


def test_the_same_seed_gives_the_same_episode(env):
    rewards = []
    for _ in range(2):
        env.reset(seed=7777)
        rewards.append([env.step(1)[1] for _ in range(4)])
    assert np.allclose(rewards[0], rewards[1])


def test_it_writes_only_its_own_scenario_files(env):
    env.reset(seed=4242)
    generated = glob.glob(os.path.join(PROJECT_ROOT, "sumo", "routes",
                                       "heavy" + PHASE_SUFFIX + "_test*"))
    assert generated, "the environment must build its own inputs"
    # The Stage 1 and Stage 2 files keep their own names.
    for name in ("heavy.rou.xml", "heavy_rl.rou.xml"):
        assert not any(path.endswith(name) for path in generated)


def test_the_observation_sees_the_direction_about_to_be_served(env):
    observation, _ = env.reset(seed=4242)
    assert observation[6] == 1.0, "the episode starts on north-south green"
    observation, _, _, _, _ = env.step(1)
    assert observation[6] == 0.0, "after one decision it is east-west's turn"
