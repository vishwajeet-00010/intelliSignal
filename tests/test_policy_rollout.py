"""
Tests for `rl/policy_rollout.py` and `agents/phase_ensemble.py`.

There are two kinds of trained policy in this project and they are not
interchangeable. A lock-and-extend network has two outputs meaning
"extend this green" and "end it"; a duration-menu network has one output
per green length. Handing the first to the duration-menu environment
does not raise anything - its two outputs are simply read as the two
shortest greens on the menu, and the signal alternates between them
whatever the traffic does. That is what happened to the promoted model
on the dashboard and in Hybrid before this module existed.

So these tests pin the thing that went wrong: the action space is read
from the checkpoint, the environment is chosen from it, and a policy
that does not fit is refused rather than quietly mis-run.
"""

import os
import sys

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.phase_dqn_agent import PhaseDQNAgent                # noqa: E402
from agents.phase_ensemble import (EnsembleError, PhaseEnsemble,  # noqa: E402
                                   load_policy)
from rl.phase_extend_env import PhaseExtendEnv                  # noqa: E402
from rl.phase_environment import PhaseDurationEnv               # noqa: E402
from rl.phase_state import PhaseStateExtractor                  # noqa: E402
from rl.policy_rollout import (ACTION_SPACE_EXTEND,             # noqa: E402
                               ACTION_SPACE_MENU, PolicyMismatchError,
                               action_space_of, describe, extension_of,
                               greedy_rollout, ladder_of, make_env,
                               profile_of)
from rl.timing_profile import get_profile                       # noqa: E402

PROFILE = get_profile("standard")
MENU = list(PROFILE.durations)


def menu_agent(seed=1):
    return PhaseDQNAgent(
        observation_size=10, num_actions=len(MENU), seed=seed,
        scenario="profile:standard|candidate:A", durations=MENU,
        observation_contract=PhaseStateExtractor().contract(),
        action_space=ACTION_SPACE_MENU)


def extend_agent(seed=1, extension=3.0):
    return PhaseDQNAgent(
        observation_size=10, num_actions=2, seed=seed,
        scenario="profile:standard|candidate:B", durations=MENU,
        observation_contract=PhaseStateExtractor().contract(),
        action_space=ACTION_SPACE_EXTEND, extension=extension)


# ---------------------------------------------------------------------------
# Reading what a checkpoint is
# ---------------------------------------------------------------------------
def test_each_kind_reports_its_own_action_space():
    assert action_space_of(menu_agent()) == ACTION_SPACE_MENU
    assert action_space_of(extend_agent()) == ACTION_SPACE_EXTEND


def test_an_older_checkpoint_is_read_from_what_the_trainer_recorded():
    """Written before the action space was stored, candidate B is still
    recognisable from the scenario string the trainer put in it."""
    agent = extend_agent()
    agent.action_space = ""
    assert action_space_of(agent) == ACTION_SPACE_EXTEND

    agent = menu_agent()
    agent.action_space = ""
    assert action_space_of(agent) == ACTION_SPACE_MENU


def test_the_extension_is_what_it_was_trained_with_not_the_menu_spacing():
    """A lock-and-extend checkpoint stores the PROFILE's menu, whose
    spacing has nothing to do with how much one extension adds."""
    agent = extend_agent(extension=4.0)
    assert extension_of(agent) == 4.0
    assert ladder_of(agent) == [15.0, 19.0, 23.0, 27.0, 30.0]

    agent = extend_agent(extension=3.0)
    assert ladder_of(agent) == [15.0, 18.0, 21.0, 24.0, 27.0, 30.0]


def test_a_lock_and_extend_policy_knows_the_profile_it_belongs_to():
    assert profile_of(extend_agent()).name == "standard"
    assert profile_of(menu_agent()) is None, \
        "a duration menu is read from the checkpoint, not from a profile"


def test_describe_says_what_the_controller_actually_does():
    story = describe(extend_agent())
    assert story["action_space"] == ACTION_SPACE_EXTEND
    assert "15 s" in story["summary"] and "3 s" in story["summary"]
    assert story["greens"] == [15.0, 18.0, 21.0, 24.0, 27.0, 30.0]

    story = describe(menu_agent())
    assert story["action_space"] == ACTION_SPACE_MENU
    assert "menu" in story["summary"].lower()
    assert story["greens"] == MENU


# ---------------------------------------------------------------------------
# Choosing the environment
# ---------------------------------------------------------------------------
def test_each_policy_gets_the_environment_its_actions_mean_something_in():
    extend = make_env(None, extend_agent(), profile=PROFILE, tag="_t1")
    try:
        assert isinstance(extend, PhaseExtendEnv)
        assert extend.action_space.n == 2
    finally:
        extend.close()

    menu = make_env(None, menu_agent(), profile=PROFILE, tag="_t2")
    try:
        assert isinstance(menu, PhaseDurationEnv)
        assert menu.action_space.n == len(MENU)
    finally:
        menu.close()


def test_a_two_output_network_is_refused_by_the_duration_menu():
    broken = extend_agent()
    broken.action_space = ACTION_SPACE_MENU          # mislabelled
    with pytest.raises(PolicyMismatchError) as error:
        make_env(None, broken, profile=PROFILE)
    assert "one output per menu entry" in str(error.value)


def test_a_menu_network_is_refused_as_lock_and_extend():
    broken = menu_agent()
    broken.action_space = ACTION_SPACE_EXTEND
    with pytest.raises(PolicyMismatchError) as error:
        make_env(None, broken, profile=PROFILE)
    assert "two outputs" in str(error.value)


def test_a_model_trained_for_other_green_limits_is_refused():
    """A 4-20 s policy under a 15 s minimum is not a cautious choice."""
    old = PhaseDQNAgent(
        observation_size=10, num_actions=10, seed=3,
        scenario="profile:responsive|candidate:A",
        durations=list(get_profile("responsive").durations),
        observation_contract=PhaseStateExtractor().contract(),
        action_space=ACTION_SPACE_MENU)
    with pytest.raises(PolicyMismatchError) as error:
        make_env(None, old, profile=PROFILE)
    assert "menu" in str(error.value)


def test_a_lock_and_extend_model_is_refused_under_other_green_limits():
    agent = extend_agent()
    with pytest.raises(PolicyMismatchError) as error:
        make_env(None, agent, profile=get_profile("responsive"))
    assert "responsive" in str(error.value)


# ---------------------------------------------------------------------------
# A real rollout
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_workload():
    from training.workload_library import build

    return build("balanced", "light", 180.0, 90001)


def test_a_lock_and_extend_policy_really_produces_long_greens(tiny_workload):
    """The bug this module exists for: run through the wrong environment
    the same checkpoint produces only the two shortest greens."""
    rollout = greedy_rollout(tiny_workload, extend_agent(seed=5), seed=90001,
                             profile=PROFILE, tag="_tr1")
    assert rollout.action_space == ACTION_SPACE_EXTEND
    assert rollout.greens, "the policy must have served some greens"
    served = rollout.completed_greens
    assert min(served) >= PROFILE.min_green - 1e-6
    assert max(served) <= PROFILE.max_green + 1e-6
    assert set(served) - {15.0, 16.0}, \
        "every green being the two shortest menu entries is the bug"
    # The last green is normally stopped by the clock, not the controller,
    # and is excluded from `completed_greens` for exactly that reason.
    assert len(rollout.served_greens) - len(served) <= 1
    assert rollout.compliance.get("compliant") is True
    for row in rollout.greens:
        assert set(row["q_values"]) == {"extend", "end"}


def test_a_duration_menu_policy_rolls_out_and_reports_its_choices(
        tiny_workload):
    rollout = greedy_rollout(tiny_workload, menu_agent(seed=7), seed=90001,
                             profile=PROFILE, tag="_tr2")
    assert rollout.action_space == ACTION_SPACE_MENU
    assert rollout.greens
    for row in rollout.greens:
        assert set(row["q_values"]) == {"{:g}s".format(d) for d in MENU}
    assert min(rollout.completed_greens) >= PROFILE.min_green - 1e-6


def test_an_observer_watches_without_changing_the_run(tiny_workload):
    from simulation.recorder import ReplayRecorder

    agent = extend_agent(seed=11)
    plain = greedy_rollout(tiny_workload, agent, seed=90001, profile=PROFILE,
                           tag="_tr3")
    recorder = ReplayRecorder(tiny_workload)
    watched = greedy_rollout(tiny_workload, agent, seed=90001,
                             profile=PROFILE, tag="_tr4", observer=recorder)
    assert watched.served_greens == plain.served_greens
    assert watched.summary["average_waiting_time"] == \
        plain.summary["average_waiting_time"]
    assert recorder.replay()["steps"] == 180


# ---------------------------------------------------------------------------
# Ensembles
# ---------------------------------------------------------------------------
def test_an_ensemble_averages_its_members():
    members = [menu_agent(seed=s) for s in (1, 2, 3)]
    ensemble = PhaseEnsemble(members)
    observation = np.zeros(10, dtype=np.float32)
    expected = np.mean([m.q_values(observation) for m in members], axis=0)
    assert np.allclose(ensemble.q_values(observation), expected)
    assert ensemble.act(observation) == int(np.argmax(expected))


def test_an_ensemble_keeps_what_its_members_agree_on():
    members = [menu_agent(seed=s) for s in (1, 2, 3)]
    ensemble = PhaseEnsemble(members)
    observation = np.zeros(10, dtype=np.float32)
    share = ensemble.agreement(observation)
    assert 0.0 < share <= 1.0


def test_an_ensemble_looks_like_the_controller_its_members_are():
    ensemble = PhaseEnsemble([extend_agent(seed=s) for s in (1, 2)])
    assert ensemble.action_space == ACTION_SPACE_EXTEND
    assert ensemble.extension == 3.0
    assert action_space_of(ensemble) == ACTION_SPACE_EXTEND
    assert len(ensemble) == 2


def test_an_ensemble_honours_a_restricted_action_set():
    ensemble = PhaseEnsemble([extend_agent(seed=s) for s in (1, 2)])
    observation = np.zeros(10, dtype=np.float32)
    assert ensemble.act(observation, valid_actions=[1]) == 1


def test_members_that_are_not_the_same_controller_are_refused():
    with pytest.raises(EnsembleError):
        PhaseEnsemble([menu_agent(), extend_agent()])
    with pytest.raises(EnsembleError):
        PhaseEnsemble([])
    with pytest.raises(EnsembleError):
        PhaseEnsemble([extend_agent(extension=3.0),
                       extend_agent(extension=4.0)])


def test_an_ensemble_cannot_be_trained():
    ensemble = PhaseEnsemble([menu_agent(seed=1), menu_agent(seed=2)])
    with pytest.raises(EnsembleError):
        ensemble.act(np.zeros(10, dtype=np.float32), training=True)


def test_one_path_loads_a_network_and_several_load_an_ensemble(tmp_path):
    first = str(tmp_path / "a.pt")
    second = str(tmp_path / "b.pt")
    menu_agent(seed=1).save(first)
    menu_agent(seed=2).save(second)

    assert isinstance(load_policy(first), PhaseDQNAgent)
    assert isinstance(load_policy([first]), PhaseDQNAgent)
    both = load_policy([first, second])
    assert isinstance(both, PhaseEnsemble) and len(both) == 2


def test_the_measurement_starts_before_the_clock_does(tiny_workload):
    """The lock of the first green must not run unmeasured.

    The lock-and-extend environment serves its minimum green inside
    `reset`, so that the first extend-or-end question is asked at a real
    decision point. A collector attached after `reset` returned missed
    those fifteen seconds: the recording the audit reads began part-way
    through the first green, and the audit duly reported a three-second
    green on a controller whose minimum is fifteen.
    """
    from simulation.recorder import ReplayRecorder, audit_signal

    recorder = ReplayRecorder(tiny_workload)
    rollout = greedy_rollout(tiny_workload, extend_agent(seed=13),
                             seed=90001, profile=PROFILE, tag="_tr5",
                             observer=recorder)
    replay = recorder.replay()
    # Every simulated second is in the recording, from the first one.
    assert replay["steps"] == int(tiny_workload.duration_s)
    assert rollout.summary["simulation_time"] == tiny_workload.duration_s

    audit = audit_signal(replay["signal"], PROFILE,
                         float(tiny_workload.signal["yellow"]),
                         float(tiny_workload.signal["all_red"]),
                         tiny_workload.step_length)
    assert audit["compliant"] is True, audit["violations"]
    assert audit["shortest_green"] >= PROFILE.min_green
