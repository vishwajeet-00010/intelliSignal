"""Tests for the duration action space and the semi-Markov DQN that uses it.

The three things these tests exist to pin down, because each one is a bug the
project has actually had:

  * a decision's cost must be charged over the time it occupies, and its
    future discounted by that same time - `gamma ** seconds`, not `gamma`
  * the observation must be written from the served direction's point of
    view, so one rule covers both greens
  * a checkpoint must carry its duration menu and its normalisation scales,
    so a policy can never be evaluated against a different action space or
    different scaling than the one it learned with
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.phase_dqn_agent import (EpsilonSchedule,  # noqa: E402
                                    PhaseDQNAgent, n_step_transitions)
from agents.replay_buffer import ReplayBuffer, Transition  # noqa: E402
from rl.phase_action import (DEFAULT_DURATIONS, PHASE_EW,  # noqa: E402
                             PHASE_NS, PhaseDurationController)
from rl.phase_state import (OBSERVATION_LABELS,  # noqa: E402
                            OBSERVATION_SIZE, PhaseStateExtractor,
                            extractor_from_contract)
from utils.scenario_builder import (ALL_RED, EW_GREEN, EW_YELLOW,  # noqa: E402
                                    NS_GREEN, NS_YELLOW)


def controller(durations=(4, 5, 6, 7, 8), yellow=4.0, all_red=0.0,
               min_green=4.0, max_green=60.0):
    return PhaseDurationController(durations=durations, yellow_time=yellow,
                                   all_red_time=all_red, min_green=min_green,
                                   max_green=max_green)


# ===========================================================================
# The action space
# ===========================================================================
def test_the_menu_must_be_sorted_unique_and_inside_the_safety_bounds():
    with pytest.raises(ValueError, match="must not be empty"):
        controller(durations=())
    with pytest.raises(ValueError, match="must not repeat"):
        controller(durations=(5, 5, 6))
    with pytest.raises(ValueError, match="increasing order"):
        controller(durations=(6, 5, 4))
    with pytest.raises(ValueError, match="below min_green"):
        controller(durations=(3, 5), min_green=4)
    with pytest.raises(ValueError, match="above max_green"):
        controller(durations=(5, 90), max_green=60)


def test_an_action_is_an_index_into_the_menu():
    ctrl = controller(durations=(4, 7, 20))
    assert ctrl.num_actions == 3
    assert ctrl.duration_for(0) == 4.0
    assert ctrl.duration_for(2) == 20.0
    with pytest.raises(ValueError, match="Invalid action"):
        ctrl.duration_for(3)
    with pytest.raises(ValueError, match="Invalid action"):
        ctrl.duration_for(-1)


def test_the_plan_is_green_then_the_mandatory_clearance():
    ctrl = controller(durations=(7,), yellow=4.0, all_red=0.0)
    plan = ctrl.plan(0)
    assert plan == [(NS_GREEN, 7), (NS_YELLOW, 4)]
    # The green of one direction can never become the green of the other
    # without the clearance in between - the menu cannot express it.
    states = [state for state, _ in plan]
    assert states[-1] in (NS_YELLOW, EW_YELLOW, ALL_RED)


def test_an_all_red_segment_is_included_when_configured():
    ctrl = controller(durations=(6,), yellow=3.0, all_red=2.0)
    assert ctrl.plan(0) == [(NS_GREEN, 6), (NS_YELLOW, 3), (ALL_RED, 2)]
    assert ctrl.decision_seconds(0) == 11.0


def test_a_decision_costs_its_green_plus_the_clearance():
    ctrl = controller(durations=(4, 20), yellow=4.0)
    assert ctrl.decision_seconds(0) == 8.0
    assert ctrl.decision_seconds(1) == 24.0


def test_committing_changes_phase_and_records_the_green():
    ctrl = controller(durations=(5, 7))
    assert ctrl.phase == PHASE_NS
    ctrl.commit(1)                      # 7 s of north-south green
    assert ctrl.phase == PHASE_EW
    assert ctrl.last_green[PHASE_NS] == 7.0
    assert ctrl.switch_count == 1
    assert ctrl.plan(0)[0][0] == EW_GREEN


def test_seven_against_six_is_two_actions():
    """The plan the Stage 2 action space could not express at all."""
    ctrl = controller(durations=DEFAULT_DURATIONS)
    ns = DEFAULT_DURATIONS.index(7)
    ew = DEFAULT_DURATIONS.index(6)
    served = []
    for _ in range(4):
        action = ns if ctrl.phase == PHASE_NS else ew
        served.append((ctrl.phase, ctrl.duration_for(action)))
        ctrl.commit(action)
    assert served == [(PHASE_NS, 7.0), (PHASE_EW, 6.0),
                      (PHASE_NS, 7.0), (PHASE_EW, 6.0)]


def test_no_approach_can_be_starved_because_every_action_ends_the_phase():
    ctrl = controller(durations=(4, 8))
    phases = []
    for _ in range(6):
        phases.append(ctrl.phase)
        ctrl.commit(0)
    assert phases == [PHASE_NS, PHASE_EW] * 3


# ===========================================================================
# The observation
# ===========================================================================
def raw_reading(north=0, east=0, south=0, west=0,
                waiting=None, vehicles=None):
    queues = {"north": float(north), "east": float(east),
              "south": float(south), "west": float(west)}
    waiting = waiting or {k: 0.0 for k in queues}
    vehicles = vehicles or dict(queues)
    return {
        "approach_queues": queues,
        "approach_waiting": waiting,
        "queues": {PHASE_NS: queues["north"] + queues["south"],
                   PHASE_EW: queues["east"] + queues["west"]},
        "waiting": {PHASE_NS: waiting["north"] + waiting["south"],
                    PHASE_EW: waiting["east"] + waiting["west"]},
        "vehicles": {PHASE_NS: vehicles["north"] + vehicles["south"],
                     PHASE_EW: vehicles["east"] + vehicles["west"]},
    }


def test_the_observation_has_the_documented_shape():
    assert OBSERVATION_SIZE == 10
    assert len(OBSERVATION_LABELS) == OBSERVATION_SIZE
    obs = PhaseStateExtractor().observe(raw_reading(), PHASE_NS, 0.0, 0.0)
    assert obs.shape == (OBSERVATION_SIZE,)
    assert obs.dtype == np.float32


def test_the_observation_is_written_from_the_served_directions_point_of_view():
    extractor = PhaseStateExtractor(max_queue_per_direction=20.0)
    raw = raw_reading(north=6, south=4, east=1, west=1)

    ns_view = extractor.observe(raw, PHASE_NS, 0.0, 0.0)
    ew_view = extractor.observe(raw, PHASE_EW, 0.0, 0.0)

    assert ns_view[0] == pytest.approx(10 / 20)     # served = north+south
    assert ns_view[1] == pytest.approx(2 / 20)      # other  = east+west
    # The same road, the other green: the two entries swap.
    assert ew_view[0] == pytest.approx(2 / 20)
    assert ew_view[1] == pytest.approx(10 / 20)


def test_the_north_south_flag_survives_so_an_asymmetric_plan_is_expressible():
    extractor = PhaseStateExtractor()
    raw = raw_reading(north=3, south=3, east=3, west=3)
    assert extractor.observe(raw, PHASE_NS, 0, 0)[6] == 1.0
    assert extractor.observe(raw, PHASE_EW, 0, 0)[6] == 0.0


def test_approaching_traffic_is_vehicles_that_are_not_yet_halting():
    extractor = PhaseStateExtractor(max_vehicles_per_direction=10.0)
    raw = raw_reading(north=2, south=1, east=0, west=0,
                      vehicles={"north": 5, "south": 3, "east": 0, "west": 0})
    obs = extractor.observe(raw, PHASE_NS, 0, 0)
    assert obs[2] == pytest.approx((8 - 3) / 10)


def test_the_worst_single_approach_is_visible_behind_the_direction_totals():
    """Ten halting vehicles can be five and five, or ten and none."""
    extractor = PhaseStateExtractor(max_queue_per_approach=10.0)
    balanced = extractor.observe(raw_reading(north=5, south=5), PHASE_NS, 0, 0)
    lopsided = extractor.observe(raw_reading(north=10, south=0), PHASE_NS, 0, 0)
    assert balanced[0] == lopsided[0], "the direction total is the same"
    assert balanced[7] == pytest.approx(0.5)
    assert lopsided[7] == pytest.approx(1.0), "the spillback risk is not"


def test_the_observation_is_clipped_into_the_declared_box():
    extractor = PhaseStateExtractor(max_queue_per_direction=4.0)
    obs = extractor.observe(raw_reading(north=50, south=50), PHASE_NS, 0, 0)
    assert obs.min() >= 0.0 and obs.max() <= 1.0


def test_the_contract_round_trips():
    extractor = PhaseStateExtractor(max_queue_per_direction=17.0,
                                    max_green_seconds=12.0)
    rebuilt = extractor_from_contract(extractor.contract())
    assert rebuilt.max_queue == 17.0
    assert rebuilt.max_green == 12.0
    raw = raw_reading(north=3, south=2, east=1, west=1)
    assert np.allclose(extractor.observe(raw, PHASE_NS, 5, 6),
                       rebuilt.observe(raw, PHASE_NS, 5, 6))


# ===========================================================================
# The replay buffer carries each transition's duration
# ===========================================================================
def test_duration_defaults_to_one_second_for_callers_that_do_not_set_it():
    buffer = ReplayBuffer(capacity=4, seed=0, num_actions=2)
    buffer.push(np.zeros(3), 0, -1.0, np.zeros(3), False)
    assert buffer.sample(1).durations[0] == pytest.approx(1.0)
    # Constructing a Transition directly still works, as the old tests do.
    assert Transition(np.zeros(3), 0, 0.0, np.zeros(3), False,
                      np.ones(2, dtype=np.bool_)).duration == 1.0


def test_duration_is_stored_and_sampled_back():
    buffer = ReplayBuffer(capacity=4, seed=0, num_actions=2)
    buffer.push(np.zeros(3), 0, -1.0, np.zeros(3), False, duration=11.0)
    assert buffer.sample(1).durations[0] == pytest.approx(11.0)


def test_a_transition_covering_no_time_is_refused():
    buffer = ReplayBuffer(capacity=4, seed=0, num_actions=2)
    with pytest.raises(ValueError, match="duration must be positive"):
        buffer.push(np.zeros(3), 0, -1.0, np.zeros(3), False, duration=0.0)


# ===========================================================================
# Multi-step returns, discounted by seconds
# ===========================================================================
def step(reward, duration, done=False, state=None, next_state=None, action=0):
    return {"state": state if state is not None else np.zeros(3),
            "action": action, "reward": reward,
            "next_state": next_state if next_state is not None else np.ones(3),
            "done": done, "duration": duration}


def test_one_step_is_the_transition_itself():
    collapsed = n_step_transitions([step(-2.0, 11.0)], 1, 0.98)
    assert collapsed["reward"] == pytest.approx(-2.0)
    assert collapsed["duration"] == pytest.approx(11.0)


def test_a_two_step_return_discounts_by_seconds_not_by_steps():
    gamma = 0.98
    window = [step(-2.0, 11.0), step(-3.0, 24.0)]
    collapsed = n_step_transitions(window, 2, gamma)
    # The second reward is discounted by the ELEVEN seconds already spent,
    # not by one step of gamma.
    assert collapsed["reward"] == pytest.approx(-2.0 + (gamma ** 11) * -3.0)
    assert collapsed["reward"] != pytest.approx(-2.0 + gamma * -3.0)
    # The collapsed transition spans both decisions, so the bootstrap term
    # is discounted over all 35 seconds.
    assert collapsed["duration"] == pytest.approx(35.0)


def test_a_collapsed_transition_starts_and_ends_where_the_window_does():
    first, last = np.full(3, 0.5), np.full(3, 0.9)
    window = [step(-1.0, 8.0, state=first, action=4),
              step(-1.0, 8.0, next_state=last, done=True)]
    collapsed = n_step_transitions(window, 2, 0.98)
    assert np.allclose(collapsed["state"], first)
    assert collapsed["action"] == 4
    assert np.allclose(collapsed["next_state"], last)
    assert collapsed["done"] is True


def test_collapsing_nothing_is_an_error():
    with pytest.raises(ValueError):
        n_step_transitions([], 1, 0.98)
    with pytest.raises(ValueError, match="n must be at least 1"):
        n_step_transitions([step(-1.0, 8.0)], 0, 0.98)


# ===========================================================================
# The agent
# ===========================================================================
def agent(**kwargs):
    defaults = dict(observation_size=OBSERVATION_SIZE, num_actions=4,
                    hidden_layers=(8, 8), batch_size=2, learning_starts=2,
                    seed=0, durations=(4, 6, 8, 10))
    defaults.update(kwargs)
    return PhaseDQNAgent(**defaults)


def test_the_target_discounts_by_elapsed_time():
    model = agent(gamma_per_second=0.98)
    rewards = torch.tensor([-2.0, -2.0])
    next_q = torch.tensor([-10.0, -10.0])
    dones = torch.tensor([0.0, 0.0])
    durations = torch.tensor([8.0, 24.0])

    target = model.bootstrap_target(rewards, next_q, dones, durations)

    assert target[0].item() == pytest.approx(-2.0 + (0.98 ** 8) * -10.0, rel=1e-5)
    assert target[1].item() == pytest.approx(-2.0 + (0.98 ** 24) * -10.0, rel=1e-5)
    # The longer decision must value the same future less, or the agent is
    # being told that twenty-four seconds cost no more than eight.
    assert target[1].item() > target[0].item()


def test_a_terminal_transition_keeps_only_its_reward():
    model = agent(gamma_per_second=0.98)
    target = model.bootstrap_target(
        torch.tensor([-5.0]), torch.tensor([-99.0]),
        torch.tensor([1.0]), torch.tensor([12.0]))
    assert target.item() == pytest.approx(-5.0)


def test_gamma_one_still_discounts_nothing_however_long_the_decision():
    model = agent(gamma_per_second=1.0)
    target = model.bootstrap_target(
        torch.tensor([-1.0]), torch.tensor([-4.0]),
        torch.tensor([0.0]), torch.tensor([37.0]))
    assert target.item() == pytest.approx(-5.0)


def test_double_q_learning_scores_the_policy_networks_choice():
    """The policy network chooses the action; the target network prices it."""
    model = agent(double_dqn=True)

    with torch.no_grad():
        for net, values in ((model.policy_net, [0.0, 9.0, 0.0, 0.0]),
                            (model.target_net, [5.0, -5.0, 5.0, 5.0])):
            for layer in net.model:
                if isinstance(layer, torch.nn.Linear):
                    layer.weight.zero_()
                    layer.bias.zero_()
            net.model[-1].bias.copy_(torch.tensor(values))

    next_states = torch.zeros(1, OBSERVATION_SIZE)
    best = model.policy_net(next_states).argmax(dim=1, keepdim=True)
    assert best.item() == 1, "the policy network prefers action 1"
    scored = model.target_net(next_states).gather(1, best).squeeze(1)
    assert scored.item() == pytest.approx(-5.0), (
        "and it must be priced at the TARGET network's -5, not at the "
        "target network's own maximum of +5")


def test_a_gradient_step_runs_and_stays_finite():
    model = agent()
    for index in range(8):
        model.remember(np.full(OBSERVATION_SIZE, 0.1 * index), index % 4,
                       -1.0 - index, np.full(OBSERVATION_SIZE, 0.2),
                       index == 7, duration=8.0 + index)
    loss = model.learn()
    assert loss is not None and np.isfinite(loss)
    assert model.learn_steps == 1


def test_learning_waits_until_there_is_enough_experience():
    model = agent(learning_starts=10)
    model.remember(np.zeros(OBSERVATION_SIZE), 0, -1.0,
                   np.zeros(OBSERVATION_SIZE), False, duration=8.0)
    assert model.learn() is None


def test_greedy_action_is_the_argmax_and_epsilon_one_explores():
    model = agent()
    observation = np.full(OBSERVATION_SIZE, 0.3, dtype=np.float32)
    expected = int(np.argmax(model.q_values(observation)))
    assert model.act(observation, training=False) == expected

    model.set_epsilon(1.0)
    chosen = {model.act(observation, training=True) for _ in range(60)}
    assert len(chosen) > 1, "epsilon = 1 must not keep returning the argmax"


def test_a_checkpoint_carries_its_menu_and_its_scales(tmp_path):
    contract = PhaseStateExtractor(max_queue_per_direction=19.0).contract()
    model = agent(durations=(4, 6, 8, 10), observation_contract=contract,
                  gamma_per_second=0.97)
    path = model.save(str(tmp_path / "best.pt"))

    restored = PhaseDQNAgent.load(path)
    assert restored.durations == [4.0, 6.0, 8.0, 10.0]
    assert restored.observation_contract["max_queue_per_direction"] == 19.0
    assert restored.gamma_per_second == pytest.approx(0.97)

    observation = np.full(OBSERVATION_SIZE, 0.4, dtype=np.float32)
    assert np.allclose(model.q_values(observation),
                       restored.q_values(observation))


def test_explain_names_the_green_it_chose():
    model = agent(durations=(4, 6, 8, 10))
    explanation = model.explain(np.full(OBSERVATION_SIZE, 0.2, dtype=np.float32))
    assert explanation["green_seconds"] in (4.0, 6.0, 8.0, 10.0)
    assert len(explanation["q_values"]) == 4
    assert "{:g} s green".format(explanation["green_seconds"]) in explanation["reason"]


# ===========================================================================
# Exploration schedule
# ===========================================================================
def test_epsilon_decays_linearly_and_then_holds():
    schedule = EpsilonSchedule(start=1.0, end=0.1, decay_steps=100)
    assert schedule.value(0) == pytest.approx(1.0)
    assert schedule.value(50) == pytest.approx(0.55)
    assert schedule.value(100) == pytest.approx(0.1)
    assert schedule.value(10000) == pytest.approx(0.1)


def test_an_impossible_epsilon_range_is_refused():
    with pytest.raises(ValueError):
        EpsilonSchedule(start=0.1, end=0.9, decay_steps=10)
    with pytest.raises(ValueError):
        EpsilonSchedule(decay_steps=0)
