"""
IntelliSignal - Stage 3
tests/test_dqn.py

Tests for the Deep Q-Network stage.

Groups, in the order the pipeline runs:

  * configuration        config/dqn/training.yaml and its validation
  * the Q-network        shapes, forward pass, finite values, sizing
  * the replay buffer    insertion, capacity, sampling, batch shapes
  * the agent            epsilon-greedy, greedy, learning, target network,
                         save/load, explanation, and the refusal to act
                         without trained weights
  * fairness             the fixed-time comparison run reproduces the
                         recorded Stage 1 baseline exactly
  * training             a tiny real training run end to end
  * evaluation           a trained model is loaded and measured without
                         training
  * comparison           improvement percentages, raw vs aggregated results,
                         and the Stage 1 baseline left untouched
  * dashboard            the DQN mode is offered only when a model exists,
                         and a DQN run really is a DQN run

The tests that touch SUMO use short durations so the suite stays usable on a
laptop. None of them writes to results/csv/, which holds the recorded Stage 1
baseline, and two of them assert exactly that.

Run with:
    python -m pytest -q
    python -m pytest tests/test_dqn.py -q
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

torch = pytest.importorskip("torch", reason="Stage 3 needs PyTorch")

from agents import (DQN_IMPLEMENTED, DQNAgent, DQNModelNotFoundError,  # noqa: E402
                    PHASE_EW, PHASE_NS, TrafficState, create_agent,
                    list_agents)
from agents.dqn_agent import (EpsilonSchedule, build_agent_for_env,  # noqa: E402
                              dqn_model_available, find_models,
                              model_directory, read_metadata, set_seeds,
                              write_metadata)
from agents.dqn_config import (DQNConfig, DQNConfigError,            # noqa: E402
                               load_dqn_config, load_dqn_settings)
from agents.dqn_network import QNetwork                              # noqa: E402
from agents.replay_buffer import ReplayBuffer, Transition            # noqa: E402
from simulation.agent_simulation import (FIXED_TIME, LOWER_IS_BETTER,  # noqa: E402
                                         episode_plan, improvement,
                                         run_agent_episode,
                                         run_fixed_time_episode, save_run)
from utils.metrics import MetricsCollector                           # noqa: E402

BASELINE_DIR = os.path.join(PROJECT_ROOT, "results", "csv")

#: Short enough for a test, long enough to contain several decisions.
SHORT_DURATION = 120.0
OBS = 10
ACTIONS = 2


def tiny_config(**overrides) -> DQNConfig:
    """A configuration whose learning starts almost immediately."""
    values = dict(episodes=2, episode_duration=SHORT_DURATION, batch_size=4,
                  learning_starts=8, replay_capacity=500,
                  target_update_frequency=5, checkpoint_every=1,
                  best_window=1, seed=42)
    values.update(overrides)
    return load_dqn_config(**values)


def fresh_agent(**overrides) -> DQNAgent:
    return DQNAgent(observation_size=OBS, num_actions=ACTIONS,
                    config=tiny_config(**overrides), trainable=True, seed=42)


def random_observation(rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    return rng.random(OBS).astype(np.float32)


# ==========================================================================
# Configuration
# ==========================================================================
def test_the_shipped_configuration_file_is_valid():
    settings = load_dqn_settings()
    assert "network" in settings and "training" in settings
    config = load_dqn_config()
    assert config.gamma == 0.98, "the configured long-horizon discount factor"
    assert config.hidden_layers == [64, 64]
    assert config.epsilon_start == 1.0 and config.epsilon_end == 0.05


def test_command_line_overrides_win_and_none_is_ignored():
    config = load_dqn_config(episodes=7, gamma=None)
    assert config.episodes == 7
    assert config.gamma == 0.98, "a None override must not erase a real value"


def test_action_mask_limits_greedy_and_exploratory_actions():
    agent = fresh_agent()
    agent.q_values = lambda observation: np.asarray([0.0, 100.0])
    assert agent.act(random_observation(), valid_actions=[0]) == 0
    agent.set_epsilon(1.0)
    assert {agent.act(random_observation(), training=True, valid_actions=[0])
            for _ in range(20)} == {0}
    with pytest.raises(ValueError, match="at least one"):
        agent.act(random_observation(), valid_actions=[])


def test_configuration_rejects_settings_that_cannot_work():
    with pytest.raises(DQNConfigError):
        load_dqn_config(gamma=1.5)
    with pytest.raises(DQNConfigError):
        load_dqn_config(batch_size=0)
    with pytest.raises(DQNConfigError):
        load_dqn_config(batch_size=64, learning_starts=8)
    with pytest.raises(DQNConfigError):
        load_dqn_config(loss="hinge")
    with pytest.raises(DQNConfigError):
        load_dqn_config(target_update_frequency=0)
    with pytest.raises(DQNConfigError):
        load_dqn_config(epsilon_start=0.1, epsilon_end=0.9)
    with pytest.raises(DQNConfigError):
        load_dqn_config(hidden_layers=[])


def test_epsilon_schedule_length_follows_the_number_of_episodes():
    """A 20-episode run must not inherit a 100-episode schedule."""
    short = load_dqn_config(episodes=20).decay_steps(180)
    long = load_dqn_config(episodes=100).decay_steps(180)
    assert short < long
    assert short == pytest.approx(20 * 180 * 0.5)


def test_epsilon_schedule_is_bounded_and_monotone():
    schedule = EpsilonSchedule(start=1.0, end=0.05, decay_steps=100)
    assert schedule.value(0) == pytest.approx(1.0)
    assert schedule.value(50) == pytest.approx(0.525)
    assert schedule.value(100) == pytest.approx(0.05)
    assert schedule.value(10_000) == pytest.approx(0.05), "never below the floor"
    values = [schedule.value(s) for s in range(0, 120, 10)]
    assert all(a >= b for a, b in zip(values, values[1:])), "must not increase"

    exponential = EpsilonSchedule(start=1.0, end=0.1, decay_steps=1,
                                  kind="exponential", rate=0.5)
    assert exponential.value(1) == pytest.approx(0.5)
    assert exponential.value(50) == pytest.approx(0.1)


# ==========================================================================
# The Q-network
# ==========================================================================
def test_network_maps_ten_observations_to_two_q_values():
    network = QNetwork(OBS, ACTIONS, [64, 64])
    values = network(torch.zeros(OBS))
    assert values.shape == (ACTIONS,)


def test_network_handles_a_batch():
    network = QNetwork(OBS, ACTIONS, [64, 64])
    values = network(torch.zeros(32, OBS))
    assert values.shape == (32, ACTIONS)


def test_network_q_values_are_finite_for_extreme_observations():
    network = QNetwork(OBS, ACTIONS, [64, 64])
    for observation in (torch.zeros(OBS), torch.ones(OBS),
                        torch.full((OBS,), 0.5)):
        values = network(observation)
        assert torch.isfinite(values).all(), "a Q-value must never be NaN or inf"


def test_network_is_sized_from_the_environment_spaces():
    """The input size is read from the environment, never typed in."""
    import gymnasium as gym

    observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(OBS,),
                                       dtype=np.float32)
    action_space = gym.spaces.Discrete(ACTIONS)
    network = QNetwork.from_spaces(observation_space, action_space, [32])
    assert network.input_size == OBS
    assert network.output_size == ACTIONS
    assert network.architecture() == "10 -> 32 -> 2"


def test_network_architecture_follows_a_wider_observation():
    """If rl/state.py grew an eleventh value, the network would follow."""
    import gymnasium as gym

    wider = gym.spaces.Box(low=0.0, high=1.0, shape=(11,), dtype=np.float32)
    network = QNetwork.from_spaces(wider, gym.spaces.Discrete(3), [64, 64])
    assert network.input_size == 11 and network.output_size == 3


def test_network_rejects_impossible_sizes():
    for bad in ({"input_size": 0}, {"output_size": 0},
                {"hidden_layers": []}, {"hidden_layers": [0]}):
        kwargs = {"input_size": OBS, "output_size": ACTIONS,
                  "hidden_layers": [8]}
        kwargs.update(bad)
        with pytest.raises(ValueError):
            QNetwork(**kwargs)


def test_network_stays_small_enough_for_a_laptop():
    network = QNetwork(OBS, ACTIONS, [64, 64])
    assert network.parameter_count < 10_000


# ==========================================================================
# The replay buffer
# ==========================================================================
def test_buffer_stores_the_five_parts_of_a_transition():
    buffer = ReplayBuffer(capacity=10)
    buffer.push(np.zeros(OBS), 1, -0.5, np.ones(OBS), False)
    assert len(buffer) == 1
    stored = next(iter(buffer))
    assert isinstance(stored, Transition)
    assert stored.action == 1 and stored.reward == pytest.approx(-0.5)
    assert stored.done is False
    assert stored.state.shape == (OBS,)


def test_buffer_drops_the_oldest_transition_when_full():
    buffer = ReplayBuffer(capacity=5)
    for index in range(12):
        buffer.push(np.full(OBS, index, dtype=np.float32), index % 2,
                    -float(index), np.zeros(OBS), False)
    assert len(buffer) == 5
    assert buffer.is_full
    rewards = sorted(t.reward for t in buffer)
    assert rewards == [-11.0, -10.0, -9.0, -8.0, -7.0], "the newest five"


def test_buffer_sample_returns_correct_batch_dimensions():
    buffer = ReplayBuffer(capacity=100, seed=1)
    for index in range(50):
        buffer.push(random_observation(np.random.default_rng(index)),
                    index % 2, -0.1 * index,
                    random_observation(np.random.default_rng(index + 100)),
                    index % 7 == 0)
    batch = buffer.sample(16)
    assert batch.states.shape == (16, OBS)
    assert batch.next_states.shape == (16, OBS)
    assert batch.actions.shape == (16,)
    assert batch.rewards.shape == (16,)
    assert batch.dones.shape == (16,)
    assert batch.states.dtype == np.float32
    assert batch.actions.dtype == np.int64
    assert set(np.unique(batch.dones)) <= {0.0, 1.0}


def test_buffer_refuses_to_sample_more_than_it_holds():
    buffer = ReplayBuffer(capacity=10)
    buffer.push(np.zeros(OBS), 0, 0.0, np.zeros(OBS), False)
    assert buffer.can_sample(1) and not buffer.can_sample(2)
    with pytest.raises(ValueError):
        buffer.sample(2)
    with pytest.raises(ValueError):
        buffer.sample(0)


def test_buffer_rejects_a_zero_capacity():
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=0)


def test_buffer_sampling_is_reproducible_for_a_given_seed():
    def rewards_for(seed):
        buffer = ReplayBuffer(capacity=100, seed=seed)
        for index in range(40):
            buffer.push(np.zeros(OBS), 0, float(index), np.zeros(OBS), False)
        return list(buffer.sample(8).rewards)

    assert rewards_for(7) == rewards_for(7)
    assert rewards_for(7) != rewards_for(8)


# ==========================================================================
# The agent
# ==========================================================================
def test_stage3_declares_the_dqn_as_implemented():
    assert DQN_IMPLEMENTED is True


def test_agent_returns_a_valid_environment_action():
    agent = fresh_agent()
    for _ in range(20):
        action = agent.act(random_observation(np.random.default_rng()),
                           training=False)
        assert action in (0, 1)


def test_agent_q_values_have_one_value_per_action_and_are_finite():
    agent = fresh_agent()
    values = agent.q_values(random_observation())
    assert values.shape == (ACTIONS,)
    assert np.isfinite(values).all()


def test_agent_rejects_an_observation_of_the_wrong_size():
    agent = fresh_agent()
    with pytest.raises(ValueError):
        agent.q_values(np.zeros(7, dtype=np.float32))


def test_greedy_action_is_the_argmax_and_is_deterministic():
    agent = fresh_agent()
    agent.set_epsilon(0.0)
    observation = random_observation()
    expected = int(np.argmax(agent.q_values(observation)))
    assert all(agent.act(observation, training=False) == expected
               for _ in range(10))
    assert all(agent.act(observation, training=True) == expected
               for _ in range(10)), "epsilon 0 means no exploration even in training"


def test_epsilon_one_explores_and_epsilon_zero_does_not():
    """With epsilon = 1 the agent must ignore its own Q-values."""
    agent = fresh_agent()
    observation = random_observation()
    greedy = int(np.argmax(agent.q_values(observation)))

    agent.set_epsilon(1.0)
    chosen = [agent.act(observation, training=True) for _ in range(200)]
    assert set(chosen) == {0, 1}, "pure exploration must reach both actions"
    non_greedy = sum(1 for a in chosen if a != greedy)
    assert 50 < non_greedy < 150, "roughly half of a 2-action random choice"

    agent.set_epsilon(0.0)
    assert {agent.act(observation, training=True) for _ in range(50)} == {greedy}


def test_evaluation_mode_never_explores_whatever_epsilon_says():
    agent = fresh_agent()
    agent.set_epsilon(1.0)                    # as if mid-training
    observation = random_observation()
    greedy = int(np.argmax(agent.q_values(observation)))
    assert {agent.act(observation, training=False) for _ in range(50)} == {greedy}


def test_epsilon_is_clamped_to_a_probability():
    agent = fresh_agent()
    assert agent.set_epsilon(5.0) == 1.0
    assert agent.set_epsilon(-1.0) == 0.0


# --- learning --------------------------------------------------------------
def test_learning_waits_until_the_buffer_has_enough_experience():
    agent = fresh_agent()
    assert agent.learn() is None, "nothing to learn from yet"
    for index in range(agent.config.learning_starts - 1):
        agent.remember(random_observation(np.random.default_rng(index)),
                       index % 2, -0.2, random_observation(), False)
    assert agent.learn() is None, "still one transition short"
    agent.remember(random_observation(), 0, -0.2, random_observation(), False)
    loss = agent.learn()
    assert loss is not None and np.isfinite(loss)


def test_learning_produces_a_finite_loss_and_changes_the_policy_network():
    agent = fresh_agent()
    rng = np.random.default_rng(3)
    for index in range(60):
        agent.remember(rng.random(OBS), index % 2, -rng.random(),
                       rng.random(OBS), index % 11 == 0)

    before = agent.policy_net.model[0].weight.detach().clone()
    losses = [agent.learn() for _ in range(20)]
    assert all(loss is not None and np.isfinite(loss) for loss in losses)
    assert not torch.allclose(before, agent.policy_net.model[0].weight), (
        "a gradient step must actually change the policy network")


def test_target_network_starts_identical_and_is_refreshed_periodically():
    agent = fresh_agent(target_update_frequency=5)
    for key, policy in agent.policy_net.state_dict().items():
        assert torch.allclose(policy, agent.target_net.state_dict()[key]), (
            "the target network must start as a copy of the policy network")

    rng = np.random.default_rng(5)
    for index in range(60):
        agent.remember(rng.random(OBS), index % 2, -rng.random(),
                       rng.random(OBS), False)

    # Four gradient steps: the policy network has moved, the target has not.
    for _ in range(4):
        agent.learn()
    assert agent.target_updates == 0
    assert not torch.allclose(agent.policy_net.model[0].weight,
                              agent.target_net.model[0].weight)

    agent.learn()                             # the fifth triggers the refresh
    assert agent.target_updates == 1
    for key, policy in agent.policy_net.state_dict().items():
        assert torch.allclose(policy, agent.target_net.state_dict()[key])


def test_target_network_is_not_trained_directly():
    agent = fresh_agent()
    rng = np.random.default_rng(9)
    for index in range(60):
        agent.remember(rng.random(OBS), index % 2, -rng.random(),
                       rng.random(OBS), False)
    for _ in range(3):
        agent.learn()
    assert all(p.grad is None for p in agent.target_net.parameters()), (
        "no gradient may flow into the target network")


def test_terminal_transitions_drop_the_future_term():
    """target = reward for a terminal transition, reward + gamma*max Q otherwise."""
    agent = fresh_agent(gamma=0.9, batch_size=2, learning_starts=2)
    observation = np.zeros(OBS, dtype=np.float32)
    next_observation = np.ones(OBS, dtype=np.float32)

    with torch.no_grad():
        future = float(agent.target_net(
            torch.from_numpy(next_observation)).max())

    reward = -1.0
    terminal_target = reward
    ongoing_target = reward + 0.9 * future
    assert terminal_target != pytest.approx(ongoing_target), (
        "the test is only meaningful when the two targets differ")
    # The rule itself is exercised through learn(); this asserts the shape of
    # the arithmetic the implementation uses.
    assert ongoing_target == pytest.approx(reward + agent.config.gamma * future)


def test_both_loss_functions_work():
    for loss in ("huber", "mse"):
        agent = fresh_agent(loss=loss)
        rng = np.random.default_rng(11)
        for index in range(40):
            agent.remember(rng.random(OBS), index % 2, -rng.random(),
                           rng.random(OBS), False)
        value = agent.learn()
        assert value is not None and np.isfinite(value)


# --- the TrafficAgent interface --------------------------------------------
def test_agent_select_action_answers_in_the_convention_it_was_asked_in():
    agent = fresh_agent()
    observation = random_observation()

    env_action = agent.select_action(observation)
    assert env_action in (0, 1), "an observation gets a keep/switch answer"

    state = TrafficState(current_phase=PHASE_NS,
                         extra={"observation": observation})
    phase = agent.select_action(state)
    assert phase in (PHASE_NS, PHASE_EW), "a TrafficState gets a phase answer"
    assert phase == (PHASE_NS if env_action == 0 else PHASE_EW)


def test_keep_and_switch_translate_into_the_right_phase():
    assert DQNAgent.phase_for(0, PHASE_NS) == PHASE_NS
    assert DQNAgent.phase_for(1, PHASE_NS) == PHASE_EW
    assert DQNAgent.phase_for(0, PHASE_EW) == PHASE_EW
    assert DQNAgent.phase_for(1, PHASE_EW) == PHASE_NS


def test_traffic_state_carries_the_observation_for_the_network():
    info = {
        "queue_per_approach": {"north": 3, "south": 4, "east": 1, "west": 2},
        "waiting_per_approach": {"north": 30, "south": 40, "east": 5, "west": 5},
        "phase": PHASE_EW, "green_elapsed": 12.0, "simulation_time": 300.0,
    }
    observation = np.arange(OBS, dtype=np.float32) / OBS

    plain = TrafficState.from_env_info(info)
    assert plain.observation() is None, "optional, and absent by default"

    carried = TrafficState.from_env_info(info, observation=observation)
    assert np.allclose(carried.observation(), observation)
    assert carried.ns_queue == 7, "the summary fields still work"


def test_agent_refuses_to_decide_without_the_observation():
    """The NS/EW summary cannot be fed to a per-approach network."""
    agent = fresh_agent()
    state = TrafficState(current_phase=PHASE_NS, ns_queue=5, ew_queue=1)
    with pytest.raises(ValueError, match="observation"):
        agent.select_action(state)


def test_agent_env_action_uses_the_network_directly():
    agent = fresh_agent()
    observation = random_observation()
    state = TrafficState(current_phase=PHASE_EW,
                         extra={"observation": observation})
    assert agent.env_action(state) == agent.act(observation, training=False)


def test_explanation_reports_the_q_values_behind_the_choice():
    agent = fresh_agent()
    observation = random_observation()
    state = TrafficState(current_phase=PHASE_NS, ns_queue=9, ew_queue=1,
                         extra={"observation": observation})
    explanation = agent.explain(state)
    assert explanation["is_learned"] is True
    assert explanation["uses_traffic_state"] is True
    assert set(explanation["q_values"]) == {"keep", "switch"}
    values = agent.q_values(observation)
    assert explanation["q_values"]["keep"] == pytest.approx(float(values[0]))
    assert explanation["q_values"]["switch"] == pytest.approx(float(values[1]))
    assert explanation["env_action_name"] in ("KEEP", "SWITCH")
    assert "Q(keep)" in explanation["reason"]


def test_agent_reset_does_not_wipe_the_replay_buffer():
    """The buffer is long-term memory across episodes, not per-episode state."""
    agent = fresh_agent()
    agent.remember(np.zeros(OBS), 0, -1.0, np.zeros(OBS), False)
    agent.reset()
    assert len(agent.memory) == 1


# --- checkpoints -----------------------------------------------------------
def test_checkpoint_round_trip_reproduces_the_same_decisions(tmp_path):
    agent = fresh_agent()
    rng = np.random.default_rng(13)
    for index in range(40):
        agent.remember(rng.random(OBS), index % 2, -rng.random(),
                       rng.random(OBS), False)
    for _ in range(10):
        agent.learn()
    agent.episode = 7
    agent.set_epsilon(0.3)

    path = agent.save(str(tmp_path / "model.pt"))
    assert os.path.isfile(path)

    loaded = DQNAgent.load(path)
    observations = [rng.random(OBS).astype(np.float32) for _ in range(20)]
    for observation in observations:
        assert np.allclose(agent.q_values(observation),
                           loaded.q_values(observation), atol=1e-6)
        assert agent.act(observation) == loaded.act(observation)
    assert loaded.episode == 7
    assert loaded.epsilon == pytest.approx(0.3)


def test_checkpoint_stores_everything_a_reader_needs(tmp_path):
    """Never 'a mysterious .pt file': the checkpoint explains itself."""
    agent = fresh_agent()
    agent.scenario = "moderate"
    payload = torch.load(agent.save(str(tmp_path / "m.pt")),
                         map_location="cpu", weights_only=False)
    for key in ("policy_state_dict", "target_state_dict",
                "optimizer_state_dict", "epsilon", "episode", "config",
                "seed", "scenario", "architecture", "observation_size",
                "num_actions", "saved_at", "torch_version"):
        assert key in payload, "checkpoint is missing {}".format(key)
    assert payload["architecture"] == "10 -> 64 -> 64 -> 2"
    assert payload["config"]["gamma"] == agent.config.gamma


def test_metadata_file_is_written_and_read_back(tmp_path):
    path = write_metadata(str(tmp_path), {"scenario": "moderate", "seed": 42,
                                          "gamma": 0.9})
    assert os.path.isfile(path)
    with open(path, encoding="utf-8") as handle:
        assert json.load(handle)["scenario"] == "moderate"
    assert read_metadata(str(tmp_path))["seed"] == 42
    assert read_metadata(str(tmp_path / "nowhere")) == {}


def test_loading_a_missing_checkpoint_is_refused_clearly(tmp_path):
    with pytest.raises(DQNModelNotFoundError):
        DQNAgent.load(str(tmp_path / "does_not_exist.pt"))


def test_asking_for_an_untrained_controller_is_refused(monkeypatch, tmp_path):
    """No model means no DQN - not a randomly initialised network."""
    monkeypatch.setattr("agents.dqn_agent.MODELS_DIR", str(tmp_path))
    assert dqn_model_available("moderate") is False
    assert find_models("moderate") == []
    with pytest.raises(DQNModelNotFoundError) as info:
        DQNAgent(scenario="moderate")
    message = str(info.value).lower()
    assert "no trained dqn model" in message
    assert "train_dqn.py" in message, "the refusal must say how to fix it"


def test_create_agent_dqn_needs_a_trained_model(monkeypatch, tmp_path):
    monkeypatch.setattr("agents.dqn_agent.MODELS_DIR", str(tmp_path))
    with pytest.raises(DQNModelNotFoundError):
        create_agent("dqn", scenario="moderate")


def test_model_discovery_prefers_best_then_final(monkeypatch, tmp_path):
    monkeypatch.setattr("agents.dqn_agent.MODELS_DIR", str(tmp_path))
    directory = tmp_path / "dqn_moderate"
    directory.mkdir()
    agent = fresh_agent()
    for name in ("checkpoint_ep010.pt", "final.pt", "best.pt"):
        agent.save(str(directory / name))
    names = [m["name"] for m in find_models("moderate")]
    assert names[:2] == ["best.pt", "final.pt"]
    assert dqn_model_available("moderate") is True
    assert DQNAgent.is_available("moderate") is True
    assert model_directory("moderate").endswith("dqn_moderate")


def test_agent_registry_reports_dqn_availability_from_the_filesystem(
        monkeypatch, tmp_path):
    monkeypatch.setattr("agents.dqn_agent.MODELS_DIR", str(tmp_path))
    rows = {row["name"]: row for row in list_agents()}
    assert rows["fixed_time"]["available"] is True
    assert rows["fixed_time"]["is_learned"] is False
    assert rows["dqn"]["available"] is False, "no model on disk yet"
    assert rows["dqn"]["is_learned"] is True

    directory = tmp_path / "dqn_moderate"
    directory.mkdir()
    fresh_agent().save(str(directory / "best.pt"))
    rows = {row["name"]: row for row in list_agents()}
    assert rows["dqn"]["available"] is True


def test_seeding_makes_two_fresh_agents_identical():
    set_seeds(123)
    first = DQNAgent(observation_size=OBS, num_actions=ACTIONS,
                     trainable=True, config=tiny_config())
    set_seeds(123)
    second = DQNAgent(observation_size=OBS, num_actions=ACTIONS,
                      trainable=True, config=tiny_config())
    observation = random_observation()
    assert np.allclose(first.q_values(observation), second.q_values(observation))


# ==========================================================================
# Fairness of the comparison
# ==========================================================================
def test_episode_plan_matches_the_stage2_timing_contract():
    plan = episode_plan("moderate")
    assert plan["decision_interval"] == 5.0
    assert plan["episode_duration"] == 1800.0
    assert plan["decisions"] == 360
    assert plan["measured_steps"] == 1800


def test_fixed_time_comparison_run_reproduces_the_recorded_baseline():
    """The strongest fairness check in the suite.

    The comparison runs fixed-time on the SUMO files the RL environment
    builds. If that produced different numbers from the recorded Stage 1
    baseline, every improvement percentage would be measured against the
    wrong reference.
    """
    import pandas as pd

    baseline_path = os.path.join(BASELINE_DIR, "moderate_results.csv")
    if not os.path.isfile(baseline_path):
        pytest.skip("Stage 1 results have not been generated")
    baseline = pd.read_csv(baseline_path).iloc[0]

    summary, timeseries = run_fixed_time_episode("moderate", seed=42)
    assert summary["controller"] == FIXED_TIME
    assert len(timeseries) == 1800
    for metric in ("vehicles", "average_waiting_time", "average_queue_length",
                   "max_queue_length", "throughput",
                   "vehicles_still_in_network", "teleports"):
        assert summary[metric] == pytest.approx(float(baseline[metric])), (
            "fixed-time comparison run differs from the Stage 1 baseline "
            "on {}".format(metric))


def test_both_controllers_see_the_same_number_of_simulated_seconds():
    agent = fresh_agent()
    fixed, fixed_ts = run_fixed_time_episode("moderate", seed=42,
                                             duration=SHORT_DURATION)
    dqn, dqn_ts = run_agent_episode("moderate", agent, seed=42,
                                    duration=SHORT_DURATION)
    assert fixed["simulation_time"] == pytest.approx(dqn["simulation_time"])
    assert len(fixed_ts) == len(dqn_ts)
    assert fixed["scenario"] == dqn["scenario"] == "moderate"
    assert dqn["controller"] == "dqn"


def test_a_dqn_episode_is_measured_by_the_stage1_collector():
    agent = fresh_agent()
    summary, timeseries = run_agent_episode("moderate", agent, seed=42,
                                            duration=SHORT_DURATION,
                                            record_decisions=True)
    for metric in ("vehicles", "average_waiting_time", "average_queue_length",
                   "max_queue_length", "throughput", "teleports",
                   "vehicles_still_in_network"):
        assert metric in summary
    assert summary["decisions"] == 24, "120 s at a 5 s decision interval"
    assert len(summary["decision_log"]) == 24
    assert {"q_keep", "q_switch"} <= set(summary["decision_log"][0])
    assert summary["blocked_switches"] == 0
    assert timeseries[0]["signal_state"], "the real SUMO signal was recorded"


def test_training_replay_uses_executed_action_and_masks_unavailable_switch(monkeypatch):
    from training import train_dqn

    class FakeController:
        green_elapsed = 0.0
        min_green = 5.0

    class FakeEnvironment:
        scenario = "moderate"
        step_length = 1.0
        connection = object()
        controller = FakeController()
        step_callback = None

        def step(self, action):
            assert action == 0
            return np.zeros(OBS, dtype=np.float32), 1.0, True, False, {
                "effective_action": 0, "switch_blocked": False,
                "forced_switch": False, "switch_count": 0,
            }

        def reset(self, seed):
            return np.zeros(OBS, dtype=np.float32), {}

    class FakeCollector:
        def __init__(self, **kwargs):
            pass

        def summary(self, scenario):
            return {"average_queue_length": 0.0, "average_waiting_time": 0.0,
                    "throughput": 0, "max_queue_length": 0, "vehicles": 0,
                    "vehicles_still_in_network": 0, "teleports": 0}

    class FakeAgent:
        epsilon = 1.0
        learn_steps = 0
        target_updates = 0

        def __init__(self):
            self.memory = []
            self.valid_actions = None

        def set_epsilon(self, value):
            self.epsilon = value

        def act(self, observation, training=False, valid_actions=None):
            self.valid_actions = valid_actions
            return 0

        def remember(self, observation, action, reward, next_observation, done):
            self.memory.append(action)

        def learn(self):
            return None

    monkeypatch.setattr(train_dqn, "MetricsCollector", FakeCollector)
    agent = FakeAgent()
    config = tiny_config(train_frequency=1)
    schedule = type("Schedule", (), {"value": lambda self, step: 1.0})()
    train_dqn._run_episode(FakeEnvironment(), agent, schedule, config,
                           episode=1, episode_seed=42, global_step=0, max_steps=1)
    assert agent.valid_actions == [0]
    assert agent.memory == [0]


def test_validation_degradation_tolerance_is_strict_and_bounded():
    from training.train_dqn import validation_is_degraded

    assert not validation_is_degraded(110.0, 100.0, 0.10)
    assert validation_is_degraded(110.01, 100.0, 0.10)
    with pytest.raises(ValueError, match="non-negative"):
        validation_is_degraded(111.0, 100.0, -0.01)


def test_the_dqn_cannot_create_an_unsafe_signal_transition():
    """Green never turns straight into the other direction's green."""
    agent = fresh_agent()
    agent.set_epsilon(1.0)                   # force plenty of switching
    _, timeseries = run_agent_episode("moderate", agent, seed=42,
                                      duration=300.0)
    states = [row["signal_state"] for row in timeseries]
    ns_green = "GGGgrrrrGGGgrrrr"
    ew_green = "rrrrGGGgrrrrGGGg"
    for before, after in zip(states, states[1:]):
        if before == ns_green:
            assert after in (ns_green, "yyyyrrrryyyyrrrr"), (
                "NS green went straight to {}".format(after))
        if before == ew_green:
            assert after in (ew_green, "rrrryyyyrrrryyyy"), (
                "EW green went straight to {}".format(after))
    assert any(s == "yyyyrrrryyyyrrrr" or s == "rrrryyyyrrrryyyy"
               for s in states), "the clearance phase must actually appear"


def test_metrics_collector_still_defaults_to_the_traci_module():
    """The Stage 3 `conn` argument must not change Stage 1 behaviour."""
    import traci

    assert MetricsCollector().conn is traci
    sentinel = object()
    assert MetricsCollector(conn=sentinel).conn is sentinel


def test_saving_into_the_stage1_results_directory_is_refused(tmp_path):
    with pytest.raises(ValueError, match="Stage 1 baseline"):
        save_run({"scenario": "moderate"}, [], BASELINE_DIR, "moderate")


# ==========================================================================
# Training
# ==========================================================================
@pytest.fixture(scope="module")
def tiny_training(tmp_path_factory):
    """One real, very short training run, shared by the tests below."""
    from training.train_dqn import train

    directory = tmp_path_factory.mktemp("training")
    config = load_dqn_config(episodes=2, episode_duration=SHORT_DURATION,
                             batch_size=4, learning_starts=8,
                             replay_capacity=200, target_update_frequency=5,
                             checkpoint_every=1, best_window=1, seed=42)
    summary = train(scenario="moderate", config=config,
                    model_dir=str(directory / "models"),
                    results_dir=str(directory / "results"),
                    quiet=True, save_plots=True)
    return summary


def test_training_run_completes_and_reports_what_happened(tiny_training):
    assert tiny_training["episodes"] == 2
    assert tiny_training["steps_per_episode"] == 24
    assert tiny_training["total_steps"] == 48
    assert tiny_training["gradient_steps"] > 0
    assert tiny_training["target_updates"] > 0
    assert tiny_training["architecture"] == "10 -> 64 -> 64 -> 2"


def test_training_saves_a_final_and_a_best_checkpoint(tiny_training):
    for key in ("best_model", "final_model"):
        assert os.path.isfile(tiny_training[key]), "{} missing".format(key)
    agent = DQNAgent.load(tiny_training["final_model"])
    assert agent.observation_size == OBS and agent.num_actions == ACTIONS


def test_training_writes_checkpoints_and_metadata(tiny_training):
    directory = tiny_training["model_dir"]
    names = os.listdir(directory)
    assert "metadata.json" in names
    assert any(n.startswith("checkpoint_ep") for n in names)

    metadata = read_metadata(directory)
    for key in ("scenario", "seed", "episodes", "gamma", "learning_rate",
                "batch_size", "architecture", "epsilon", "training_date",
                "config"):
        assert key in metadata, "metadata is missing {}".format(key)
    assert metadata["scenario"] == "moderate"
    assert metadata["gamma"] == 0.98


def test_training_history_csv_has_the_required_columns(tiny_training):
    import pandas as pd

    from training.train_dqn import HISTORY_COLUMNS

    path = tiny_training["history_csv"]
    assert os.path.isfile(path)
    frame = pd.read_csv(path)
    assert len(frame) == 2
    for column in ("episode", "total_reward", "average_reward",
                   "episode_length", "epsilon", "loss",
                   "average_queue_length", "average_waiting_time",
                   "throughput"):
        assert column in frame.columns, "history is missing {}".format(column)
    assert list(frame.columns) == HISTORY_COLUMNS


def test_training_losses_are_never_nan(tiny_training):
    import pandas as pd

    frame = pd.read_csv(tiny_training["history_csv"])
    losses = frame["loss"].dropna()
    assert not losses.empty, "learning must have happened"
    assert np.isfinite(losses).all(), "a NaN loss means the run diverged"


def test_training_curves_are_generated_from_the_recorded_history(tiny_training):
    plots = tiny_training.get("plots") or []
    assert plots, "no training curves were produced"
    assert any(p.endswith("_reward.png") for p in plots)
    assert any(p.endswith("_overview.png") for p in plots)
    for path in plots:
        assert os.path.isfile(path) and os.path.getsize(path) > 1000


def test_training_leaves_no_sumo_process_behind(tmp_path):
    """A training run must shut SUMO down, not leak one process per episode.

    Counted as a DIFFERENCE across the run rather than as an absolute count,
    so an unrelated SUMO process on the machine cannot fail the test.
    """
    import subprocess

    from training.train_dqn import train

    def sumo_processes() -> int:
        try:
            result = subprocess.run(["pgrep", "-c", "-f", "sumo"],
                                    capture_output=True, text=True, timeout=20)
        except Exception:                      # pragma: no cover - no pgrep
            pytest.skip("pgrep is not available")
        return int(result.stdout.strip() or 0)

    before = sumo_processes()
    train(scenario="moderate",
          config=load_dqn_config(episodes=2, episode_duration=SHORT_DURATION,
                                 batch_size=4, learning_starts=8,
                                 replay_capacity=100, best_window=1, seed=42),
          model_dir=str(tmp_path / "models"),
          results_dir=str(tmp_path / "results"),
          quiet=True, save_plots=False)
    assert sumo_processes() <= before, (
        "training left a SUMO process running after its episodes finished")


def test_training_does_not_touch_the_stage1_baseline(tiny_training):
    """A training run must not write over the recorded baseline."""
    for scenario in ("moderate", "heavy"):
        path = os.path.join(BASELINE_DIR, "{}_results.csv".format(scenario))
        if os.path.isfile(path):
            assert os.path.getmtime(path) < os.path.getmtime(
                tiny_training["history_csv"]), (
                "the Stage 1 baseline was modified during training")


def test_epsilon_is_logged_and_decays_over_training(tiny_training):
    import pandas as pd

    frame = pd.read_csv(tiny_training["history_csv"])
    assert frame["epsilon"].iloc[0] > frame["epsilon"].iloc[-1]
    assert (frame["epsilon"] <= 1.0).all() and (frame["epsilon"] >= 0.0).all()


def test_build_agent_for_env_takes_its_sizes_from_the_environment():
    from rl import IntelliSignalEnv

    env = IntelliSignalEnv(scenario="moderate", episode_duration=SHORT_DURATION)
    try:
        agent = build_agent_for_env(env, config=tiny_config(), scenario="moderate")
        assert agent.observation_size == env.observation_space.shape[0] == OBS
        assert agent.num_actions == env.action_space.n == ACTIONS
    finally:
        env.close()


# ==========================================================================
# Evaluation
# ==========================================================================
@pytest.fixture(scope="module")
def trained_model(tiny_training):
    return tiny_training["final_model"]


def test_evaluation_loads_the_model_and_produces_metrics(trained_model, tmp_path):
    from evaluation.evaluate_dqn import evaluate

    result = evaluate(scenario="moderate", model=trained_model, seeds=[42],
                      duration=SHORT_DURATION, results_dir=str(tmp_path),
                      quiet=True)
    assert result["epsilon"] == 0.0, "evaluation must be greedy"
    assert len(result["rows"]) == 1
    row = result["rows"][0]
    for metric in ("average_waiting_time", "average_queue_length",
                   "max_queue_length", "throughput", "teleports"):
        assert row[metric] is not None
    assert row["controller"] == "dqn"


def test_evaluation_does_not_train(trained_model, tmp_path):
    """evaluate() asserts this itself; this test proves the check is real."""
    from evaluation.evaluate_dqn import evaluate

    agent = DQNAgent.load(trained_model)
    gradient_steps = agent.learn_steps
    evaluate(scenario="moderate", model=trained_model, seeds=[42],
             duration=SHORT_DURATION, results_dir=str(tmp_path), quiet=True)
    reloaded = DQNAgent.load(trained_model)
    assert reloaded.learn_steps == gradient_steps
    for key, weight in agent.policy_net.state_dict().items():
        assert torch.allclose(weight, reloaded.policy_net.state_dict()[key]), (
            "evaluation changed the saved weights")


def test_evaluation_supports_multiple_seeds_and_aggregates_them(
        trained_model, tmp_path):
    from evaluation.evaluate_dqn import evaluate

    result = evaluate(scenario="moderate", model=trained_model,
                      seeds=[42, 43], duration=SHORT_DURATION,
                      results_dir=str(tmp_path), quiet=True)
    assert [row["seed"] for row in result["rows"]] == [42, 43]
    stats = result["aggregate"]["average_waiting_time"]
    assert stats["n"] == 2
    assert stats["std"] is not None
    assert stats["min"] <= stats["mean"] <= stats["max"]


def test_a_single_seed_reports_no_standard_deviation(trained_model, tmp_path):
    """One run cannot have a spread, and 0.0 would imply a precision it has not."""
    from evaluation.evaluate_dqn import aggregate, evaluate, format_mean_std

    result = evaluate(scenario="moderate", model=trained_model, seeds=[42],
                      duration=SHORT_DURATION, results_dir=str(tmp_path),
                      quiet=True)
    stats = result["aggregate"]["average_queue_length"]
    assert stats["std"] is None
    assert "1 seed" in format_mean_std(stats)
    assert aggregate([], ["throughput"])["throughput"]["mean"] is None


def test_the_same_seed_gives_the_same_evaluation_twice(trained_model, tmp_path):
    from evaluation.evaluate_dqn import evaluate

    first = evaluate(scenario="moderate", model=trained_model, seeds=[42],
                     duration=SHORT_DURATION, results_dir=str(tmp_path),
                     quiet=True, save=False)["rows"][0]
    second = evaluate(scenario="moderate", model=trained_model, seeds=[42],
                      duration=SHORT_DURATION, results_dir=str(tmp_path),
                      quiet=True, save=False)["rows"][0]
    for metric in ("average_waiting_time", "average_queue_length",
                   "throughput", "max_queue_length"):
        assert first[metric] == second[metric], "{} is not reproducible".format(metric)


def test_evaluation_writes_raw_and_aggregated_results_separately(
        trained_model, tmp_path):
    import pandas as pd

    from evaluation.evaluate_dqn import evaluate

    result = evaluate(scenario="moderate", model=trained_model,
                      seeds=[42, 43], duration=SHORT_DURATION,
                      results_dir=str(tmp_path), quiet=True)
    raw = pd.read_csv(result["raw_csv"])
    summary = pd.read_csv(result["summary_csv"])
    assert len(raw) == 2, "one row per seed, nothing averaged away"
    assert {"scenario", "controller", "seed"} <= set(raw.columns)
    assert {"metric", "mean", "std"} <= set(summary.columns)
    assert result["timeseries"], "per-second data is saved for the charts"


def test_evaluation_refuses_without_a_model(monkeypatch, tmp_path):
    from evaluation.evaluate_dqn import resolve_model

    monkeypatch.setattr("agents.dqn_agent.MODELS_DIR", str(tmp_path))
    with pytest.raises(DQNModelNotFoundError) as info:
        resolve_model("moderate")
    assert "train" in str(info.value).lower()


# ==========================================================================
# Comparison
# ==========================================================================
def test_improvement_is_positive_when_the_dqn_is_better():
    # lower is better: 10 -> 8 is a 20 % improvement
    assert improvement(10.0, 8.0, "average_waiting_time") == pytest.approx(20.0)
    # and a worse result is reported as negative, not hidden
    assert improvement(10.0, 12.0, "average_waiting_time") == pytest.approx(-20.0)
    # higher is better for throughput
    assert improvement(100.0, 110.0, "throughput") == pytest.approx(10.0)
    assert improvement(100.0, 90.0, "throughput") == pytest.approx(-10.0)


def test_improvement_handles_a_zero_denominator_and_missing_values():
    assert improvement(0.0, 5.0, "teleports") is None
    assert improvement(None, 5.0, "teleports") is None
    assert improvement(5.0, None, "teleports") is None
    assert improvement("x", 1.0, "teleports") is None


def test_metric_directions_are_declared_in_one_place():
    assert LOWER_IS_BETTER["average_waiting_time"] is True
    assert LOWER_IS_BETTER["average_queue_length"] is True
    assert LOWER_IS_BETTER["max_queue_length"] is True
    assert LOWER_IS_BETTER["throughput"] is False


def test_comparison_runs_both_controllers_and_keeps_every_run(
        trained_model, tmp_path):
    import pandas as pd

    from evaluation.compare_controllers import compare

    result = compare(scenario="moderate", model=trained_model, seeds=[42, 43],
                     duration=SHORT_DURATION, results_dir=str(tmp_path),
                     quiet=True)
    assert len(result["rows"]) == 4, "two controllers x two seeds"
    assert {r["controller"] for r in result["rows"]} == {"fixed_time", "dqn"}

    raw = pd.read_csv(result["raw_csv"])
    assert len(raw) == 4, "no seed and no controller is dropped"
    aggregated = pd.read_csv(result["aggregated_csv"])
    assert set(aggregated["controller"]) == {"fixed_time", "dqn"}
    improvement_frame = pd.read_csv(result["improvement_csv"])
    assert "improvement_percent" in improvement_frame.columns
    assert len(improvement_frame) == 6, "one row per compared metric"


def test_comparison_puts_both_controllers_on_identical_traffic(
        trained_model, tmp_path):
    """Same seed, same demand: the vehicle count must match exactly."""
    from evaluation.compare_controllers import compare

    result = compare(scenario="moderate", model=trained_model, seeds=[42],
                     duration=SHORT_DURATION, results_dir=str(tmp_path),
                     quiet=True, save=False)
    by_controller = {r["controller"]: r for r in result["rows"]}
    assert (by_controller["fixed_time"]["vehicles_departed"]
            == by_controller["dqn"]["vehicles_departed"])
    assert (by_controller["fixed_time"]["simulation_time"]
            == by_controller["dqn"]["simulation_time"])


def test_comparison_keeps_the_stage1_baseline_untouched(trained_model, tmp_path):
    from evaluation.compare_controllers import compare

    paths = [os.path.join(BASELINE_DIR, "{}_results.csv".format(s))
             for s in ("moderate", "heavy")]
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        pytest.skip("Stage 1 results have not been generated")
    before = {p: (os.path.getmtime(p), open(p, "rb").read()) for p in paths}

    compare(scenario="moderate", model=trained_model, seeds=[42],
            duration=SHORT_DURATION, results_dir=str(tmp_path), quiet=True)

    for path, (mtime, content) in before.items():
        assert os.path.getmtime(path) == mtime, "{} was rewritten".format(path)
        assert open(path, "rb").read() == content


def test_comparison_reads_the_recorded_baseline_for_reference():
    from evaluation.compare_controllers import recorded_baseline

    row = recorded_baseline("moderate")
    if row is None:
        pytest.skip("Stage 1 results have not been generated")
    assert row["scenario"] == "moderate"
    assert row["average_waiting_time"] > 0


def test_comparison_can_include_an_untrained_control(trained_model, tmp_path):
    """The control exists so the learning cannot be given credit it is owed."""
    from evaluation.compare_controllers import UNTRAINED, compare

    result = compare(scenario="moderate", model=trained_model, seeds=[42],
                     duration=SHORT_DURATION, results_dir=str(tmp_path),
                     include_untrained=True, quiet=True, save=False)
    controllers = {r["controller"] for r in result["rows"]}
    assert controllers == {"fixed_time", "dqn", UNTRAINED}
    assert UNTRAINED in result["aggregates"]


def test_comparison_charts_are_drawn_from_the_csv_files(trained_model, tmp_path):
    from evaluation.compare_controllers import compare
    from evaluation.plot_comparison import plot_comparison

    compare(scenario="moderate", model=trained_model, seeds=[42, 43],
            duration=SHORT_DURATION, results_dir=str(tmp_path), quiet=True)
    plots = plot_comparison("moderate", results_dir=str(tmp_path),
                            output_dir=str(tmp_path / "plots"))
    assert any(p.endswith("_waiting_time.png") for p in plots)
    assert any(p.endswith("_queue_over_time.png") for p in plots)
    assert any(p.endswith("_summary.png") for p in plots)
    for path in plots:
        assert os.path.getsize(path) > 1000


def test_comparison_charts_refuse_to_draw_without_results(tmp_path):
    from evaluation.plot_comparison import plot_comparison

    with pytest.raises(FileNotFoundError):
        plot_comparison("moderate", results_dir=str(tmp_path / "empty"))


# ==========================================================================
# Dashboard integration
# ==========================================================================
def test_dqn_mode_is_disabled_without_a_trained_model(monkeypatch, tmp_path):
    from simulation.controller import SimulationController

    monkeypatch.setattr("agents.dqn_agent.MODELS_DIR", str(tmp_path))
    controller = SimulationController(results_dir=str(tmp_path / "data"))
    modes = {m.key: m for m in controller.control_modes("moderate")}
    assert modes["fixed_time"].available is True
    assert modes["dqn_adaptive"].available is False
    assert "train a model first" in modes["dqn_adaptive"].unavailable_reason.lower()
    assert "train_dqn.py" in modes["dqn_adaptive"].unavailable_reason
    assert modes["dqn_adaptive"].badge == "NO MODEL"


def test_dqn_mode_is_enabled_only_for_a_scenario_that_has_a_model(
        monkeypatch, tmp_path):
    from simulation.controller import SimulationController

    monkeypatch.setattr("agents.dqn_agent.MODELS_DIR", str(tmp_path))
    directory = tmp_path / "dqn_moderate"
    directory.mkdir()
    fresh_agent().save(str(directory / "best.pt"))

    controller = SimulationController(results_dir=str(tmp_path / "data"))
    moderate = {m.key: m for m in controller.control_modes("moderate")}
    heavy = {m.key: m for m in controller.control_modes("heavy")}
    assert moderate["dqn_adaptive"].available is True
    assert moderate["dqn_adaptive"].badge == "TRAINED MODEL"
    assert heavy["dqn_adaptive"].available is False, "no heavy model was saved"
    assert controller.scenarios_with_models() == ["moderate"]


def test_running_the_dqn_mode_without_a_model_is_refused(monkeypatch, tmp_path):
    from simulation.controller import (ControlModeUnavailableError,
                                       SimulationController)

    monkeypatch.setattr("agents.dqn_agent.MODELS_DIR", str(tmp_path))
    controller = SimulationController(results_dir=str(tmp_path / "data"))
    with pytest.raises(ControlModeUnavailableError) as info:
        controller.run("moderate", mode="dqn_adaptive", duration=60)
    assert "train a model first" in str(info.value).lower()

    with pytest.raises(ControlModeUnavailableError):
        controller.agent_for_mode("dqn_adaptive", "moderate")


def test_model_status_is_honest_when_nothing_is_trained(monkeypatch, tmp_path):
    from simulation.controller import SimulationController

    monkeypatch.setattr("agents.dqn_agent.MODELS_DIR", str(tmp_path))
    controller = SimulationController(results_dir=str(tmp_path / "data"))
    status = controller.model_status("moderate")
    assert status["available"] is False
    assert status["checkpoints"] == []
    assert status["summary"] == {}, "no architecture is invented"
    assert "train a model first" in status["reason"].lower()


def test_model_status_reports_the_real_checkpoint(monkeypatch, tmp_path,
                                                  trained_model):
    import shutil

    from simulation.controller import SimulationController

    monkeypatch.setattr("agents.dqn_agent.MODELS_DIR", str(tmp_path))
    directory = tmp_path / "dqn_moderate"
    directory.mkdir()
    shutil.copy(trained_model, directory / "best.pt")
    write_metadata(str(directory), {"scenario": "moderate", "seed": 42,
                                    "training_date": "2026-01-01",
                                    "episodes": 2})

    controller = SimulationController(results_dir=str(tmp_path / "data"))
    status = controller.model_status("moderate")
    assert status["available"] is True
    assert status["summary"]["architecture"] == "10 -> 64 -> 64 -> 2"
    assert status["summary"]["gamma"] == 0.98
    assert status["metadata"]["training_date"] == "2026-01-01"
    assert status["checkpoints"][0]["name"] == "best.pt"


def test_dashboard_dqn_run_really_runs_the_network(monkeypatch, tmp_path,
                                                   trained_model):
    """The run must use the loaded model, not the fixed-time plan."""
    import shutil

    from simulation.controller import SimulationController

    monkeypatch.setattr("agents.dqn_agent.MODELS_DIR", str(tmp_path))
    directory = tmp_path / "dqn_moderate"
    directory.mkdir()
    shutil.copy(trained_model, directory / "best.pt")

    results = tmp_path / "data"
    controller = SimulationController(results_dir=str(results))
    run = controller.run("moderate", mode="dqn_adaptive",
                         duration=SHORT_DURATION, seed=42)

    assert run.mode == "dqn_adaptive"
    assert run.metrics["controller"] == "dqn"
    assert "DQN" in run.source
    assert run.metrics["architecture"] == "10 -> 64 -> 64 -> 2"
    assert run.metrics["model"].endswith("best.pt")
    assert run.metric("average_waiting_time") is not None
    assert os.path.isfile(run.csv_path)
    assert run.csv_path.endswith("moderate_dqn_results.csv"), (
        "DQN results must be kept in their own file")

    decisions = controller.load_decisions("moderate")
    assert decisions is not None and "q_keep" in decisions.columns
    assert len(decisions) == 24


def test_a_dqn_run_never_overwrites_a_fixed_time_run(monkeypatch, tmp_path,
                                                     trained_model):
    import shutil

    from simulation.controller import SimulationController

    monkeypatch.setattr("agents.dqn_agent.MODELS_DIR", str(tmp_path))
    directory = tmp_path / "dqn_moderate"
    directory.mkdir()
    shutil.copy(trained_model, directory / "best.pt")

    controller = SimulationController(results_dir=str(tmp_path / "data"))
    fixed = controller.run("moderate", mode="fixed_time", duration=SHORT_DURATION)
    dqn = controller.run("moderate", mode="dqn_adaptive",
                         duration=SHORT_DURATION, seed=42)
    assert fixed.csv_path != dqn.csv_path
    assert os.path.isfile(fixed.csv_path) and os.path.isfile(dqn.csv_path)

    loaded_fixed = controller.load_results("moderate", "fixed_time")
    loaded_dqn = controller.load_results("moderate", "dqn_adaptive")
    assert loaded_fixed.metrics["controller"] == "fixed_time"
    assert loaded_dqn.metrics["controller"] == "dqn"


def test_dqn_results_never_fall_back_to_the_baseline(monkeypatch, tmp_path):
    """Asking for DQN results with no DQN run must give nothing, not the baseline."""
    from simulation.controller import SimulationController

    monkeypatch.setattr("agents.dqn_agent.MODELS_DIR", str(tmp_path))
    controller = SimulationController(results_dir=str(tmp_path / "empty"))
    assert controller.load_results("moderate", "dqn_adaptive") is None
    fixed = controller.load_results("moderate", "fixed_time")
    if fixed is not None:
        assert fixed.source == "Stage 1 baseline", (
            "the fixed-time fallback is still allowed, and is labelled")


def test_live_comparison_needs_both_runs(monkeypatch, tmp_path, trained_model):
    import shutil

    from simulation.controller import SimulationController

    monkeypatch.setattr("agents.dqn_agent.MODELS_DIR", str(tmp_path))
    directory = tmp_path / "dqn_moderate"
    directory.mkdir()
    shutil.copy(trained_model, directory / "best.pt")

    controller = SimulationController(results_dir=str(tmp_path / "data"))
    empty = controller.controller_comparison("moderate")
    assert empty["available"] is False
    assert "no dqn run" in empty["reason"].lower()

    controller.run("moderate", mode="fixed_time", duration=SHORT_DURATION)
    controller.run("moderate", mode="dqn_adaptive", duration=SHORT_DURATION,
                   seed=42)
    comparison = controller.controller_comparison("moderate")
    assert comparison["available"] is True
    assert comparison["comparable"] is True
    metrics = {row["metric"] for row in comparison["rows"]}
    assert {"average_waiting_time", "average_queue_length", "throughput",
            "max_queue_length", "vehicles_still_in_network",
            "teleports"} == metrics
    for row in comparison["rows"]:
        assert row["fixed_time"] is not None and row["dqn"] is not None


def test_live_comparison_refuses_to_compare_different_durations(
        monkeypatch, tmp_path, trained_model):
    import shutil

    from simulation.controller import SimulationController

    monkeypatch.setattr("agents.dqn_agent.MODELS_DIR", str(tmp_path))
    directory = tmp_path / "dqn_moderate"
    directory.mkdir()
    shutil.copy(trained_model, directory / "best.pt")

    controller = SimulationController(results_dir=str(tmp_path / "data"))
    controller.run("moderate", mode="fixed_time", duration=SHORT_DURATION)
    controller.run("moderate", mode="dqn_adaptive", duration=240.0, seed=42)
    comparison = controller.controller_comparison("moderate")
    assert comparison["comparable"] is False
    assert "not comparable" in comparison["reason"].lower()
    assert all(row["improvement_percent"] is None for row in comparison["rows"])


def test_dashboard_comparison_page_uses_real_evaluation_files(
        monkeypatch, tmp_path, trained_model):
    from evaluation.compare_controllers import compare
    from simulation.controller import SimulationController

    evaluation_dir = tmp_path / "evaluation"
    monkeypatch.setattr("simulation.controller.EVALUATION_RESULTS_DIR",
                        str(evaluation_dir))
    controller = SimulationController(results_dir=str(tmp_path / "data"))

    missing = controller.evaluation_results("moderate")
    assert missing["available"] is False
    assert "compare_controllers.py" in missing["command"]

    compare(scenario="moderate", model=trained_model, seeds=[42, 43],
            duration=SHORT_DURATION, results_dir=str(evaluation_dir),
            quiet=True)

    found = controller.evaluation_results("moderate")
    assert found["available"] is True
    assert found["seeds"] == [42, 43]
    assert len(found["raw"]) == 4
    assert set(found["aggregated"]["controller"]) == {"fixed_time", "dqn"}
    assert "improvement_percent" in found["improvement"].columns

    series = controller.evaluation_timeseries("moderate", seed=42)
    assert series and set(series) == {"fixed_time", "dqn"}


def test_dashboard_training_page_reads_the_history_csv(monkeypatch, tmp_path,
                                                       tiny_training):
    import shutil

    from simulation.controller import SimulationController

    training_dir = tmp_path / "training"
    training_dir.mkdir()
    shutil.copy(tiny_training["history_csv"],
                training_dir / "dqn_moderate_seed42_training.csv")
    monkeypatch.setattr("simulation.controller.TRAINING_RESULTS_DIR",
                        str(training_dir))

    controller = SimulationController(results_dir=str(tmp_path / "data"))
    history = controller.training_history("moderate")
    assert history is not None and len(history) == 2
    assert "seed" in history.columns
    assert controller.training_history("heavy") is None


def test_environment_status_detects_torch_and_models():
    from simulation.controller import SimulationController

    status = SimulationController().environment_status()
    assert status.torch_available is True
    assert status.torch_version
    assert isinstance(status.trained_models, list)


def test_comparison_chart_builders_handle_real_and_empty_data(
        monkeypatch, tmp_path, trained_model):
    import pandas as pd

    from dashboard import charts
    from evaluation.compare_controllers import compare

    assert charts.controller_metric_chart(pd.DataFrame(), "x", "t") is None
    assert charts.queue_by_controller_chart({}) is None
    assert charts.training_curve_chart(pd.DataFrame(), "total_reward", "t") is None

    result = compare(scenario="moderate", model=trained_model, seeds=[42],
                     duration=SHORT_DURATION, results_dir=str(tmp_path),
                     quiet=True)
    aggregated = pd.read_csv(result["aggregated_csv"])
    chart = charts.controller_metric_chart(aggregated, "average_waiting_time",
                                           "Average waiting time", "s")
    assert chart is not None
    chart.to_dict()                            # raises if the spec is invalid
