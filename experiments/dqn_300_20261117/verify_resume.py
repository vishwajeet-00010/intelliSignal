"""CPU-only recovery verification with real DQN/replay and synthetic episodes.

No SUMO processes are started and no project model/result files are changed.
"""
from pathlib import Path
import copy
import json
import random
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import resumable_training as subject


class SyntheticEnv:
    def __init__(self, **kwargs):
        self.observation_space = SimpleNamespace(shape=(10,))
        self.action_space = SimpleNamespace(n=2)
        self.max_decisions = 8
        self.episode_duration = 80.0
        self.decision_interval = 10.0

    def close(self):
        pass


def synthetic_episode(env, agent, schedule, config, episode, episode_seed,
                      global_step, max_steps):
    # Exercise global Python/NumPy/Torch RNG, independent action RNG,
    # independent replay sampling RNG, optimizer momentum, and target updates.
    losses, total, actions = [], 0.0, []
    for step in range(max_steps):
        observation = np.random.random(10).astype(np.float32)
        observation += torch.rand(10).numpy() + random.random()
        agent.set_epsilon(schedule.value(global_step))
        action = agent.act(observation, training=True)
        reward = random.random() + float(torch.rand(())) - action
        next_observation = np.random.random(10).astype(np.float32)
        agent.remember(observation, action, reward, next_observation, step == max_steps - 1)
        loss = agent.learn()
        if loss is not None:
            losses.append(loss)
        total += reward
        actions.append(action)
        global_step += 1
    row = {key: 0 for key in subject._write_history.__globals__["HISTORY_COLUMNS"]}
    row.update(episode=episode, total_reward=total,
               average_reward=total / max_steps, episode_length=max_steps,
               epsilon=agent.epsilon, loss=sum(losses) / len(losses),
               replay_size=len(agent.memory), gradient_steps=agent.learn_steps,
               target_updates=agent.target_updates, sumo_seed=episode_seed,
               action_trace=actions)
    return row, global_step


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            # These describe wall time/output location, not learning state.
            if key not in {"saved_at", "wall_clock_seconds", "elapsed_seconds", "snapshot_path"}:
                equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for x, y in zip(left, right):
            equal(x, y)
    else:
        assert left == right, (left, right)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def main():
    torch.set_num_threads(1)
    subject.IntelliSignalEnv = SyntheticEnv
    subject._run_episode = synthetic_episode
    config = subject.load_dqn_config(episodes=6, seed=42, batch_size=4,
                                    learning_starts=4, replay_capacity=100,
                                    best_window=2, target_update_frequency=5,
                                    checkpoint_every=2, hidden_layers=[8, 8])
    with tempfile.TemporaryDirectory(prefix="intellisignal_resume_") as temporary:
        folder = Path(temporary)

        def run(name, cfg=config, scenario="heavy"):
            return subject.train(scenario=scenario, config=cfg,
                                 model_dir=str(folder / name / "models"),
                                 results_dir=str(folder / name / "training"), quiet=True)

        run("uninterrupted")
        expected = load(folder / "uninterrupted" / "resume.pt")

        def interrupt_after_mutation(*args, **kwargs):
            result = synthetic_episode(*args, **kwargs)
            if args[4] == 4:
                raise RuntimeError("Synthetic interruption after unsaved episode4")
            return result

        subject._run_episode = interrupt_after_mutation
        try:
            run("resumed")
        except RuntimeError as error:
            assert "Synthetic interruption" in str(error)
        else:
            raise AssertionError("Interruption did not occur")
        assert load(folder / "resumed" / "resume.pt")["agent"]["episode"] == 3
        # Simulate a public artifact write newer than the committed snapshot.
        (folder / "resumed" / "models" / "best.pt").write_bytes(b"uncommitted")
        subject._run_episode = synthetic_episode
        run("resumed")
        actual = load(folder / "resumed" / "resume.pt")
        equal(expected, actual)
        equal(load(folder / "uninterrupted" / "models" / "best.pt"),
              load(folder / "resumed" / "models" / "best.pt"))
        equal(load(folder / "uninterrupted" / "models" / "final.pt"),
              load(folder / "resumed" / "models" / "final.pt"))

        # A completed rerun regenerates final artifacts without extra training.
        run("resumed")
        equal(actual, load(folder / "resumed" / "resume.pt"))
        changed = copy.deepcopy(config)
        changed.gamma = 0.8
        for cfg, scenario in ((changed, "heavy"), (config, "moderate")):
            try:
                run("resumed", cfg, scenario)
            except ValueError:
                pass
            else:
                raise AssertionError("Changed config/scenario accepted")
    print(json.dumps({"result": "passed", "checks": [
        "uninterrupted vs interrupted/resumed CPU weights exactly equal",
        "optimizer, target network, replay, all RNG states and action traces equal",
        "best checkpoint rollback and final checkpoint equal",
        "completed resume performs no extra episodes",
        "changed configuration/scenario refused"], "sumo_runs": 0}))


if __name__ == "__main__":
    main()
