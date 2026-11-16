"""
IntelliSignal - Stage 3
training/train_dqn.py

TRAINS the Deep Q-Network against the existing Stage 2 environment.

Usage
-----
    # quick demonstration (a few minutes)
    python training/train_dqn.py --scenario moderate --episodes 20 --seed 42

    # the default training run
    python training/train_dqn.py --scenario moderate --episodes 100 --seed 42
    python training/train_dqn.py --scenario heavy    --episodes 100 --seed 42

    # a second and third seed, to check the result is not one lucky run
    python training/train_dqn.py --scenario moderate --episodes 100 --seed 43
    python training/train_dqn.py --scenario moderate --episodes 100 --seed 44

The loop
--------
    reset the environment  -> first observation
    repeat until the episode ends:
        epsilon = schedule(global step)
        action  = epsilon-greedy over Q(observation)       agents/dqn_agent.py
        observation', reward, terminated, truncated = env.step(action)
        store (observation, action, reward, observation', terminated)
        every train_frequency steps: one gradient step from a random batch
        every target_update_frequency gradient steps: target <- policy
    record the episode's metrics, decay epsilon, save checkpoints

What this script does NOT do
---------------------------
  * It does not define the reward, the observation, the action or the safety
    rules. Those are Stage 2's and are used unchanged.
  * It does not touch results/csv/ - the recorded Stage 1 baseline. Training
    output goes to results/training/ and models/.
  * It does not claim the policy is good. It writes down what happened.
    Traffic reinforcement learning is noisy; 100 episodes is a small budget
    chosen so the run finishes on a student laptop, and the evaluation stage
    is what decides whether the learned policy is actually any better than
    the fixed-time plan.

Metrics written per episode
---------------------------
The learning numbers (return, loss, epsilon) come from the loop itself. The
traffic numbers come from `utils.metrics.MetricsCollector` - the same class
that produced the Stage 1 baseline - attached to the running simulation
through the environment's per-second hook, so a training episode and a
baseline run are measured identically:

    average_waiting_time   mean ACCUMULATED waiting time per vehicle  [s]
    average_queue_length   mean halting vehicles on the 8 incoming lanes
    throughput             vehicles that completed their route
    max_queue_length, teleports, vehicles, vehicles_still_in_network
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.dqn_agent import (EpsilonSchedule, build_agent_for_env,  # noqa: E402
                              model_directory, set_seeds, write_metadata)
from agents.dqn_config import DQNConfig, load_dqn_config          # noqa: E402
from rl.environment import IntelliSignalEnv                       # noqa: E402
from simulation.agent_simulation import (QUIET_SUMO_ARGS,          # noqa: E402
                                         _LateCollector)
from utils.metrics import MetricsCollector                         # noqa: E402

TRAINING_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "training")

#: The canonical project seed. A run with this seed owns models/dqn_<scenario>;
#: any other seed gets its own directory so the two never overwrite each other.
PRIMARY_SEED = 42

#: Columns of the training history CSV, in this order.
HISTORY_COLUMNS = [
    "episode", "total_reward", "average_reward", "episode_length",
    "epsilon", "loss", "average_queue_length", "average_waiting_time",
    "throughput", "max_queue_length", "vehicles", "vehicles_still_in_network",
    "teleports", "switch_count", "forced_switches", "blocked_switches",
    "replay_size", "gradient_steps", "target_updates", "sumo_seed",
    "wall_clock_seconds",
]


# ===========================================================================
def default_model_dir(scenario: str, seed: int) -> str:
    """Where this run's checkpoints go when --model-dir is not given."""
    if int(seed) == PRIMARY_SEED:
        return model_directory(scenario)
    return os.path.join(PROJECT_ROOT, "models",
                        "dqn_{}_seed{}".format(scenario, int(seed)))


def history_path(scenario: str, seed: int, directory: str = None) -> str:
    return os.path.join(directory or TRAINING_RESULTS_DIR,
                        "dqn_{}_seed{}_training.csv".format(scenario, int(seed)))


# ===========================================================================
def train(scenario: str = "moderate", config: Optional[DQNConfig] = None,
          model_dir: Optional[str] = None, results_dir: Optional[str] = None,
          quiet: bool = False, progress_every: int = 1,
          save_plots: bool = True) -> Dict[str, Any]:
    """Run the training loop and return a summary of what happened."""
    config = config or load_dqn_config()
    seed = int(config.seed)
    set_seeds(seed)

    model_dir = model_dir or default_model_dir(scenario, seed)
    results_dir = results_dir or TRAINING_RESULTS_DIR
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    env = IntelliSignalEnv(
        scenario=scenario,
        episode_duration=config.episode_duration,
        seed=seed,
        extra_sumo_args=QUIET_SUMO_ARGS,
    )

    try:
        # The network is sized from the environment's own spaces - the input
        # size is never typed in by hand.
        agent = build_agent_for_env(env, config=config, scenario=scenario,
                                    seed=seed)

        steps_per_episode = int(env.max_decisions)
        if config.max_steps_per_episode:
            steps_per_episode = min(steps_per_episode,
                                    int(config.max_steps_per_episode))
        schedule = EpsilonSchedule(
            start=config.epsilon_start, end=config.epsilon_end,
            decay_steps=config.decay_steps(steps_per_episode),
            kind=config.epsilon_decay, rate=config.epsilon_decay_rate)

        if not quiet:
            _print_header(scenario, config, agent, env, steps_per_episode,
                          schedule, model_dir)

        history: List[Dict[str, Any]] = []
        best_average = None
        best_episode = 0
        global_step = 0
        started = time.time()

        for episode in range(1, int(config.episodes) + 1):
            episode_seed = (seed + episode - 1) if config.vary_episode_seed else seed
            row, global_step = _run_episode(
                env, agent, schedule, config, episode, episode_seed,
                global_step, steps_per_episode)
            row["wall_clock_seconds"] = round(time.time() - started, 1)
            history.append(row)
            agent.episode = episode

            # best.pt follows the mean return of the last `best_window`
            # episodes, not a single episode - one lucky episode is not a
            # better policy.
            window = history[-int(config.best_window):]
            window_mean = sum(r["total_reward"] for r in window) / len(window)
            if len(history) >= int(config.best_window) and (
                    best_average is None or window_mean > best_average):
                best_average = window_mean
                best_episode = episode
                agent.save(os.path.join(model_dir, "best.pt"),
                           extra={"selection": "best mean return over last "
                                                "{} episodes".format(config.best_window),
                                  "window_mean_return": round(window_mean, 4),
                                  "episode": episode})

            if config.checkpoint_every and episode % int(config.checkpoint_every) == 0:
                agent.save(os.path.join(
                    model_dir, "checkpoint_ep{:03d}.pt".format(episode)))

            if not quiet and (episode % max(int(progress_every), 1) == 0
                              or episode == 1):
                _print_episode(row, config.episodes)

            _write_history(history, history_path(scenario, seed, results_dir))

        # --- final artefacts ------------------------------------------
        final_path = agent.save(os.path.join(model_dir, "final.pt"),
                                extra={"selection": "last episode"})
        csv_path = _write_history(history, history_path(scenario, seed, results_dir))

        elapsed = time.time() - started
        returns = [r["total_reward"] for r in history]
        losses = [r["loss"] for r in history if r["loss"] is not None]
        summary = {
            "scenario": scenario,
            "seed": seed,
            "episodes": len(history),
            "steps_per_episode": steps_per_episode,
            "total_steps": global_step,
            "gradient_steps": agent.learn_steps,
            "target_updates": agent.target_updates,
            "architecture": agent.architecture(),
            "parameters": agent.policy_net.parameter_count,
            "final_reward": returns[-1] if returns else None,
            "best_reward": max(returns) if returns else None,
            "mean_reward_first_5": (sum(returns[:5]) / min(5, len(returns))
                                    if returns else None),
            "mean_reward_last_5": (sum(returns[-5:]) / min(5, len(returns))
                                   if returns else None),
            "best_window_mean": best_average,
            "best_window_episode": best_episode,
            "final_loss": losses[-1] if losses else None,
            "final_epsilon": history[-1]["epsilon"] if history else None,
            "model_dir": model_dir,
            "best_model": os.path.join(model_dir, "best.pt"),
            "final_model": final_path,
            "history_csv": csv_path,
            "wall_clock_seconds": round(elapsed, 1),
            "config": config.as_dict(),
        }

        metadata = {
            "scenario": scenario,
            "seed": seed,
            "episodes": len(history),
            "episode_duration": float(env.episode_duration),
            "decision_interval": float(env.decision_interval),
            "steps_per_episode": steps_per_episode,
            "gamma": config.gamma,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "architecture": agent.architecture(),
            "parameters": agent.policy_net.parameter_count,
            "observation_size": agent.observation_size,
            "num_actions": agent.num_actions,
            "epsilon": {
                "start": config.epsilon_start,
                "end": config.epsilon_end,
                "schedule": schedule.describe(),
            },
            "replay_capacity": config.replay_capacity,
            "target_update_frequency": config.target_update_frequency,
            "loss": config.loss,
            "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "final_reward": summary["final_reward"],
            "best_reward": summary["best_reward"],
            "mean_reward_last_5": summary["mean_reward_last_5"],
            "best_model_episode": best_episode,
            "history_csv": os.path.relpath(csv_path, PROJECT_ROOT),
            "wall_clock_seconds": summary["wall_clock_seconds"],
            "config": config.as_dict(),
        }
        write_metadata(model_dir, metadata)

        if save_plots:
            try:
                from training.plot_training import plot_training_history
                summary["plots"] = plot_training_history(csv_path)
            except Exception as exc:                # pragma: no cover
                summary["plots"] = []
                if not quiet:
                    print("  (training curves were not generated: {})".format(exc))

        if not quiet:
            _print_summary(summary)
        return summary
    finally:
        env.close()


# ===========================================================================
def _run_episode(env, agent, schedule, config, episode: int, episode_seed: int,
                 global_step: int, max_steps: int):
    """One episode: collect experience, learn from it, measure the traffic."""
    hook = _LateCollector()
    env.step_callback = hook

    observation, info = env.reset(seed=int(episode_seed))
    collector = hook.arm(MetricsCollector(step_length=env.step_length,
                                          conn=env.connection))

    total_reward = 0.0
    losses: List[float] = []
    forced = 0
    blocked = 0
    steps = 0

    while steps < max_steps:
        epsilon = schedule.value(global_step)
        agent.set_epsilon(epsilon)

        action = agent.act(observation, training=True)
        next_observation, reward, terminated, truncated, info = env.step(action)

        # `terminated` is a real end of the traffic (no vehicle left and none
        # expected) and has no future to bootstrap from. `truncated` is only
        # the 1800 s time limit, so that transition keeps its future term -
        # treating a time limit as terminal would teach the agent that the
        # world ends at 1800 s.
        agent.remember(observation, action, reward, next_observation, terminated)

        observation = next_observation
        total_reward += float(reward)
        forced += int(bool(info.get("forced_switch")))
        blocked += int(bool(info.get("switch_blocked")))
        steps += 1
        global_step += 1

        if global_step % int(config.train_frequency) == 0:
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)

        if terminated or truncated:
            break

    env.step_callback = None
    traffic = collector.summary(env.scenario)

    row = {
        "episode": episode,
        "total_reward": round(total_reward, 4),
        "average_reward": round(total_reward / steps, 5) if steps else None,
        "episode_length": steps,
        "epsilon": round(float(agent.epsilon), 5),
        "loss": round(sum(losses) / len(losses), 6) if losses else None,
        "average_queue_length": traffic["average_queue_length"],
        "average_waiting_time": traffic["average_waiting_time"],
        "throughput": traffic["throughput"],
        "max_queue_length": traffic["max_queue_length"],
        "vehicles": traffic["vehicles"],
        "vehicles_still_in_network": traffic["vehicles_still_in_network"],
        "teleports": traffic["teleports"],
        "switch_count": int(info.get("switch_count", 0)),
        "forced_switches": forced,
        "blocked_switches": blocked,
        "replay_size": len(agent.memory),
        "gradient_steps": agent.learn_steps,
        "target_updates": agent.target_updates,
        "sumo_seed": int(episode_seed),
    }
    return row, global_step


# ===========================================================================
def _write_history(history: List[Dict[str, Any]], path: str) -> str:
    """Write the training history CSV. Called after every episode, so an
    interrupted run still leaves a usable record."""
    import pandas as pd

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    frame = pd.DataFrame(history)
    for column in HISTORY_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    frame[HISTORY_COLUMNS].to_csv(path, index=False)
    return path


# ===========================================================================
# console output
# ===========================================================================
def _print_header(scenario, config, agent, env, steps_per_episode, schedule,
                  model_dir) -> None:
    print("=" * 74)
    print("  IntelliSignal - Stage 3  |  DQN training")
    print("=" * 74)
    print("  Scenario            : {}".format(scenario))
    print("  Episode             : {:.0f} simulated seconds = {} decisions"
          .format(env.episode_duration, steps_per_episode))
    print("  Observation         : {} values (from env.observation_space)"
          .format(agent.observation_size))
    print("  Actions             : {} (0 = keep green, 1 = switch)"
          .format(agent.num_actions))
    print("  Network             : {}  ({} parameters)"
          .format(agent.architecture(), agent.policy_net.parameter_count))
    print("  Exploration         : {}".format(schedule.describe()))
    print("  Models              : {}".format(os.path.relpath(model_dir, PROJECT_ROOT)))
    print("-" * 74)
    print(config.describe())
    print("-" * 74)
    print("  ep      return    avg loss   eps     queue   wait(s)  thru  switch")


def _print_episode(row, total_episodes) -> None:
    print("  {ep:>3}/{tot:<3} {ret:>9.2f}  {loss:>9}  {eps:>5.3f}  "
          "{queue:>6.2f}  {wait:>7.2f}  {thru:>4}  {sw:>4}".format(
              ep=row["episode"], tot=total_episodes,
              ret=row["total_reward"],
              loss=("{:.5f}".format(row["loss"]) if row["loss"] is not None
                    else "   -"),
              eps=row["epsilon"], queue=row["average_queue_length"],
              wait=row["average_waiting_time"], thru=row["throughput"],
              sw=row["switch_count"]))


def _print_summary(summary) -> None:
    print("-" * 74)
    print("  TRAINING FINISHED - {} episodes in {:.0f} s of real time".format(
        summary["episodes"], summary["wall_clock_seconds"]))
    print("-" * 74)
    print("  Architecture            : {}".format(summary["architecture"]))
    print("  Gradient steps          : {}".format(summary["gradient_steps"]))
    print("  Target network updates  : {}".format(summary["target_updates"]))
    print("  Return, first 5 episodes: {:.2f}".format(
        summary["mean_reward_first_5"] or 0.0))
    print("  Return, last 5 episodes : {:.2f}".format(
        summary["mean_reward_last_5"] or 0.0))
    print("  Best single episode     : {:.2f}".format(summary["best_reward"] or 0.0))
    print("  Final episode           : {:.2f}".format(summary["final_reward"] or 0.0))
    print("  Final mean loss         : {}".format(
        "{:.6f}".format(summary["final_loss"]) if summary["final_loss"] is not None
        else "n/a"))
    print("  Final epsilon           : {}".format(summary["final_epsilon"]))
    print("-" * 74)
    print("  Model (best)  : {}".format(
        os.path.relpath(summary["best_model"], PROJECT_ROOT)))
    print("  Model (final) : {}".format(
        os.path.relpath(summary["final_model"], PROJECT_ROOT)))
    print("  History       : {}".format(
        os.path.relpath(summary["history_csv"], PROJECT_ROOT)))
    for plot in summary.get("plots", []):
        print("  Plot          : {}".format(os.path.relpath(plot, PROJECT_ROOT)))
    print("=" * 74)
    print("  A finished training run is NOT a result. Evaluate it against the")
    print("  fixed-time baseline before claiming anything:")
    print("    python evaluation/compare_controllers.py --scenario {} "
          "--seeds 42 43 44".format(summary["scenario"]))
    print("=" * 74)


# ===========================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IntelliSignal Stage 3 - train a DQN traffic-signal "
                    "controller against the existing SUMO environment.")
    parser.add_argument("--scenario", "-s", default="moderate",
                        help="Scenario to train on (moderate or heavy).")
    parser.add_argument("--episodes", "-e", type=int, default=None,
                        help="Number of training episodes "
                             "(default: config/dqn/training.yaml).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for Python, NumPy, PyTorch and SUMO.")
    parser.add_argument("--episode-duration", type=float, default=None,
                        dest="episode_duration",
                        help="Simulated seconds per episode "
                             "(default: the scenario's 1800 s).")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Discount factor (default 0.9).")
    parser.add_argument("--learning-rate", type=float, default=None,
                        dest="learning_rate", help="Adam learning rate.")
    parser.add_argument("--batch-size", type=int, default=None,
                        dest="batch_size", help="Mini-batch size.")
    parser.add_argument("--replay-capacity", type=int, default=None,
                        dest="replay_capacity", help="Replay buffer capacity.")
    parser.add_argument("--target-update-frequency", type=int, default=None,
                        dest="target_update_frequency",
                        help="Gradient steps between target-network updates.")
    parser.add_argument("--epsilon-end", type=float, default=None,
                        dest="epsilon_end", help="Final exploration rate.")
    parser.add_argument("--hidden", type=int, nargs="+", default=None,
                        dest="hidden_layers",
                        help="Hidden layer sizes, e.g. --hidden 64 64.")
    parser.add_argument("--model-dir", default=None, dest="model_dir",
                        help="Where to save checkpoints "
                             "(default: models/dqn_<scenario>).")
    parser.add_argument("--results-dir", default=None, dest="results_dir",
                        help="Where to save the training history CSV "
                             "(default: results/training).")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating the training curves.")
    parser.add_argument("--quiet", action="store_true",
                        help="Print only the final summary.")
    parser.add_argument("--progress-every", type=int, default=1,
                        dest="progress_every",
                        help="Print every Nth episode (default: every one).")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = load_dqn_config(
        episodes=args.episodes,
        seed=args.seed,
        episode_duration=args.episode_duration,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        target_update_frequency=args.target_update_frequency,
        epsilon_end=args.epsilon_end,
        hidden_layers=args.hidden_layers,
    )
    train(scenario=args.scenario, config=config, model_dir=args.model_dir,
          results_dir=args.results_dir, quiet=args.quiet,
          progress_every=args.progress_every, save_plots=not args.no_plots)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
