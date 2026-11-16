"""
IntelliSignal - Stage 2
rl/run_environment.py

Manual demonstration of the reinforcement-learning environment.

**This script does not train anything.** There is no agent, no neural network
and no learning. It drives the environment with a simple, chosen sequence of
actions and prints what happens, so the environment can be seen working before
a DQN is introduced in a later stage.

Usage
-----
    python rl/run_environment.py
    python rl/run_environment.py --scenario heavy
    python rl/run_environment.py --scenario moderate --policy random --steps 20
    python rl/run_environment.py --scenario heavy --policy alternate --gui

Policies (none of them are learned - they are fixed rules for demonstration)
    random     : a random action each decision (the default)
    alternate  : keep, keep, switch, repeat
    keep       : always ask to keep; max_green forces the switches
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np  # noqa: E402

from rl.action import ACTION_KEEP, ACTION_SWITCH  # noqa: E402
from rl.environment import IntelliSignalEnv       # noqa: E402
from rl.state import APPROACH_ORDER               # noqa: E402

POLICIES = ("random", "alternate", "keep")


def choose_action(policy: str, step: int, rng) -> int:
    if policy == "random":
        return int(rng.integers(0, 2))
    if policy == "alternate":
        return ACTION_SWITCH if step % 3 == 2 else ACTION_KEEP
    return ACTION_KEEP


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the IntelliSignal RL environment manually. "
                    "No training and no agent - this only demonstrates that "
                    "reset(), step() and the reward work.")
    parser.add_argument("--scenario", "-s", default="moderate",
                        help="moderate or heavy (default: moderate)")
    parser.add_argument("--policy", "-p", default="random", choices=POLICIES,
                        help="which fixed rule chooses the actions")
    parser.add_argument("--steps", "-n", type=int, default=15,
                        help="number of environment steps (default: 15)")
    parser.add_argument("--duration", type=float, default=None,
                        help="episode length in simulated seconds "
                             "(default: enough for the requested steps)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed (default: 42)")
    parser.add_argument("--gui", action="store_true",
                        help="run the simulation in sumo-gui")
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    env = IntelliSignalEnv(
        scenario=args.scenario,
        episode_duration=args.duration,
        render_mode="human" if args.gui else None,
    )
    if args.duration is None:
        # Give the episode room for the requested number of decisions.
        needed = (args.steps + 1) * env.decision_interval
        env.close()
        env = IntelliSignalEnv(
            scenario=args.scenario,
            episode_duration=max(needed, env.decision_interval * 2),
            render_mode="human" if args.gui else None,
        )

    _print_header(env, args)

    observation, info = env.reset(seed=args.seed)
    print("  reset() -> observation of shape {}, inside observation_space: {}"
          .format(observation.shape, env.observation_space.contains(observation)))
    print("  starting phase: {} ({})".format(info["phase"], info["phase_name"]))
    print()
    _print_table_header()

    total_reward = 0.0
    terminated = truncated = False
    for step in range(args.steps):
        action = choose_action(args.policy, step, rng)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        _print_row(step + 1, action, reward, info)
        if terminated or truncated:
            print("\n  Episode finished: terminated={}, truncated={}"
                  .format(terminated, truncated))
            break

    print("-" * 104)
    print("  Steps run          : {}".format(min(args.steps, step + 1)))
    print("  Total reward       : {:.3f}".format(total_reward))
    print("  Mean reward / step : {:.3f}".format(total_reward / (step + 1)))
    print("  Phase switches     : {}".format(info["switch_count"]))
    print("  Simulated time     : {:.0f} s".format(info["simulation_time"]))
    print("  Vehicles in network: {}".format(info["vehicles_in_network"]))
    print()
    print("  Final observation:")
    print("    " + ", ".join("{}={:.3f}".format(label, value) for label, value
                             in zip(_labels(), observation)))

    env.close()
    print()
    print("  env.close() called - SUMO shut down.")
    print("  No agent was trained. Stage 2 provides the environment only.")
    return 0


def _labels():
    from rl.state import OBSERVATION_LABELS
    return OBSERVATION_LABELS


def _print_header(env, args) -> None:
    print("=" * 104)
    print("  IntelliSignal - Stage 2: RL environment demonstration (no training)")
    print("=" * 104)
    print("  Scenario          : {}".format(args.scenario))
    print("  Action policy     : {} (a fixed rule, not a learned agent)".format(args.policy))
    print("  Observation space : {}".format(env.observation_space))
    print("  Action space      : {}   0 = keep phase, 1 = switch phase"
          .format(env.action_space))
    print("  Decision interval : {:.0f} s per step".format(env.decision_interval))
    print("  Min / max green   : {:.0f} s / {:.0f} s".format(
        env.controller.min_green, env.controller.max_green))
    print("  Yellow / all-red  : {:.0f} s / {:.0f} s".format(
        env.controller.yellow_time, env.controller.all_red_time))
    print("  Episode length    : {:.0f} s  ({} decisions)".format(
        env.episode_duration, env.max_decisions))
    print("  Reward            : {}".format(env.reward_fn.describe()))
    print("-" * 104)


def _print_table_header() -> None:
    print("{:>4}  {:>6}  {:>8}  {:>9}  {:>5}  {:>9}  {:>6}  {:>9}  {}".format(
        "Step", "Action", "Applied", "Sim time", "Phase", "Reward",
        "Queue", "Waiting", "Queue per approach N/E/S/W"))
    print("-" * 104)


def _print_row(step: int, action: int, reward: float, info: dict) -> None:
    per_approach = "/".join(
        "{:.0f}".format(info["queue_per_approach"][a]) for a in APPROACH_ORDER)
    note = ""
    if info["switch_blocked"]:
        note = "  (switch refused: min_green not reached)"
    elif info["forced_switch"]:
        note = "  (switch forced: max_green reached)"
    print("{:>4}  {:>6}  {:>8}  {:>7.0f}s  {:>5}  {:>9.3f}  {:>6.0f}  {:>9.1f}  {}{}".format(
        step,
        info["requested_action_name"],
        info["effective_action_name"],
        info["simulation_time"],
        info["phase"],
        reward,
        info["total_queue"],
        info["total_waiting_time"],
        per_approach,
        note,
    ))


if __name__ == "__main__":
    raise SystemExit(main())
