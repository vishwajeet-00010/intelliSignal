"""
IntelliSignal
training/train_phase_dqn.py

Trains the duration-choosing Deep Q-Network.

    python training/train_phase_dqn.py --scenario heavy --episodes 200 \
        --seed 42 --out output/run

The loop
--------
    reset the environment -> first observation
    repeat until 1800 simulated seconds have passed:
        epsilon = schedule(decisions so far)
        action  = epsilon-greedy over Q(observation)
        run that green plus its clearance
        store (observation, action, integrated reward, observation',
               done, duration in seconds)
        one gradient step per decision once the buffer has filled
    every `validate_every` episodes: run the GREEDY policy on held-out
    seeds and keep the checkpoint with the lowest mean waiting time

What is measured, and by what
-----------------------------
The reported traffic numbers come from `utils.metrics.MetricsCollector` -
the class that produced the Stage 1 fixed-time baseline - attached to the
running simulation through the environment's per-second hook. A training
episode, a validation episode and a fixed-time baseline run are therefore
measured by one piece of code over the same 1800 steps, which is what makes
"the DQN reached X s" comparable to "the fixed plan reached Y s" at all.

Model selection
---------------
`best.pt` is the checkpoint with the lowest mean waiting time on the
VALIDATION seeds, which are disjoint from both the training seeds and the
confirmatory seeds used in the final report. Selecting on training episodes
would be selecting on the noise the agent was exploring in.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from collections import deque
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch                                                   # noqa: E402

from agents.phase_dqn_agent import (EpsilonSchedule,              # noqa: E402
                                    PhaseDQNAgent, n_step_transitions,
                                    set_seeds)
from rl.phase_action import DEFAULT_DURATIONS                     # noqa: E402
from rl.phase_environment import PhaseDurationEnv                 # noqa: E402
from rl.phase_state import PhaseStateExtractor                    # noqa: E402
from utils.metrics import MetricsCollector                        # noqa: E402

HISTORY_COLUMNS = [
    "episode", "total_reward", "decisions", "epsilon", "loss",
    "average_waiting_time", "average_queue_length", "throughput",
    "max_queue_length", "vehicles", "vehicles_still_in_network", "teleports",
    "mean_green", "mean_green_ns", "mean_green_ew", "distinct_greens",
    "replay_size", "gradient_steps", "sumo_seed", "wall_clock_seconds",
]


# ---------------------------------------------------------------------------
def run_episode(env, agent, epsilon: Optional[float],
                schedule=None, global_step: int = 0, n_step: int = 1,
                learn: bool = True, seed: Optional[int] = None,
                measure: bool = True):
    """One episode. Returns (summary, global_step).

    `measure` attaches the Stage 1 `MetricsCollector`. It is on for
    validation, because that is the number the project reports, and off for
    training episodes, because the collector reads every vehicle's
    accumulated waiting time on every simulated second - roughly 200,000
    TraCI round-trips per episode, which is an order of magnitude more than
    everything else the episode does put together. Training does not need
    it: the reward comes from the environment, and the queue integral the
    environment already keeps is the same mean queue the collector reports.
    """
    observation, _ = env.reset(seed=seed)
    collector = collector_factory(env) if measure else None
    env.step_callback = collector.collect_step if collector else None

    pending: deque = deque()
    losses: List[float] = []
    last_info: Dict[str, Any] = {}
    total_reward = 0.0
    greens: List[float] = []
    greens_by_phase: Dict[int, List[float]] = {0: [], 1: []}
    decisions = 0

    while True:
        phase = env.controller.phase
        if epsilon is not None:
            agent.set_epsilon(epsilon)
        elif schedule is not None:
            agent.set_epsilon(schedule.value(global_step))
        action = agent.act(observation, training=learn)

        next_observation, reward, terminated, truncated, info = env.step(action)
        duration = float(info["duration"])
        greens.append(float(info["green_served"]))
        greens_by_phase[phase].append(float(info["green_served"]))
        total_reward += float(reward)
        decisions += 1
        global_step += 1

        if learn:
            pending.append({
                "state": observation, "action": action, "reward": reward,
                "next_state": next_observation, "done": terminated,
                "duration": duration,
            })
            while len(pending) >= n_step:
                collapsed = n_step_transitions(list(pending), n_step,
                                               agent.gamma_per_second)
                agent.remember(**collapsed)
                pending.popleft()

            loss = agent.learn()
            if loss is not None:
                losses.append(loss)

        observation = next_observation
        if terminated or truncated:
            last_info = info
            break

    if learn:
        # Flush whatever is left: the tail of the episode is shorter than
        # `n_step`, and dropping it would silently discard the decisions
        # taken closest to the end of every episode.
        while pending:
            collapsed = n_step_transitions(list(pending), len(pending),
                                           agent.gamma_per_second)
            agent.remember(**collapsed)
            pending.popleft()

    if collector is not None:
        summary = collector.summary(env.scenario)
    else:
        summary = {
            "scenario": env.scenario,
            "average_waiting_time": None,
            "average_queue_length": last_info.get("episode_mean_queue"),
            "throughput": None,
            "max_queue_length": None,
            "vehicles": None,
            "vehicles_still_in_network": None,
            "teleports": None,
        }
    summary.update({
        "total_reward": total_reward,
        "decisions": decisions,
        "loss": statistics.fmean(losses) if losses else None,
        "mean_green": statistics.fmean(greens) if greens else 0.0,
        "mean_green_ns": (statistics.fmean(greens_by_phase[0])
                          if greens_by_phase[0] else 0.0),
        "mean_green_ew": (statistics.fmean(greens_by_phase[1])
                          if greens_by_phase[1] else 0.0),
        "distinct_greens": len(set(greens)),
        "greens": greens,
    })
    return summary, global_step


# ---------------------------------------------------------------------------
def collector_factory(env):
    return MetricsCollector(step_length=env.step_length, conn=env.connection)


# ---------------------------------------------------------------------------
def evaluate(agent, env, seeds: Sequence[int]) -> Dict[str, Any]:
    """Greedy rollouts on held-out seeds, measured with MetricsCollector."""
    waits, throughputs, queues, greens = [], [], [], []
    for seed in seeds:
        summary, _ = run_episode(env, agent, epsilon=0.0, learn=False,
                                 seed=int(seed), measure=True)
        waits.append(float(summary["average_waiting_time"]))
        throughputs.append(float(summary["throughput"]))
        queues.append(float(summary["average_queue_length"]))
        greens.append(float(summary["mean_green"]))
    return {
        "seeds": [int(s) for s in seeds],
        "mean_waiting_time": statistics.fmean(waits),
        "sd_waiting_time": statistics.pstdev(waits) if len(waits) > 1 else 0.0,
        "mean_throughput": statistics.fmean(throughputs),
        "mean_queue_length": statistics.fmean(queues),
        "mean_green": statistics.fmean(greens),
        "per_seed_waiting": waits,
    }


# ---------------------------------------------------------------------------
def pin_threads(threads: int = 1) -> None:
    """Keep one training process on one core's worth of linear algebra.

    The networks here are tiny - 10 inputs, two hidden layers - so torch's
    intra-op parallelism buys nothing, and by default it opens a thread per
    core. Eight training runs on a sixteen-core machine then ask for 128
    threads, and the runs spend their time context-switching: measured at
    51 s per episode against 6 s for a single run. Pinning each process to
    one thread is what makes running the variants in parallel worthwhile.
    """
    threads = max(int(threads), 1)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        # Already set, or parallel work has already started; not fatal.
        pass


def train(scenario: str = "heavy", episodes: int = 200, seed: int = 42,
          durations: Sequence[float] = DEFAULT_DURATIONS,
          out_dir: str = "output/phase_dqn",
          gamma_per_second: float = 0.99,
          learning_rate: float = 5e-4,
          batch_size: int = 128,
          hidden_layers: Sequence[int] = (128, 128),
          n_step: int = 2,
          epsilon_end: float = 0.05,
          exploration_fraction: float = 0.5,
          learning_starts: int = 1000,
          target_update_frequency: int = 400,
          replay_capacity: int = 100000,
          double_dqn: bool = True,
          waiting_weight: float = 1.0,
          queue_weight: float = 1.0,
          reward_scale: float = 0.01,
          validation_seeds: Sequence[int] = (2001, 2002, 2003, 2004, 2005),
          validate_every: int = 5,
          measure_training: bool = False,
          obs_scales: Optional[Dict[str, float]] = None,
          threads: int = 1,
          tag: str = "",
          quiet: bool = False) -> Dict[str, Any]:
    pin_threads(threads)
    set_seeds(seed)
    os.makedirs(out_dir, exist_ok=True)
    model_dir = os.path.join(out_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    extractor = PhaseStateExtractor(
        max_green_seconds=float(max(durations)),
        **{key: float(value) for key, value in (obs_scales or {}).items()})
    env = PhaseDurationEnv(
        scenario=scenario, durations=durations, seed=seed,
        waiting_weight=waiting_weight, queue_weight=queue_weight,
        reward_scale=reward_scale, state_extractor=extractor,
        gamma_per_second=gamma_per_second,
        tag=tag or "_s{}".format(seed))

    try:
        agent = PhaseDQNAgent(
            observation_size=int(env.observation_space.shape[0]),
            num_actions=int(env.action_space.n),
            hidden_layers=hidden_layers,
            learning_rate=learning_rate,
            gamma_per_second=gamma_per_second,
            batch_size=batch_size,
            replay_capacity=replay_capacity,
            learning_starts=learning_starts,
            target_update_frequency=target_update_frequency,
            double_dqn=double_dqn,
            seed=seed,
            scenario=scenario,
            durations=list(env.controller.durations),
            observation_contract=extractor.contract(),
        )

        decisions_per_episode = max(
            int(env.episode_duration //
                (statistics.fmean(durations) + env.controller.yellow_time
                 + env.controller.all_red_time)), 1)
        schedule = EpsilonSchedule(
            start=1.0, end=epsilon_end,
            decay_steps=max(int(episodes * decisions_per_episode
                                * exploration_fraction), 1))

        if not quiet:
            print("=" * 78)
            print("Phase-duration DQN - {} - seed {}".format(scenario, seed))
            print("=" * 78)
            print("  {}".format(env.controller.describe()))
            print("  observation         : {} values, phase-relative"
                  .format(env.observation_space.shape[0]))
            print("  network             : {}".format(agent.architecture()))
            print("  gamma               : {}/second  ({:.3f} over a {:.0f} s "
                  "decision)".format(gamma_per_second,
                                     gamma_per_second ** 11, 11))
            print("  double DQN          : {}".format(double_dqn))
            print("  n-step return       : {}".format(n_step))
            print("  reward              : -{:g} x integral of ({:g} x waiting"
                  " + {:g} x queue)".format(reward_scale, waiting_weight,
                                            queue_weight))
            print("  epsilon             : {}".format(schedule.describe()))
            print("  ~decisions/episode  : {}".format(decisions_per_episode))
            print("  validation seeds    : {}".format(list(validation_seeds)))
            print("=" * 78, flush=True)

        history: List[Dict[str, Any]] = []
        validation_history: List[Dict[str, Any]] = []
        best_wait = None
        best_episode = 0
        global_step = 0
        started = time.time()

        for episode in range(1, int(episodes) + 1):
            episode_seed = seed + episode - 1
            summary, global_step = run_episode(
                env, agent, epsilon=None, schedule=schedule,
                global_step=global_step, n_step=n_step, learn=True,
                seed=episode_seed, measure=measure_training)

            row = {column: summary.get(column) for column in HISTORY_COLUMNS}
            row.update({
                "episode": episode,
                "epsilon": round(agent.epsilon, 4),
                "replay_size": len(agent.memory),
                "gradient_steps": agent.learn_steps,
                "sumo_seed": episode_seed,
                "wall_clock_seconds": round(time.time() - started, 1),
            })
            if row["loss"] is not None and not math.isfinite(row["loss"]):
                raise RuntimeError(
                    "Training stopped: non-finite loss at episode {}."
                    .format(episode))
            history.append(row)
            agent.episode = episode

            if episode % int(validate_every) == 0 or episode == episodes:
                validation = evaluate(agent, env, validation_seeds)
                validation["episode"] = episode
                validation_history.append(validation)
                if best_wait is None or validation["mean_waiting_time"] < best_wait:
                    best_wait = validation["mean_waiting_time"]
                    best_episode = episode
                    agent.save(os.path.join(model_dir, "best.pt"), extra={
                        "selection": "lowest mean waiting on validation seeds",
                        "validation": validation,
                        "episode": episode,
                    })
                _write_json(validation_history,
                            os.path.join(out_dir, "validation_history.json"))
                if not quiet:
                    print("  ep {:>4d}  eps {:.3f}  loss {:>8}  queue "
                          "{:>7.3f}  VALID {:>7.3f}  (best {:>7.3f} @ {})  "
                          "green ns/ew {:.1f}/{:.1f}".format(
                              episode, agent.epsilon,
                              "{:.4f}".format(row["loss"]) if row["loss"] else "-",
                              float(row["average_queue_length"] or 0.0),
                              validation["mean_waiting_time"], best_wait,
                              best_episode, row["mean_green_ns"],
                              row["mean_green_ew"]), flush=True)
            elif not quiet:
                print("  ep {:>4d}  eps {:.3f}  loss {:>8}  queue "
                      "{:>7.3f}  green ns/ew {:.1f}/{:.1f}  distinct {}".format(
                          episode, agent.epsilon,
                          "{:.4f}".format(row["loss"]) if row["loss"] else "-",
                          float(row["average_queue_length"] or 0.0),
                          row["mean_green_ns"], row["mean_green_ew"],
                          row["distinct_greens"]), flush=True)

            _write_history(history, os.path.join(out_dir, "training.csv"))

        agent.save(os.path.join(model_dir, "final.pt"),
                   extra={"selection": "last episode"})

        result = {
            "scenario": scenario,
            "seed": seed,
            "episodes": len(history),
            "durations": list(env.controller.durations),
            "best_validation_waiting": best_wait,
            "best_validation_episode": best_episode,
            "best_model": os.path.join(model_dir, "best.pt"),
            "final_model": os.path.join(model_dir, "final.pt"),
            "gamma_per_second": gamma_per_second,
            "n_step": n_step,
            "double_dqn": double_dqn,
            "reward": {"scale": reward_scale, "waiting": waiting_weight,
                       "queue": queue_weight},
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "hidden_layers": list(hidden_layers),
            "observation_contract": extractor.contract(),
            "wall_clock_seconds": round(time.time() - started, 1),
        }
        _write_json(result, os.path.join(out_dir, "summary.json"))
        if not quiet:
            print("\nbest validation waiting {:.3f} s at episode {}".format(
                best_wait if best_wait is not None else float("nan"),
                best_episode))
        return result
    finally:
        env.close()
        env.cleanup_generated_inputs()


# ---------------------------------------------------------------------------
def _write_history(history, path: str) -> str:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)
    return path


def _write_json(payload, path: str) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=float)
    return path


# ---------------------------------------------------------------------------
def parse_numbers(text: str) -> List[float]:
    values: List[float] = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk[1:]:
            start, end = chunk.rsplit("-", 1)
            values.extend(float(v) for v in range(int(float(start)),
                                                  int(float(end)) + 1))
        else:
            values.append(float(chunk))
    return values


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="heavy")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--durations", default=None,
                        help="comma list, e.g. 4,5,6,7,8,9,10,12,15,20")
    parser.add_argument("--out", default="output/phase_dqn")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="per-SECOND discount")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", default="128,128")
    parser.add_argument("--n-step", type=int, default=2)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--exploration-fraction", type=float, default=0.5)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--target-update", type=int, default=400)
    parser.add_argument("--no-double", action="store_true")
    parser.add_argument("--waiting-weight", type=float, default=1.0)
    parser.add_argument("--queue-weight", type=float, default=1.0)
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument("--validation-seeds", default="2001-2005")
    parser.add_argument("--validate-every", type=int, default=5)
    parser.add_argument("--measure-training", action="store_true",
                        help="attach MetricsCollector to training episodes "
                             "too; about ten times slower, and only the "
                             "validation number is ever reported")
    parser.add_argument("--obs-queue", type=float, default=None,
                        help="max_queue_per_direction for the observation")
    parser.add_argument("--obs-waiting", type=float, default=None,
                        help="max_waiting_per_direction")
    parser.add_argument("--obs-vehicles", type=float, default=None,
                        help="max_vehicles_per_direction")
    parser.add_argument("--obs-approach-queue", type=float, default=None,
                        help="max_queue_per_approach")
    parser.add_argument("--threads", type=int, default=1,
                        help="torch intra-op threads; 1 is right when "
                             "several runs share a machine")
    parser.add_argument("--tag", default="")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    durations = (parse_numbers(args.durations) if args.durations
                 else list(DEFAULT_DURATIONS))
    obs_scales = {name: value for name, value in (
        ("max_queue_per_direction", args.obs_queue),
        ("max_waiting_per_direction", args.obs_waiting),
        ("max_vehicles_per_direction", args.obs_vehicles),
        ("max_queue_per_approach", args.obs_approach_queue),
    ) if value is not None}
    train(scenario=args.scenario, episodes=args.episodes, seed=args.seed,
          durations=durations, out_dir=args.out,
          gamma_per_second=args.gamma, learning_rate=args.lr,
          batch_size=args.batch_size,
          hidden_layers=[int(h) for h in args.hidden.split(",")],
          n_step=args.n_step, epsilon_end=args.epsilon_end,
          exploration_fraction=args.exploration_fraction,
          learning_starts=args.learning_starts,
          target_update_frequency=args.target_update,
          double_dqn=not args.no_double,
          waiting_weight=args.waiting_weight, queue_weight=args.queue_weight,
          reward_scale=args.reward_scale,
          validation_seeds=[int(s) for s in parse_numbers(args.validation_seeds)],
          validate_every=args.validate_every,
          measure_training=args.measure_training, obs_scales=obs_scales,
          threads=args.threads, tag=args.tag, quiet=args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
