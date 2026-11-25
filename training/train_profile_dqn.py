"""
IntelliSignal
training/train_profile_dqn.py

Trains a duration-choosing DQN under a timing profile, on varied traffic.

    python training/train_profile_dqn.py --profile standard --episodes 120 \
        --seed 42 --out output/run --pilot

Three things this does that `training/train_phase_dqn.py` does not:

1. **It obeys a timing profile.** The menu, the minimum and maximum green
   and the red limit all come from `rl/timing_profile.py`, and they are
   enforced on the same code path during exploration, validation and
   anything the dashboard runs later. A model trained here cannot emit a
   green its profile forbids.

2. **It trains on varied traffic.** Episodes cycle through the eighteen
   workloads of `training/workload_library.py` - balanced, directional,
   lane-skewed, bursty, ramping and turn-heavy, at light, moderate and
   heavy intensity. The previous models saw two steady balanced patterns
   and could not generalise beyond them.

3. **It selects on delay per REQUESTED vehicle**, from SUMO's own trip
   information, not on mean waiting over vehicles that got in. A
   controller that blocks a saturated entry looks good on the second
   measure and bad on the first, and the first is the honest one.

Training, validation and test workloads are disjoint, and so are their
SUMO seeds.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.phase_dqn_agent import (EpsilonSchedule, PhaseDQNAgent,
                                    n_step_transitions, set_seeds)
from evaluation.harness import (aggregate, greedy_agent_policy,
                                run_extend_policy, run_policy)
from rl.phase_environment import PhaseDurationEnv
from rl.phase_extend_env import PhaseExtendEnv
from rl.phase_state import SATURATION_SOFT, PhaseStateExtractor
from rl.policy_rollout import ACTION_SPACE_EXTEND, ACTION_SPACE_MENU
from rl.timing_profile import get_profile
from training.workload_library import split, sumo_seeds

HISTORY_COLUMNS = ["episode", "workload", "total_reward", "decisions",
                   "epsilon", "loss", "mean_green", "interventions",
                   "compliant", "sumo_seed", "wall_clock_s"]


# ---------------------------------------------------------------------------
def build_extractor(profile) -> PhaseStateExtractor:
    """Scales wide enough for the declared training range, softly saturated.

    The scales are set from the heaviest workload the agent will be shown
    rather than from one scenario, and the soft mode means that even
    beyond them the readings stay ordered instead of pinning at 1.0.
    """
    return PhaseStateExtractor(
        max_queue_per_direction=25.0,
        max_waiting_per_direction=250.0,
        max_vehicles_per_direction=40.0,
        max_queue_per_approach=15.0,
        max_green_seconds=float(profile.max_green),
        saturation=SATURATION_SOFT,
    )


# ---------------------------------------------------------------------------
def train_episode(env, agent, schedule, global_step: int, seed: int,
                  n_step: int = 2, masked: bool = False) -> Dict[str, Any]:
    """One episode on whatever workload the environment currently holds.

    `masked` is for the lock-and-extend action space, where EXTEND
    becomes illegal at the maximum green. The mask is stored with the
    transition so the learning target honours it too.
    """
    observation, _ = env.reset(seed=int(seed))
    env.step_callback = None
    pending: List[Dict[str, Any]] = []
    losses: List[float] = []
    greens: List[float] = []
    total_reward = 0.0
    interventions = 0
    last_info: Dict[str, Any] = {}

    while True:
        agent.set_epsilon(schedule.value(global_step))
        valid = env.valid_actions() if masked else None
        action = agent.act(observation, training=True, valid_actions=valid)
        next_observation, reward, terminated, truncated, info = env.step(action)
        global_step += 1
        total_reward += float(reward)
        if masked:
            if info.get("phase_ended"):
                greens.append(float(info["green_served"]))
        else:
            greens.append(float(info["green_served"]))
        interventions += 1 if info.get("intervention") else 0

        pending.append({"state": observation, "action": action,
                        "reward": reward, "next_state": next_observation,
                        "done": terminated, "duration": float(info["duration"]),
                        "next_valid_actions": (list(info["valid_actions"])
                                               if masked else None)})
        while len(pending) >= n_step:
            collapsed = n_step_transitions(pending, n_step,
                                           agent.gamma_per_second)
            collapsed["next_valid_actions"] = pending[n_step - 1].get(
                "next_valid_actions")
            agent.remember(**collapsed)
            pending.pop(0)
        loss = agent.learn()
        if loss is not None:
            losses.append(loss)

        observation = next_observation
        if terminated or truncated:
            last_info = info
            break

    while pending:
        collapsed = n_step_transitions(pending, len(pending),
                                       agent.gamma_per_second)
        collapsed["next_valid_actions"] = pending[-1].get("next_valid_actions")
        agent.remember(**collapsed)
        pending.pop(0)

    compliance = last_info.get("timing_compliance", {})
    return {
        "global_step": global_step,
        "total_reward": total_reward,
        "decisions": len(greens),
        "loss": statistics.fmean(losses) if losses else None,
        "mean_green": statistics.fmean(greens) if greens else 0.0,
        "interventions": interventions,
        "compliant": bool(compliance.get("compliant", True)),
    }


# ---------------------------------------------------------------------------
def validate(agent, profile, workloads, seeds, extractor,
             candidate: str = "A", extension: float = 3.0) -> Dict[str, Any]:
    """Greedy rollouts on the validation split, measured by the harness."""
    runs = []
    for index, workload in enumerate(workloads):
        seed = seeds[index % len(seeds)]
        if candidate == "B":
            runs.append(run_extend_policy(
                workload, agent, profile, seed, extension=extension,
                controller="candidate B", extractor=extractor, tag="_valx"))
        else:
            runs.append(run_policy(
                workload, greedy_agent_policy(agent), profile, seed,
                controller="candidate A", extractor=extractor, tag="_val"))
    summary = aggregate(runs)
    summary["per_workload"] = [
        {"workload": run.detail.get("name", workload.name),
         "id": workload.workload_id[:8],
         "total_delay": run.mean_total_delay,
         "waiting": run.mean_waiting,
         "completed": run.completed,
         "pending": run.waiting_to_enter}
        for run, workload in zip(runs, workloads)]
    return summary


# ---------------------------------------------------------------------------
def train(candidate: str = "A", extension: float = 3.0,
          profile_name: str = "standard", episodes: int = 120,
          seed: int = 42, out_dir: str = "output/profile_dqn",
          duration_s: float = 900.0, gamma_per_second: float = 0.99,
          learning_rate: float = 5e-4, batch_size: int = 128,
          hidden_layers: Sequence[int] = (128, 128), n_step: int = 2,
          epsilon_end: float = 0.05, exploration_fraction: float = 0.6,
          learning_starts: int = 600, target_update: int = 400,
          reward_scale: float = 0.005, validate_every: int = 10,
          threads: int = 1, pilot: bool = False,
          quiet: bool = False) -> Dict[str, Any]:
    torch.set_num_threads(max(int(threads), 1))
    set_seeds(seed)
    profile = get_profile(profile_name)

    os.makedirs(out_dir, exist_ok=True)
    model_dir = os.path.join(out_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    train_loads = split("train", duration_s=duration_s)
    val_loads = split("validation", duration_s=duration_s)
    train_seeds = sumo_seeds("train", 32)
    val_seeds = sumo_seeds("validation", len(val_loads))

    extractor = build_extractor(profile)
    masked = candidate == "B"
    if masked:
        env = PhaseExtendEnv(
            workload=train_loads[0], profile=profile.name,
            extension=extension, state_extractor=extractor,
            gamma_per_second=gamma_per_second, reward_scale=reward_scale,
            tag="_tx{}".format(seed))
    else:
        env = PhaseDurationEnv(
            workload=train_loads[0], profile=profile.name,
            state_extractor=extractor, gamma_per_second=gamma_per_second,
            reward_scale=reward_scale, tag="_tp{}".format(seed))

    agent = PhaseDQNAgent(
        observation_size=int(env.observation_space.shape[0]),
        num_actions=int(env.action_space.n), hidden_layers=hidden_layers,
        learning_rate=learning_rate, gamma_per_second=gamma_per_second,
        batch_size=batch_size, learning_starts=learning_starts,
        target_update_frequency=target_update, double_dqn=True, seed=seed,
        scenario="profile:{}|candidate:{}".format(profile.name, candidate),
        durations=list(env.controller.durations),
        observation_contract=extractor.contract(),
        action_space=(ACTION_SPACE_EXTEND if masked else ACTION_SPACE_MENU),
        extension=(float(extension) if masked else 0.0))

    # How many decisions an episode holds depends on the action space -
    # a lock-and-extend policy is asked something several times per green
    # and a duration menu once - so it is measured on the first episode
    # rather than estimated from the green length. Estimating it made
    # candidate B stop exploring at episode 116 of 320 while candidate A
    # explored until 241, and then the two were compared as equals.
    clearance = float(env.controller.yellow_time
                      + env.controller.all_red_time)
    per_episode = max(int(duration_s / (profile.min_green + clearance)), 1)
    schedule = EpsilonSchedule(
        start=1.0, end=epsilon_end,
        decay_steps=max(int(episodes * per_episode * exploration_fraction), 1))

    if not quiet:
        print("=" * 76)
        print("Profile DQN  |  candidate {}  |  profile {}  |  seed {}".format(
            candidate, profile.name, seed))
        print("=" * 76)
        print("  {}".format(profile.describe()))
        print("  red limit     : {}".format(
            profile.red_limit_analysis(env.controller.yellow_time)["explanation"]))
        print("  observation   : {} values, soft saturation (version {})".format(
            env.observation_space.shape[0], extractor.observation_version))
        print("  training      : {} workloads x {} episodes".format(
            len(train_loads), episodes))
        print("  validation    : {} workloads, seeds {}".format(
            len(val_loads), val_seeds))
        print("  selection     : lowest mean total delay per REQUESTED vehicle")
        print("=" * 76, flush=True)

    history: List[Dict[str, Any]] = []
    validation_history: List[Dict[str, Any]] = []
    best_delay: Optional[float] = None
    best_episode = 0
    global_step = 0
    started = time.time()
    budget = int(3 if pilot else episodes)

    try:
        for episode in range(1, budget + 1):
            workload = train_loads[(episode - 1) % len(train_loads)]
            if (workload.duration_s != env.episode_duration
                    or workload.step_length != env.step_length):
                raise RuntimeError(
                    "workload {} is {:.0f} s at {:.2f} s steps but the "
                    "environment was built for {:.0f} s at {:.2f} s; the "
                    "horizon would be wrong.".format(
                        workload.name, workload.duration_s,
                        workload.step_length, env.episode_duration,
                        env.step_length))
            env.workload = workload
            result = train_episode(
                env, agent, schedule, global_step,
                seed=train_seeds[(episode - 1) % len(train_seeds)],
                n_step=n_step, masked=masked)
            if episode == 1:
                # Now the real number is known, re-size the schedule so
                # both action spaces explore for the same share of the
                # training budget.
                measured = max(result["global_step"], 1)
                schedule = EpsilonSchedule(
                    start=1.0, end=epsilon_end,
                    decay_steps=max(int(episodes * measured
                                        * exploration_fraction), 1))
                if not quiet:
                    print("  exploration  : {} decisions in episode 1, so "
                          "epsilon reaches {:.2f} at episode {}".format(
                              measured, epsilon_end,
                              int(episodes * exploration_fraction)),
                          flush=True)
            global_step = result["global_step"]

            row = {
                "episode": episode, "workload": workload.name,
                "total_reward": round(result["total_reward"], 3),
                "decisions": result["decisions"],
                "epsilon": round(agent.epsilon, 4),
                "loss": result["loss"], "mean_green": round(result["mean_green"], 2),
                "interventions": result["interventions"],
                "compliant": result["compliant"],
                "sumo_seed": train_seeds[(episode - 1) % len(train_seeds)],
                "wall_clock_s": round(time.time() - started, 1),
            }
            if row["loss"] is not None and not math.isfinite(row["loss"]):
                raise RuntimeError("non-finite loss at episode {}".format(episode))
            history.append(row)
            agent.episode = episode

            due = (episode % validate_every == 0 or episode == budget)
            if due and not pilot:
                check = validate(agent, profile, val_loads, val_seeds,
                                 extractor, candidate=candidate,
                                 extension=extension)
                check["episode"] = episode
                validation_history.append(check)
                if best_delay is None or check["mean_total_delay"] < best_delay:
                    best_delay = check["mean_total_delay"]
                    best_episode = episode
                    agent.save(os.path.join(model_dir, "best.pt"), extra={
                        "selection": "lowest mean total delay per requested "
                                     "vehicle on the validation split",
                        "profile": profile.to_dict(),
                        "candidate": candidate, "extension": extension,
                        "validation": check, "episode": episode})
                _write_json(validation_history,
                            os.path.join(out_dir, "validation_history.json"))
                if not quiet:
                    print("  ep {:>4d} {:<22} eps {:.3f}  loss {:>8}  green "
                          "{:>5.1f}  VALID delay {:>7.2f}  wait {:>6.2f}  "
                          "(best {:>7.2f} @ {})".format(
                              episode, workload.name, agent.epsilon,
                              "{:.4f}".format(row["loss"]) if row["loss"] else "-",
                              row["mean_green"], check["mean_total_delay"],
                              check["mean_waiting"], best_delay, best_episode),
                          flush=True)
            elif not quiet:
                print("  ep {:>4d} {:<22} eps {:.3f}  loss {:>8}  green "
                      "{:>5.1f}  reward {:>9.1f}  compliant {}".format(
                          episode, workload.name, agent.epsilon,
                          "{:.4f}".format(row["loss"]) if row["loss"] else "-",
                          row["mean_green"], row["total_reward"],
                          row["compliant"]), flush=True)

            _write_history(history, os.path.join(out_dir, "training.csv"))

        agent.save(os.path.join(model_dir, "final.pt"),
                   extra={"selection": "last episode",
                          "profile": profile.to_dict(),
                          "candidate": candidate, "extension": extension})
    finally:
        env.close()
        env.cleanup_generated_inputs()

    elapsed = time.time() - started
    result = {
        "profile": profile.to_dict(),
        "candidate": candidate, "extension": extension,
        "seed": seed, "episodes": len(history),
        "pilot": bool(pilot),
        "seconds_per_episode": round(elapsed / max(len(history), 1), 2),
        "best_validation_total_delay": best_delay,
        "best_validation_episode": best_episode,
        "best_model": os.path.join(model_dir, "best.pt"),
        "final_model": os.path.join(model_dir, "final.pt"),
        "observation_contract": extractor.contract(),
        "durations": list(profile.durations),
        "wall_clock_s": round(elapsed, 1),
    }
    _write_json(result, os.path.join(out_dir, "summary.json"))
    if not quiet:
        print("\n{:.1f} s total, {:.2f} s per episode".format(
            elapsed, result["seconds_per_episode"]))
        if pilot:
            print("PILOT ONLY - no model was selected. Estimated time for "
                  "{} episodes: {:.0f} min".format(
                      episodes, episodes * result["seconds_per_episode"] / 60))
        else:
            print("best validation total delay {:.2f} s at episode {}".format(
                best_delay if best_delay is not None else float("nan"),
                best_episode))
    return result


# ---------------------------------------------------------------------------
def _write_history(history, path):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)
    return path


def _write_json(payload, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=float)
    return path


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="A", choices=["A", "B"],
                        help="A: choose the whole green from a menu. "
                             "B: minimum-green lock, then extend or end.")
    parser.add_argument("--extension", type=float, default=3.0,
                        help="candidate B only: seconds added per EXTEND")
    parser.add_argument("--profile", default="standard")
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=900.0)
    parser.add_argument("--out", default="output/profile_dqn")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", default="128,128")
    parser.add_argument("--n-step", type=int, default=2)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--exploration-fraction", type=float, default=0.6)
    parser.add_argument("--learning-starts", type=int, default=600)
    parser.add_argument("--target-update", type=int, default=400)
    parser.add_argument("--reward-scale", type=float, default=0.005)
    parser.add_argument("--validate-every", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--pilot", action="store_true",
                        help="three episodes, to measure runtime and catch "
                             "failures before committing to a budget")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    train(candidate=args.candidate, extension=args.extension,
          profile_name=args.profile, episodes=args.episodes, seed=args.seed,
          out_dir=args.out, duration_s=args.duration,
          gamma_per_second=args.gamma, learning_rate=args.lr,
          batch_size=args.batch_size,
          hidden_layers=[int(h) for h in args.hidden.split(",")],
          n_step=args.n_step, epsilon_end=args.epsilon_end,
          exploration_fraction=args.exploration_fraction,
          learning_starts=args.learning_starts,
          target_update=args.target_update, reward_scale=args.reward_scale,
          validate_every=args.validate_every, threads=args.threads,
          pilot=args.pilot, quiet=args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
