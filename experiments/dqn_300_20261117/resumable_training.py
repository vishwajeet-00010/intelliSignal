"""Experiment-local training with exact episode-boundary recovery.

The training loop and summary/metadata layout are adapted from
training/train_dqn.py. Core learning, episode execution, and selection rules
are unchanged. Full-state snapshots add replay/RNG recovery; original
project models and training modules are not modified.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from agents.dqn_agent import EpsilonSchedule, build_agent_for_env, set_seeds, write_metadata
from agents.dqn_config import DQNConfig, load_dqn_config
from rl.environment import IntelliSignalEnv
from simulation.agent_simulation import QUIET_SUMO_ARGS
from training.train_dqn import (
    PROJECT_ROOT, TRAINING_RESULTS_DIR, default_model_dir, history_path,
    _run_episode, _write_history, _print_header, _print_episode, _print_summary,
)


def replace_with_retry(source, destination, attempts=20):
    """Tolerate temporary Windows file locks; preserve the source on failure."""
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(min(0.25 * (attempt + 1), 2.0))


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    replace_with_retry(temporary, path)


def capture_rng():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("Snapshot requires CUDA RNG but CUDA is unavailable.")
        torch.cuda.set_rng_state_all(state["cuda"])


def _restore_agent(agent, snapshot):
    payload = snapshot["agent"]
    agent._restore(payload, str(snapshot["snapshot_path"]))
    # Unlike the core compatibility loader, strict recovery must never swallow
    # optimizer restoration errors.
    agent.optimizer.load_state_dict(payload["optimizer_state_dict"])
    agent.memory.clear()
    agent.memory._storage.extend(snapshot["replay"])
    agent.memory._random.setstate(snapshot["replay_rng"])
    agent._random.setstate(snapshot["agent_rng"])
    agent._last_loss = snapshot["last_loss"]
    agent.policy_net.train(snapshot["policy_training"])
    agent.target_net.train(snapshot["target_training"])


def _save_resume(path, agent, scenario, config, environment, history,
                 global_step, best_average, best_episode, best_payload, elapsed):
    payload = {
        "version": 1, "snapshot_path": str(path), "scenario": scenario,
        "config": config.as_dict(), "environment": environment,
        "agent": agent.state_dict(), "replay": list(agent.memory),
        "replay_rng": agent.memory._random.getstate(),
        "agent_rng": agent._random.getstate(), "rng": capture_rng(),
        "last_loss": agent._last_loss,
        "policy_training": agent.policy_net.training,
        "target_training": agent.target_net.training,
        "global_step": global_step, "history": history,
        "best_average": best_average, "best_episode": best_episode,
        "best_payload": best_payload, "elapsed_seconds": elapsed,
    }
    atomic_save(payload, path)


def train(scenario: str = "moderate", config: Optional[DQNConfig] = None,
          model_dir: Optional[str] = None, results_dir: Optional[str] = None,
          quiet: bool = False, progress_every: int = 1,
          save_plots: bool = False) -> Dict[str, Any]:
    """Run the training loop and return a summary of what happened."""
    if save_plots:
        raise ValueError("Generate plots separately with an explicit isolated output_dir.")
    if model_dir is None or results_dir is None:
        raise ValueError("Explicit experiment model_dir and results_dir are required.")
    config = config or load_dqn_config()
    config.validate()
    resume_path = Path(model_dir).resolve().parent / "resume.pt"
    existing = None
    if resume_path.exists():
        existing = torch.load(resume_path, map_location="cpu", weights_only=False)
        if existing["version"] != 1 or existing["scenario"] != scenario:
            raise ValueError("Resume snapshot version/scenario mismatch.")
        if existing["config"] != config.as_dict():
            raise ValueError("Resume configuration differs from the saved experiment.")
    elif any(Path(model_dir).glob("*.pt")) or Path(history_path(scenario, int(config.seed), results_dir)).exists():
        raise ValueError("Existing training artifacts have no full-state resume snapshot.")
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
        best_payload = None
        elapsed_before = 0.0
        env_contract = {
            "scenario_config": getattr(env, "scenario_config", None),
            "rl_config": getattr(env, "rl_config", None),
            "episode_duration": float(env.episode_duration),
            "decision_interval": float(env.decision_interval),
            "steps_per_episode": steps_per_episode,
        }
        if existing is not None:
            if existing["environment"] != env_contract:
                raise ValueError("Resume traffic/environment configuration changed.")
            _restore_agent(agent, existing)
            history = existing["history"]
            global_step = existing["global_step"]
            best_average = existing["best_average"]
            best_episode = existing["best_episode"]
            best_payload = existing["best_payload"]
            elapsed_before = existing["elapsed_seconds"]
            if len(history) != agent.episode or agent.episode > config.episodes:
                raise ValueError("Resume episode/history mismatch.")
            # Roll back a best.pt/history write that may have preceded a failed snapshot.
            if best_payload is not None:
                atomic_save(best_payload, Path(model_dir) / "best.pt")
            _write_history(history, history_path(scenario, seed, results_dir))
            restore_rng(existing["rng"])
        started = time.time()
        if existing is None:
            _save_resume(resume_path, agent, scenario, config, env_contract, history,
                         global_step, best_average, best_episode, best_payload, 0.0)

        for episode in range(len(history) + 1, int(config.episodes) + 1):
            episode_seed = (seed + episode - 1) if config.vary_episode_seed else seed
            row, global_step = _run_episode(
                env, agent, schedule, config, episode, episode_seed,
                global_step, steps_per_episode)
            row["wall_clock_seconds"] = round(elapsed_before + time.time() - started, 1)
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
                best_payload = copy.deepcopy(agent.state_dict())
                best_payload["extra"] = {
                    "selection": "best mean return over last {} episodes".format(config.best_window),
                    "window_mean_return": round(window_mean, 4), "episode": episode}
                atomic_save(best_payload, Path(model_dir) / "best.pt")

            if config.checkpoint_every and episode % int(config.checkpoint_every) == 0:
                atomic_save(agent.state_dict(), Path(model_dir) / "checkpoint_ep{:03d}.pt".format(episode))

            if not quiet and (episode % max(int(progress_every), 1) == 0
                              or episode == 1):
                _print_episode(row, config.episodes)

            _write_history(history, history_path(scenario, seed, results_dir))
            _save_resume(resume_path, agent, scenario, config, env_contract, history,
                         global_step, best_average, best_episode, best_payload,
                         elapsed_before + time.time() - started)

        # --- final artefacts ------------------------------------------
        final_path = str(Path(model_dir) / "final.pt")
        final_payload = agent.state_dict()
        final_payload["extra"] = {"selection": "last episode"}
        atomic_save(final_payload, final_path)
        csv_path = _write_history(history, history_path(scenario, seed, results_dir))

        elapsed = elapsed_before + time.time() - started
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

