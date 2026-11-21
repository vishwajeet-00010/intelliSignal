"""Collect raw per-approach state distributions from greedy DQN rollouts."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.base_agent import TrafficState
from agents.dqn_agent import DQNAgent, default_model_path, set_seeds
from rl.environment import IntelliSignalEnv


def collect(scenario: str, model: str | None, seeds: list[int]) -> dict:
    model_path = model or default_model_path(scenario)
    if not model_path:
        raise FileNotFoundError("No trained model found for {}".format(scenario))
    agent = DQNAgent.load(model_path, scenario=scenario)
    agent.set_epsilon(0.0)
    values = {kind: {name: [] for name in
                     ("north", "east", "south", "west")}
              for kind in ("queues", "waiting")}
    for seed in seeds:
        set_seeds(seed)
        env = IntelliSignalEnv(scenario=scenario, seed=seed)
        def sample():
            raw = env.state_extractor.raw(env.connection)
            for kind in values:
                for approach, value in raw[kind].items():
                    values[kind][approach].append(float(value))
        env.step_callback = sample
        try:
            observation, info = env.reset(seed=seed)
            while True:
                state = TrafficState.from_env_info(info, observation)
                action = agent.env_action(state)
                observation, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
        finally:
            env.close()
    summary = {"scenario": scenario, "model": model_path, "seeds": seeds,
               "samples_per_approach": {}, "percentiles": {}}
    for kind, approaches in values.items():
        summary["percentiles"][kind] = {}
        summary["samples_per_approach"][kind] = {}
        for approach, samples in approaches.items():
            summary["samples_per_approach"][kind][approach] = len(samples)
            summary["percentiles"][kind][approach] = {
                "p95": float(np.percentile(samples, 95)),
                "p99": float(np.percentile(samples, 99)),
                "max": float(max(samples)),
            }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=("moderate", "heavy"))
    parser.add_argument("--model")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=list(range(2001, 2011)))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = collect(args.scenario, args.model, args.seeds)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result["percentiles"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
