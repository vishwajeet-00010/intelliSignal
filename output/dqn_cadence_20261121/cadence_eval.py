"""Evaluate the 11 s-cadence heavy DQN against every baseline that matters.

Each controller runs in the configuration it was designed for:

  * the new DQN under the 11 s cadence it was trained with (greens 7/18/29...)
  * `always_switch_11s`, the best plan that cadence can express without
    learning anything - the floor the DQN must beat to have earned its keep
  * the tuned fixed-time 7 s plan through SUMO's own TLS program, which is the
    strongest non-learned baseline measured so far (12.638 s)
  * the original 30 s fixed-time plan

The Stage 4 incumbent is NOT re-run here. It was trained under the 5 s cadence
and `models/active_models.json` records that contract, so running it at 11 s
would evaluate it outside the environment it learned. Its recorded confirmatory
figure (16.469 s on these same seeds) is quoted from
`output/dqn_stage4_20261120/heavy/confirmatory/confirmatory_rows.csv`.

Usage:
    .venv-stage3/Scripts/python.exe output/dqn_cadence_20261121/cadence_eval.py
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))

from agents.base_agent import (ACTION_EW_GREEN, ACTION_NS_GREEN,  # noqa: E402
                               PHASE_NS, TrafficAgent)
from agents.dqn_agent import DQNAgent, set_seeds                  # noqa: E402
from evaluation.fixed_time_sweep import run_fixed_time_plan       # noqa: E402
from evaluation.gates import switch_saturation                    # noqa: E402
from simulation.agent_simulation import (run_agent_episode,       # noqa: E402
                                         run_fixed_time_episode)

SCENARIO = "heavy"
SEEDS = list(range(6101, 6121))
TUNED_GREEN = 7.0

#: Recorded on these same seeds by Stage 4, under the 5 s cadence.
STAGE4_INCUMBENT_WAITING = 16.469
STAGE4_INCUMBENT_THROUGHPUT = 1420.55

METRICS = ("average_waiting_time", "average_queue_length", "max_queue_length",
           "throughput", "vehicles_still_in_network", "teleports")


class AlwaysSwitch(TrafficAgent):
    """Ends every green at the first opportunity the safety layer allows.

    Under the 11 s cadence that produces 7 s greens, so this is the tuned
    fixed-time plan expressed inside the environment - the zero-learning floor.
    """

    name = "always-switch"
    description = "Ends every green as soon as the safety rules permit."
    is_learned = False

    def select_action(self, state):
        return ACTION_EW_GREEN if state.current_phase == PHASE_NS else ACTION_NS_GREEN


def summarise(rows):
    out = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        if values:
            out[metric] = statistics.fmean(values)
    sat = [switch_saturation(float(row["switch_count"]), float(row["decisions"]),
                             decision_interval=float(row["decision_interval"]),
                             yellow_time=4.0, all_red_time=0.0,
                             min_green=float(row["min_green"]))
           for row in rows if row.get("decisions")]
    if sat:
        out["switch_saturation"] = statistics.fmean(sat)
    return out


def run_env_policy(label, policy, interval, min_green):
    rows = []
    for seed in SEEDS:
        set_seeds(seed)
        if isinstance(policy, DQNAgent):
            policy.set_epsilon(0.0)
        summary, _ = run_agent_episode(SCENARIO, policy, seed=seed, duration=1800)
        summary["controller"] = label
        summary["seed"] = seed
        summary["decision_interval"] = interval
        summary["min_green"] = min_green
        rows.append(summary)
        print("  {:<22s} seed={} wait={:7.3f} tp={}".format(
            label, seed, summary["average_waiting_time"], summary["throughput"]),
            flush=True)
    return rows


def main() -> int:
    model = OUT / "models" / "dqn_heavy" / "best.pt"
    if not model.exists():
        raise SystemExit("trained model not found: {}".format(model))

    from rl.config import load_rl_config
    from utils.config_loader import load_config
    env_cfg = load_rl_config(load_config(SCENARIO))["environment"]
    interval = float(env_cfg["decision_interval"])
    min_green = float(env_cfg["min_green"])
    print("heavy cadence: decision_interval={:g}s min_green={:g}s "
          "-> shortest green {:g}s\n".format(
              interval, min_green, interval - 4.0))

    results = {}

    print("new DQN (11 s cadence)")
    agent = DQNAgent.load(str(model), scenario=SCENARIO)
    results["dqn_cadence11"] = run_env_policy("dqn_cadence11", agent,
                                              interval, min_green)

    print("\nalways-switch under the same cadence (zero-learning floor)")
    results["always_switch_11s"] = run_env_policy("always_switch_11s",
                                                  AlwaysSwitch(), interval,
                                                  min_green)

    print("\ntuned fixed-time {:g}s green, SUMO TLS".format(TUNED_GREEN))
    rows = []
    for seed in SEEDS:
        row = run_fixed_time_plan(SCENARIO, TUNED_GREEN, seed, quiet=False)
        row["controller"] = "fixed_time_tuned_7s"
        rows.append(row)
    results["fixed_time_tuned_7s"] = rows

    print("\noriginal fixed-time 30s green, SUMO TLS")
    rows = []
    for seed in SEEDS:
        summary, _ = run_fixed_time_episode(SCENARIO, seed=seed, duration=1800)
        summary["controller"] = "fixed_time_original"
        summary["seed"] = seed
        rows.append(summary)
    results["fixed_time_original"] = rows

    from evaluation.fixed_time_sweep import cleanup_generated_inputs
    cleanup_generated_inputs(SCENARIO)

    all_rows = [row for rows in results.values() for row in rows]
    columns = list(dict.fromkeys(key for row in all_rows for key in row
                                 if not isinstance(row.get(key), (list, dict))))
    with (OUT / "cadence_rows.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    means = {name: summarise(rows) for name, rows in results.items()}
    means["stage4_incumbent_5s_cadence"] = {
        "average_waiting_time": STAGE4_INCUMBENT_WAITING,
        "throughput": STAGE4_INCUMBENT_THROUGHPUT,
        "note": "recorded by Stage 4 under the 5 s cadence; not re-run here",
    }

    summary = {
        "experiment": "dqn_cadence_20261121",
        "scenario": SCENARIO,
        "seeds": SEEDS,
        "cadence": {"decision_interval": interval, "min_green": min_green,
                    "shortest_green": interval - 4.0,
                    "reachable_greens": [interval - 4.0 + k * interval
                                         for k in range(5)]},
        "means": means,
    }
    with (OUT / "cadence_results.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
        fh.write("\n")

    print("\n" + "=" * 72)
    print("HEAVY - mean over {} confirmatory seeds".format(len(SEEDS)))
    print("=" * 72)
    ordered = sorted(means.items(),
                     key=lambda kv: kv[1].get("average_waiting_time", 9e9))
    for name, values in ordered:
        sat = values.get("switch_saturation")
        print("  {:<28s} wait={:>8.3f}  tp={:>8.2f}{}".format(
            name, values.get("average_waiting_time", float("nan")),
            values.get("throughput", float("nan")),
            "  switch_saturation={:.3f}".format(sat) if sat is not None else ""))
    print("\nwrote cadence_rows.csv and cadence_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
