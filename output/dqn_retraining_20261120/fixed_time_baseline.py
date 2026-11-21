"""Measure the brief's tuned fixed-time greens on specified SUMO seeds."""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from simulation.run_simulation import prepare_sumo_imports, sumo_binary
from utils.config_loader import load_config
from utils.scenario_builder import build_scenario
from utils.metrics import MetricsCollector

prepare_sumo_imports()
import traci


def run_one(scenario: str, green: float, seed: int, duration: float = 1800.0):
    config = load_config(scenario)
    config["scenario"]["name"] = scenario + "_tuned_sweep"
    config["simulation"]["duration"] = float(duration)
    config["simulation"]["seed"] = int(seed)
    config["signal"]["ns_green"] = float(green)
    config["signal"]["ew_green"] = float(green)
    files = build_scenario(config)
    step_length = float(config["simulation"]["step_length"])
    collector = MetricsCollector(step_length=step_length)
    label = "tuned_{}_{}_{}".format(scenario, int(green), int(seed))
    traci.start([sumo_binary(False), "-c", files["sumocfg"],
                 "--no-warnings", "true", "--no-step-log", "true"], label=label)
    try:
        for _ in range(int(round(duration / step_length))):
            traci.simulationStep()
            collector.collect_step()
    finally:
        try:
            traci.close()
        except Exception:
            pass
    row = collector.summary(scenario)
    row.update({"green": green, "seed": seed,
                "cycle": 2 * (green + float(config["signal"]["yellow"]))})
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=("moderate", "heavy"))
    parser.add_argument("--green", required=True, type=float)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = [run_one(args.scenario, args.green, seed) for seed in args.seeds]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for pattern in ("sumo/routes/*_tuned_sweep*", "sumo/config/*_tuned_sweep*",
                    "sumo/additional/*_tuned_sweep*"):
        for path in glob.glob(os.path.join(ROOT, pattern)):
            os.remove(path)
    print("saved {} run(s) to {}".format(len(rows), args.out))


if __name__ == "__main__":
    main()
