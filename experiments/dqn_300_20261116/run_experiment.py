"""Isolated 300-episode experiments; uses the existing training/evaluation APIs."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("MPLBACKEND", "Agg")

from agents.dqn_config import load_dqn_config
from training.train_dqn import train
from training.plot_training import plot_training_history
from evaluation.compare_controllers import compare
from evaluation.evaluate_dqn import evaluate
import evaluation.plot_comparison as comparison_plots


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def protected_hashes():
    hashes = {}
    for folder in ("models", "config", "results/csv", "results/training",
                   "results/evaluation", "results/plots"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file():
                hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def status(stage, **extra):
    record = {"stage": stage, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}
    write_json(OUT / "status.json", record)
    print("EXPERIMENT_STATUS", json.dumps(record), flush=True)


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, value):
        self.console.write(value)
        self.log.write(value)
        self.log.flush()

    def flush(self):
        self.console.flush()
        self.log.flush()


def main():
    if (OUT / "manifest.json").exists():
        if list(OUT.glob("*/models/*.pt")) or list(OUT.glob("*/training/*.csv")):
            raise RuntimeError("Training artifacts already exist; refusing to overwrite or pretend to resume.")
        previous = json.loads((OUT / "protected_before.json").read_text(encoding="utf-8"))
        if previous != protected_hashes():
            raise RuntimeError("Protected artifacts changed since initial attempt.")
        print("Retrying before the first completed episode; no model/history artifacts exist.", flush=True)
    before = protected_hashes()
    write_json(OUT / "protected_before.json", before)
    write_json(OUT / "manifest.json", {
        "scenarios": ["heavy", "moderate"], "episodes": 300, "training_seed": 42,
        "training_traffic_seeds": [42, 341], "duration_seconds": 1800,
        "reference_seeds": [42, 43, 44], "holdout_seeds": list(range(1001, 1011)),
        "checkpoint_selection": "best.pt selected by existing 5-episode mean training return",
        "training_mode": "fresh; default epsilon schedule stretches to 27000 decisions",
        "promotion_rule": "On holdouts: lower mean waiting, waiting wins on >5/10 seeds, no worse mean queue, throughput or teleports vs incumbent. Report reference results and all tradeoffs. No automatic replacement.",
        "python": sys.version,
        "packages": {n: importlib.metadata.version(n) for n in
                     ["eclipse-sumo", "traci", "sumolib", "torch", "numpy", "gymnasium"]},
        "config": load_dqn_config(episodes=300, seed=42).as_dict(),
    })
    for scenario in ("heavy", "moderate"):
        folder = OUT / scenario
        status("training", scenario=scenario, episodes=300)
        summary = train(scenario=scenario, config=load_dqn_config(episodes=300, seed=42),
                        model_dir=str(folder / "models"), results_dir=str(folder / "training"),
                        save_plots=False, progress_every=10)
        summary["plots"] = plot_training_history(summary["history_csv"],
                                                output_dir=str(folder / "plots" / "training"))
        write_json(folder / "training_summary.json", summary)
        candidate = str(folder / "models" / "best.pt")
        incumbent = str(ROOT / "models" / ("dqn_" + scenario) / "best.pt")
        for label, seeds in (("reference", [42, 43, 44]), ("holdout", list(range(1001, 1011)))):
            status("evaluating_candidate", scenario=scenario, seed_set=label)
            comparison_plots.PLOTS_DIR = str(folder / "plots" / label)
            candidate_result = compare(scenario=scenario, model=candidate, seeds=seeds,
                                       duration=1800, include_untrained=True,
                                       results_dir=str(folder / "evaluation" / label / "candidate"))
            write_json(folder / (label + "_candidate.json"), candidate_result)
            status("evaluating_incumbent", scenario=scenario, seed_set=label)
            incumbent_result = evaluate(scenario=scenario, model=incumbent, seeds=seeds,
                                        duration=1800,
                                        results_dir=str(folder / "evaluation" / label / "incumbent"))
            write_json(folder / (label + "_incumbent.json"), incumbent_result)
        status("scenario_complete", scenario=scenario)
    after = protected_hashes()
    changed = [p for p in sorted(set(before) | set(after)) if before.get(p) != after.get(p)]
    write_json(OUT / "integrity.json", {"protected_files": len(before), "changed": changed})
    if changed:
        raise RuntimeError("Protected artifacts changed: " + repr(changed))
    status("complete", protected_files=len(before), protected_artifacts_unchanged=True)


if __name__ == "__main__":
    with (OUT / "run.log").open("a", encoding="utf-8") as log:
        sys.stdout = Tee(sys.stdout, log)
        sys.stderr = Tee(sys.stderr, log)
        try:
            main()
        except BaseException as exc:
            status("failed", error=repr(exc))
            traceback.print_exc()
            raise
