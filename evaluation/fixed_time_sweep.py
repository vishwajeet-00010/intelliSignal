"""
IntelliSignal
evaluation/fixed_time_sweep.py

Measures a fixed-time signal plan by running SUMO's own traffic-light program,
and sweeps the green duration to find the best plan for a scenario.

Why this exists
---------------
A fixed-time baseline must be run the way a fixed-time controller actually
runs: as a traffic-light program inside SUMO, exactly like the Stage 1
baseline. `simulation/run_simulation.py::run_scenario` already does that.

The alternative - driving `agents.fixed_time_agent.FixedTimeAgent` through
`IntelliSignalEnv` - does NOT produce the plan you asked for. The environment
only decides at `decision_interval` boundaries and refuses a switch before
`min_green`, so the green duration is quantised. With the project's 5 s cadence
and 4 s yellow, a switch leaves 1 s of green served, so the reachable greens are
{6, 11, 16, 21, ...}. A request for a 7-second green becomes an 11-second green.

That is what went wrong in `output/dqn_stage4_20261120/stage4_eval.py`: its
`fixed_time_tuned` row is an 11-second controller measured at ~36.8 s mean
waiting, while the 7-second plan this module measures reaches ~12.7 s. See
`output/dqn_stage4_20261120/CORRECTION.md`.

Use this module for every fixed-time baseline. Use the environment only for
controllers that are genuinely meant to decide at the environment's cadence.

Command line
------------
    python evaluation/fixed_time_sweep.py --scenario heavy \
        --greens 5,6,7,8,9,10,12,16,20,30 --seeds 3001,3002,3003 \
        --out results/csv/heavy_fixed_time_sweep.csv

    python evaluation/fixed_time_sweep.py --scenario heavy --greens 7 \
        --seeds 6101-6120 --out out.csv

Nothing in `results/csv/` is overwritten unless you point `--out` there
yourself; the generated SUMO inputs use a private scenario name and are removed
when the run finishes.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from simulation.run_simulation import (prepare_sumo_imports,  # noqa: E402
                                       sumo_binary)
from utils.config_loader import load_config                   # noqa: E402
from utils.metrics import MetricsCollector                    # noqa: E402
from utils.scenario_builder import build_scenario             # noqa: E402

#: Generated SUMO inputs get this suffix so no Stage 1 or RL file is touched.
SWEEP_SUFFIX = "_fixed_time_sweep"

#: Tuned greens measured with this module over seeds 3001-3020.
#: heavy: 12.736 s mean waiting, throughput 1441.0
#: moderate: 3.007 s mean waiting, throughput 689.4
TUNED_GREEN_SECONDS: Dict[str, float] = {"moderate": 5.0, "heavy": 7.0}


# ---------------------------------------------------------------------------
def tuned_green(scenario: str) -> float:
    """The measured best fixed-time green for a scenario, in seconds."""
    try:
        return TUNED_GREEN_SECONDS[scenario]
    except KeyError:
        raise ValueError(
            "No tuned green recorded for scenario {!r}. Run a sweep first."
            .format(scenario))


# ---------------------------------------------------------------------------
def run_fixed_time_plan(scenario: str, green: float, seed: int,
                        duration: float = 1800.0,
                        yellow: Optional[float] = None,
                        all_red: Optional[float] = None,
                        quiet: bool = True) -> Dict[str, object]:
    """Run one episode under a symmetric fixed-time plan and measure it.

    `yellow` and `all_red` default to the scenario's own `signal:` values, so
    the plan differs from the Stage 1 baseline only in green duration. Override
    them only when the clearance interval is itself the thing under test, and
    say so in the report - a baseline with a different clearance is not
    comparable to the controllers it is measured against.
    """
    prepare_sumo_imports()
    import traci

    config = load_config(scenario)
    config["scenario"]["name"] = scenario + SWEEP_SUFFIX
    config["simulation"]["duration"] = float(duration)
    config["simulation"]["seed"] = int(seed)
    config["signal"]["ns_green"] = float(green)
    config["signal"]["ew_green"] = float(green)
    if yellow is not None:
        config["signal"]["yellow"] = float(yellow)
    if all_red is not None:
        config["signal"]["all_red"] = float(all_red)
    files = build_scenario(config)

    step_length = float(config["simulation"]["step_length"])
    collector = MetricsCollector(step_length=step_length)
    label = "fts_{}_{:g}_{}".format(scenario, green, seed)

    traci.start([sumo_binary(False), "-c", files["sumocfg"],
                 "--no-warnings", "--no-step-log"], label=label)
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
    sig = config["signal"]
    row.update({
        "controller": "fixed_time_green{:g}".format(green),
        "green": float(green),
        "yellow": float(sig["yellow"]),
        "all_red": float(sig["all_red"]),
        "cycle": 2.0 * (float(green) + float(sig["yellow"]) + float(sig["all_red"])),
        "seed": int(seed),
        "signal_program": "sumo_tls",
    })
    if not quiet:
        print("{} green={:g} cycle={:g} seed={} wait={:.3f} queue={:.3f} "
              "throughput={}".format(scenario, green, row["cycle"], seed,
                                     row["average_waiting_time"],
                                     row["average_queue_length"],
                                     row["throughput"]), flush=True)
    return row


# ---------------------------------------------------------------------------
def sweep(scenario: str, greens: Sequence[float], seeds: Sequence[int],
          duration: float = 1800.0, quiet: bool = True) -> List[Dict[str, object]]:
    """Run every (green, seed) combination and return the rows."""
    rows = []
    for green in greens:
        for seed in seeds:
            rows.append(run_fixed_time_plan(scenario, green, seed,
                                            duration=duration, quiet=quiet))
    return rows


# ---------------------------------------------------------------------------
def cleanup_generated_inputs(scenario: Optional[str] = None) -> None:
    """Remove the throwaway SUMO inputs this module generated.

    `scenario` scopes the deletion to one scenario's files. Leave it unset only
    when nothing else is running: two sweeps started concurrently share this
    directory, and an unscoped cleanup would delete the other run's inputs
    between its `build_scenario` call and SUMO reading them.
    """
    stem = "{}{}".format(scenario, SWEEP_SUFFIX) if scenario else "*" + SWEEP_SUFFIX
    for pattern in ("sumo/routes/{}*".format(stem),
                    "sumo/config/{}*".format(stem),
                    "sumo/additional/{}*".format(stem)):
        for path in glob.glob(os.path.join(PROJECT_ROOT, pattern)):
            try:
                os.remove(path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
def write_rows(rows: Iterable[Dict[str, object]], path: str) -> str:
    rows = list(rows)
    if not rows:
        raise ValueError("no rows to write")
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


# ---------------------------------------------------------------------------
def parse_number_list(text: str) -> List[float]:
    """Accept "5,6,7" and "6101-6120" and mixtures of both."""
    values: List[float] = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk[1:]:
            start, end = chunk.rsplit("-", 1)
            first, last = int(float(start)), int(float(end))
            if last < first:
                raise ValueError("range {!r} counts downwards".format(chunk))
            values.extend(float(value) for value in range(first, last + 1))
        else:
            values.append(float(chunk))
    if not values:
        raise ValueError("no values parsed from {!r}".format(text))
    return values


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure fixed-time signal plans through SUMO's own TLS "
                    "program, sweeping the green duration.")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--greens", default=None,
                        help="Comma list and/or ranges, e.g. 5,6,7 or 5-12. "
                             "Defaults to the recorded tuned green.")
    parser.add_argument("--seeds", default="3001-3020",
                        help="Comma list and/or ranges, e.g. 3001-3020.")
    parser.add_argument("--duration", type=float, default=1800.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    greens = (parse_number_list(args.greens) if args.greens
              else [tuned_green(args.scenario)])
    seeds = [int(value) for value in parse_number_list(args.seeds)]

    try:
        rows = sweep(args.scenario, greens, seeds, duration=args.duration,
                     quiet=args.quiet)
    finally:
        cleanup_generated_inputs(args.scenario)

    write_rows(rows, args.out)

    by_green: Dict[float, List[float]] = {}
    for row in rows:
        by_green.setdefault(float(row["green"]), []).append(
            float(row["average_waiting_time"]))
    print("\n{} - mean waiting by green, {} seed(s)".format(
        args.scenario, len(seeds)))
    for green in sorted(by_green):
        waits = by_green[green]
        print("  green={:>5g}s  cycle={:>5g}s  wait={:>8.3f}".format(
            green, 2 * (green + 4), sum(waits) / len(waits)))
    best = min(by_green, key=lambda g: sum(by_green[g]) / len(by_green[g]))
    print("  best green = {:g} s".format(best))
    print("saved {} row(s) to {}".format(len(rows), args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
