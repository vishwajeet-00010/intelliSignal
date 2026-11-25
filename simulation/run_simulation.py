"""
IntelliSignal - Stage 1
run_simulation.py

Runs one traffic scenario in SUMO, controls nothing more than a fixed-time
traffic signal, and records real traffic metrics through TraCI.

Usage
-----
    python simulation/run_simulation.py --scenario moderate
    python simulation/run_simulation.py --scenario heavy
    python simulation/run_simulation.py --scenario moderate --gui
    python simulation/run_simulation.py --scenario moderate --plot

Pipeline
--------
    config/<scenario>.yaml
        -> SUMO input files (network, routes, fixed-time signal)
        -> SUMO simulation, driven step by step through TraCI
        -> metrics collected from the live simulation
        -> results/csv/<scenario>_results.csv
        -> results/plots/*.png   (with --plot)

There is no reinforcement learning in this stage. The traffic light follows
the fixed phase plan defined in the YAML file. This run is the BASELINE the
future DQN controller will be compared against.
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow "python simulation/run_simulation.py" to import the utils package.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.config_loader import cycle_time, load_config, total_demand  # noqa: E402
from utils.metrics import MetricsCollector  # noqa: E402
from utils.scenario_builder import TLS_ID, TLS_PROGRAM_ID, build_scenario  # noqa: E402
from utils.sumo_launch import start_sumo  # noqa: E402

RESULTS_CSV_DIR = os.path.join(PROJECT_ROOT, "results", "csv")


# --------------------------------------------------------------------------
# SUMO / TraCI discovery
# --------------------------------------------------------------------------
def prepare_sumo_imports() -> None:
    """Make sure the SUMO python tools are importable.

    SUMO is found either through the SUMO_HOME environment variable (the
    normal installation) or through the 'sumo' pip package.
    """
    if "SUMO_HOME" in os.environ:
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        if os.path.isdir(tools) and tools not in sys.path:
            sys.path.append(tools)
    try:
        import traci  # noqa: F401
        import sumolib  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment problem
        raise SystemExit(
            "Could not import TraCI/sumolib.\n"
            "Set the SUMO_HOME environment variable to your SUMO installation,\n"
            "or install the python packages with:  pip install eclipse-sumo traci sumolib\n"
            "Original error: {}".format(exc)
        )


def sumo_binary(use_gui: bool) -> str:
    """Return the path of the sumo or sumo-gui executable."""
    import sumolib

    name = "sumo-gui" if use_gui else "sumo"
    try:
        return sumolib.checkBinary(name)
    except Exception as exc:  # pragma: no cover - environment problem
        raise SystemExit("Could not locate '{}': {}".format(name, exc))


# --------------------------------------------------------------------------
# The simulation itself
# --------------------------------------------------------------------------
def run_scenario(scenario: str, use_gui: bool = False, quiet: bool = False,
                 duration: float = None, results_dir: str = None) -> dict:
    """Run one scenario end to end and return its summary metrics.

    `duration` overrides the simulated time from the YAML file (handy for a
    quick demonstration or for the tests); `results_dir` overrides where the
    CSV files are written.
    """
    prepare_sumo_imports()
    import traci

    config = load_config(scenario)
    if duration is not None:
        config["simulation"]["duration"] = float(duration)
    files = build_scenario(config)

    sim = config["simulation"]
    duration = float(sim["duration"])
    step_length = float(sim["step_length"])
    total_steps = int(round(duration / step_length))

    if not quiet:
        _print_header(config, files)

    cmd = [
        sumo_binary(use_gui),
        "-c", files["sumocfg"],
        "--start",            # start immediately when the GUI opens
        "--quit-on-end",      # close the GUI when the simulation ends
    ]

    collector = MetricsCollector(step_length=step_length)
    label = "intellisignal_{}".format(scenario)

    try:
        start_sumo(cmd, label=label)
    except Exception as exc:
        raise SystemExit("SUMO failed to start: {}".format(exc))

    try:
        # Confirm that SUMO really loaded and activated our fixed-time program.
        program = traci.trafficlight.getProgram(TLS_ID)
        if program != TLS_PROGRAM_ID:
            raise SystemExit(
                "Traffic light '{}' is running program '{}' but the scenario "
                "defines '{}'.".format(TLS_ID, program, TLS_PROGRAM_ID))
        logic = [l for l in traci.trafficlight.getAllProgramLogics(TLS_ID)
                 if l.programID == program][0]
        phases = logic.phases
        if not quiet:
            print("  Traffic light '{}' running program '{}' with {} phases:".format(
                TLS_ID, program, len(phases)))
            for i, phase in enumerate(phases):
                print("    phase {}  {:>4.0f}s  {}".format(i, phase.duration, phase.state))
            print()

        step = 0
        while step < total_steps and traci.simulation.getMinExpectedNumber() >= 0:
            traci.simulationStep()
            collector.collect_step()
            step += 1
            if not quiet and step % 300 == 0:
                print("    t = {:>5.0f}s   running = {:>3d}   queue = {:>3d}".format(
                    traci.simulation.getTime(),
                    traci.vehicle.getIDCount(),
                    collector.timeseries[-1]["queue_length"],
                ))
    except traci.exceptions.FatalTraCIError as exc:
        print("\n  SUMO connection lost: {}".format(exc), file=sys.stderr)
        raise
    finally:
        # Always shut SUMO down, even when the run failed.
        try:
            traci.close()
        except Exception:
            pass
        if not quiet:
            print("\n  SUMO closed cleanly.")

    summary = collector.summary(config["scenario"]["name"])
    summary["config_file"] = os.path.basename(config["_path"])
    csv_path = save_results(summary, collector, config, results_dir)

    if not quiet:
        _print_summary(summary, csv_path, config)

    return summary


# --------------------------------------------------------------------------
# Result storage
# --------------------------------------------------------------------------
def save_results(summary: dict, collector: MetricsCollector, config: dict,
                 results_dir: str = None) -> str:
    """Write the summary CSV and the per-second time-series CSV."""
    import pandas as pd

    target_dir = results_dir or RESULTS_CSV_DIR
    os.makedirs(target_dir, exist_ok=True)
    name = config["scenario"]["name"]

    columns = [
        "scenario", "vehicles", "average_waiting_time", "average_queue_length",
        "throughput", "simulation_time", "max_queue_length", "vehicles_departed",
        "vehicles_still_in_network", "teleports", "config_file",
    ]
    summary_path = os.path.join(target_dir, "{}_results.csv".format(name))
    pd.DataFrame([summary])[columns].to_csv(summary_path, index=False)

    ts_path = os.path.join(target_dir, "{}_timeseries.csv".format(name))
    pd.DataFrame(collector.timeseries).to_csv(ts_path, index=False)

    return summary_path


# --------------------------------------------------------------------------
# Console output
# --------------------------------------------------------------------------
def _print_header(config: dict, files: dict) -> None:
    sc = config["scenario"]
    sig = config["signal"]
    print("=" * 66)
    print("  IntelliSignal - Stage 1 (fixed-time baseline)")
    print("=" * 66)
    print("  Scenario     : {}".format(sc["name"]))
    print("  Description  : {}".format(sc.get("description", "-")))
    print("  Duration     : {:.0f} s".format(float(config["simulation"]["duration"])))
    print("  Total demand : {:.0f} vehicles/hour".format(total_demand(config)))
    print("  Signal cycle : {:.0f} s  (NS green {}s, EW green {}s, yellow {}s)".format(
        cycle_time(config), sig["ns_green"], sig["ew_green"], sig["yellow"]))
    print("  SUMO config  : {}".format(os.path.relpath(files["sumocfg"], PROJECT_ROOT)))
    print("  Routes       : {}".format(os.path.relpath(files["routes"], PROJECT_ROOT)))
    print("  Signal file  : {}".format(os.path.relpath(files["tls"], PROJECT_ROOT)))
    print("-" * 66)


def _print_summary(summary: dict, csv_path: str, config: dict) -> None:
    print("-" * 66)
    print("  RESULTS - {} traffic".format(summary["scenario"]))
    print("-" * 66)
    print("  Vehicles in simulation     : {}".format(summary["vehicles"]))
    print("  Average waiting time       : {:.2f} s / vehicle".format(
        summary["average_waiting_time"]))
    print("  Average queue length       : {:.2f} vehicles".format(
        summary["average_queue_length"]))
    print("  Maximum queue length       : {} vehicles".format(summary["max_queue_length"]))
    print("  Throughput (completed)     : {} vehicles".format(summary["throughput"]))
    print("  Still in network at end    : {}".format(summary["vehicles_still_in_network"]))
    print("  Teleports (gridlock sign)  : {}".format(summary["teleports"]))
    print("  Simulation time            : {:.0f} s".format(summary["simulation_time"]))
    print("-" * 66)
    print("  Saved: {}".format(os.path.relpath(csv_path, PROJECT_ROOT)))
    print("=" * 66)


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="IntelliSignal Stage 1 - run a fixed-time SUMO traffic scenario.",
    )
    parser.add_argument("--scenario", "-s", required=True,
                        help="Scenario name, e.g. moderate or heavy "
                             "(must match a file in config/).")
    parser.add_argument("--gui", action="store_true",
                        help="Run with sumo-gui so the intersection is visible.")
    parser.add_argument("--plot", action="store_true",
                        help="Regenerate the result graphs after the run.")
    parser.add_argument("--duration", type=float, default=None,
                        help="Override the simulated time in seconds "
                             "(default: the value in the scenario YAML file).")
    parser.add_argument("--quiet", action="store_true",
                        help="Print less output.")
    args = parser.parse_args(argv)

    run_scenario(args.scenario, use_gui=args.gui, quiet=args.quiet,
                 duration=args.duration)

    if args.plot:
        from simulation.plot_results import generate_plots
        print()
        generate_plots()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
