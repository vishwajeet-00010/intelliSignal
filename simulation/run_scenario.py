"""
IntelliSignal - Stage 1
run_scenario.py

One-command demonstration wrapper.

    python simulation/run_scenario.py moderate
    python simulation/run_scenario.py heavy
    python simulation/run_scenario.py all

Each command:
    1. loads the scenario configuration
    2. generates the SUMO input files from it
    3. starts SUMO
    4. runs the simulation step by step through TraCI
    5. collects the traffic metrics
    6. stops SUMO safely (also if something goes wrong)
    7. saves the results to results/csv/
    8. prints the metrics and regenerates the graphs in results/plots/
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from simulation.plot_results import generate_plots      # noqa: E402
from simulation.run_simulation import run_scenario      # noqa: E402
from utils.config_loader import available_scenarios     # noqa: E402

# Scenarios are run from the lightest demand to the heaviest, so that the
# comparison table reads in the same order as the presentation.
PREFERRED_ORDER = ["moderate", "heavy"]


def ordered_scenarios() -> list:
    names = available_scenarios()
    return sorted(names, key=lambda n: (PREFERRED_ORDER.index(n)
                                        if n in PREFERRED_ORDER else 99, n))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run an IntelliSignal Stage 1 scenario and build the graphs.")
    parser.add_argument("scenario",
                        help="Scenario name ({}) or 'all'.".format(
                            ", ".join(ordered_scenarios())))
    parser.add_argument("--gui", action="store_true",
                        help="Show the simulation in sumo-gui.")
    parser.add_argument("--duration", type=float, default=None,
                        help="Override the simulated time, in seconds.")
    parser.add_argument("--no-plots", action="store_true",
                        help="Do not regenerate the graphs afterwards.")
    args = parser.parse_args(argv)

    if args.scenario == "all":
        scenarios = ordered_scenarios()
    else:
        scenarios = [args.scenario]

    summaries = []
    for scenario in scenarios:
        summaries.append(run_scenario(scenario, use_gui=args.gui,
                                      duration=args.duration))
        print()

    if not args.no_plots:
        generate_plots()
        print()

    if len(summaries) > 1:
        _print_comparison(summaries)
    return 0


def _print_comparison(summaries: list) -> None:
    print("=" * 74)
    print("  COMPARISON - fixed-time signal under different traffic demand")
    print("=" * 74)
    header = "{:<22}" + "{:>16}" * len(summaries)
    print(header.format("Metric", *[s["scenario"] for s in summaries]))
    print("-" * 74)
    rows = [
        ("Vehicles", "vehicles", "{:.0f}"),
        ("Avg waiting time (s)", "average_waiting_time", "{:.2f}"),
        ("Avg queue (vehicles)", "average_queue_length", "{:.2f}"),
        ("Max queue (vehicles)", "max_queue_length", "{:.0f}"),
        ("Throughput", "throughput", "{:.0f}"),
        ("Still in network", "vehicles_still_in_network", "{:.0f}"),
        ("Teleports", "teleports", "{:.0f}"),
    ]
    for label, key, fmt in rows:
        values = [fmt.format(float(s[key])) for s in summaries]
        print(("{:<22}" + "{:>16}" * len(values)).format(label, *values))
    print("=" * 74)
    print("  All values measured in SUMO through TraCI. See results/csv/.")


if __name__ == "__main__":
    raise SystemExit(main())
