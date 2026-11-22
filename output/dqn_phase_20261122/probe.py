"""Headroom probe: can ANY adaptive controller beat the best fixed plan?

Before training anything, measure what a phase-duration controller can reach
on heavy traffic. The best fixed plan measured through SUMO's own traffic
light program is north-south 7 s / east-west 6 s at 5.754 s mean waiting
(seeds 6101-6120). If a simple vehicle-actuated controller cannot beat that,
then no amount of DQN tuning will either, and the problem is the formulation
rather than the learning.

This runner drives the signal the way the new environment will: one decision
per green, choosing that green's duration, then a mandatory clearance. It
measures with the project's own `MetricsCollector` over exactly 1800 steps,
the same as `evaluation/fixed_time_sweep.py`, so the numbers are directly
comparable to the 5.754 s figure.

    python output/dqn_phase_20261122/probe.py --seeds 6101-6110
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from typing import Callable, Dict, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from simulation.run_simulation import prepare_sumo_imports, sumo_binary  # noqa: E402
from utils.config_loader import load_config                             # noqa: E402
from utils.metrics import MetricsCollector                              # noqa: E402
from utils.scenario_builder import (ALL_RED, EW_GREEN, EW_YELLOW,       # noqa: E402
                                    NS_GREEN, NS_YELLOW, TLS_ID,
                                    build_scenario)

PHASE_NS, PHASE_EW = 0, 1
GREEN = {PHASE_NS: NS_GREEN, PHASE_EW: EW_GREEN}
YELLOW = {PHASE_NS: NS_YELLOW, PHASE_EW: EW_YELLOW}

LANES = {
    PHASE_NS: ["N2C_0", "N2C_1", "S2C_0", "S2C_1"],
    PHASE_EW: ["E2C_0", "E2C_1", "W2C_0", "W2C_1"],
}

SUFFIX = "_phase_probe"


# ---------------------------------------------------------------------------
def read_phase(conn, phase: int) -> Dict[str, float]:
    """Queue and waiting time on both directions, from the served side's view."""
    out = {}
    for name, index in (("served", phase), ("other", 1 - phase)):
        lanes = LANES[index]
        out[name + "_queue"] = sum(
            conn.lane.getLastStepHaltingNumber(lane) for lane in lanes)
        out[name + "_waiting"] = sum(
            conn.lane.getWaitingTime(lane) for lane in lanes)
    return out


# ---------------------------------------------------------------------------
# Controllers. Each returns the green duration, in seconds, for `phase`.
# ---------------------------------------------------------------------------
def make_fixed(ns: float, ew: float) -> Callable:
    def policy(conn, phase, state):
        return ns if phase == PHASE_NS else ew
    policy.name = "fixed ns={:g} ew={:g}".format(ns, ew)
    return policy


def make_queue_proportional(seconds_per_vehicle: float, lo: float, hi: float,
                            base: float = 0.0) -> Callable:
    """Green time proportional to the queue waiting on the served direction."""
    def policy(conn, phase, state):
        want = base + seconds_per_vehicle * state["served_queue"]
        return max(lo, min(hi, want))
    policy.name = "queue-proportional {:g}s/veh base={:g} [{:g},{:g}]".format(
        seconds_per_vehicle, base, lo, hi)
    return policy


def make_ratio(lo: float, hi: float, slope: float) -> Callable:
    """Longer green when the served side is busier than the other side."""
    def policy(conn, phase, state):
        served, other = state["served_queue"], state["other_queue"]
        if served + other <= 0:
            return lo
        share = served / float(served + other)
        return max(lo, min(hi, lo + slope * share * (hi - lo)))
    policy.name = "pressure-ratio slope={:g} [{:g},{:g}]".format(slope, lo, hi)
    return policy


# ---------------------------------------------------------------------------
def make_gap_out(lo: float, hi: float, gap: float) -> Callable:
    """Vehicle-actuated: extend while vehicles keep arriving, else gap out.

    Implemented as a marker; the runner handles it specially because it has
    to re-read SUMO every second rather than commit to a duration up front.
    """
    def policy(conn, phase, state):
        raise AssertionError("handled by the runner")
    policy.name = "gap-out min={:g} max={:g} gap={:g}".format(lo, hi, gap)
    policy.gap_out = (lo, hi, gap)
    return policy


# ---------------------------------------------------------------------------
def run(policy, scenario: str, seed: int, duration: float = 1800.0,
        yellow: float = 4.0, all_red: float = 0.0,
        tag: str = "") -> Dict[str, object]:
    """One episode under `policy`, measured exactly like the fixed-time sweep."""
    prepare_sumo_imports()
    import traci

    config = load_config(scenario)
    config["scenario"]["name"] = scenario + SUFFIX + str(tag)
    config["simulation"]["duration"] = float(duration)
    config["simulation"]["seed"] = int(seed)
    files = build_scenario(config)

    step_length = float(config["simulation"]["step_length"])
    total_steps = int(round(duration / step_length))
    collector = MetricsCollector(step_length=step_length)
    label = "probe_{}_{}_{}".format(scenario, seed, tag or id(policy))

    gap_out = getattr(policy, "gap_out", None)
    phase = PHASE_NS
    greens: List[float] = []
    steps_done = 0

    traci.start([sumo_binary(False), "-c", files["sumocfg"],
                 "--no-warnings", "--no-step-log"], label=label)
    try:
        while steps_done < total_steps:
            state = read_phase(traci, phase)

            # ---- green -------------------------------------------------
            traci.trafficlight.setRedYellowGreenState(TLS_ID, GREEN[phase])
            if gap_out is None:
                want = float(policy(traci, phase, state))
                served = 0
                while served < want and steps_done < total_steps:
                    traci.simulationStep()
                    collector.collect_step()
                    served += step_length
                    steps_done += 1
            else:
                lo, hi, gap = gap_out
                served, idle = 0.0, 0.0
                while steps_done < total_steps:
                    traci.simulationStep()
                    collector.collect_step()
                    served += step_length
                    steps_done += 1
                    if served >= hi:
                        break
                    moving = sum(traci.lane.getLastStepVehicleNumber(lane)
                                 for lane in LANES[phase])
                    idle = 0.0 if moving > 0 else idle + step_length
                    if served >= lo and idle >= gap:
                        break
            greens.append(served)

            # ---- clearance ---------------------------------------------
            traci.trafficlight.setRedYellowGreenState(TLS_ID, YELLOW[phase])
            for _ in range(int(round(yellow / step_length))):
                if steps_done >= total_steps:
                    break
                traci.simulationStep()
                collector.collect_step()
                steps_done += 1
            if all_red > 0:
                traci.trafficlight.setRedYellowGreenState(TLS_ID, ALL_RED)
                for _ in range(int(round(all_red / step_length))):
                    if steps_done >= total_steps:
                        break
                    traci.simulationStep()
                    collector.collect_step()
                    steps_done += 1

            phase = 1 - phase
    finally:
        try:
            traci.close()
        except Exception:
            pass

    row = collector.summary(scenario)
    row.update({
        "controller": policy.name,
        "seed": int(seed),
        "greens": len(greens),
        "mean_green": round(statistics.fmean(greens), 2) if greens else 0.0,
    })
    return row


# ---------------------------------------------------------------------------
def cleanup(scenario: str, tag: str = "") -> None:
    import glob
    stem = "{}{}{}".format(scenario, SUFFIX, tag)
    for pattern in ("sumo/routes/{}*", "sumo/config/{}*", "sumo/additional/{}*"):
        for path in glob.glob(os.path.join(PROJECT_ROOT, pattern.format(stem))):
            try:
                os.remove(path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
def parse_seeds(text: str) -> List[int]:
    out: List[int] = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk[1:]:
            a, b = chunk.rsplit("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(chunk))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="heavy")
    parser.add_argument("--seeds", default="6101-6110")
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)

    seeds = parse_seeds(args.seeds)
    policies = [
        make_fixed(7, 6),                       # the plan to beat: 5.754 s
        make_fixed(6, 5),
        make_fixed(5, 5),
        make_gap_out(5, 20, 2.0),
        make_gap_out(5, 20, 3.0),
        make_gap_out(4, 15, 2.0),
        make_queue_proportional(1.2, 5, 20, base=3.0),
        make_ratio(5, 16, 1.0),
    ]

    print("{} - {} seeds, 1800 s, measured with MetricsCollector\n".format(
        args.scenario, len(seeds)))
    print("{:<42} {:>9} {:>11} {:>9} {:>7}".format(
        "controller", "wait", "throughput", "queue", "green"))
    print("-" * 82)

    results = []
    try:
        for policy in policies:
            rows = [run(policy, args.scenario, seed, tag=args.tag)
                    for seed in seeds]
            wait = statistics.fmean(r["average_waiting_time"] for r in rows)
            thru = statistics.fmean(r["throughput"] for r in rows)
            queue = statistics.fmean(r["average_queue_length"] for r in rows)
            green = statistics.fmean(r["mean_green"] for r in rows)
            sd = statistics.pstdev([r["average_waiting_time"] for r in rows])
            results.append((wait, policy.name, thru, queue, green, sd))
            print("{:<42} {:>9.3f} {:>11.1f} {:>9.3f} {:>7.2f}".format(
                policy.name, wait, thru, queue, green), flush=True)
    finally:
        cleanup(args.scenario, args.tag)

    print("\nranked by mean waiting:")
    for wait, name, thru, queue, green, sd in sorted(results):
        print("  {:>8.3f}  (sd {:>5.3f})  thr {:>6.1f}   {}".format(
            wait, sd, thru, name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
