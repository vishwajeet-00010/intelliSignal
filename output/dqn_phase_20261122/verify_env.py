"""Check that the learning environment reproduces the fixed-time baseline.

A learned controller is only comparable to a fixed-time plan if the
environment it learns in produces the same traffic as the harness the
baseline was measured in. So: drive `PhaseDurationEnv` with a scripted
policy that always asks for 7 s north-south and 6 s east-west, and check it
lands on the number `evaluation/fixed_time_sweep.py` reports for the same
plan and the same seeds (5.754 s over 6101-6120).

It also prints the size of the reward, which is what `--reward-scale` has to
be chosen against: a per-decision reward in the tens makes the Q values
hundreds, and Huber loss on hundreds is most of why the previous heavy run
diverged.
"""

from __future__ import annotations

import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl.phase_action import PHASE_NS                       # noqa: E402
from rl.phase_environment import PhaseDurationEnv          # noqa: E402
from rl.phase_state import PhaseStateExtractor             # noqa: E402
from utils.metrics import MetricsCollector                 # noqa: E402

DURATIONS = (4, 5, 6, 7, 8, 9, 10, 12, 15, 20)
NS_INDEX = DURATIONS.index(7)
EW_INDEX = DURATIONS.index(6)


def main() -> int:
    seeds = [6101 + i for i in range(10)]
    env = PhaseDurationEnv(scenario="heavy", durations=DURATIONS,
                           state_extractor=PhaseStateExtractor(
                               max_green_seconds=max(DURATIONS)),
                           reward_scale=1.0, tag="_verify")
    waits, throughputs, rewards, step_costs = [], [], [], []
    try:
        for seed in seeds:
            env.reset(seed=seed)
            collector = MetricsCollector(step_length=env.step_length,
                                         conn=env.connection)
            env.step_callback = collector.collect_step
            total_reward = 0.0
            while True:
                action = (NS_INDEX if env.controller.phase == PHASE_NS
                          else EW_INDEX)
                _, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step_costs.append(-reward / max(info["duration"], 1.0))
                if terminated or truncated:
                    break
            summary = collector.summary("heavy")
            waits.append(float(summary["average_waiting_time"]))
            throughputs.append(float(summary["throughput"]))
            rewards.append(total_reward)
            print("seed {}  wait {:>7.3f}  throughput {:>6}  "
                  "episode reward (scale 1) {:>12.1f}".format(
                      seed, summary["average_waiting_time"],
                      summary["throughput"], total_reward), flush=True)
    finally:
        env.close()
        env.cleanup_generated_inputs()

    mean_wait = statistics.fmean(waits)
    print("\nscripted ns=7 / ew=6 inside PhaseDurationEnv")
    print("  mean waiting   : {:.3f} s  (sd {:.3f})".format(
        mean_wait, statistics.pstdev(waits)))
    print("  mean throughput: {:.1f}".format(statistics.fmean(throughputs)))
    print("  fixed_time_sweep over 6101-6120 reports 5.754 s / 1466.5")
    print("  difference     : {:+.3f} s".format(mean_wait - 5.754))
    print("\nreward sizing, at scale 1.0")
    print("  cost per simulated second : {:.1f}".format(
        statistics.fmean(step_costs)))
    print("  -> a decision of ~11 s costs about {:.0f}".format(
        statistics.fmean(step_costs) * 11))
    for scale in (0.01, 0.005, 0.002, 0.001):
        print("     reward_scale {:<6g} -> per-decision reward about {:>7.2f}"
              .format(scale, -scale * statistics.fmean(step_costs) * 11))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
