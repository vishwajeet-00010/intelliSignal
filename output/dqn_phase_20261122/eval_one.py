"""Evaluate one duration-choosing checkpoint on the confirmatory seeds.

One model per process, so the candidates can be measured in parallel without
their generated SUMO inputs colliding. The fixed-time baseline is not re-run
here - it is a constant for these seeds (ns=7/ew=6, 5.754 s, throughput
1466.5) and `final_table.py` measures it again for the published table.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
for path in (PROJECT_ROOT, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from evaluation.evaluate_phase_dqn import evaluate_agent  # noqa: E402
from probe import parse_seeds                            # noqa: E402

FIXED_76 = 5.754


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--scenario", default="heavy")
    parser.add_argument("--seeds", default="6101-6120")
    parser.add_argument("--baseline", type=float, default=FIXED_76)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    torch.set_num_threads(1)
    seeds = [int(s) for s in parse_seeds(args.seeds)]
    result = evaluate_agent(args.model, args.scenario, seeds,
                            tag="_ev" + args.name)
    result["name"] = args.name
    result["baseline_mean_waiting_time"] = args.baseline
    # Write the result BEFORE deriving anything from it. The measurement is
    # twenty SUMO episodes; a formatting mistake must not be able to throw
    # it away.
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=float)

    if args.baseline > 0:
        delta = 100.0 * (result["mean_waiting_time"] - args.baseline) / args.baseline
        result["delta_percent"] = delta
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, default=float)
        print("\n{}: wait {:.3f} (sd {:.3f})  throughput {:.1f}  "
              "vs baseline {:.3f}  -> {:+.1f}%".format(
                  args.name, result["mean_waiting_time"],
                  result["sd_waiting_time"], result["mean_throughput"],
                  args.baseline, delta))
    else:
        print("\n{}: wait {:.3f} (sd {:.3f})  throughput {:.1f}".format(
            args.name, result["mean_waiting_time"],
            result["sd_waiting_time"], result["mean_throughput"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
