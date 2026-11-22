"""Confirm a shortlist of fixed plans on the confirmatory seeds, via SUMO's TLS."""
import argparse, json, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from evaluation.fixed_time_sweep import (cleanup_generated_inputs,
                                         parse_number_list, run_fixed_time_plan)

p = argparse.ArgumentParser()
p.add_argument("--scenario", required=True)
p.add_argument("--plans", required=True, help="ns:ew,ns:ew,...")
p.add_argument("--seeds", default="6101-6120")
p.add_argument("--tag", default="cf")
p.add_argument("--out", required=True)
a = p.parse_args()

seeds = [int(s) for s in parse_number_list(a.seeds)]
out = []
try:
    for plan in a.plans.split(","):
        ns, ew = (float(v) for v in plan.split(":"))
        rows = [run_fixed_time_plan(a.scenario, ns, s, ew_green=ew, tag=a.tag)
                for s in seeds]
        waits = [r["average_waiting_time"] for r in rows]
        rec = {"ns": ns, "ew": ew,
               "wait": round(statistics.fmean(waits), 3),
               "sd": round(statistics.pstdev(waits), 3),
               "throughput": round(statistics.fmean(r["throughput"] for r in rows), 1),
               "per_seed": waits}
        out.append(rec)
        print("ns={:g} ew={:g}  wait {:.3f} (sd {:.3f})  thr {:.1f}".format(
            ns, ew, rec["wait"], rec["sd"], rec["throughput"]), flush=True)
finally:
    cleanup_generated_inputs(a.scenario, a.tag)
json.dump(out, open(a.out, "w"), indent=2)
print("saved", a.out)
