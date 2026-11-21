"""Independent arithmetic, action-safety and observation audit of saved runs.

No traffic simulation is started. Reads validation/final JSON and writes
review_data_audit.json. Run with --partition validation or final after runs.
"""
from pathlib import Path
import argparse
import json
import math
import statistics
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from agents.dqn_agent import DQNAgent

AGENTS = {}


def close(actual, expected, tol=1e-6):
    assert math.isclose(float(actual), float(expected), abs_tol=tol, rel_tol=tol), (actual, expected)


def verify(path):
    data = json.loads(path.read_text())
    summary = data["summary"]
    assert summary["seed"] == int(path.stem)
    assert summary["policy"] == path.parent.name
    rows = data.get("decisions", [])
    assert summary["simulation_time"] == 1800
    timeseries = data["timeseries"]
    assert len(timeseries) == 1800
    assert [r["time"] for r in timeseries] == list(range(1, 1801))
    assert summary["throughput"] == timeseries[-1]["arrived_cumulative"]
    assert summary["max_queue_length"] == max(r["queue_length"] for r in timeseries)
    close(summary["average_queue_length"], round(sum(r["queue_length"] for r in timeseries) / 1800, 2))
    assert summary["vehicles_still_in_network"] == max(summary["vehicles"] - summary["throughput"], 0)
    for name in ["average_waiting_time", "average_queue_length", "throughput", "max_queue_length", "vehicles_departed", "vehicles_still_in_network", "teleports"]:
        assert math.isfinite(float(summary[name]))
    if not rows:
        return {"path": str(path.relative_to(HERE)), "decisions": 0, "controller": "fixed_time"}
    assert len(rows) == 180
    assert summary["rng_unchanged"]
    manifest = json.loads((HERE / "manifest.json").read_text())
    policy = summary["policy"]
    is_untrained = policy in {"untrained", "untrained_historical0"}
    initialization_seed = 0 if policy == "untrained_historical0" else 42
    matches = [e for e in manifest["checkpoints"] if e["scenario"] == summary["scenario"] and e["policy"] == policy]
    checkpoint_path = ROOT / matches[0]["path"] if matches else None
    if checkpoint_path is None and not is_untrained:
        frozen = json.loads((HERE / "freeze.json").read_text())
        frozen_entries = [e for e in frozen["entries"][summary["scenario"]] if e["label"] == policy]
        assert len(frozen_entries) == 1, "Missing frozen policy provenance"
        checkpoint_path = ROOT / frozen_entries[0]["path"]
    gamma = (torch.load(checkpoint_path, map_location="cpu", weights_only=False)["config"]["gamma"]
             if checkpoint_path else 0.9)
    key = (summary["scenario"], policy)
    if key not in AGENTS:
        AGENTS[key] = (DQNAgent.load(str(checkpoint_path), scenario=key[0])
                       if checkpoint_path else DQNAgent(observation_size=10, scenario=key[0], seed=initialization_seed, trainable=False))
    agent = AGENTS[key]
    observations = torch.tensor([r["observation"] for r in rows], dtype=torch.float32)
    with torch.no_grad():
        actual_q = agent.policy_net(observations).numpy()
        actual_next_targets = agent.target_net(observations[1:]).max(dim=1).values.numpy()
    assert np.allclose(actual_q, [[r["q_keep"], r["q_switch"]] for r in rows], atol=1e-5, rtol=1e-6)
    # The final next-observation is not persisted. Remaining 179 target values
    # are reproducible directly from the next decision's saved observation.
    assert np.allclose(actual_next_targets, [r["target_next_max"] for r in rows[:-1]], atol=1e-5, rtol=1e-6)
    for i, row in enumerate(rows):
        values = [row["q_keep"], row["q_switch"]]
        requested = int(np.argmax(values))
        assert requested == row["requested_action"]
        elapsed = row["green_elapsed"]
        blocked = requested == 1 and elapsed < 10
        forced = requested == 0 and elapsed >= 60
        effective = 0 if blocked else (1 if forced else requested)
        assert row["switch_blocked"] == blocked
        assert row["forced_switch"] == forced
        assert row["effective_action"] == effective
        assert row["probes_requested"][policy] == requested
        assert row["probes_effective"][policy] == effective
        assert np.allclose(row["observation"], np.clip(row["raw_normalized"], 0, 1), atol=1e-7)
        close(row["reward"], row["waiting_penalty"] + row["queue_penalty"] + row["fairness_penalty"])
        # Execution subtracts NumPy float32 Q from a scalar target. NumPy2
        # scalar promotion rounds that operation to float32; use an explicit
        # scale-aware float32 error bound for the independent float64 check.
        target = row["reward"] + gamma * row["target_next_max"] * (not row["terminated"])
        td_tolerance = 4 * np.finfo(np.float32).eps * max(1, abs(target), abs(values[requested]))
        close(row["td_error_offline"], target - values[requested], tol=td_tolerance)
        if i + 1 < len(rows):
            following = rows[i + 1]
            close(following["green_elapsed"], 6 if effective else elapsed + 10)
            assert following["phase"] == (1 - row["phase"] if effective else row["phase"])
            close(following["time"], row["time"] + 10)
    close(summary["episode_return"], sum(r["reward"] for r in rows))
    close(summary["queue_unrounded"], -sum(r["queue_penalty"] for r in rows) / 1.8)
    assert not any(r["terminated"] or r["truncated"] for r in rows[:-1])
    assert rows[-1]["truncated"]
    return {"path": str(path.relative_to(HERE)), "decisions": len(rows), "controller": policy}


def verify_final_decision():
    """Recompute final means and guardrails from persisted paired records."""
    frozen = json.loads((HERE / "freeze.json").read_text())
    reported = json.loads((HERE / "final_decision.json").read_text())
    fields = ["average_waiting_time", "average_queue_length", "max_queue_length", "throughput", "teleports", "vehicles_departed", "vehicles_still_in_network"]

    def value(summary, field):
        if field == "average_waiting_time" and "waiting_unrounded" in summary:
            return summary["waiting_unrounded"]
        if field == "average_queue_length" and "queue_unrounded" in summary:
            return summary["queue_unrounded"]
        return summary[field]

    audited = {}
    for scenario, entries in frozen["entries"].items():
        groups = {}
        for label in [e["label"] for e in entries] + ["fixed_time", "untrained", "untrained_historical0"]:
            groups[label] = [json.loads((HERE / "final" / scenario / label / f"{seed}.json").read_text())["summary"] for seed in range(3001, 3021)]
        selected_labels = [e["label"] for e in entries if e.get("role") == "selected"]
        control_labels = [e["label"] for e in entries if e.get("role") == "control"]

        def aggregate(labels, field):
            return [statistics.fmean(value(groups[label][i], field) for label in labels) for i in range(20)]

        for role, labels in [("selected", selected_labels), ("incumbent", ["incumbent"]), ("fixed_time", ["fixed_time"]), ("untrained", ["untrained"]), ("untrained_historical0", ["untrained_historical0"])] + ([("control", control_labels)] if control_labels else []):
            for field in fields:
                close(reported[scenario]["means"][role][field], statistics.fmean(aggregate(labels, field)))
        comparisons = [("incumbent", ["incumbent"]), ("fixed_time", ["fixed_time"]), ("untrained", ["untrained_historical0"])] + ([("control", control_labels)] if control_labels else [])
        for role, reference_labels in comparisons:
            output = reported[scenario][role + "_comparison"]
            delta = {field: [x-y for x,y in zip(aggregate(selected_labels, field), aggregate(reference_labels, field))] for field in fields}
            for field in fields:
                close(output["mean_deltas"][field], statistics.fmean(delta[field]))
            wins = sum(d < -1e-10 for d in delta["average_waiting_time"])
            checks = {"lower_waiting": statistics.fmean(delta["average_waiting_time"]) < -1e-10, "waiting_wins_more_than_half": wins > 10, "queue_no_worse": statistics.fmean(delta["average_queue_length"]) <= 1e-10, "throughput_no_worse": statistics.fmean(delta["throughput"]) >= -1e-10, "teleports_no_worse": statistics.fmean(delta["teleports"]) <= 1e-10}
            assert output["checks"] == checks and output["passes"] == all(checks.values())
            assert output["waiting_wins"] == wins
        if not control_labels:
            eligible = frozen["selection_eligibility"][scenario]["validation_eligible"]
            expected = eligible and reported[scenario]["incumbent_comparison"]["passes"]
            assert reported[scenario]["promotion_criteria_passed"] == expected
            assert reported[scenario]["training_uncertainty"] is None
        audited[scenario] = {"traffic_seeds": 20, "learned_training_replicates": len(selected_labels), "means_and_paired_guardrails_recomputed": True}
    return audited


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=["validation", "final"], default="validation")
    args = parser.parse_args()
    paths = sorted((HERE / args.partition).glob("*/*/*.json"))
    assert paths, "No completed run artifacts exist yet"
    outputs = [verify(p) for p in paths]
    result = {"reviewer": "/root/review", "partition": args.partition, "result": "passed", "runs": len(outputs), "decisions": sum(r["decisions"] for r in outputs), "checks": ["1800-second metric/time-series coverage", "greedy request and safety override accounting", "clipping from raw observation", "reward components and exact queue relationship", "offline TD arithmetic with termination-only mask", "all saved Q values and 179/180 next-target values independently recomputed from checkpoints", "phase/elapsed transitions and end-of-episode mask"], "records": outputs}
    if args.partition == "final" and (HERE / "final_decision.json").exists():
        result["decision_audit"] = verify_final_decision()
    path = HERE / ("review_data_audit_" + args.partition + ".json")
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k != "records"}))


if __name__ == "__main__":
    main()
