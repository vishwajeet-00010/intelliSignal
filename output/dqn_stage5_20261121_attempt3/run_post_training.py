"""Continue the frozen Stage 5 quality/evaluation pipeline after training."""
from __future__ import annotations

import json
import argparse
import csv
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
STATUS = OUT / "training_batch_status_final.json"
LOG = OUT / "post_training_pipeline.log"


def log(message: str) -> None:
    line = time.strftime("%Y-%m-%d %H:%M:%S") + " " + message
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run(label: str, command: list[str], cwd: Path = ROOT) -> int:
    log(f"START {label}: {' '.join(command)}")
    with LOG.open("a", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=cwd, stdout=handle,
                                stderr=subprocess.STDOUT, check=False)
    log(f"END {label}: exit={result.returncode}")
    return result.returncode


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def append_current_status(path: Path, marker: str, body: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in text:
        text = text.split(marker, 1)[0].rstrip()
    path.write_text(text + "\n\n" + marker + "\n" + body.rstrip() + "\n",
                    encoding="utf-8")


def update_handoff_docs(manifest: dict, decision: dict, checklist: dict) -> None:
    status = "technically complete; awaiting user sign-off" if checklist["technical_gates_pass"] else "incomplete; see Stage 5 findings"
    scenario_lines = []
    for scenario, value in decision.get("scenarios", {}).items():
        scenario_lines.append(f"- {scenario.title()}: {value.get('scenario_decision', 'decision unavailable')}.")
    details = "\n".join(scenario_lines) or "- Scenario decisions are unavailable."
    body = (f"## Latest Stage 5 status (21 Nov 2026)\n\n"
            f"Stage 5 is **{status}**.\n\n{details}\n\n"
            f"- Training: {manifest.get('training', {}).get('final_batch_completed', 0)}/20 cells completed; "
            f"{manifest.get('training', {}).get('final_batch_failed', 0)} failed.\n"
            f"- Tests: {manifest.get('tests', {}).get('status', 'not recorded')} "
            f"({manifest.get('tests', {}).get('result', 'result unavailable')}).\n"
            f"- Protected-file integrity: {manifest.get('integrity', {}).get('passed')}.\n"
            f"- Report: `output/dqn_stage5_20261121_attempt3/intellisignal_stage5_dqn_stability_report_attempt3.pdf`.\n"
            f"- PDF visual review and explicit user sign-off are still required. Stage 6 is not authorized.\n")
    # Preserve existing user edits in README/HANDOFF. Apply this generated
    # update only after reviewing the complete evidence and both documents.
    (OUT / "handoff_update.md").write_text(body, encoding="utf-8")


def run_artifact_evidence(scenario: str, arm: str, seed: int) -> dict:
    run_dir = OUT / scenario / arm / f"seed{seed}"
    summary_path = run_dir / "training_summary.json"
    history_path = run_dir / "training" / f"dqn_{scenario}_seed{seed}_training.csv"
    episodes = None
    summary_valid = False
    if summary_path.exists():
        try:
            summary = load_json(summary_path)
            episodes = int(summary.get("episodes", 0))
            summary_valid = episodes > 0
        except (ValueError, TypeError, OSError, json.JSONDecodeError):
            pass
    history_rows = 0
    if history_path.exists():
        try:
            with history_path.open("r", encoding="utf-8-sig", newline="") as handle:
                history_rows = sum(1 for _ in csv.DictReader(handle))
        except (OSError, csv.Error):
            pass
    best = (run_dir / "models" / "best.pt").is_file()
    validation = (run_dir / "models" / "validation_history.json").is_file()
    artifacts_exist = run_dir.exists() and any(run_dir.iterdir())
    if summary_valid and history_rows == episodes and best and validation:
        status = "complete"
    elif artifacts_exist or history_rows or episodes:
        status = "partial"
    else:
        status = "no artifacts"
    return {"status": status, "episodes": episodes, "history_rows": history_rows,
            "best_checkpoint": best, "validation_history": validation}


def write_attempt_reconciliation(jobs: list[dict], runner_alive: bool = False) -> Path:
    rows = ["# Stage 5 attempt 3 run reconciliation", "",
            "Parent: `output/dqn_stage5_20261121/`. This attempt has fresh "
            "counters and artifacts; the parent and the sandbox-blocked "
            "attempt 2 remain preserved.", "",
            f"Runner live at reconciliation: **{runner_alive}**.", "",
            "| Scenario | Arm | Seed | Batch status | Artifact status | Episodes | History rows | Best | Validation |",
            "|---|---|---:|---|---|---:|---:|---|---|"]
    counts = {"complete": 0, "partial": 0, "not_started": 0}
    for job in jobs:
        evidence = run_artifact_evidence(job["scenario"], job["arm"], int(job["seed"]))
        status = evidence["status"]
        counts[{"no artifacts": "not_started"}.get(status, status)] += 1
        rows.append(f"| {job['scenario']} | {job['arm']} | {job['seed']} | "
                    f"{job.get('status')} (exit {job.get('return_code', 'n/a')}) | "
                    f"{status} | {evidence['episodes'] if evidence['episodes'] is not None else '—'} | "
                    f"{evidence['history_rows']} | {evidence['best_checkpoint']} | "
                    f"{evidence['validation_history']} |")
    rows += ["", "Artifact totals: **{complete} complete, {partial} partial, "
             "{not_started} not started**.".format(**counts), ""]
    target = OUT / "run_reconciliation.md"
    target.write_text("\n".join(rows), encoding="utf-8")
    return target


def write_run_reconciliation(jobs: list[dict], runner_alive: bool = False) -> Path:
    rows = ["# Stage 5 run reconciliation", "",
            "This report compares the preserved batch-status snapshots with "
            "per-run summary, history, checkpoint, and validation artifacts. "
            "The raw JSON and stdout/stderr logs are not rewritten.", "",
            "## First batch snapshot (excluded pilot/replacement seeds)", "",
            "These seeds were excluded before the primary 6207–6211 run set. "
            "Status and artifact evidence are both shown because the seed-6202 "
            "summaries conflict with an archived failure/interruption record.", "",
            "| Scenario | Arm | Seed | First snapshot status | Artifact status | Episodes | History rows | Best | Validation |",
            "|---|---|---:|---|---|---:|---:|---|---|"]
    first_batch_path = OUT / "training_batch_status.json"
    if first_batch_path.exists():
        first_batch = load_json(first_batch_path).get("jobs", [])
        for job in first_batch:
            evidence = run_artifact_evidence(
                job["scenario"], job["arm"], int(job["seed"]))
            episodes = evidence["episodes"] if evidence["episodes"] is not None else "—"
            rows.append(f"| {job['scenario']} | {job['arm']} | {job['seed']} | "
                        f"{job.get('status')} (exit {job.get('return_code', 'n/a')}) | "
                        f"{evidence['status']} | {episodes} | {evidence['history_rows']} | "
                        f"{evidence['best_checkpoint']} | {evidence['validation_history']} |")
    rows += ["", "## Archived replacement batch (excluded seeds)", "",
            "The replacement log's 6202–6206 arguments are invalid under the "
            "current frozen CLI, which accepts 6207–6211. This records the "
            "archived invocation failure; it is separate from the primary "
            "6207–6211 run set.", "",
            "| Scenario | Arm | Seed | Archived status | Artifact evidence |",
            "|---|---|---:|---|---|"]
    archived_path = OUT / "training_batch_status_replacement.json"
    if archived_path.exists():
        archived = load_json(archived_path).get("jobs", [])
        for job in archived:
            evidence = run_artifact_evidence(
                job["scenario"], job["arm"], int(job["seed"]))
            summary = ("{} episodes".format(evidence["episodes"])
                       if evidence["episodes"] is not None else "no summary")
            artifacts = ("{}; history rows {}; best {}; validation {}".format(
                evidence["status"], evidence["history_rows"],
                evidence["best_checkpoint"], evidence["validation_history"]))
            rows.append(f"| {job['scenario']} | {job['arm']} | {job['seed']} | "
                        f"{job.get('status')} (exit {job.get('return_code', 'n/a')}) | "
                        f"{summary}; {artifacts} |")
    rows += ["", "## Primary batch (seeds 6207–6211)", "",
            f"Runner process observed: **{'yes' if runner_alive else 'no'}**.", "",
            "| Scenario | Arm | Seed | Snapshot status | Artifact status | Episodes | History rows | Best | Validation |",
            "|---|---|---:|---|---|---:|---:|---|---|"]
    for job in jobs:
        evidence = run_artifact_evidence(
            job["scenario"], job["arm"], int(job["seed"]))
        episodes = evidence["episodes"] if evidence["episodes"] is not None else "—"
        rows.append(f"| {job['scenario']} | {job['arm']} | {job['seed']} | "
                    f"{job.get('status')} | {evidence['status']} | {episodes} | "
                    f"{evidence['history_rows']} | {evidence['best_checkpoint']} | "
                    f"{evidence['validation_history']} |")
    counts = {"complete": 0, "partial": 0, "not_started": 0}
    for job in jobs:
        evidence_status = run_artifact_evidence(
            job["scenario"], job["arm"], int(job["seed"]))["status"]
        counts[{"no artifacts": "not_started"}.get(evidence_status,
                                                      evidence_status)] += 1
    rows += ["", "Artifact-derived primary totals: **{complete} complete, "
             "{partial} partial, {not_started} not started**. These totals "
             "come from the files above, not from the older manifest counters."
             .format(**counts), "",
             "The archived final snapshot says one complete, one running, and "
             "18 queued. The runner was not live when reconciled; the running "
             "cell has partial artifacts and is not counted complete." if not runner_alive
             else "The archived final snapshot is still running; reconcile again after the runner exits.", "",
             "A completed summary alongside a failed archived status is shown "
             "as a discrepancy; neither record is silently rewritten.", ""]
    target = OUT / "run_reconciliation.md"
    target.write_text("\n".join(rows), encoding="utf-8")
    return target


def process_is_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def runner_making_progress(batch: dict) -> bool:
    """Use artifact heartbeat because Windows may deny cross-session PID probes."""
    if process_is_alive(batch.get("runner_pid")):
        return True
    candidates = [STATUS]
    for job in batch.get("jobs", []):
        if job.get("status") == "running":
            run_dir = OUT / job["scenario"] / job["arm"] / f"seed{job['seed']}"
            candidates += [run_dir / "frozen_run_config.json",
                           run_dir / "training" / f"dqn_{job['scenario']}_seed{job['seed']}_training.csv"]
    latest = max((path.stat().st_mtime for path in candidates if path.exists()),
                 default=0.0)
    return latest > 0 and time.time() - latest < 15 * 60


def reconcile_incomplete_batch(batch: dict) -> Path:
    jobs = batch.get("jobs", [])
    runner_alive = runner_making_progress(batch)
    reconciliation_path = write_attempt_reconciliation(jobs, runner_alive)
    manifest_path = OUT / "experiment_manifest.json"
    manifest = load_json(manifest_path)
    training = manifest.setdefault("training", {})
    training.setdefault("reported_counts_before_reconciliation", {
        "completed_cells": training.get("completed_cells"),
        "failed_cells": training.get("failed_cells"),
        "status": training.get("status"),
    })
    counts = {"complete": 0, "partial": 0, "not_started": 0}
    for job in jobs:
        evidence_status = run_artifact_evidence(
            job["scenario"], job["arm"], int(job["seed"]))["status"]
        counts[{"no artifacts": "not_started"}.get(evidence_status,
                                                      evidence_status)] += 1
    training["status"] = "incomplete_live_runner" if runner_alive else "incomplete_no_live_runner"
    training["completed_cells"] = counts["complete"]
    training["partial_cells"] = counts["partial"]
    training["not_started_cells"] = counts["not_started"]
    training["failed_cells"] = 0
    manifest["status"] = "stage5_incomplete_reconciled"
    manifest["run_reconciliation"] = str(reconciliation_path.relative_to(ROOT))
    manifest["reconciliation"] = {
        "basis": "training_batch_status_final.json plus per-run summary, history, checkpoint, and validation artifacts",
        "runner_pid": batch.get("runner_pid"),
        "runner_alive": runner_alive,
        "primary_cells": counts,
        "parent_attempt": "output/dqn_stage5_20261121/",
        "preceding_attempt": "output/dqn_stage5_20261121_attempt2/",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    checklist_path = OUT / "stage5_completion_checklist.json"
    checklist_path.write_text(json.dumps({
        "training_cells_complete": counts["complete"] == 20,
        "primary_cells": counts,
        "run_reconciliation": str(reconciliation_path.relative_to(ROOT)),
        "tests_passed": "not run during status reconciliation",
        "validation_complete": False,
        "confirmatory_complete_when_required": False,
        "technical_gates_pass": False,
        "user_signoff": "pending",
        "stage6_allowed": False,
    }, indent=2), encoding="utf-8")
    return reconciliation_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconcile-only", action="store_true",
                        help="write a status/artifact reconciliation without running tests or evaluations")
    args = parser.parse_args()
    technical_pass = False
    automated_gates_pass = False
    if args.reconcile_only:
        batch = load_json(STATUS) if STATUS.exists() else {"jobs": []}
        path = reconcile_incomplete_batch(batch)
        print("Wrote status reconciliation: {}".format(path), flush=True)
        return 0

    log("Waiting for all 20 frozen Stage 5 training cells to reach a terminal state.")
    while True:
        batch = load_json(STATUS) if STATUS.exists() else {"jobs": []}
        jobs = batch.get("jobs", [])
        if STATUS.exists():
            if len(jobs) == 20 and all(j.get("status") in {"completed", "failed"}
                                        for j in jobs):
                break
            if not runner_making_progress(batch):
                path = reconcile_incomplete_batch(batch)
                log("Training runner is absent; marked Stage 5 incomplete and "
                    "wrote {}. No tests or evaluations were started.".format(path))
                return 1
        else:
            path = reconcile_incomplete_batch(batch)
            log("Training status is missing; marked Stage 5 incomplete and "
                "wrote {}.".format(path))
            return 1
        time.sleep(30)

    failures = [j for j in jobs if j.get("status") != "completed"]
    reconciliation_path = write_attempt_reconciliation(jobs)
    artifact_failures = []
    for job in jobs:
        run_dir = OUT / job["scenario"] / job["arm"] / f"seed{job['seed']}"
        summary_path = run_dir / "training_summary.json"
        if job.get("status") == "completed":
            try:
                summary = load_json(summary_path)
                valid = (int(summary.get("episodes", 0)) > 0 and
                         (run_dir / "models" / "best.pt").is_file() and
                         (run_dir / "models" / "validation_history.json").is_file())
            except (ValueError, TypeError, OSError, json.JSONDecodeError):
                valid = False
            if not valid:
                artifact_failures.append(job)
    audit_rc = run("independent training artifact audit", [sys.executable,
                   str(OUT / "audit_training.py")])
    if audit_rc != 0:
        audit = load_json(OUT / "training_artifact_audit.json")
        artifact_failures.extend(cell for cell in audit["cells"]
                                 if cell["status"] != "complete")
    failures.extend(artifact_failures)
    failed_keys = {(j["scenario"], j["arm"], int(j["seed"])) for j in failures}
    manifest_path = OUT / "experiment_manifest.json"
    manifest = load_json(manifest_path)
    manifest["training"]["final_batch_status"] = str(STATUS)
    manifest["training"]["final_batch_completed"] = len(jobs) - len(failed_keys)
    manifest["training"]["final_batch_failed"] = len(failed_keys)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    results = {"training_artifact_audit": audit_rc}
    tests_dir = OUT / "pytest-stage5-final"
    test_rc = run("full pytest suite", [sys.executable, "-m", "pytest", "-q",
                                         "--basetemp", str(tests_dir)])
    test_text = LOG.read_text(encoding="utf-8", errors="replace")
    test_summaries = re.findall(r"\d+ passed[^\r\n]*", test_text)
    test_result = test_summaries[-1] if test_summaries else "summary unavailable; inspect pipeline log"
    manifest = load_json(manifest_path)
    manifest["tests"] = {"command": f"{sys.executable} -m pytest -q --basetemp {tests_dir}",
                          "result": test_result,
                          "status": "passed" if test_rc == 0 else "failed",
                          "exit_code": test_rc}
    results["tests"] = test_rc
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not failures:
        eval_script = str(OUT / "stage5_eval.py")
        all_eval_ok = True
        for scenario in ("moderate", "heavy"):
            rc = run(f"{scenario} validation", [sys.executable, eval_script,
                      "validation", "--scenario", scenario])
            all_eval_ok = all_eval_ok and rc == 0
        results["validation"] = 0 if all_eval_ok else 1
        confirmatory_ok = True
        if all_eval_ok:
            for scenario in ("moderate", "heavy"):
                selection_path = OUT / scenario / "validation" / "validation_selection.json"
                selection = load_json(selection_path)
                if selection.get("selected_candidate") is not None:
                    rc = run(f"{scenario} confirmatory", [sys.executable, eval_script,
                              "confirmatory", "--scenario", scenario])
                    confirmatory_ok = confirmatory_ok and rc == 0
        results["confirmatory"] = 0 if confirmatory_ok else 1
        analysis_rc = run("Stage 5 decision analysis", [sys.executable,
                          str(OUT / "analyze_stage5.py")])
        results["analysis"] = analysis_rc
        integrity_rc = run("protected file integrity after", [sys.executable,
                              str(OUT / "hash_snapshot.py"), "after"])
        results["integrity_snapshot"] = integrity_rc
        before = load_json(OUT / "integrity_before.json")
        after = load_json(OUT / "integrity_after.json")
        before_map = {row["path"]: row["sha256"] for row in before["files"]}
        after_map = {row["path"]: row["sha256"] for row in after["files"]}
        integrity_passed = before_map == after_map
        manifest = load_json(manifest_path)
        manifest["integrity"].update({"after": "integrity_after.json",
                                      "passed": integrity_passed,
                                      "unchanged_file_count": len(before_map) if integrity_passed else None})
        manifest["environment"] = {"python": sys.version,
                                   "platform": sys.platform}
        manifest["pipeline_results"] = results
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        report_rc = run("build Stage 5 PDF", [sys.executable, str(OUT / "build_report.py")])
        results["report"] = report_rc
        report_path = OUT / "intellisignal_stage5_dqn_stability_report_attempt3.pdf"
        rendered = False
        converter = shutil.which("pdftoppm")
        if report_rc == 0 and report_path.exists() and converter:
            render_dir = OUT / "pdf_render"
            render_dir.mkdir(parents=True, exist_ok=True)
            render_rc = run("render every PDF page for visual QA",
                            [converter, "-png", "-r", "130", str(report_path),
                             str(render_dir / "page")])
            rendered = render_rc == 0 and any(render_dir.glob("page-*.png"))
            results["pdf_render"] = render_rc
        elif not converter:
            results["pdf_render"] = "pdftoppm unavailable"

        decision_path = OUT / "stage5_decision.json"
        decision = load_json(decision_path) if decision_path.exists() else {}
        automated_gates_pass = (not failures and test_rc == 0 and all_eval_ok and
                                confirmatory_ok and analysis_rc == 0 and
                                integrity_rc == 0 and integrity_passed and
                                report_rc == 0 and report_path.exists() and rendered)
        checklist = {
            "training_cells_complete": not failures,
            "tests_passed": test_rc == 0,
            "validation_complete": bool(results.get("validation") == 0),
            "confirmatory_complete_when_required": confirmatory_ok,
            "protected_integrity_passed": integrity_passed,
            "decision_analysis_created": analysis_rc == 0 and decision_path.exists(),
            "pdf_created": report_rc == 0 and report_path.exists(),
            "pdf_pages_rendered_for_review": rendered,
            "pdf_visual_review": "pending_manual_review",
            "automated_gates_pass": automated_gates_pass,
            "technical_gates_pass": technical_pass,
            "user_signoff": "pending_explicit_confirmation",
            "stage6_allowed": False,
        }
        (OUT / "stage5_completion_checklist.json").write_text(
            json.dumps(checklist, indent=2), encoding="utf-8")
        manifest["report"] = str(report_path) if report_path.exists() else None
        manifest["pipeline_results"] = results
        manifest["status"] = "stage5_awaiting_pdf_visual_review" if automated_gates_pass else "stage5_incomplete_review_required"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        update_handoff_docs(manifest, decision, checklist)
    else:
        log("Training failures recorded; evaluation and report generation require manual review.")

    manifest = load_json(manifest_path)
    if failures:
        checklist = {
            "training_cells_complete": False,
            "incomplete_training_cells": failures,
            "tests_passed": test_rc == 0,
            "validation_complete": False,
            "confirmatory_complete_when_required": False,
            "run_reconciliation": str(reconciliation_path),
            "technical_gates_pass": False,
            "user_signoff": "pending_explicit_confirmation",
            "stage6_allowed": False,
        }
        (OUT / "stage5_completion_checklist.json").write_text(
            json.dumps(checklist, indent=2), encoding="utf-8")
        decision = {}
        update_handoff_docs(manifest, decision, checklist)
        manifest["status"] = "stage5_incomplete_review_required"
        manifest["run_reconciliation"] = str(reconciliation_path)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    else:
        manifest["run_reconciliation"] = str(reconciliation_path)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log("Post-training pipeline reached its final automated step. Review the log and PDF visually.")
    return 0 if automated_gates_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
