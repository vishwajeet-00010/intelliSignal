"""Run the frozen final test with at most one worker per traffic scenario.

Each worker uses one Torch thread and scenario-private generated SUMO paths.
No training or selection occurs here. Existing per-seed results are preserved.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import final_evaluate as final
from artifact_io import write_json

HERE = Path(__file__).resolve().parent


def main():
    frozen = json.loads((HERE / "freeze.json").read_text(encoding="utf-8"))
    final.verify_freeze(frozen)
    assert frozen["execution"]["max_concurrent_scenario_workers"] == 2
    assert frozen["execution"]["torch_threads_per_worker"] == 1
    import msvcrt
    with (HERE / "final.lock").open("a+b") as lock:
        lock.seek(0)
        lock.write(b"0")
        lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            attempt = str(time.time_ns())
            children = []
            launch = {"launcher_pid": os.getpid(), "attempt": attempt,
                      "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "freeze_sha256": final.base.digest(HERE / "freeze.json"),
                      "workers": []}
            for scenario in final.base.SCENARIOS:
                command = [sys.executable, "-B", "-u", str(HERE / "final_evaluate.py"),
                           "--evaluate", "--scenario", scenario]
                log_path = HERE / ("final-" + scenario + ".log")
                log = log_path.open("a", encoding="utf-8")
                log.write("\nFINAL ATTEMPT " + attempt + "\n")
                log.flush()
                try:
                    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                             cwd=str(final.base.ROOT),
                                             creationflags=subprocess.CREATE_NO_WINDOW)
                except Exception:
                    log.close()
                    raise
                children.append((scenario, child, log))
                launch["workers"].append({"scenario": scenario, "pid": child.pid,
                                          "command": command, "log": log_path.name})
                print("FINAL_WORKER", scenario, child.pid, log_path.name, flush=True)
            write_json(HERE / "launches" / ("final_" + attempt + ".json"), launch, True)
            outcome = {"attempt": attempt, "worker_exit_codes": {}}
            for scenario, child, log in children:
                code = child.wait()
                log.close()
                outcome["worker_exit_codes"][scenario] = code
                print("FINAL_WORKER_DONE", scenario, code, flush=True)
            outcome["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            outcome["all_workers_succeeded"] = all(code == 0 for code in outcome["worker_exit_codes"].values())
            write_json(HERE / "launches" / ("final_" + attempt + "_outcome.json"), outcome, True)
            if not outcome["all_workers_succeeded"]:
                raise RuntimeError("A final worker failed; per-seed artifacts and logs are preserved.")
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == "__main__":
    main()
