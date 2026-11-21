"""Independent recovery checks; no SUMO runs or incumbent-artifact writes.

Run from the project root with the verified interpreter and -B.
Writes only review_verification.json beside this script. The inherited
recovery test creates and removes its own temporary directory.
"""
from pathlib import Path
import json
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PRIOR = ROOT / "experiments" / "dqn_300_20261117"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PRIOR))
sys.path.insert(0, str(HERE))
import resumable_training as subject
import artifact_io


def verify_retry():
    real_replace = subject.os.replace
    real_sleep = subject.time.sleep
    checks = []
    try:
        with tempfile.TemporaryDirectory(prefix="intellisignal_lock_review_") as d:
            source = Path(d) / "snapshot.tmp"
            destination = Path(d) / "snapshot.pt"
            source.write_bytes(b"new")
            destination.write_bytes(b"old")
            calls = []

            def delayed(a, b):
                calls.append(1)
                if len(calls) < 3:
                    raise PermissionError("synthetic transient file lock")
                return real_replace(a, b)

            subject.os.replace = delayed
            subject.time.sleep = lambda _: None
            subject.replace_with_retry(source, destination, attempts=4)
            assert len(calls) == 3
            assert destination.read_bytes() == b"new" and not source.exists()
            checks.append("transient locks retried and atomic replacement succeeded")

            source.write_bytes(b"newer")

            def locked(a, b):
                raise PermissionError("synthetic persistent file lock")

            subject.os.replace = locked
            try:
                subject.replace_with_retry(source, destination, attempts=2)
            except PermissionError:
                pass
            else:
                raise AssertionError("persistent file lock did not raise")
            assert source.read_bytes() == b"newer"
            assert destination.read_bytes() == b"new"
            checks.append("exhausted retry preserves complete source and prior destination")
    finally:
        subject.os.replace = real_replace
        subject.time.sleep = real_sleep
    return checks


def main():
    result = subprocess.run(
        [sys.executable, "-B", str(PRIOR / "verify_resume.py")],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    resume = json.loads(result.stdout.strip().splitlines()[-1])
    assert resume["result"] == "passed"
    artifact_checks = []
    with tempfile.TemporaryDirectory(prefix="intellisignal_atomic_review_") as d:
        directory = Path(d)
        path = directory / "result.json"
        payload = {"summary": {}, "timeseries": []}
        artifact_io.write_json(path, payload, exclusive=True)
        original = path.read_bytes()
        artifact_io.write_json(path, payload, exclusive=True)
        assert path.read_bytes() == original
        try:
            artifact_io.write_json(path, {"different": True}, exclusive=True)
        except FileExistsError:
            pass
        else:
            raise AssertionError("different completed artifact was overwritten")
        assert path.read_bytes() == original
        artifact_checks.append("identical completed JSON is idempotent; different artifact refused")
        for name, content in [("malformed.json", b'{"summary":'), ("null.json", b'null')]:
            partial = directory / name
            partial.write_bytes(content)
            assert not artifact_io.valid_or_preserve(partial, ("summary", "timeseries"))
            assert not partial.exists()
            recovered = list((directory / "recovery").glob(name + ".invalid.*"))
            assert len(recovered) == 1 and recovered[0].read_bytes() == content
        artifact_checks.append("malformed and null partial JSON preserved before isolated retry")
    output = {
        "reviewer": "/root/review",
        "result": "passed",
        "command": r".\.venv-stage3\Scripts\python.exe -B experiments/controlled_20261119/verify_review.py",
        "resume": resume,
        "retry_checks": verify_retry(),
        "artifact_checks": artifact_checks,
        "sumo_runs": 0,
        "scope": "CPU full-state recovery and retained file-lock retry; synthetic episodes, not real SUMO replay",
    }
    path = HERE / "review_verification.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
