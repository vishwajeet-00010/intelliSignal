"""Hash protected model/config/baseline files for the Stage 4 integrity check."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def snapshot(label: str) -> Path:
    protected = []
    for folder in (ROOT / "models", ROOT / "results" / "csv"):
        protected.extend(path for path in folder.rglob("*")
                         if path.is_file() and path.suffix.lower() in
                         {".pt", ".pth", ".csv", ".json"})
    protected.extend((ROOT / "config" / name for name in
                      ("moderate.yaml", "heavy.yaml")))
    protected.extend((OUT.parent / "dqn_retraining_20261120" / "models" /
                      f"dqn_{scenario}" / "best.pt" for scenario in
                      ("moderate", "heavy")))
    rows = []
    for path in sorted(set(p.resolve() for p in protected)):
        if path.is_file():
            rows.append({"path": str(path), "sha256": digest(path),
                         "size_bytes": path.stat().st_size})
    target = OUT / f"integrity_{label}.json"
    target.write_text(json.dumps({"label": label, "file_count": len(rows),
                                  "files": rows}, indent=2), encoding="utf-8")
    print(f"Wrote {target} ({len(rows)} protected files)")
    return target


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"before", "after"}:
        raise SystemExit("usage: hash_snapshot.py before|after")
    snapshot(sys.argv[1])
