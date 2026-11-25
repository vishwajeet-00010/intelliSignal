"""
IntelliSignal
tools/clean_workspace.py

Remove what the project regenerates, and nothing else.

    python tools/clean_workspace.py              # say what would go
    python tools/clean_workspace.py --apply      # do it, with a backup

What this will and will not touch
---------------------------------
**It never deletes a file git is tracking.** Committed results, reports,
figures and promoted checkpoints are the project's record; removing them
would not even shrink the repository, because the objects stay in its
history. Everything swept here is untracked and either regenerable or a
by-product of a run that has already been read.

**It never deletes a registered model.** The paths in
`models/active_models.json` are resolved first and excluded by name, so
a promoted checkpoint cannot be caught by a glob - including the member
files of a promoted ensemble.

**It refuses to run while anything is simulating.** SUMO reads its input
files throughout a run and writes its trip information at the end, so a
sweep during training would kill runs and silently empty the delay
accounting of others.

The categories, each of which can be skipped
--------------------------------------------
    tripinfo    SUMO trip files the harness has already read and deleted;
                what remains is the leftovers of workers that were killed
    sumo        generated route, config and traffic-light files for runs
                that have finished - every one is rebuilt on demand
    pycache     compiled Python and the pytest cache
    weights     intermediate `checkpoint_ep*.pt` and `final.pt` from runs
                that were never promoted, where `best.pt` is kept
    scratch     pilots and raw per-episode logs of finished experiments

Everything removed is first written to a dated zip OUTSIDE the
repository, so a mistake costs one unzip.
"""

from __future__ import annotations

import argparse
import datetime
import fnmatch
import json
import os
import subprocess
import sys
import zipfile
from typing import Dict, Iterable, List, Set, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

BACKUP_DIR = os.path.join(os.path.dirname(PROJECT_ROOT),
                          "intelliSignal_cleanup_backups")

#: Never swept, whatever a pattern says.
PROTECTED_GLOBS = (
    "models/*",                      # the registry and the Stage 3 models
    "sumo/network/*",                # the hand-built network
    "config/*",
    "*/FINDINGS.md", "*/REPORT.md", "*/CORRECTION.md", "*/PARKED.md",
    "*/HANDOFF.md", "*/protocol.md", "*/audit.md",
    "*/summary.json", "*/validation_history.json", "*/metadata.json",
    "*/selection.json", "*/experiment_manifest.json",
    "*.csv", "*.md", "*.pdf", "*.png", "*.yaml", "*.yml", "*.py",
)


# ---------------------------------------------------------------------------
def tracked_files() -> Set[str]:
    out = subprocess.run(["git", "ls-files"], cwd=PROJECT_ROOT,
                         capture_output=True, text=True, check=False)
    return {line.strip().replace("\\", "/") for line in out.stdout.splitlines()
            if line.strip()}


def registered_paths() -> Set[str]:
    """Every checkpoint any registry section points at, as repo paths."""
    found: Set[str] = set()
    path = os.path.join(PROJECT_ROOT, "models", "active_models.json")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return found
    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "path" and isinstance(value, str):
                    found.add(value.replace("\\", "/"))
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)
    walk(payload)
    return found


def busy() -> List[str]:
    """Anything running that would be hurt by a sweep."""
    reasons = []
    try:
        out = subprocess.run(["tasklist"], capture_output=True, text=True,
                             check=False).stdout.lower()
        if "sumo.exe" in out:
            reasons.append("SUMO is running")
    except OSError:
        pass
    try:
        from simulation.hybrid_job import active_jobs
        if active_jobs():
            reasons.append("a Hybrid search is running")
    except Exception:
        pass
    return reasons


def is_protected(relative: str) -> bool:
    return any(fnmatch.fnmatch(relative, pattern)
               for pattern in PROTECTED_GLOBS)


def walk_files(root: str) -> Iterable[str]:
    base = os.path.join(PROJECT_ROOT, root)
    if not os.path.isdir(base):
        return
    for folder, dirs, names in os.walk(base):
        dirs[:] = [d for d in dirs if d not in (".git", ".venv-stage3")]
        for name in names:
            full = os.path.join(folder, name)
            yield os.path.relpath(full, PROJECT_ROOT).replace("\\", "/")


# ---------------------------------------------------------------------------
def collect(skip: Set[str], keep_newer_than: float) -> Dict[str, List[str]]:
    """The files each category would remove, already filtered."""
    tracked = tracked_files()
    registered = registered_paths()
    found: Dict[str, List[str]] = {}

    def add(name: str, paths: Iterable[str]) -> None:
        if name in skip:
            return
        chosen = []
        for relative in paths:
            if relative in tracked or relative in registered:
                continue
            if is_protected(relative):
                continue
            full = os.path.join(PROJECT_ROOT, relative)
            try:
                if os.path.getmtime(full) > keep_newer_than:
                    continue
            except OSError:
                continue
            chosen.append(relative)
        if chosen:
            found[name] = sorted(chosen)

    # --- SUMO trip information, already read ---------------------------
    add("tripinfo", (p for p in walk_files("results/tripinfo")
                     if p.endswith(".xml")))

    # --- generated SUMO inputs -----------------------------------------
    def generated():
        for folder in ("sumo/routes", "sumo/config", "sumo/additional"):
            for relative in walk_files(folder):
                name = os.path.basename(relative)
                if (name.startswith("custom_") or name.startswith("stress_")
                        or "_rl" in name or "_phase_rl" in name
                        or "_ext_rl" in name or name.startswith("car.")):
                    yield relative
    add("sumo", generated())

    # --- compiled Python ------------------------------------------------
    def caches():
        for folder, dirs, names in os.walk(PROJECT_ROOT):
            dirs[:] = [d for d in dirs if d != ".venv-stage3"]
            if os.path.basename(folder) in ("__pycache__", ".pytest_cache"):
                for name in names:
                    yield os.path.relpath(
                        os.path.join(folder, name),
                        PROJECT_ROOT).replace("\\", "/")
    add("pycache", caches())

    # --- weights nothing points at ---------------------------------------
    def weights():
        for folder in ("output", "experiments"):
            for relative in walk_files(folder):
                name = os.path.basename(relative)
                if not name.endswith(".pt"):
                    continue
                if name.startswith("checkpoint_ep") or name == "final.pt" \
                        or name.startswith("resume"):
                    yield relative
    add("weights", weights())

    # --- pilots and raw logs of finished experiments ---------------------
    def scratch():
        for relative in walk_files("output"):
            parts = relative.split("/")
            if any(part.startswith("pilot") for part in parts[1:]):
                yield relative
            elif relative.endswith(".log") and "/logs/" in relative:
                yield relative
            elif relative.endswith(".log") and parts[-1] in (
                    "train.log", "train_heavy.log", "eval.log", "asym.log",
                    "verify76.log"):
                yield relative
    add("scratch", scratch())
    return found


def total_bytes(paths: Iterable[str]) -> int:
    size = 0
    for relative in paths:
        try:
            size += os.path.getsize(os.path.join(PROJECT_ROOT, relative))
        except OSError:
            pass
    return size


# ---------------------------------------------------------------------------
def back_up(found: Dict[str, List[str]], stamp: str) -> str:
    os.makedirs(BACKUP_DIR, exist_ok=True)
    path = os.path.join(BACKUP_DIR, "cleanup_{}.zip".format(stamp))
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        for name, paths in sorted(found.items()):
            for relative in paths:
                full = os.path.join(PROJECT_ROOT, relative)
                if os.path.isfile(full):
                    archive.write(full, relative)
        archive.writestr("MANIFEST.json", json.dumps(
            {name: paths for name, paths in sorted(found.items())}, indent=1))
    return path


def remove(found: Dict[str, List[str]]) -> Tuple[int, int]:
    removed = failed = 0
    for paths in found.values():
        for relative in paths:
            try:
                os.remove(os.path.join(PROJECT_ROOT, relative))
                removed += 1
            except OSError:
                failed += 1
    # take the empty directories with them
    for folder in ("results/tripinfo", "sumo/routes", "sumo/config",
                   "sumo/additional", "output", "experiments"):
        base = os.path.join(PROJECT_ROOT, folder)
        for root, dirs, names in os.walk(base, topdown=False):
            if not dirs and not names and os.path.abspath(root) != \
                    os.path.abspath(base):
                try:
                    os.rmdir(root)
                except OSError:
                    pass
    return removed, failed


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="actually delete; without it nothing changes")
    parser.add_argument("--skip", default="",
                        help="categories not to touch, comma separated")
    parser.add_argument("--min-age-minutes", type=float, default=30.0,
                        help="leave anything written more recently alone")
    parser.add_argument("--force", action="store_true",
                        help="sweep even while something is simulating")
    args = parser.parse_args(argv)

    import time

    reasons = busy()
    if reasons and not args.force:
        print("Refusing to sweep: {}.".format(", ".join(reasons)))
        print("A run reads its input files throughout and writes its trip "
              "information at the end, so a sweep now would break it.")
        return 1

    cutoff = time.time() - args.min_age_minutes * 60
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    found = collect(skip, cutoff)
    if not found:
        print("Nothing to clean.")
        return 0

    print("{:<10} {:>7} {:>10}".format("category", "files", "MB"))
    print("-" * 30)
    grand = 0
    for name, paths in sorted(found.items()):
        size = total_bytes(paths)
        grand += size
        print("{:<10} {:>7} {:>10.1f}".format(name, len(paths), size / 1e6))
    print("-" * 30)
    print("{:<10} {:>7} {:>10.1f}".format(
        "total", sum(len(p) for p in found.values()), grand / 1e6))

    if not args.apply:
        print("\nNothing was deleted. Add --apply to do it.")
        for name, paths in sorted(found.items()):
            print("\n{} - first few of {}:".format(name, len(paths)))
            for relative in paths[:5]:
                print("   {}".format(relative))
        return 0

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = back_up(found, stamp)
    print("\nbacked up to {} ({:.1f} MB)".format(
        archive, os.path.getsize(archive) / 1e6))
    removed, failed = remove(found)
    print("removed {} files, {} could not be removed".format(removed, failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
