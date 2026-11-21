"""Finalize Stage 4's manifest from the evaluation artifacts."""
from __future__ import annotations

import importlib.metadata
import json
import platform
from pathlib import Path

OUT = Path(__file__).resolve().parent


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    manifest_path = OUT / "experiment_manifest.json"
    manifest = read(manifest_path)
    validation = {}
    actuated = {}
    for scenario in ("moderate", "heavy"):
        selection_path = OUT / scenario / "validation" / "validation_selection.json"
        selection = read(selection_path)
        validation[scenario] = {
            "selection_file": str(selection_path),
            "candidate_checks": selection["candidate_checks"],
            "all_independent_candidates_pass": selection["all_independent_candidates_pass"],
            "confirmatory_candidate_allowed": selection["confirmatory_candidate_allowed"],
            "selected_candidate_seed": selection["selected_candidate_seed"],
            "observation_stats": selection["observation_stats"],
        }
        actuated[scenario] = selection["selected_actuated"]
        manifest.setdefault("confirmatory", {}).setdefault(scenario, {})[
            "rows_csv"] = str(OUT / scenario / "confirmatory" / "confirmatory_rows.csv")
    manifest["validation"] = validation
    manifest["actuated_baseline"] = actuated
    before = read(OUT / "integrity_before.json")
    after = read(OUT / "integrity_after.json")
    before_map = {row["path"]: row["sha256"] for row in before["files"]}
    after_map = {row["path"]: row["sha256"] for row in after["files"]}
    manifest["integrity"] = {
        "before": str(OUT / "integrity_before.json"),
        "after": str(OUT / "integrity_after.json"),
        "protected_files_before": len(before_map),
        "protected_files_after": len(after_map),
        "changed_files": sorted(path for path in before_map.keys() & after_map.keys()
                                 if before_map[path] != after_map[path]),
        "missing_after": sorted(before_map.keys() - after_map.keys()),
        "new_after": sorted(after_map.keys() - before_map.keys()),
        "passed": before_map == after_map,
    }
    manifest["tests"] = {
        "command": ".venv-stage3\\Scripts\\python.exe -m pytest -q --basetemp tmp/pytest-stage4-final-20261121",
        "result": "224 passed, 1 skipped",
        "duration_seconds": 102.0,
        "status": "passed",
    }
    manifest["environment"] = {
        "python": platform.python_version(),
        "sumo": "1.28.0",
        "packages": {name: importlib.metadata.version(name) for name in
                      ("pytest", "torch", "gymnasium", "numpy", "reportlab", "pypdf")},
    }
    manifest["commands"] = [
        ".venv-stage3\\Scripts\\python.exe -m pytest -q --basetemp tmp/pytest-stage4-final-20261121",
        ".venv-stage3\\Scripts\\python.exe output\\dqn_stage4_20261120\\stage4_eval.py validation --scenario moderate",
        ".venv-stage3\\Scripts\\python.exe output\\dqn_stage4_20261120\\stage4_eval.py validation --scenario heavy",
        ".venv-stage3\\Scripts\\python.exe output\\dqn_stage4_20261120\\stage4_eval.py confirmatory --scenario moderate",
        ".venv-stage3\\Scripts\\python.exe output\\dqn_stage4_20261120\\stage4_eval.py confirmatory --scenario heavy",
        ".venv-stage3\\Scripts\\python.exe output\\dqn_stage4_20261120\\analyze_confirmatory.py",
    ]
    manifest["report"] = str(OUT / "stage4_validation_report.pdf")
    manifest["status"] = "complete_no_candidate_promoted"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Finalized {manifest_path}")
    print(json.dumps(manifest["integrity"], indent=2))


if __name__ == "__main__":
    main()
