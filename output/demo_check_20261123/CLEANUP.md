# Repository cleanup, 23 Nov 2026

Everything below was checked against the repository before anything was
deleted: references from code, the model registry, config loaders, tests,
documentation, CLI commands and reproduction scripts, plus a repo-wide
duplicate-content scan over every file above 50 KB.

**Reclaimed: 25.2 MB.** Nothing tracked by git was removed. The full suite
is green afterwards (382 passed, 1 skipped), Moderate and Heavy reproduce
their saved seed-6101 results exactly, and a custom workload runs end to
end.

---

## Removed

| what | files | size | why it is safe |
|---|---|---|---|
| `tmp/` | 810 | 12.97 MB | pytest scratch directories from six earlier runs, plus a mid-run copy of two checkpoints. Untracked; pytest recreates what it needs. |
| `output/dqn_phase_20261122/logs/` | 89 | 9.96 MB | Raw SUMO stdout from the 22 Nov sweeps. Untracked. Every measurement in those logs is already in the CSV and JSON files beside them, which are committed. |
| `__pycache__/` everywhere | 117 | 2.20 MB | Compiled bytecode. Already matched by `.gitignore`. |
| Generated SUMO inputs | 60 | 0.09 MB | `sumo/routes/*.rou.xml`, `sumo/config/*.sumocfg`, `sumo/additional/*_tls.add.xml`. All written at run time by `utils/scenario_builder.py` or `custom_traffic/builder.py`. `git ls-files` returns nothing for those three directories, so none was tracked. Verified regenerated: the seed-6101 checks below rebuilt them and matched the saved numbers to three decimals. |

`sumo/network/intersection.net.xml` is **not** generated at run time and was
not touched.

---

## New ignore rules

Narrow, and only for things the code writes:

```gitignore
tmp/
results/custom_runs/
results/custom_workloads/
results/hybrid/
output/**/logs/
```

---

## Left in place, and why

| what | size | why it stays |
|---|---|---|
| `experiments/controlled_20261119/` | 208.5 MB | A reported result. It has its own `REPORT.md`, `REVIEW.md`, a frozen protocol and an integrity snapshot (`protected_source_before.json`), and all 820 files under `experiments/` are tracked. The checkpoints under `final/` and `validation/` are the evidence behind that report. |
| `experiments/dqn_300_20261116,17/` | 44.9 MB | Same: tracked evidence for earlier reported results. |
| `output/dqn_1s_20261121/models/`, `output/dqn_cadence_20261121/models/` | 2.1 MB | Untracked checkpoints from the 21 Nov experiments. They back the numbers in `output/dqn_1s_20261121/FINDINGS.md`. Untracked and user-created, so left alone. |
| `output/dqn_phase_20261122/runs/*/models/*.pt` for the eleven runs that were **not** promoted | 7.0 MB | These back the robustness claim ("12 of 13 beat the best fixed plan"). Deleting them would mean that table could not be re-verified without retraining. |
| `results/evaluation/comparison_<x>_dqn_seed<n>_timeseries.csv` and `dqn_<x>_seed<n>_timeseries.csv` | 0.7 MB | **Byte-identical pairs**, produced by two different scripts under two naming conventions. Not removed: both names may be read by `evaluation/evaluate_dqn.py` and `evaluation/compare_controllers.py` depending on which produced the run, and establishing which is dead would take a change to those scripts rather than a deletion. |
| Copies of the same timeseries CSVs under `experiments/`, `verification/` and `results/` | 6.4 MB | Each evidence folder is deliberately self-contained — `verification/windows_20261116/` is a snapshot of what was measured on that machine on that date. De-duplicating by symlink or by deletion would make those folders no longer stand alone. |
| `presentation/` (2.8 MB), `verification/` (2.0 MB) | 4.8 MB | Tracked deliverables and measurement snapshots. |

---

## Duplicate scan

1,077 files above 50 KB were hashed. 45 duplicate groups were found,
accounting for 7.08 MB. After `tmp/` was removed, the remainder is the two
classes in the table above — cross-folder evidence snapshots and the
two-naming-convention pair — and neither is safe to collapse by deleting a
file.

---

## Verification after cleanup

```
1. Moderate and Heavy on seed 6101, against the saved results
   heavy     ns=7 ew=6  saved 6.670 s  now 6.670 s   MATCH
   moderate  ns=6 ew=4  saved 2.790 s  now 2.790 s   MATCH

2. A custom workload, end to end, under both controllers
   workload 33eddcd49f48f374   528 vehicles over 900 s
   fixed_time   10.78 s   528 requested, 528 entered, 502 completed,  0 pending
   dqn          16.43 s   528 requested, 487 entered, 435 completed, 41 pending
   preview total matches what was requested of SUMO: yes
   saved run reloads from disk: yes
```

Reproduce with `python output/demo_check_20261123/verify_demo.py`.

---

## One thing noticed and not fixed

Running the test suite, and running `evaluation/evaluate_dqn.py`, rewrite
eleven committed PNGs under `results/plots/`. The images are regenerated
with the same data, so the plots do not change meaning, but the files do
change and show up in `git status`. They were restored with
`git checkout -- results/plots/` after each run here.

It is pre-existing and it is a side effect of the plotting helpers writing
to the project's results directory rather than to a scratch directory.
Fixing it means changing where those helpers write, which touches code the
historical results were produced with, so it is reported rather than
changed. **After running the suite, check `git status`.**
