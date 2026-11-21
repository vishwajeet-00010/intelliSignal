# Stage 5 run reconciliation

This report compares the preserved batch-status snapshots with per-run summary, history, checkpoint, and validation artifacts. The raw JSON and stdout/stderr logs are not rewritten.

## First batch snapshot (excluded pilot/replacement seeds)

These seeds were excluded before the primary 6207–6211 run set. Status and artifact evidence are both shown because the seed-6202 summaries conflict with an archived failure/interruption record.

| Scenario | Arm | Seed | First snapshot status | Artifact status | Episodes | History rows | Best | Validation |
|---|---|---:|---|---|---:|---:|---|---|
| moderate | control | 6201 | completed (exit 0) | complete | 100 | 100 | True | True |
| moderate | treatment | 6201 | completed (exit 0) | complete | 100 | 100 | True | True |
| moderate | control | 6202 | failed (exit 1) | complete | 100 | 100 | True | True |
| moderate | treatment | 6202 | completed (exit 0) | complete | 100 | 100 | True | True |
| moderate | control | 6203 | failed (exit 2) | no artifacts | — | 0 | False | False |
| moderate | treatment | 6203 | failed (exit 2) | no artifacts | — | 0 | False | False |
| moderate | control | 6204 | failed (exit 2) | no artifacts | — | 0 | False | False |
| moderate | treatment | 6204 | failed (exit 2) | no artifacts | — | 0 | False | False |
| moderate | control | 6205 | failed (exit 2) | no artifacts | — | 0 | False | False |
| moderate | treatment | 6205 | failed (exit 2) | no artifacts | — | 0 | False | False |
| heavy | control | 6201 | failed (exit 2) | no artifacts | — | 0 | False | False |
| heavy | treatment | 6201 | failed (exit 2) | no artifacts | — | 0 | False | False |
| heavy | control | 6202 | failed (exit 2) | no artifacts | — | 0 | False | False |
| heavy | treatment | 6202 | failed (exit 2) | no artifacts | — | 0 | False | False |
| heavy | control | 6203 | failed (exit 2) | no artifacts | — | 0 | False | False |
| heavy | treatment | 6203 | failed (exit 2) | no artifacts | — | 0 | False | False |
| heavy | control | 6204 | failed (exit 2) | no artifacts | — | 0 | False | False |
| heavy | treatment | 6204 | failed (exit 2) | no artifacts | — | 0 | False | False |
| heavy | control | 6205 | failed (exit 2) | no artifacts | — | 0 | False | False |
| heavy | treatment | 6205 | failed (exit 2) | no artifacts | — | 0 | False | False |

## Archived replacement batch (excluded seeds)

The replacement log's 6202–6206 arguments are invalid under the current frozen CLI, which accepts 6207–6211. This records the archived invocation failure; it is separate from the primary 6207–6211 run set.

| Scenario | Arm | Seed | Archived status | Artifact evidence |
|---|---|---:|---|---|
| moderate | control | 6202 | completed (exit 0) | 100 episodes; complete; history rows 100; best True; validation True |
| moderate | treatment | 6202 | failed (exit 2) | 100 episodes; complete; history rows 100; best True; validation True |
| moderate | control | 6203 | failed (exit 2) | no summary; no artifacts; history rows 0; best False; validation False |
| moderate | treatment | 6203 | failed (exit 2) | no summary; no artifacts; history rows 0; best False; validation False |
| moderate | control | 6204 | failed (exit 2) | no summary; no artifacts; history rows 0; best False; validation False |
| moderate | treatment | 6204 | failed (exit 2) | no summary; no artifacts; history rows 0; best False; validation False |
| moderate | control | 6205 | failed (exit 2) | no summary; no artifacts; history rows 0; best False; validation False |
| moderate | treatment | 6205 | failed (exit 2) | no summary; no artifacts; history rows 0; best False; validation False |
| moderate | control | 6206 | failed (exit 2) | no summary; no artifacts; history rows 0; best False; validation False |
| moderate | treatment | 6206 | failed (exit 2) | no summary; no artifacts; history rows 0; best False; validation False |
| heavy | control | 6202 | failed (exit 2) | no summary; no artifacts; history rows 0; best False; validation False |
| heavy | treatment | 6202 | failed (exit 2) | no summary; no artifacts; history rows 0; best False; validation False |
| heavy | control | 6203 | failed (exit 2) | no summary; no artifacts; history rows 0; best False; validation False |
| heavy | treatment | 6203 | failed (exit 2) | no summary; no artifacts; history rows 0; best False; validation False |
| heavy | control | 6204 | failed (exit 2) | no summary; no artifacts; history rows 0; best False; validation False |
| heavy | treatment | 6204 | failed (exit 2) | no summary; no artifacts; history rows 0; best False; validation False |
| heavy | control | 6205 | failed (exit 2) | no summary; no artifacts; history rows 0; best False; validation False |
| heavy | treatment | 6205 | failed (exit 2) | no summary; no artifacts; history rows 0; best False; validation False |
| heavy | control | 6206 | failed (exit 2) | no summary; no artifacts; history rows 0; best False; validation False |
| heavy | treatment | 6206 | failed (exit 2) | no summary; no artifacts; history rows 0; best False; validation False |

## Primary batch (seeds 6207–6211)

Runner process observed: **no**.

| Scenario | Arm | Seed | Snapshot status | Artifact status | Episodes | History rows | Best | Validation |
|---|---|---:|---|---|---:|---:|---|---|
| moderate | control | 6207 | completed | complete | 100 | 100 | True | True |
| moderate | treatment | 6207 | running | partial | — | 19 | True | True |
| moderate | control | 6208 | queued | no artifacts | — | 0 | False | False |
| moderate | treatment | 6208 | queued | no artifacts | — | 0 | False | False |
| moderate | control | 6209 | queued | no artifacts | — | 0 | False | False |
| moderate | treatment | 6209 | queued | no artifacts | — | 0 | False | False |
| moderate | control | 6210 | queued | no artifacts | — | 0 | False | False |
| moderate | treatment | 6210 | queued | no artifacts | — | 0 | False | False |
| moderate | control | 6211 | queued | no artifacts | — | 0 | False | False |
| moderate | treatment | 6211 | queued | no artifacts | — | 0 | False | False |
| heavy | control | 6207 | queued | no artifacts | — | 0 | False | False |
| heavy | treatment | 6207 | queued | no artifacts | — | 0 | False | False |
| heavy | control | 6208 | queued | no artifacts | — | 0 | False | False |
| heavy | treatment | 6208 | queued | no artifacts | — | 0 | False | False |
| heavy | control | 6209 | queued | no artifacts | — | 0 | False | False |
| heavy | treatment | 6209 | queued | no artifacts | — | 0 | False | False |
| heavy | control | 6210 | queued | no artifacts | — | 0 | False | False |
| heavy | treatment | 6210 | queued | no artifacts | — | 0 | False | False |
| heavy | control | 6211 | queued | no artifacts | — | 0 | False | False |
| heavy | treatment | 6211 | queued | no artifacts | — | 0 | False | False |

Artifact-derived primary totals: **1 complete, 1 partial, 18 not started**. These totals come from the files above, not from the older manifest counters.

The archived final snapshot says one complete, one running, and 18 queued. The runner was not live when reconciled; the running cell has partial artifacts and is not counted complete.

A completed summary alongside a failed archived status is shown as a discrepancy; neither record is silently rewritten.
