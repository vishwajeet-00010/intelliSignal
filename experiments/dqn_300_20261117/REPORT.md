# 300-episode DQN experiment

Completed scenarios in this report: **heavy, moderate**.

Fresh training uses seed 42, traffic seeds 42–341, 300 episodes, and the default epsilon schedule stretched to 27,000 decisions. This compares complete training recipes, not a continuation of the old checkpoint or an isolated change in episode count.

Reference seeds 42/43/44 overlap training and are reproduction checks. Holdout seeds 1001–1010 are outside the training range. Every evaluation uses 1800 simulated seconds, greedy actions, and the same metric collector. The candidate comparison includes fixed-time and an untrained network; the incumbent is evaluated on the same seeds.

Declared rule from the experiment manifest: On holdouts: lower mean waiting, waiting wins on >5/10 seeds, no worse mean queue, throughput or teleports vs incumbent. Report reference results and all tradeoffs. No automatic replacement.

Only **one training seed** is used per scenario. Holdout traffic seeds assess demand variability, not reproducibility across independent training runs; no statistical significance or general superiority is claimed.

Inserted and remaining vehicle counts are context metrics: changing admission into the network can change the remaining count. Read them together with throughput and waiting.

Protected-artifact check: 163 files checked; 0 changed.

## Heavy traffic

Training: 300 episodes, seed 42, 8392.1 seconds. Selected `best.pt` is from episode 273 using the existing five-episode mean training-return rule. No evaluation results were used to select this checkpoint.

### Reference seeds: 42, 43, 44

Means over matched traffic seeds; delta is candidate minus incumbent. Standard deviations and per-metric seed counts are in `paired_metrics.csv`.

| Metric | Fixed-time | Untrained | Incumbent | Candidate | Delta |
|---|---:|---:|---:|---:|---:|
| Waiting (s/vehicle) | 52.597 | 88.207 | 47.910 | 47.640 | -0.270 |
| Average queue (vehicles) | 40.983 | 57.950 | 37.750 | 37.567 | -0.183 |
| Maximum queue (vehicles) | 71.333 | 101.000 | 69.667 | 69.667 | 0.000 |
| Completed vehicles | 1351.333 | 1089.333 | 1343.667 | 1343.000 | -0.667 |
| Inserted vehicles | 1447.000 | 1200.333 | 1459.333 | 1462.000 | 2.667 |
| Vehicles remaining | 95.667 | 111.000 | 115.667 | 119.000 | 3.333 |
| Teleports | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Waiting-time wins/ties/losses: **2/0/1**. Mean waiting improvement versus incumbent: **0.564%**.

| Seed | Incumbent waiting | Candidate waiting | Delta |
|---|---:|---:|---:|
| 42 | 47.920 | 45.850 | -2.070 |
| 43 | 49.900 | 47.740 | -2.160 |
| 44 | 45.910 | 49.330 | +3.420 |

### Holdout seeds: 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010

Means over matched traffic seeds; delta is candidate minus incumbent. Standard deviations and per-metric seed counts are in `paired_metrics.csv`.

| Metric | Fixed-time | Untrained | Incumbent | Candidate | Delta |
|---|---:|---:|---:|---:|---:|
| Waiting (s/vehicle) | 53.851 | 96.962 | 46.676 | 47.725 | 1.049 |
| Average queue (vehicles) | 41.851 | 62.074 | 36.805 | 37.667 | 0.862 |
| Maximum queue (vehicles) | 72.300 | 101.200 | 68.100 | 69.500 | 1.400 |
| Completed vehicles | 1347.700 | 1063.400 | 1345.500 | 1344.900 | -0.600 |
| Inserted vehicles | 1443.300 | 1175.300 | 1458.400 | 1462.600 | 4.200 |
| Vehicles remaining | 95.600 | 111.900 | 112.900 | 117.700 | 4.800 |
| Teleports | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Waiting-time wins/ties/losses: **4/0/6**. Mean waiting improvement versus incumbent: **-2.247%**.

| Seed | Incumbent waiting | Candidate waiting | Delta |
|---|---:|---:|---:|
| 1001 | 45.780 | 48.500 | +2.720 |
| 1002 | 47.670 | 47.450 | -0.220 |
| 1003 | 45.790 | 46.570 | +0.780 |
| 1004 | 47.550 | 51.580 | +4.030 |
| 1005 | 45.720 | 43.560 | -2.160 |
| 1006 | 47.510 | 47.030 | -0.480 |
| 1007 | 46.510 | 46.180 | -0.330 |
| 1008 | 47.710 | 50.760 | +3.050 |
| 1009 | 47.250 | 49.160 | +1.910 |
| 1010 | 45.270 | 46.460 | +1.190 |

### Promotion assessment

**DO NOT PROMOTE: candidate fails one or more declared holdout guardrails.**

- FAIL: Lower mean waiting.
- FAIL: Waiting wins on more than half of holdout seeds.
- FAIL: No worse mean queue.
- FAIL: No worse mean throughput.
- PASS: No worse mean teleports.

This report does not replace any model. The original checkpoint remains the dashboard default.

### Training trend

Descriptive 50-episode windows; higher return means less negative return. Traffic seeds and exploration change during training, so these trends do not prove convergence.

| Episodes | Mean return | Mean waiting | Mean queue | Mean loss | Mean epsilon |
|---|---:|---:|---:|---:|---:|
| 1–50 | -1133.819 | 56.249 | 44.340 | 1.583 | 0.839 |
| 51–100 | -863.364 | 52.500 | 41.370 | 1.496 | 0.522 |
| 101–150 | -675.311 | 49.216 | 38.859 | 1.371 | 0.205 |
| 151–200 | -615.111 | 48.557 | 38.253 | 1.338 | 0.050 |
| 201–250 | -612.663 | 48.047 | 37.935 | 1.232 | 0.050 |
| 251–300 | -612.032 | 48.088 | 37.895 | 1.187 | 0.050 |

Mean return changed by **+0.630** between the last two 50-episode windows. A single run cannot establish that longer training will improve the policy.

Artifacts: `heavy/models/`, `heavy/training/`, `heavy/evaluation/`, and `heavy/plots/`.

## Moderate traffic

Training: 300 episodes, seed 42, 78321.2 seconds. Selected `best.pt` is from episode 248 using the existing five-episode mean training-return rule. No evaluation results were used to select this checkpoint.

### Reference seeds: 42, 43, 44

Means over matched traffic seeds; delta is candidate minus incumbent. Standard deviations and per-metric seed counts are in `paired_metrics.csv`.

| Metric | Fixed-time | Untrained | Incumbent | Candidate | Delta |
|---|---:|---:|---:|---:|---:|
| Waiting (s/vehicle) | 9.910 | 10.910 | 5.713 | 5.713 | 0.000 |
| Average queue (vehicles) | 3.657 | 4.087 | 2.083 | 2.083 | 0.000 |
| Maximum queue (vehicles) | 9.667 | 11.000 | 7.000 | 7.000 | 0.000 |
| Completed vehicles | 693.000 | 688.667 | 691.000 | 691.000 | 0.000 |
| Inserted vehicles | 702.000 | 702.000 | 702.000 | 702.000 | 0.000 |
| Vehicles remaining | 9.000 | 13.333 | 11.000 | 11.000 | 0.000 |
| Teleports | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Waiting-time wins/ties/losses: **0/3/0**. Mean waiting improvement versus incumbent: **-0.000%**.

| Seed | Incumbent waiting | Candidate waiting | Delta |
|---|---:|---:|---:|
| 42 | 5.780 | 5.780 | +0.000 |
| 43 | 5.600 | 5.600 | +0.000 |
| 44 | 5.760 | 5.760 | +0.000 |

### Holdout seeds: 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010

Means over matched traffic seeds; delta is candidate minus incumbent. Standard deviations and per-metric seed counts are in `paired_metrics.csv`.

| Metric | Fixed-time | Untrained | Incumbent | Candidate | Delta |
|---|---:|---:|---:|---:|---:|
| Waiting (s/vehicle) | 9.862 | 10.866 | 5.726 | 5.726 | 0.000 |
| Average queue (vehicles) | 3.650 | 4.046 | 2.082 | 2.082 | 0.000 |
| Maximum queue (vehicles) | 10.300 | 11.200 | 7.300 | 7.300 | 0.000 |
| Completed vehicles | 693.400 | 688.700 | 690.600 | 690.600 | 0.000 |
| Inserted vehicles | 702.000 | 702.000 | 702.000 | 702.000 | 0.000 |
| Vehicles remaining | 8.600 | 13.300 | 11.400 | 11.400 | 0.000 |
| Teleports | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Waiting-time wins/ties/losses: **0/10/0**. Mean waiting improvement versus incumbent: **-0.000%**.

| Seed | Incumbent waiting | Candidate waiting | Delta |
|---|---:|---:|---:|
| 1001 | 5.920 | 5.920 | +0.000 |
| 1002 | 5.770 | 5.770 | +0.000 |
| 1003 | 5.740 | 5.740 | +0.000 |
| 1004 | 5.870 | 5.870 | +0.000 |
| 1005 | 5.830 | 5.830 | +0.000 |
| 1006 | 5.620 | 5.620 | +0.000 |
| 1007 | 5.750 | 5.750 | +0.000 |
| 1008 | 5.350 | 5.350 | +0.000 |
| 1009 | 5.710 | 5.710 | +0.000 |
| 1010 | 5.700 | 5.700 | +0.000 |

### Promotion assessment

**DO NOT PROMOTE: candidate fails one or more declared holdout guardrails.**

- FAIL: Lower mean waiting.
- FAIL: Waiting wins on more than half of holdout seeds.
- PASS: No worse mean queue.
- PASS: No worse mean throughput.
- PASS: No worse mean teleports.

This report does not replace any model. The original checkpoint remains the dashboard default.

### Training trend

Descriptive 50-episode windows; higher return means less negative return. Traffic seeds and exploration change during training, so these trends do not prove convergence.

| Episodes | Mean return | Mean waiting | Mean queue | Mean loss | Mean epsilon |
|---|---:|---:|---:|---:|---:|
| 1–50 | -83.777 | 9.145 | 3.385 | 0.027 | 0.839 |
| 51–100 | -54.529 | 7.406 | 2.722 | 0.014 | 0.522 |
| 101–150 | -37.010 | 6.215 | 2.264 | 0.011 | 0.205 |
| 151–200 | -31.726 | 5.798 | 2.107 | 0.010 | 0.050 |
| 201–250 | -31.778 | 5.808 | 2.108 | 0.010 | 0.050 |
| 251–300 | -31.842 | 5.833 | 2.115 | 0.009 | 0.050 |

Mean return changed by **-0.065** between the last two 50-episode windows. A single run cannot establish that longer training will improve the policy.

Artifacts: `moderate/models/`, `moderate/training/`, `moderate/evaluation/`, and `moderate/plots/`.
