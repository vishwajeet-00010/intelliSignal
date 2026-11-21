# IntelliSignal controlled experiment - 19-20 November 2026

Incumbent models remain unchanged. Heavy: fails the declared final criteria; retain the incumbent. Moderate: passes the declared final criteria; eligible for human consideration only.

## Frozen final comparison

Twenty paired final traffic seeds, 1800 seconds per episode.

| Scenario | Controller | Waiting | Queue | Completed | Inserted | Remaining | Teleports |
| --- | --- | --- | --- | --- | --- | --- | --- |
| heavy | Incumbent | 46.974 | 37.033 | 1345.000 | 1458.850 | 113.850 | 0.000 |
| heavy | Frozen challenger | 47.624 | 37.547 | 1345.850 | 1461.250 | 115.400 | 0.000 |
| heavy | Fixed-time | 52.472 | 40.853 | 1351.550 | 1446.250 | 94.700 | 0.000 |
| heavy | Untrained (seed 42) | 61.301 | 47.965 | 1346.850 | 1445.750 | 98.900 | 0.000 |
| heavy | Untrained (seed 0) | 95.293 | 61.090 | 1063.150 | 1175.400 | 112.250 | 0.000 |
| moderate | Incumbent | 5.719 | 2.077 | 690.000 | 702.000 | 12.000 | 0.000 |
| moderate | Frozen challenger | 5.431 | 1.999 | 690.350 | 702.000 | 11.650 | 0.000 |
| moderate | Fixed-time | 9.869 | 3.650 | 692.850 | 702.000 | 9.150 | 0.000 |
| moderate | Untrained (seed 42) | 19.893 | 7.477 | 683.200 | 702.000 | 18.800 | 0.000 |
| moderate | Untrained (seed 0) | 10.909 | 4.076 | 688.150 | 702.000 | 13.850 | 0.000 |

| Scenario | Frozen challenger | Waiting W/T/L | Failed incumbent criteria |
| --- | --- | --- | --- |
| Heavy | prior_best | 7/0/13 | lower_waiting, waiting_wins_more_than_half, queue_no_worse |
| Moderate | incumbent_final | 18/0/2 | None |

## Paired differences and variability

Challenger minus incumbent; SD is over paired traffic-seed differences.

| Scenario | Metric | Mean delta | Paired SD |
| --- | --- | --- | --- |
| Heavy | Waiting (s/vehicle) | +0.650 | 2.465 |
| Heavy | Mean queue (vehicles) | +0.514 | 1.815 |
| Heavy | Maximum queue | -0.300 | 1.689 |
| Heavy | Completed vehicles | +0.850 | 7.569 |
| Heavy | Inserted vehicles | +2.400 | 6.754 |
| Heavy | Remaining vehicles | +1.550 | 3.620 |
| Heavy | Teleports | +0.000 | 0.000 |
| Moderate | Waiting (s/vehicle) | -0.288 | 0.184 |
| Moderate | Mean queue (vehicles) | -0.078 | 0.065 |
| Moderate | Maximum queue | +0.500 | 0.889 |
| Moderate | Completed vehicles | +0.350 | 1.424 |
| Moderate | Inserted vehicles | +0.000 | 0.000 |
| Moderate | Remaining vehicles | -0.350 | 1.424 |
| Moderate | Teleports | +0.000 | 0.000 |

| Scenario | Controller | Waiting mean | Traffic SD |
| --- | --- | --- | --- |
| heavy | Incumbent | 46.974 | 1.301 |
| heavy | Frozen challenger | 47.624 | 2.119 |
| heavy | Fixed-time | 52.472 | 1.484 |
| heavy | Untrained (seed 42) | 61.301 | 1.169 |
| heavy | Untrained (seed 0) | 95.293 | 12.101 |
| moderate | Incumbent | 5.719 | 0.136 |
| moderate | Frozen challenger | 5.431 | 0.094 |
| moderate | Fixed-time | 9.869 | 0.147 |
| moderate | Untrained (seed 42) | 19.893 | 0.206 |
| moderate | Untrained (seed 0) | 10.909 | 0.358 |

No new training was run: existing checkpoints passed the validation gate in both scenarios, satisfying the predeclared condition for avoiding retraining. The conditional control/lower-learning-rate experiment was therefore not triggered. The evaluated historical checkpoints use training seed 42 only; training-seed variability is unavailable, not zero.

Traffic SD describes variability over 20 matched traffic seeds. Shared traffic seeds and multiple checkpoints are correlated observations. They do not create independent training replicates. Fixed-time waiting/queue retain the collector's two-decimal per-run precision; learned-policy guardrails use unrounded values. Inserted and remaining vehicles are context measures and must be read with throughput. Zero observed teleports is a simulator result.

## Diagnosis before training

| Scenario / model | Requested switches | Effective switches | Blocked | Forced | Green mean (min-max) s |
| --- | --- | --- | --- | --- | --- |
| Heavy / Incumbent best | 1701 | 890 | 811 | 0 | 16.0 (16-20) |
| Heavy / Prior best | 1428 | 877 | 551 | 0 | 16.4 (16-30) |
| Moderate / Incumbent best | 1522 | 880 | 642 | 0 | 16.4 (16-26) |
| Moderate / Incumbent final | 1789 | 890 | 899 | 0 | 16.0 (16-20) |

| Scenario / model | Max queue / wait clip rate | Elapsed clip | Q range | Mean Q gap | Abs TD mean / p95 |
| --- | --- | --- | --- | --- | --- |
| Heavy / Incumbent best | 0.00% / 0.00% | 0.00% | -57.51 to -9.10 | 3.730 | 1.508 / 4.261 |
| Heavy / Prior best | 0.00% / 0.00% | 0.00% | -58.71 to -9.95 | 3.492 | 1.435 / 4.086 |
| Moderate / Incumbent best | 0.00% / 0.00% | 0.00% | -3.45 to -1.42 | 0.278 | 0.096 / 0.237 |
| Moderate / Incumbent final | 0.00% / 0.00% | 0.00% | -2.97 to -1.43 | 0.290 | 0.087 / 0.211 |

Each diagnostic row covers 1800 decisions over ten validation episodes. Green values are completed durations; censored final greens are retained in diagnostics.json. Clip rates above are the maximum across the four queue/wait dimensions, not pooled rates. Exact-boundary saturation is separate. The binary phase is categorical. Q gaps and TD values are descriptive; TD is offline residual error under the stored target network and checkpoint gamma, bootstrapping at time limits. Original training-minibatch TD errors were not logged.

Heavy: current-waiting penalties account for 88.49% of incumbent reward cost. Descriptive reward correlations: waiting -0.988, queue -0.991, throughput 0.665.
Moderate: current-waiting penalties account for 87.88% of incumbent reward cost. Descriptive reward correlations: waiting -0.964, queue -0.991, throughput 0.292.

Moderate incumbent/prior-best effective actions agree on 100.000% of common-observation probes from their two rollout corpora. This measures decisions in the tested states, not equality on all possible states. weight_comparison.json separately records tensor and closed-loop trace comparisons. Their weights differ; identical effective decisions do not imply identical networks. Best versus incumbent-final effective agreement is 97.222% over 12600 evaluated state visits across all diagnostic rollouts; the best-checkpoint comparison does not describe every moderate policy.

Measured: reward uses interval-average current waiting on incoming lanes; reported waiting uses accumulated waiting per entered vehicle. Queue samples are shared, but throughput is not rewarded directly. There were 4 paired checkpoint/seed reward-waiting ranking reversals in this validation corpus. Possible explanation: these objective differences and similar action choices can limit the benefit of more training. The measurements do not prove the cause of any performance difference.

## Selection, safeguards and decision

Validation: seeds 2001-2010, greedy actions, six predeclared existing checkpoints per scenario plus fixed-time and seed-42 untrained controls. The repository seed audit found no prior recorded use of validation or final seeds; deleted/external runs cannot be ruled out. Historical seeds 1001-1010 were not treated as untouched. Training traffic, if used, is confined to seeds 42-343.

Validation eligibility requires lower mean waiting, more than 5/10 waiting wins, and no worse mean queue, throughput or teleports against the incumbent. Eligible checkpoints rank by waiting, queue, throughput and declared order. If none qualify, the challenger is diagnostic only. No selection uses exploratory training return alone.

freeze.json fixes every checkpoint hash, source hash and recipe before the 200 final evaluation episodes on seeds 3001-3020. Promotion requires lower waiting, more than 10/20 waiting wins and no worse mean queue, throughput or teleports. A newly trained recipe must also beat its matched validation-selected control, improve in at least two of three training seeds, and have a negative upper bound for the predeclared 95% t interval across training-seed mean waiting differences. No thresholds were relaxed; no model is replaced automatically.

Frozen challengers: heavy: `experiments/dqn_300_20261117/heavy/models/best.pt`; moderate: `models/dqn_moderate/final.pt`. Full checkpoint hashes are in freeze.json. Final results do not reopen the training or selection branch.

| Scenario | Validation checkpoint | Waiting | Queue | Completed | Gate |
| --- | --- | --- | --- | --- | --- |
| heavy | incumbent_best | 48.053 | 37.855 | 1344.100 | baseline |
| heavy | incumbent_final | 47.188 | 37.298 | 1348.100 | PASS |
| heavy | prior_best | 47.165 | 37.267 | 1348.800 | PASS |
| heavy | prior_final | 48.096 | 37.941 | 1346.900 | FAIL |
| heavy | prior_ep100 | 47.763 | 37.593 | 1343.800 | FAIL |
| heavy | prior_ep200 | 47.923 | 37.850 | 1347.500 | FAIL |
| moderate | incumbent_best | 5.760 | 2.096 | 689.400 | baseline |
| moderate | incumbent_final | 5.441 | 2.002 | 690.300 | PASS |
| moderate | prior_best | 5.760 | 2.096 | 689.400 | FAIL |
| moderate | prior_final | 5.441 | 2.002 | 690.300 | PASS |
| moderate | prior_ep100 | 5.760 | 2.096 | 689.400 | FAIL |
| moderate | prior_ep200 | 5.441 | 2.002 | 690.300 | PASS |

Protected files: 745; changed: 0; missing: 0. Scenario settings, safety logic, baselines, incumbent models and all prior experiment artifacts were preserved. Greedy evaluation verifies RNG, weights, replay and learning-counter invariance. Independent review and reproducible full-state/retry checks are retained.

Good and bad: Heavy - good: frozen-challenger waiting is 9.24% below fixed-time on the final seeds. Against the incumbent, waiting changes by +0.650 s and completed vehicles by +0.85. Maximum queue changes by -0.30 vehicles; throughput is 5.70 below fixed-time. Bad / limit: failed incumbent guardrails: lower waiting, waiting wins more than half, queue no worse. Moderate - good: frozen-challenger waiting is 44.97% below fixed-time on the final seeds. Against the incumbent, waiting changes by -0.288 s and completed vehicles by +0.35. Maximum queue changes by +0.50 vehicles; throughput is 2.50 below fixed-time. Bad / limit: passing this finite test does not establish general superiority.

## Reproduce and inspect

See COMMANDS.md for exact execution and verification commands. Primary artifacts: manifest.json, selection.json, freeze.json, diagnostics.json, weight_comparison.json, validation_metrics.csv, final_metrics.csv, paired_final_metrics.csv, training_traffic_variability.csv, final_decision.json, integrity.json and REVIEW.md. Every per-seed measurement and decision trace is retained under validation/ and final/.
