# IntelliSignal Stage 4 experiment protocol

Protocol frozen before Stage 4 training. Outputs live only under this experiment directory. The previous experiment trees, protected baseline CSVs, and incumbent checkpoints remain read-only.

## Audit findings and failure safeguards

| Observed failure or risk | Evidence | Stage 4 safeguard |
|---|---|---|
| A longer run can look better on reference seeds yet regress on fresh traffic and key metrics. | `experiments/dqn_300_20261117/REPORT.md`: the heavy candidate was slightly better on seeds 42-44 but failed holdout seeds 1001-1010 for mean waiting, queue, and throughput; it was not promoted. | Select checkpoints only on the predeclared validation seeds; compare the final selected policy on a separate, fresh confirmatory set. Keep wait, queue, throughput, max queue, unfinished vehicles, teleports, and action safety together. |
| One training seed does not establish run-to-run reliability. | The 300-episode experiment manifest and report explicitly state one training seed per scenario. | Train two independent seeds per scenario and report both; never present the best run as the distribution. |
| Training-return checkpoint selection can select a checkpoint that does not generalize. | The 300-episode manifest records selection by five-episode mean training return, while the heavy candidate failed its holdout gates. | Stage 4 candidate checkpoints are selected by mean greedy validation waiting across five fixed validation seeds. |
| A previous process stopped before completion and could not be resumed exactly. | `experiments/dqn_300_20261116/status.json`: 233 of 300 episodes; legacy checkpoints lacked replay and RNG state. | Run in a new isolated directory, save periodic checkpoints/history, record failures, and do not assume a partial run is resumable unless full state is captured. The current validated training loop writes periodic weights and CSV history; any interruption means reject that run unless exact recovery is available. |
| Single-run late training curves remain noisy. | Historical Stage 3 report and 300-episode report describe noisy late heavy training. | Evaluate greedy checkpoints every 10 episodes; stop after three consecutive validation evaluations more than 10% worse than the best validation mean. Reject non-finite losses or metrics. |
| Normalization can clip observations or collapse feature ranges. | `output/dqn_retraining_20261120/experiment_manifest.json` marks the p99 range check partial. | Collect post-training observation distributions for the newly selected policies on fresh calibration seeds; report clipping and low-range dimensions, without silently retuning on confirmatory seeds. |

## Predeclared experiment design

- Scenarios: moderate and heavy, existing 1800-second demand configuration.
- Independent training seeds per scenario: 55 and 56; 100 episodes each unless validation early stopping triggers. Episode traffic varies by episode using the configured seed schedule.
- Validation seeds (used for checkpoint selection and actuated-controller tuning): 5101-5105.
- Observation-distribution calibration seeds (diagnostics only; not used to retune scales): 5201-5210.
- Confirmatory seeds (sealed until candidate/controller choices are frozen): 6101-6120.
- Previously used seeds (not Stage 4 holdout): 42-44, 1001-1010, 2001-2010, and 3001-3020.
- Controllers: previous DQN incumbent from `output/dqn_retraining_20261120/models/`; Stage 4 candidates; fixed-time using the existing scenario plan; tuned fixed-time greens (moderate 5 s, heavy 7 s); and a project-specific sampled queue/wait pressure heuristic, tuned only on validation seeds. This is a transparent heuristic comparator, not a certified industry actuated controller.
- Candidate evaluation metrics: mean accumulated waiting time, average queue, maximum queue, throughput, inserted and remaining vehicles, teleports, switch count, blocked switches, switch/decision ratio, effective actions, and green-duration counts.
- Minimum-green safety and yellow clearance are enforced by the existing environment for the actuated rule as for DQN.

## Numerical promotion gates

A Stage 4 candidate can replace the DQN incumbent only if all conditions hold on the sealed 20-seed confirmatory set:

1. Full pytest suite has no failures or errors; all candidate rollouts are finite, with zero blocked switches and zero teleports.
2. Mean waiting improves by at least 2% versus the current DQN incumbent, with paired waiting wins on at least 11 of 20 seeds.
3. Mean average queue is no more than 2% worse; mean max queue no more than 5% worse; throughput no more than 1% lower.
4. Both independent training runs pass validation versus the incumbent within a 3% waiting tolerance; if only one passes, it cannot be promoted as a robust multi-seed result.
5. Report comparison against tuned fixed-time and validation-tuned actuated control. If DQN does not beat them on the confirmatory set, state that it is not the best controller even if it improves over the previous DQN.

No candidate is automatically installed into `models/` or dashboard defaults. Failed candidates remain isolated and the previous checkpoint is retained.

## Execution sequence

1. Snapshot SHA-256 hashes of current DQN incumbent checkpoints and protected baseline/config files.
2. Run two independent candidate training runs per scenario in this new output directory; validate checkpoints on seeds 5101-5105 and apply the predeclared early-stop rule.
3. Tune the actuated-controller threshold and maximum green using only validation seeds. Freeze those values.
4. Choose the candidate per scenario from the two independent runs using validation only; if neither meets validation guardrails, do not open the confirmatory set for that scenario.
5. Run the full test suite after code changes.
6. Evaluate the frozen incumbent, candidate, fixed-time, tuned fixed-time, and tuned actuated controller on identical confirmatory seeds and duration. No changes after seeing these results.
7. Verify hashes, record environment and commands, create charts/PDF, render and inspect every page, and write the final manifest and decision.
