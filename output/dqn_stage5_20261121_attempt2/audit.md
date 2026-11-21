# Stage 5 pre-change audit

## Stage 4 evidence

- Stage 4 full tests: 224 passed, 1 skipped. All Stage 4 evaluation outputs are in `output/dqn_stage4_20261120/`.
- Moderate: Stage 4 seed 55 best validation wait was 3.386 s at episode 10 and failed validation against the incumbent. Seed 56 best was 2.980 s at episode 40 and passed individually; it stopped at episode 70 after three consecutive degradation checks. The pair therefore failed the required two-run gate.
- Heavy: seed 55 best validation wait was 18.182 s at episode 50 and failed individually. Seed 56 best was 16.982 s at episode 50 and passed individually; it stopped at episode 80 after three consecutive degradation checks. The pair failed the two-run gate.
- In both scenarios, zero blocked switches and zero teleports occurred on the Stage 4 validation rollouts. The issue was performance consistency against the incumbent, not those safety counts.
- Across Stage 4’s 20-seed confirmatory benchmark, current DQN mean waiting was 2.9185 s moderate and 16.469 s heavy; starting DQN was 29.5405 s and 19.411 s, respectively. No Stage 4 candidate passed validation; none was promoted.

## Code evidence and hypothesis

- `rl/state.py` computes the phase-age feature with `min(max_green, 4 * min_green)`. Current environment values are `max_green=60 s` and `min_green=5 s`, so feature 10 saturates after 20 s while the signal can legally remain green until 60 s. The input cannot convey whether the green is at 20, 40, or 60 seconds once clipped.
- Stage 4 queue/wait calibration found moderate waiting p99 at or slightly above its scale for some approaches (one p99/scale ratio was 1.07); heavy p99 values were within the configured scales. This motivates monitoring clipping, but Stage 5 does not change those scales.
- Stage 4 configurations reached epsilon floor by the middle of 100-episode training. This is a possible alternative explanation but is not changed in Stage 5, to preserve a one-factor experiment.
- Stage 4 late training was not consistently better than its best validation checkpoint: two of four runs triggered the predeclared early-stop safeguard, and validation-best episodes differed across scenarios/seeds. This establishes the need to retain checkpoint selection and report every seed; it does not prove a specific causal mechanism.

## Study choice and limitations

The Stage 5 treatment tests only whether scaling the final phase-age observation across the full legal green duration improves validation consistency. It is a direct, testable mismatch between the observation and the control limit. The existing scaling remains available as the control and as the inference path for existing models. Training algorithm, reward, traffic, action constraints, and all other hyperparameters are held fixed. The result can establish whether this one representation change helps in this simulator; it cannot establish generalization to other intersections or demand models.

## Execution interruption record

The first training batch's attached execution runtime was replaced while seed 6201 work was underway. An earlier audit incorrectly reported moderate/treatment seed6201 as partial; direct artifact inspection shows both moderate/control and moderate/treatment seed6201 have 100-episode histories and completed summaries. Both remain excluded as pilots because the primary seed set had already been amended. A later audit initially found moderate/control seed6202 at episode 78/100 and no paired treatment summary, but subsequent inspection found completed 100-episode summaries for both moderate seed6202 arms. The archived replacement status records the treatment and later jobs as exit-code-2 failures, conflicting with the treatment artifact. Seed 6202 remains excluded; preserve this discrepancy in the run reconciliation. The primary matched seeds remain 6207-6211.
