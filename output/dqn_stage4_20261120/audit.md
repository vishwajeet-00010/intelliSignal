# Stage 4 pre-training repository audit

## Current architecture and key paths

- SUMO network and demand are generated from `sumo/` and scenario YAML files in `config/`.
- `rl/environment.py` owns signal safety and exposes the 10-feature normalized observation described in `rl/state.py`.
- `agents/dqn_agent.py` contains the Q-network controller; `training/train_dqn.py` handles replay, target updates, periodic checkpoints, and greedy validation.
- `simulation/agent_simulation.py` and `utils/metrics.py` provide the shared DQN/agent rollout and metric collection. `evaluation/compare_controllers.py` compares the existing DQN and fixed-time controllers.
- `models/` contains the prior Stage 3 incumbent checkpoints. The latest retrained checkpoints, chosen by validation waiting time, are separately stored in `output/dqn_retraining_20261120/models/`.
- Training histories are in `results/training/` and isolated experiment folders. Protected Stage 1 results are in `results/csv/`.
- The full pytest suite covers configuration, agent/network behavior, training, evaluation, simulation, and dashboard integration.

## Relevant historical experiments

- `experiments/dqn_300_20261117/` contains an isolated 300-episode run and per-seed holdout evaluation. It used a single training seed and chose `best.pt` by training return. The report says its heavy candidate failed the fresh holdout gates for waiting, queue, and throughput; moderate did not improve over the incumbent. Neither candidate was promoted.
- `experiments/dqn_300_20261116/status.json` records an interrupted 300-episode heavy run at episode 233. Its legacy checkpoints omitted replay memory and RNG state, so exact resume was impossible. The artifacts were preserved and replaced with a fresh run.
- `experiments/controlled_20261119/weight_comparison.json` confirms prior/incumbent model tensors differ and includes rollout trace comparisons. It is audit evidence only; Stage 4 does not overwrite it.
- `output/dqn_retraining_20261120/experiment_manifest.json` records a strong single-training-seed retraining result across 20 traffic seeds, but the heavy DQN remains behind tuned fixed-time on mean waiting and its normalization p99 gate is partial.

## Stage 4 changes and isolation

- Added explicit validation seed override and optional consecutive-validation-degradation stopping to `training/train_dqn.py`. The default legacy behavior remains unchanged unless the new arguments are provided.
- Added a unit test for the degradation threshold helper.
- Candidate scripts and all experiment artifacts live under this new Stage 4 output directory. No checkpoint is automatically promoted.
- Added a standalone pressure-actuated rule-based controller to the Stage 4 evaluation script; it is tuned on validation seeds only and is not represented as a production feature.
- The Stage 4 protocol was written before candidate training. Fresh training seeds are 55 and 56; validation seeds are 5101-5105; confirmatory seeds are 6101-6120.

## Initial verification

- Full pytest suite before candidate runs: 224 passed, 1 skipped.
- A sandboxed attempt initially failed because Windows denied SUMO's local TraCI socket. The same suite completed successfully outside the sandbox after access was granted.
- The pre-training integrity snapshot covers incumbent model checkpoints, baseline CSV/JSON artifacts, scenario YAML files, and the latest retrained incumbent checkpoints.
