"""
IntelliSignal - Stage 3
training package: the DQN training loop.

    python training/train_dqn.py --scenario moderate --episodes 20 --seed 42

Nothing in this package defines the environment, the observation, the action,
the reward or the signal-safety rules. It only collects experience from the
Stage 2 environment and updates the network in agents/.
"""

TRAINING_RESULTS_SUBDIR = "training"
