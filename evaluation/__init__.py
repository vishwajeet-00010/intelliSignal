"""
IntelliSignal - Stage 3
evaluation package: measuring a trained controller, and comparing controllers.

    python evaluation/evaluate_dqn.py --scenario moderate --seeds 42 43 44
    python evaluation/compare_controllers.py --scenario moderate --seeds 42 43 44

Two rules this package exists to enforce:

  1. Evaluation NEVER trains. Epsilon is zero, the replay buffer is never
     written and no gradient step is taken, so the numbers describe the saved
     policy and not a policy that kept improving while being measured.
  2. A comparison is only a comparison when both controllers saw the same
     road, the same demand, the same duration and the same seed. Both are run
     from `simulation/agent_simulation.py`, which builds one set of SUMO input
     files and measures both runs with the Stage 1 metrics collector.
"""

#: Where evaluation output goes. results/csv/ is the Stage 1 baseline and is
#: never written by this package.
EVALUATION_RESULTS_SUBDIR = "evaluation"
