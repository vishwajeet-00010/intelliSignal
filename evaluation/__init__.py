"""
IntelliSignal
evaluation package: measuring a controller, and comparing controllers.

The current pipeline, in the order it is used:

    harness.py                  every controller is measured here, and
                                nowhere else
    select_profile_model.py     score every trained run on the validation
                                traffic, try ensembles, sweep every fixed
                                plan, then read the test traffic once
    promote_profile_model.py    write the chosen model into the registry
                                with its fingerprint and its evidence
    compare_profiles.py         candidate A against candidate B
    fixed_time_sweep.py         measure every plan on the menu
    gates.py                    the promotion conditions, as a pure
                                function that can be tested

Historical, kept because the Stage 1-3 numbers must reproduce:
`evaluate_dqn.py`, `compare_controllers.py`, `plot_comparison.py`,
`evaluate_phase_dqn.py`, `promote_phase_model.py`,
`calibrate_phase_observation.py`, `collect_observation_stats.py`.

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
