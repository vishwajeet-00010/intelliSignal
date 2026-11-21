# Stage 5 attempt 3 lineage

- Parent protocol and original artifacts: `output/dqn_stage5_20261121/`. The original primary set has one complete, one partial, and 18 unstarted cells; none is reused here.
- Attempt 2: `output/dqn_stage5_20261121_attempt2/`. The sandboxed SUMO/TraCI connection failed with Windows socket error 10013 during its first cell. Its run directory, status and log remain preserved and excluded.
- Attempt 3: fresh manifest, 20 queued cells, seeds 6207–6211, and a protected-file hash snapshot taken before training. All training and evaluation outputs are confined to this directory.
- Compared with the original pre-change Stage 5 snapshot, the attempt 3 pre-run snapshot includes the newly created `models/active_models.json` and a changed `config/dqn/training.yaml`. These pre-existing changes were inspected and retained. Attempt 3 integrity will compare its own before and after snapshots, not claim that the two snapshots taken at different project states are identical.
