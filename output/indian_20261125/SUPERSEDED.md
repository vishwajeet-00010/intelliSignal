# Superseded by `output/standard_20261125/`

**Date:** 25 November 2026.

The models trained in this directory are gone, deliberately. Two defects
found after they were trained mean their weights and their numbers
cannot be compared with anything current, and keeping them invited
exactly that comparison.

**The lock-and-extend controller was asked the wrong question.** It
serves a minimum green and then decides, every few seconds, whether to
extend. The decision was being taken from the junction as it looked
*before* that minimum green had run, so the policy was choosing whether
to extend a green whose effect it had not yet seen. Everything trained
here learned to end almost every green at the lock, which made an
adaptive controller behave like a fifteen-second fixed plan.

**Vehicles still inside the junction at the end cost nothing.** SUMO
writes a vehicle's trip record when it arrives, so traffic stranded on
the network when the run ended was missing from the delay sum while
still being counted in the denominator. The worse the controller, the
more of those it had. The delay accounting is now version 3 and charges
them; version 2 numbers are not comparable with version 3 numbers.

## What was kept

Everything that is a record rather than an artefact: `FINDINGS.md`, the
per-run `summary.json`, `training.csv` and `validation_history.json`,
the sweep scripts and the comparison JSON. The 25 November findings in
this directory are still an accurate account of *the startup-lost-time
discovery*, which is what they were written for, and that conclusion is
unaffected by either defect above.

## What replaced it

`output/standard_20261125/` - twenty runs retrained on the corrected
environment, re-scored on validation at three SUMO seeds each, with the
baseline being the best of all sixty-four fixed plans rather than the
end of one search. See its `FINDINGS.md`.

The removed checkpoints were backed up outside the repository before
deletion; they are not in git history, so they are not recoverable from
this repository.
