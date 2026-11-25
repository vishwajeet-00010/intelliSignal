# What was measured, and how the model was chosen

**Date:** 25 November 2026. Every number here is read out of
`selection.json` beside this file by `tools/write_findings.py`; none of
them was typed.

---

## 1. The headline, and the honest version of it

On traffic the model had never been trained or selected on, measured
once:

| controller | delay per vehicle | against the best fixed plan | patterns won | worst pattern |
|---|---|---|---|---|
| **the promoted model** | **14.97 s** | **-40.9 %** | **6 / 6** | -1.74 s |
| best fixed plan (ns 30 / ew 15), chosen on validation | 25.32 s | - | - | - |
| the best fixed plan with hindsight (ns 16 / ew 15) | 15.01 s | -0.2 % | - | - |

**Read the two baselines separately, because they say different things.**

The first is the fixed plan chosen the same way the model was: tuned on
one set of traffic, then measured on another. That is the only fixed
plan anyone could actually deploy, because a real junction's timings
have to be set before tomorrow's traffic arrives. Against it the model
is better by 40.9 % (14.97 s against 25.32 s).

The second is the best fixed plan *for this particular test traffic*,
found by measuring all 64 of them on it afterwards. Nobody
could choose that plan in advance - it is picked with hindsight. Against
it the model is within 0.2 % (14.97 s against 15.01 s), which on six patterns is a tie.

So the result is not that a learned controller beats fixed timing by
41 %. It is this: **the model matches the best fixed plan that
exists for the traffic it meets, without being told what that traffic
will be.** The fixed plan that was chosen honestly lands 10.31 s worse,
because the optimum moved - ns 30 / ew 15 on the
validation traffic, ns 16 / ew 15 on the test
traffic. Not having to know which one you will get is what the
adaptivity buys.

It also beats both adaptive benchmarks, which do not need to be told
the traffic either: better by 18.3 % (14.97 s against 18.33 s) against actuated control, and
better by 22.2 % (14.97 s against 19.24 s) against max-pressure. It is the only controller here
that does not lose on a single pattern.

---

## 2. What was being chosen between

| action space | runs | best | worst | best run |
|---|---|---|---|---|
| A - duration menu | 10 | 18.84 | 22.20 | A_s44 |
| B - lock and extend | 10 | 17.00 | 21.07 | B_s45 |

Each run is the same recipe with a different random seed: 320
episodes on 18 varied traffic patterns, keeping the checkpoint
that scored best on the validation patterns. The spread within a family
is the reason a single run is not evidence of a method.

Every run was then re-scored under identical conditions - 3 SUMO
seeds on each of the 6 validation patterns - because the score
a trainer reports for itself is one draw.

### Averaged ensembles

| ensemble | members | validation delay |
|---|---|---|
| ensemble Ax3 | A_s44, A_s43, A_s45 | 18.81 |
| ensemble Ax5 | A_s44, A_s43, A_s45, A_s51, A_s42 | 19.13 |
| ensemble Bx3 | B_s45, B_s44, B_s42 | 17.44 |
| ensemble Bx5 | B_s45, B_s44, B_s42, B_s46, B_s50 | 20.45 |

Averaging the Q-values of several independently trained networks usually
keeps what they agree on and cancels what is peculiar to one of them.
Here it did not beat the best single run, so no ensemble was promoted.
That is reported because it is what happened, not because it helps.

---

## 3. The rule, declared before the test traffic was read

> Lowest validation mean of the metric among single runs and averaged ensembles whose every run obeyed the timing profile. Decided before the test split was measured.

Chosen: **B_s45**.

---

## 4. The best fixed plan is the best of all of them

| ns green | ew green | validation delay |
|---|---|---|
| 30 | 15 | 19.86 |
| 30 | 17 | 20.67 |
| 30 | 16 | 21.05 |
| 25 | 16 | 21.62 |
| 25 | 15 | 22.32 |

The worst plan on the menu, ns 15 s / ew 30 s, gives
52.01 s - more than twice the best. A comparison against a plan
chosen carelessly would say more about the plan than about the model.

---

## 5. The test traffic, measured once

| controller | delay/veh | wait/veh | completed | turned away | vs fixed | won | worst |
|---|---|---|---|---|---|---|---|
| selected model | 14.97 | 7.96 | 462.9 | 0.0 | -40.9 % | 6 / 6 | -1.74 |
| best fixed plan chosen WITH hindsight on the test traffic (ns 16 / ew 15) | 15.01 | 8.04 | 462.0 | 0.0 | -40.7 % | 6 / 6 | -1.75 |
| best candidate A run (A_s44) | 15.77 | 8.58 | 462.3 | 0.0 | -37.7 % | 6 / 6 | -1.63 |
| actuated (gap 3 s) | 18.33 | 11.27 | 461.3 | 0.0 | -27.6 % | 3 / 6 | +1.85 |
| max-pressure | 19.24 | 11.94 | 460.8 | 0.0 | -24.0 % | 3 / 6 | +1.82 |
| best fixed plan | 25.32 | 16.77 | 450.1 | 0.0 | - | - | - |

`delay/veh` is total delay per requested vehicle: time lost on the road,
plus time a vehicle spent unable to get in, plus the delay of vehicles
still inside the junction when the clock ran out, divided by every
vehicle the traffic asked for. Lower is better, and it cannot be improved
by refusing traffic.

`won` is how many of the 6 test patterns the controller beat
the fixed plan on, and `worst` is its worst single pattern against it. A
controller that wins on average while losing badly somewhere is not the
same as one that never loses.

---

## 6. Two corrections this round, both of which changed the numbers

**A vehicle still inside the junction at the end was costing nothing.**
SUMO writes its trip record when a vehicle arrives, so traffic still on
the network when the run ended was absent from the delay sum while still
being counted in the denominator. The worse the controller, the more of
those it had, so the measure quietly flattered whichever controller
stranded the most traffic. SUMO is now asked for the unfinished trips as
well (`tripinfo-output.write-unfinished`) and they are charged. The delay
accounting is version 3; numbers from version 2 are not comparable.

**The lock-and-extend controller was being asked the wrong question.** It
serves a minimum green and then decides, every three seconds, whether to
extend. The decision was being taken from the junction as it looked
*before* that minimum green had run - fifteen seconds of stale
information - so the policy was choosing whether to extend a green it
had not yet seen the effect of. It learned to end almost every green at
the lock, which made an adaptive controller behave like a fifteen-second
fixed plan while reporting itself adaptive. The observation is now taken
after the lock. Related: the exploration schedule assumed one decision
per green, which is right for the duration menu and wrong for
lock-and-extend, so that family stopped exploring at roughly episode 116
of 320 while the other explored to 241. Both families now explore for the
same share of the budget.

Everything in this document was measured after both corrections. The
models trained before them were discarded, not re-used.

---

## 7. How to reproduce it

```bash
# twenty runs, two action spaces, ten seeds each
python training/train_profile_dqn.py --profile standard --candidate B \
    --episodes 320 --validate-every 20 --seed 42 --out output/run

# score every run and every fixed plan, then read the test traffic once
python evaluation/select_profile_model.py --profile standard \
    --runs output/standard_20261125/runs --out output/standard_20261125/selection --seeds 3

# write the chosen model into the registry, with this report attached
python evaluation/promote_profile_model.py --profile standard \
    --selection output/standard_20261125/selection/selection.json --selected-on validation

# the end-to-end checks, against the real simulator
python tools/verify_product.py
```

---

## 8. What this does not show

* Six test patterns at 3 seeds each is a small sample.
* The traffic is deterministic. Every vehicle is written out
  individually, which is what makes two controllers comparable, but it
  is not the same as stochastic arrivals.
* The fleet shares are representative of an Indian urban arterial, not a
  survey of one junction, and the saturation headway was not validated
  against field data.
* One junction, no coordination with a neighbour, no pedestrians.
* `moderate` and `heavy` still use a single generic car with no startup
  delay. Their numbers are kept so the earlier stages reproduce, and
  nothing in the code or the dashboard puts them in a table with
  anything measured here.


