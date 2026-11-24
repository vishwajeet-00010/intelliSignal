# The four-second green: diagnosis, fix, and what the 15–30 s profile costs

**Date:** 24 Nov 2026. **Branch:** `feature/timing-profiles`.
**Start commit:** `214dc4ee85d260aea8a19ceafe82501f22614f8c` on `main`.

---

## 1. Separate conclusions, as asked

| question | answer |
|---|---|
| **Timing behaviour fixed?** | **Yes.** Greens are bounded by a profile that is enforced in the execution controller, during exploration, evaluation, custom runs and Hybrid. Verified in real SUMO runs for every controller under both profiles: 18/18 checks. |
| **Functional regressions?** | **None found.** 446 tests pass, 1 skipped. Moderate and Heavy reproduce their saved seed-6101 results exactly (6.670 s and 2.790 s). |
| **DQN performance target achieved?** | **Yes under the standard profile**, on the untouched test split: 15.89 s against the tuned fixed plan's 22.51 s, **−29.4 %**, winning on **6 of 6** workloads. **Not under the responsive profile**, where a constrained actuated controller is better on the honest metric. |

---

## 2. What actually caused the four-second greens

Not a timing setting, not the discount, not the reward. **The observation
was saturating.**

The moderate model's scales were calibrated on moderate traffic:
`max_queue_per_direction 7`, `max_queue_per_approach 4`,
`max_waiting_per_direction 29`. A custom workload with 1312 veh/h on one
lane is far outside that, and the inputs clipped to exactly 1.0 and stayed
there:

| feature | observations pinned at 1.0 |
|---|---|
| `worst_approach` | **93.2 %** |
| `served_waiting` | **74.8 %** |
| `served_queue` | 46.6 % |
| `other_approaching` | 46.6 % |
| `served_approaching` | 40.8 % |

With a near-constant input the network returned a near-constant action:
**91 of 103 greens (88 %) were the shortest on its menu.** That reproduces
the reported 208-of-222 (94 %) closely enough to be the same failure.

**The fix** is a monotone soft saturation, `v / (v + k)`, instead of
`v / k` then clip. At the scale it still reads 0.5; at five times the
scale it reads 0.83; it never reaches 1.0 and never maps two different
congestion levels onto the same number. It is versioned
(`observation_version: 2`) and a checkpoint with no `saturation` key keeps
the clipping it was trained with, so **no existing model changed
behaviour**.

### Hypotheses that were checked and found sound

| checked | verdict |
|---|---|
| elapsed-time discounting | **already correct** — `gamma ** duration` plus within-step discounting, committed `7ad3b2d` |
| `getAccumulatedWaitingTime` / waiting-time memory | **already correct** — `waiting-time-memory` is set to the full episode; measured peak accumulated waiting 310 s, 15 vehicles above the 100 s SUMO default, so the setting is in effect |
| action masking in the learning target | **was not applicable, now implemented** — the duration menu has no illegal action; the new lock-and-extend space does, and the mask is now carried into the bootstrap, not only into action selection |
| short greens as a malfunction | **false on trained traffic.** They are the measured optimum there: the best *fixed* plans are also short (ns 7/ew 6 heavy, ns 6/ew 4 moderate) |

---

## 3. What the 15–30 s profile costs

This was measured before any training, over every plan with both greens in
15–30 s:

| | best plan under 15–30 s | best plan unconstrained | cost |
|---|---|---|---|
| heavy | ns 18 / ew 15 → **42.50 s** | ns 7 / ew 6 → 5.754 s | **7.4×** |
| moderate | ns 15 / ew 15 → **5.57 s** | ns 6 / ew 4 → 2.761 s | **2.0×** |

That is the ceiling for *any* controller under the profile. It is the
price of the operating policy, not a defect, and it is why the two
profiles are never compared with each other anywhere in the code or the
dashboard.

Both profiles are kept and selectable. `responsive` (4–20 s) is what every
model before 24 Nov was trained with; `standard` (15–30 s) is conventional
practice.

### The red limit

`max continuous red ≤ 4 × that group's preceding green`, measured from
entering red to the next green. On a two-group junction a group's red is
exactly the other group's green plus the clearance, so:

| profile | worst-case red | tightest limit | binds? |
|---|---|---|---|
| standard | 30 + 4 = **34 s** | 4 × 15 = **60 s** | **never** |
| responsive | 20 + 4 = 24 s | 4 × 4 = 16 s | **yes** |

Under the profile that was asked for the rule **can never fire**. It is
implemented and enforced anyway, because it does bind under the responsive
profile — where it was measured shortening a requested 20 s green to 12 s.
It is not claimed as an improvement.

The first red of an episode has no preceding green, so the limit is
`4 × min_green`. Stated explicitly rather than left to a default.

---

## 4. The measurement was not trustworthy, and now is

`average_waiting_time` divides the waiting of vehicles **seen** by the
number of vehicles **seen**. A vehicle SUMO never inserted is in neither
term, so a controller that blocks a saturated entry is rewarded twice.

A second, versioned accounting is now computed from SUMO's own
`tripinfo` output. The historical field is **unchanged** — same name, same
definition, same numbers — and the new one sits beside it:

```
requested · entered · completed · still on road · waiting to enter
mean stopped waiting per entered     (the historical measure)
mean time loss per entered           (a SUPERSET of stopped waiting)
mean insertion delay per requested   (disjoint from both)
mean TOTAL DELAY per requested       (time loss + insertion delay)
```

`timeLoss` already contains stopped time, so the two are never added.
Every run also carries a cross-check between the live collector and the
trip information; it agreed on completions in every verification run.

**Total delay per requested vehicle is the number the comparison leads
with**, because it is the only one a controller cannot improve by refusing
vehicles.

---

## 5. Training

Eighteen workloads, six shapes × three intensities: balanced, directional,
lane-skewed, bursty, ramping and turn-heavy at light, moderate and heavy
demand. Train / validation / test splits share **no vehicle list and no
SUMO seed**, and validation and test share no shape.

Declared budget, followed: a 3-episode pilot to measure runtime (1.0 s per
episode), then **8 runs × 300 episodes** — candidates A and B at three
seeds each on the standard profile, candidate A at two seeds on the
responsive profile. Roughly 30 minutes wall-clock.

| run | candidate | profile | best validation delay | at episode |
|---|---|---|---|---|
| A_rsp_s42 | A | responsive | 9.92 | 30 |
| A_rsp_s43 | A | responsive | 10.43 | 240 |
| **B_std_s44** | **B** | **standard** | **16.28** | **75** |
| B_std_s43 | B | standard | 16.82 | 165 |
| B_std_s42 | B | standard | 17.07 | 75 |
| A_std_s42 | A | standard | 17.28 | 150 |
| A_std_s43 | A | standard | 18.12 | 15 |
| A_std_s44 | A | standard | 19.49 | 270 |

Candidate B beat candidate A on all three standard-profile seeds.

**Candidate A** chooses a whole green from a menu. **Candidate B** serves a
15 s lock and then decides every 3 s whether to extend, up to 30 s. A
short decision interval does not become a short green: the lock runs
before the first question is asked, so the shortest reachable green is 15 s.

---

## 6. The comparison, on the untouched test split

The fixed baseline was **tuned on validation and reported on test**, with
north-south and east-west swept independently — the same discipline as the
models.

### Standard profile (15–30 s), 6 test workloads

| controller | delay/veh | sd | wait/veh | completed | waiting to enter | compliant |
|---|---|---|---|---|---|---|
| **DQN candidate A** | **15.56** | 10.77 | 8.86 | 533.2 | 0.0 | yes |
| DQN candidate B | 15.89 | 10.91 | 9.33 | 531.2 | 0.0 | yes |
| actuated (gap 3 s) | 17.11 | 8.63 | 11.16 | 531.2 | 0.0 | yes |
| max-pressure | 18.27 | 7.79 | 12.45 | 531.2 | 0.0 | yes |
| tuned fixed ns 30 / ew 15 | 22.51 | 13.78 | 17.50 | 512.7 | 8.8 | yes |

The spread is large because the test split spans light to heavy, so the
**paired** per-workload comparison is the honest statistic:

| controller | workloads won | mean difference | worst case |
|---|---|---|---|
| **DQN candidate A** | **6 / 6** | −6.94 s | −2.43 s |
| **DQN candidate B** | **6 / 6** | −6.62 s | −1.52 s |
| actuated | 4 / 6 | −5.39 s | **+1.62 s** |
| max-pressure | 3 / 6 | −4.24 s | **+1.61 s** |

Both learned candidates beat the tuned fixed plan on **every** workload and
are the only controllers that never lose on any of them. They also complete
*more* vehicles (533 and 531 against 513) and leave none waiting to enter
against the fixed plan's 8.8 — so the lower delay is not bought by
excluding traffic.

### Responsive profile (4–20 s) — the DQN does **not** win

| controller | delay/veh | wait/veh | completed | won |
|---|---|---|---|---|
| actuated (gap 3 s) | **16.06** | 9.44 | 532.5 | 4/6 |
| max-pressure | 17.51 | 10.92 | 530.7 | 3/6 |
| DQN candidate A | 18.20 | **8.38** | 521.2 | 6/6 |
| tuned fixed ns 10 / ew 4 | 26.83 | 15.87 | 494.7 | — |

This is worth stating plainly: **the responsive DQN has the best mean
waiting of any controller (8.38 s) and is third on total delay**, because
it completes 11 fewer vehicles. That is exactly the failure the new metric
was built to expose, and it is exposing it on our own model. It is
consistent on the paired view (6/6) but its mean gain is smaller than
actuated's.

---

## 7. What was promoted, and what was not

**Promoted:** `B_std_s44` for the `standard` profile, under a new
`profile_models` section of `models/active_models.json`.

It was **selected on validation** (best of six at 16.28 s) and *then*
reported on test (15.89 s). Candidate A happened to edge it on test
(15.56 s) but was not the validation winner, so it was **not** promoted —
promoting it would have been selecting on the split the number is reported
with.

**Not promoted:** anything for the responsive profile, because the learned
candidate loses to the actuated benchmark on the honest metric there.

**Kept, unchanged:** the Stage 3/4 KEEP/SWITCH models in `active_models`
and the per-scenario duration models in `phase_duration_models`. They are
historical evidence. A checkpoint whose menu falls outside a profile is
**refused** at promotion and at load time — verified: the 4–20 s model is
rejected for the standard profile.

---

## 8. Known limitations

* **The two profiles are not comparable.** 15.89 s under `standard` and
  the 5.151 s recorded on 22 Nov under the heavy scenario measure
  different operating policies on different traffic. Nothing in the code
  or the dashboard puts them in one table.
* **The red limit is dead code under `standard`.** Enforced, tested, and
  provably never triggered on a two-group junction.
* **Max-pressure is simplified.** Every downstream edge here is an exit
  with no signal, so pressure reduces to the upstream queue. It is a
  benchmark, not a claim about max-pressure control in general.
* **Demand is still deterministic.** `<flow vehsPerHour=...>` with no
  `random="true"`; custom workloads write every vehicle out explicitly.
  The comparison is exact, but a claim about real traffic needs stochastic
  arrivals and that experiment has not been run.
* **Six test workloads, one seed each.** The paired view is strong (6/6)
  but the sample is small. More seeds would tighten it.
* **Candidate A versus B is unresolved.** B won validation on all three
  seeds, A won test by 0.33 s. Within noise; B was promoted on the
  validation rule.
* **Running the test suite rewrites eleven committed PNGs** under
  `results/plots/`. Pre-existing; restore with
  `git checkout -- results/plots/` after a run.

---

## 9. Reproducing

```bash
# the cost of the profile, measured before any training
python output/dqn_phase_20261122/grid.py --scenario heavy \
    --ns 15,18,20,22,25,30 --ew 15,18,20,22,25,30 --seeds 6101,6102 \
    --out heavy_constrained.csv

# pilot, then the declared budget
python training/train_profile_dqn.py --profile standard --candidate B \
    --episodes 300 --seed 44 --pilot
python training/train_profile_dqn.py --profile standard --candidate B \
    --episodes 300 --validate-every 15 --seed 44 --out output/run

# the fair comparison, baseline tuned on validation and reported on test
python evaluation/compare_profiles.py --profile standard \
    --candidate-a output/.../A_std_s42/models/best.pt \
    --candidate-b output/.../B_std_s44/models/best.pt

# end-to-end verification
python output/audit_20261123/verify_profiles.py

# the dashboard
streamlit run dashboard.py          # http://localhost:8501
```

Environment: Python 3.12.10, torch 2.14.1, gymnasium 1.4.0, SUMO 1.28.0.
Suite: **446 passed, 1 skipped**.
