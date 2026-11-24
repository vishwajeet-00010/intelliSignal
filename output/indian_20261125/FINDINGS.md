# Why the greens were four seconds, and what fixed it

**Date:** 25 Nov 2026.

---

## 1. The short answer

The four-second green was not a tuning mistake. **The simulation had no
startup lost time**, so a four-second green really did produce nearly
four seconds of discharge, and short cycles really were optimal *in that
model*. The model was wrong, not the method.

Real vehicles do not move the instant the light changes. The first driver
takes about a second to react, the driver behind reacts to them, and the
wave takes several seconds to reach the back of the queue. Until now every
vehicle in this project was one generic car with `startupDelay` unset.

With an Indian urban fleet and real startup delay, measured on this
junction:

| green | mean waiting |
|---|---|
| **4 s** | **61.3 s** |
| 6 s | 15.4 s |
| 8 s | 9.5 s |
| 10 s | 8.8 s |
| **12 s** | **8.2 s** ← optimum |
| 15 s | 12.1 s |
| 20 s | 13.2 s |
| 30 s | 20.1 s |
| 50 s | 26.7 s |

A four-second green is now the *worst* plan on the board, by a factor of
seven. The 15 s minimum is no longer a costly constraint fighting the
model — it sits just past the optimum and costs about 49 %, which is what
a pedestrian-safe minimum is worth.

Before this change the same constraint cost **7.4×** on the old heavy
scenario. That gap was an artefact of the missing physics.

---

## 2. The Indian urban fleet

`custom_traffic/fleet.py`, selected with `traffic_model: indian_urban`.

| vehicle | share | length | starts after | PCU |
|---|---|---|---|---|
| motorcycle | 62 % | 2.2 m | 0.6 s | 0.25 |
| car | 22 % | 4.5 m | 1.2 s | 1.00 |
| auto-rickshaw | 11 % | 3.2 m | 1.0 s | 0.50 |
| bus | 5 % | 11.0 m | 2.0 s | 3.00 |

Mean 0.58 PCU per vehicle, mean 0.85 s before a stopped vehicle moves off.
Clearance is a 3 s amber plus a 2 s all-red, as Indian practice uses.

Demand is reported in **PCU as well as vehicles**, because sixty
motorcycles are not sixty cars.

### A mistake worth recording

The first version of this fleet used `tau = 0.8` for motorcycles to
represent short Indian headways. It produced **400–565 rear-end collisions
per run**. The Krauss car-following model is only collision-free while the
desired headway is at least one simulation step, and the step here is
1.0 s. Every number from that first sweep was meaningless, including a
tempting "optimum at 15 s" that matched what was wanted.

Short headways are now expressed through `minGap` and `sigma`, and
`tau ≥ 1.0` everywhere. A test pins it.

### Demand was calibrated, not guessed

With realistic headways the junction's capacity is far lower than the
collision-ridden run implied:

| ns / ew veh/h | completed | mean waiting |
|---|---|---|
| 1200 / 900 | 68 % | 92.5 s (oversaturated) |
| 900 / 700 | 87 % | 38.5 s |
| **700 / 550** | **93 %** | **16.1 s** ← used |
| 550 / 420 | 94 % | 9.9 s |

---

## 3. Results — standard profile, untouched test split

Six varied workloads, Indian fleet, 15–30 s greens. The fixed baseline was
tuned on validation (north-south and east-west independently) and reported
on test, the same discipline as the models.

| controller | delay/veh | sd | wait/veh | completed | pending |
|---|---|---|---|---|---|
| **DQN candidate A** | **14.76** | 5.92 | 8.33 | 463.5 | 0.0 |
| **DQN candidate B** | **14.94** | 4.88 | 8.39 | 463.2 | 0.0 |
| actuated (gap 3 s) | 16.91 | 5.09 | 10.77 | 461.7 | 0.0 |
| tuned fixed ns 22 / ew 15 | 17.72 | 7.73 | 11.49 | 458.5 | 0.0 |
| max-pressure | 18.05 | 4.00 | 11.67 | 462.2 | 0.0 |

Paired per workload, which is the honest statistic when the split spans
light to heavy:

| controller | won | mean difference | worst case |
|---|---|---|---|
| **DQN candidate A** | **6 / 6** | −2.96 s | −0.30 s |
| **DQN candidate B** | **6 / 6** | −2.78 s | −0.05 s |
| actuated | 2 / 6 | −0.81 s | **+4.05 s** |
| max-pressure | 1 / 6 | **+0.33 s** | +3.27 s |

**Both learned controllers beat the tuned fixed plan on every single
workload and are the only ones that never lose on any of them.** Under
the old physics the actuated benchmark was competitive; with realistic
startup lost time the learned controller separates clearly, because
knowing *when* to end a green is worth more when ending it early is
expensive.

Max-pressure is now slightly worse than the tuned plan overall.

---

## 4. Automatic fallback to Hybrid

The DQN mode no longer refuses a run when it has no validated model. It
hands over to Hybrid and says so on the Comparison page:

> **The DQN did not take the signal — Hybrid is deciding instead.**
> No DQN has been trained and validated for the 'standard' timing profile yet.

It falls back when any of these hold:

* no model is registered for the active profile;
* the registered model's green menu falls outside the profile;
* the model's promotion record does not show it beating the tuned fixed
  plan it was measured against.

Hybrid then tunes the fixed plan on the actual traffic, tests whatever DQN
exists against it, and uses whichever wins. A tie goes to fixed time.
Hybrid is also now the **default controller** for the same reason: it is
the setting that cannot produce a dishonest result.

---

## 5. What the dashboard now says

* Opens on **Indian arterial · standard profile · Hybrid**.
* Every page states the timing rules in force and the fleet on the road.
* Demand is shown in **vehicles and PCU**, with the mix.
* The signal plan is shown as a traffic engineer reads it: north-south
  green, east-west green, clearance, cycle.
* The **About** page carries the argument directly:

  | green | lost to starting off | actually moving traffic | useful share |
  |---|---|---|---|
  | 4 s | 2.8 s | 1.2 s | 29 % |
  | 8 s | 2.8 s | 5.2 s | 65 % |
  | 15 s | 2.8 s | 12.2 s | 81 % |
  | 30 s | 2.8 s | 27.2 s | 91 % |

---

## 6. Verified

* **452 tests pass**, 1 skipped.
* **18/18 end-to-end checks**: every controller obeys both profiles in real
  SUMO; a 4–20 s model is refused under the 15–30 s profile; a custom run
  saves, reloads and replays with matching numbers; the live collector and
  SUMO's trip information agree on completions.
* Moderate and Heavy still reproduce their saved seed-6101 results
  exactly (6.670 s and 2.790 s) — those scenarios keep the original single
  car and were not touched.
* Dashboard serves on `streamlit run dashboard.py`.

---

## 7. Known limitations

* **The fleet shares are representative, not from a survey.** 62/22/11/5
  is typical of an Indian urban arterial but is not a count of a specific
  junction. A real deployment would calibrate from local data.
* **Saturation headway was not separately validated** against an Indian
  field study; it emerges from `tau`, `minGap` and `startupDelay` and lands
  in a plausible range, which is not the same as being measured.
* **`moderate` and `heavy` still use the old single car.** Their numbers
  are preserved deliberately, so they are not comparable with anything
  measured on the Indian fleet, and nothing in the code puts them in one
  table.
* **Six test workloads, one seed each.** 6/6 is a strong paired result but
  a small sample.
* **A versus B is unresolved.** B won validation (15.14 against 16.70), A
  won test by 0.18 s. Within noise; B was promoted on the validation rule.
* **Demand is still deterministic** — vehicles are written out
  individually, which makes the comparison exact but is not the same as
  stochastic arrivals.
* The red limit remains provably unable to fire under the standard
  profile, as before.

---

## 8. Reproducing

```bash
# where the optimum sits once startup lost time is modelled
python output/indian_20261125/sweep_green.py \
    --greens 4,6,8,10,12,15,20,30,50 --seeds 6101,6102 --out sweep.csv

# pilot, then the declared budget (6 runs x 320 episodes, ~20 min)
python training/train_profile_dqn.py --profile standard --candidate B \
    --episodes 320 --seed 42 --pilot
python training/train_profile_dqn.py --profile standard --candidate B \
    --episodes 320 --validate-every 20 --seed 42 --out output/run

# the fair comparison
python evaluation/compare_profiles.py --profile standard \
    --candidate-a output/indian_20261125/runs/A_s44/models/best.pt \
    --candidate-b output/indian_20261125/runs/B_s42/models/best.pt

# end-to-end verification, and the dashboard
python output/audit_20261123/verify_profiles.py
streamlit run dashboard.py
```
