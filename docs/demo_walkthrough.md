> **Numbers here predate the current model family.** The walkthrough
> itself is still broadly how a demonstration is given, but every figure
> below was measured before the Indian urban fleet and the 15-30 s
> timing rules were in place, and the dashboard has been rebuilt around
> five pages since it was written. The current figures are in
> `output/standard_20261125/FINDINGS.md`, and `docs/dashboard.md`
> describes the dashboard as it is now.

# Teacher demonstration — a ten-minute walkthrough

Everything below was run on 23 Nov 2026 and the numbers are the ones the
machine produced. Start the dashboard with:

```bash
streamlit run dashboard.py
```

---

## 0 · What the audience should take away

A conventional traffic signal gives each direction a fixed green, every
cycle, whatever the traffic. This project asks whether a learned controller
does better — and answers it by measuring, not by asserting. Sometimes the
answer is no, and the demonstration shows that too.

---

## 1 · Dashboard (30 seconds)

The landing page: the junction, the traffic currently selected, the
controller currently selected, and three numbers — average waiting time,
vehicles completed, average queue length.

> "This is one four-way intersection in SUMO. Eight incoming lanes, two per
> approach. Every number on this screen came from a simulation that
> actually ran."

---

## 2 · Custom Traffic — the 200 and 300–350 example (3 minutes)

Open **Custom Traffic**. Press **Generate example**, or type it in:

| lane | what to enter |
|---|---|
| North lane 0 (kerb-side) | Vehicles **200** |
| East lane 0 (kerb-side) | min **300**, max **350** |

Leave speed, turns and arrival windows **blank**. Set the run length to
**900 s**. Press **Preview traffic**.

The preview that comes back:

```
workload 33eddcd49f48f374   528 vehicles over 900 s

North lane 0 (kerb-side)   200 veh   0-900 s    800 veh/h   63 right, 137 straight
East  lane 0 (kerb-side)   328 veh   0-900 s   1312 veh/h  103 right, 225 straight
```

Three things to point at:

* **328, not "300–350".** The range was resolved once, with the seed, and
  it is marked `·auto`. Press **Regenerate** and it changes; the 200 you
  typed does not.
* **63 right / 137 straight** is marked `·derived`. You gave a total, so
  the split came from the base scenario's turning ratios — restricted to
  what that lane can do. The kerb-side lane cannot turn left, so it is not
  offered the option.
* **1312 veh/h** is above what one lane can discharge, and the page says so
  in a warning. That is not an error; it is the reason the run will report
  vehicles *waiting to enter*.

Open **The first vehicles, as SUMO will receive them**:

> "This is the actual route file. Each vehicle individually, with its
> departure time and its lane. That is why both controllers can be given
> exactly the same traffic rather than the same random seed."

### Show that the arrival window matters

Change East lane 0's window to **0 → 300 s** and press Preview again. The
rate jumps to 3936 veh/h and the workload id changes. Measured on a smaller
example: 200 cars over 1800 s gives **9.12 s** mean waiting; the same 200
cars in the first 300 s gives **32.74 s**. Same vehicles, different traffic.

Change it back before moving on.

---

## 3 · Run it under Fixed-Time, then under the DQN (2 minutes)

Pick **Custom traffic** and **Fixed-Time** in the sidebar, press **Run**,
then switch to **DQN Adaptive** and press **Run** again. Open
**Comparison**:

| controller | waiting (s) | completed | waiting to enter |
|---|---|---|---|
| Fixed-Time | 10.78 | 502 | 0 |
| DQN | 16.43 | 435 | 41 |

> "The DQN is worse here, and the dashboard says so. This is the model
> trained on balanced moderate traffic, and this workload is nothing like
> it — all the demand is on two lanes out of eight. A controller that was
> only ever shown one kind of traffic does not automatically generalise."

The five vehicle counts are the point of this table. The DQN did not just
wait longer — it let 41 fewer vehicles onto the network. Reporting
throughput alone would have hidden that.

---

## 4 · Hybrid — letting the DQN try, with a deadline (3 minutes)

Switch the controller to **Hybrid** and press **Run**. The page moves to
**Comparison** and shows a progress bar.

> "Hybrid starts the traffic on a valid fixed plan immediately, and then
> spends up to five minutes of *real* time — not simulated time — tuning
> that plan and testing the DQN against it. If the DQN wins, it takes over.
> If it does not, the best measured fixed plan keeps the signal."

What it reported on this workload:

```
Hybrid selected tuned fixed time (ns 15 s / ew 20 s) after 1 check.
Mean waiting was 6.65 s with fixed time and 18.08 s with the DQN.

Why: Mean waiting was 18.08 s with the DQN against 6.65 s with fixed time,
     so fixed time keeps the signal. A tie goes to fixed time.
```

Two things worth saying out loud:

* **Tuning alone was worth 38 %.** The configured plan gave 10.78 s; tuning
  north–south and east–west independently found 15 s / 20 s at **6.63 s**.
  Most of the win on this traffic was in the fixed plan, not in the
  learning.
* **The DQN was rejected on evidence.** Open *Validation checks* to show the
  matched comparison: same vehicles, same seeds, and all three conditions —
  lower waiting, at least equal throughput, no increase in vehicles waiting
  to enter.

Under **Advanced** the three limits are editable: 5 minutes, 30 extra
training episodes, patience of 3 consecutive failed checks.

---

## 5 · Where the DQN does win (1 minute)

Switch the traffic to **Heavy** and show the measured result from 22 Nov,
on confirmatory seeds 6101–6120:

| controller | waiting (s) | throughput |
|---|---|---|
| duration DQN | **5.151** | **1468.5** |
| best fixed plan (ns 7 / ew 6) | 5.754 | 1466.5 |
| the deployed older DQN | 16.469 | 1420.5 |
| original 30/30 plan | 53.029 | 1349.8 |

> "On the traffic it was trained for, the learned controller beats the best
> fixed plan that exists — by 10.5 %, with higher throughput, winning on 17
> of 20 seeds. On traffic it has never seen, it loses, and Hybrid catches
> that. Both of those are the honest result."

---

## 6 · Simulation page (1 minute)

Open **Simulation** and drag the time slider.

> "The junction, the clock, and one sentence: *North–South has green.* The
> signal colours are decoded from what TraCI actually recorded. The
> Q-values the network produced at that moment are under *Decision
> details* — there, but not in the way."

---

## If something goes wrong on the day

| symptom | what to do |
|---|---|
| Run is greyed out on Custom traffic | No preview yet — press **Preview traffic** |
| "Enter traffic on at least one lane" | The form is blank; press **Generate example** |
| Hybrid seems stuck | Press **Refresh**; the search runs in the background and the page reconnects to it |
| A page shows "no results yet" | That is honest, not broken — press **Run** |
| SUMO window does not open | Uncheck **Open the SUMO window** under Advanced |

---

## The numbers quoted above, and where they came from

| claim | produced by |
|---|---|
| 9.12 s vs 32.74 s for the two arrival windows | `custom_traffic` resolver + SUMO, 200 vehicles on one lane |
| 10.78 / 16.43 s on the custom workload | `output/demo_check_20261123/verify_demo.py` |
| Hybrid selecting 15/20 s at 6.63 s | `simulation/hybrid_job.py`, 240 s budget |
| heavy 5.151 vs 5.754 | `output/dqn_phase_20261122/FINDINGS.md` |
| Moderate and Heavy reproduce on seed 6101 | `output/demo_check_20261123/verify_demo.py` |
