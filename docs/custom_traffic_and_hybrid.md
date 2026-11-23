# Custom traffic and the Hybrid controller

Two additions made for the final demonstration, and how they work.

---

## 1 · Custom traffic

### The pipeline

```
CustomSpec            what the user typed; None means "not said"
     |  custom_traffic/resolver.py
     v
ResolvedWorkload      every value decided, each tagged user / derived / auto
     |  custom_traffic/builder.py
     v
SUMO .rou.xml         one explicit <vehicle> element per vehicle
```

The middle object is the point. The preview on screen is rendered from
`ResolvedWorkload`, the route file is written from the *same*
`ResolvedWorkload`, and both controllers in a comparison are handed that one
file. "What I was shown" and "what was simulated" cannot drift apart, and a
run can be reproduced from its workload id alone.

### What the form accepts, per lane

| field | may be blank | notes |
|---|---|---|
| vehicle count | yes | exact number, **or** a min–max range |
| arrival window | yes | start and end in simulated seconds |
| desired speed | yes | km/h; SUMO sets the actual speed from the traffic |
| turning split | yes | exact counts **or** percentages, per movement |

Lane 0 is the kerb-side lane and serves **right and straight**. Lane 1 is
the centre lane and serves **straight and left**. That is read from the SUMO
network, and `custom_traffic/network.py::verify_against_network()` fails if
the two ever disagree — a vehicle on a lane that cannot serve its route does
not raise an error in SUMO, it just quietly teleports.

### The resolution rules

1. **At least one real value is required.** An empty form is a mistake, not
   a request to invent a scenario. *Generate example* fills it first.
2. **Every explicit value survives, including zero.** A lane set to 0 stays
   empty. A lane that was never ticked stays at zero demand and is never
   filled in for the user.
3. **Missing independent values are filled** from the base scenario and the
   network's limits, with any randomness drawn from the workload's seed.
4. **Dependent values are derived.** 200 vehicles with 50 assigned to turn
   right puts the other 150 on that lane's remaining permitted movements, in
   the base scenario's proportions, using largest-remainder so the parts
   still add up to exactly 200.
5. **Contradictions are reported, not silently fixed**, and each message
   names the field to change — a backwards range, turn counts that exceed
   the lane total, percentages over 100, a window that ends before it
   starts or runs past the end of the simulation.

Each decided value carries its provenance (`user`, `derived`, `auto`) and
the preview marks it. **Regenerate** redraws only the generated values;
anything typed is untouched. Per-lane random streams mean editing one lane
does not reshuffle another.

### Why explicit vehicles instead of `<flow>`

Moderate and Heavy use `<flow vehsPerHour=...>`, which asks SUMO to insert
vehicles on its own schedule. That cannot express "200 cars in the first
five minutes", does not give an exact count, and makes two controllers
comparable only because SUMO happens to be deterministic. Writing the
vehicles out fixes all three.

It also makes the arrival window visible. The same 200 cars on one lane:

| window | mean waiting | completed |
|---|---|---|
| over 1800 s (one every 9 s) | 9.12 s | 197 |
| in the first 300 s (one every 1.5 s) | 32.74 s | 200 |

### Vehicle accounting

A congested run is only readable if these five are kept apart, so every
custom run reports all of them:

**requested** (asked for) · **entered** (put on the network) ·
**completed** (finished the route) · **still on road** ·
**waiting to enter** (never inserted — the lane was full).

"Throughput 435" on its own hides the difference between a signal that is
working badly and a lane that could not accept the demand at all.

---

## 2 · The Hybrid controller

### What it does

```
Run pressed  ->  traffic starts on a valid fixed-time plan
             ->  in the background, against a deadline:
                   1. tune north-south and east-west green independently
                   2. try an existing phase-duration DQN on this workload
                   3. if it does not already qualify, train a candidate,
                      freezing and validating it at intervals
deadline     ->  whatever has not finished is abandoned, and said so
```

The deadline is **wall-clock** time, not simulated time. A 1800 s
simulation runs in about fifteen seconds, so five real minutes is room for
roughly twenty measured runs.

### The three limits (editable under Advanced)

| limit | default | meaning |
|---|---|---|
| time | 300 s | real seconds from pressing Run, covering tuning, training and validation |
| episodes | 30 | additional training episodes for a candidate |
| patience | 3 | consecutive **completed** validation checks the candidate may lose |

### The decision rule

The DQN is selected only when, across matched validation runs, it has

* lower mean waiting time per vehicle, **and**
* at least equal mean throughput, **and**
* no increase in vehicles still waiting to enter.

Anything else selects the best fixed plan that was actually measured.
**A tie goes to fixed time** — "no worse" is not "better".

The search stops early in three ways: the DQN qualifies (the checkpoint is
saved for that workload), it fails `patience` consecutive checks, or the
time or episode limit is reached. If the deadline falls inside an episode,
that episode is cancelled and not counted, and the decision says so.

`judge()` in `simulation/hybrid.py` is a pure function and is tested
without SUMO, so "a tie must lose" does not depend on how a simulation
happened to turn out.

### What "matched" means

Both controllers get the same explicit vehicle list, the same SUMO seed,
the same duration and the same `utils.metrics.MetricsCollector`. Training
seeds (8000+) and validation seeds (7001, 7002) never overlap.

For Moderate and Heavy — which define demand as flows — Hybrid first
re-expresses the scenario as an explicit vehicle list
(`workload_from_scenario`). That is the same traffic in intent but **not**
byte-identical to the flow-based scenario, so a Hybrid number is labelled
as a Hybrid run and is not compared with the published baselines.

### Safety

Unchanged, and structural rather than enforced. The duration menu is
bounded by `min_green` and `max_green`, so an unsafe green cannot be asked
for. Every action ends with the Stage 1 yellow clearance, so a green can
never become the other green directly. Every action hands the road to the
other direction next, so no approach can be starved. A controller change
happens at a phase boundary, because a decision *is* one whole green plus
its clearance.

### Running in a browser

The job lives in `simulation/hybrid_job.py`, not in the Streamlit session:

* **One job per workload.** Pressing Run twice returns the same job; two
  workers would run SUMO over the same generated files.
* **A refresh reconnects** to the running job rather than restarting it.
* **Cancelling** stops at the next safe point and removes the generated
  SUMO inputs.

### A worked outcome

On the demonstration workload (200 cars on north lane 0, 328 on east lane
0, 900 s):

```
Hybrid selected tuned fixed time (ns 15 s / ew 20 s) after 1 check.
Mean waiting was 6.65 s with fixed time and 18.08 s with the DQN.
```

Tuning alone took the configured 30/30 plan from 10.78 s to **6.63 s**. The
registered DQN — trained on balanced moderate traffic — reached 18.08 s on
this very unbalanced workload and left 58 vehicles waiting to enter, so it
was rejected. That is the rule working, not the rule failing.
