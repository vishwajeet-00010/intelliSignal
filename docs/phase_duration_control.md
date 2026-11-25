> **Numbers here predate the current model family.** The reasoning about
> the duration action space still holds and the code it describes is
> still in use, but every figure below was measured on the single
> generic car with no startup delay, under the 4-20 s `responsive`
> timing rules. The current figures are in
> `output/standard_20261125/FINDINGS.md`.

# The duration action space

This document describes the second action space in the project — choosing how
long each green lasts — and why it exists alongside the original KEEP/SWITCH
one. It is the formulation the heavy-traffic controller uses.

---

## 1. The problem it was built to solve

For most of this project's life the heavy-traffic DQN lost to a plain
fixed-time plan, and retraining it did not help. Three separate causes were
found, and all three are properties of the *formulation*, not of the training.

### 1.1 The best plan was not expressible

Measured through SUMO's own traffic-light program over seeds 6101–6120, the
best fixed plan for heavy traffic gives **north-south 7 s and east-west 6 s**,
at 5.754 s mean waiting. The surface around it is extremely narrow:

| ns / ew | 5 | 6 | 7 |
|---|---|---|---|
| **6** | 14.98 | 20.13 | 28.16 |
| **7** | **6.50** | **6.28** | 13.81 |
| **8** | 9.04 | 11.71 | 17.96 |

*(2 seeds; the full 64-cell grid is in `output/dqn_phase_20261122/grid_ns*.csv`.)*

With KEEP/SWITCH the green is a whole number of decision intervals, so the two
directions can only ever differ by a whole interval. A one-second difference
requires a one-second interval — and the environment forbade an interval
shorter than the 4 s clearance until `clearance_outside_interval` was added.

### 1.2 A one-second interval destroys the credit assignment

Allowing a one-second interval makes 7/6 reachable but produces ~1100
decisions per episode, nearly all of them KEEP, each with an almost identical
reward. The signal the agent needs — *this green should have been one second
shorter* — is spread across a hundred near-identical transitions.

### 1.3 The transitions stop being Markov

With the clearance outside the interval, KEEP advances 1 s and SWITCH advances
1 + 4 = 5 s. The old update charged both the same mean reward and discounted
both by one factor of gamma, so switching looked systematically cheaper than
it was. Training diverged: validation went 17.4 → 112 → 146 → 17.0.

---

## 2. What the duration action space is

One decision per green, taken when that green starts:

```
action i  ->  green of the current phase for durations[i] seconds
              yellow of the current phase for yellow_time seconds
              all-red for all_red_time seconds        (omitted when 0)
              the phase changes
```

The default menu is `(4, 5, 6, 7, 8, 9, 10, 12, 15, 20)`.

* 7 s against 6 s is **two actions**, not a pattern that must be held across
  eleven consecutive decisions.
* An episode has roughly 160 decisions instead of 1100, and each one owns its
  outcome.
* Safety is still structural, and is now stronger: the menu is bounded by
  `min_green` and `max_green`, so an unsafe green is not something the agent
  can ask for. Every action ends with the clearance, so a green-to-green
  transition cannot be expressed. Every action hands the road to the other
  direction next, so no approach can be starved.

Code: `rl/phase_action.py`, `rl/phase_state.py`, `rl/phase_environment.py`.

---

## 3. The three corrections that make the learning correct

### 3.1 The reward is an integral, not a mean

A decision covers `green + clearance` seconds, and that differs by action: 8 s
for the shortest, 24 s for the longest. The reward sums the congestion over
every second the decision occupies:

```
reward = -scale * SUM over the decision's seconds of
             ( waiting_weight * total current waiting time
             + queue_weight   * halting vehicles )
```

so a decision pays for exactly the congestion it presides over.

### 3.2 The future is discounted by elapsed time

`gamma` is a **per-second** rate and the target uses `gamma ** seconds`:

```
a*     = argmax_a Q_policy(s', a)            # Double Q-learning
target = r + gamma ** duration * Q_target(s', a*)
```

At gamma = 0.99 a 12 s decision discounts the future by 0.89 and a 24 s
decision by 0.79, which is the honest ratio. Doing only one of 3.1 and 3.2
would be worse than doing neither: the cost would be charged over the real
time span while the future was discounted as though no time had passed.

The congestion *within* a decision is discounted too, by how far into the
decision it happens. Without that, the last second of a 24 s decision is
charged at full weight while the state that follows it is valued at
`gamma ** 24`, which quietly overcharges long greens and pushes the policy
toward the shortest action on the menu.

### 3.3 Double Q-learning

Plain DQN takes `max_a Q_target(s', a)`, so an action whose value happens to
be over-estimated is both chosen and believed. With a menu whose entries
differ by very little — a 6 s and a 7 s green are nearly the same decision —
that bias is large relative to the signal, and a degenerate "always the same
action" policy is what it looks like from the inside. The policy network
chooses; the target network prices.

---

## 4. The observation

Ten numbers, written from the point of view of the green about to be served,
so that "give more time to the busier direction" is one rule rather than two:

| # | name | meaning |
|---|---|---|
| 0 | `served_queue` | halting vehicles on the two approaches about to get green |
| 1 | `other_queue` | halting vehicles on the two that stay red |
| 2 | `served_approaching` | vehicles present but not halting, served direction |
| 3 | `other_approaching` | the same for the direction kept at red |
| 4 | `served_waiting` | summed current waiting time, served direction |
| 5 | `other_waiting` | summed current waiting time, other direction |
| 6 | `is_north_south` | 1.0 for a north-south green |
| 7 | `worst_approach` | the longest queue on any single approach |
| 8 | `last_served_green` | the green this direction got last time |
| 9 | `last_other_green` | the green the other direction got last time |

Index 6 is kept deliberately: demand is 850 veh/h on each of north and south
against 650 on each of east and west, so the best plan is asymmetric, and a
policy that could not tell the directions apart could not express 7 against 6.

Index 7 is the only entry that is not a per-direction aggregate. North and
south are summed, so ten halting vehicles can mean five on each or ten on one.
The second spills back, and it is invisible in indices 0 and 1.

Indices 2 and 3 have no equivalent in the Stage 2 observation. Demand is
generated with SUMO `<flow vehsPerHour=...>` and **no** `random="true"`, so
vehicles are inserted on an equidistant schedule and arrive in a repeating
pattern. Knowing how much traffic is on the approach but not yet stopped is
what lets a controller end a green as the last vehicle of a platoon clears.

---

## 5. Running it

```bash
# train
python training/train_phase_dqn.py --scenario heavy --episodes 600 \
    --seed 42 --gamma 0.99 --n-step 2 --out output/my_run

# evaluate against the tuned fixed plan on the confirmatory seeds
python evaluation/evaluate_phase_dqn.py \
    --model output/my_run/models/best.pt --seeds 6101-6120
```

Two flags matter more than they look:

* `--threads 1`. The networks are tiny, so torch's intra-op parallelism buys
  nothing, and by default it opens a thread per core. Eight runs on a
  sixteen-core machine then ask for 128 threads and spend their time context
  switching — measured at 51 s per episode against 6 s pinned to one thread.
* `--measure-training` is **off** by default. `MetricsCollector` reads every
  vehicle's accumulated waiting time every simulated second, roughly 200,000
  TraCI round-trips per episode, which is an order of magnitude more than
  everything else the episode does. Training does not need it; validation,
  which is the number the project reports, always uses it.

---

## 6. What this does not change

The Stage 2 KEEP/SWITCH environment, `agents/dqn_agent.py`, the deployed
moderate-traffic model and every historical result are untouched. The two
action spaces live side by side, and `agents/replay_buffer.py` gained one
optional field (`duration`, defaulting to 1.0) that existing callers never
set. The scenario files, the yellow and all-red clearance, the demand and the
`MetricsCollector` are all exactly as they were — which is what makes the new
numbers comparable to the old ones.
