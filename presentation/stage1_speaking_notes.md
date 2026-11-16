# IntelliSignal — Stage 1 Speaking Notes

Short, plain-language notes for each slide, plus the questions faculty are most likely
to ask. Keep answers honest: Stage 1 is the simulation foundation and the fixed-time
baseline. The DQN is the **next** stage.

---

## Slide 1 — Title

> "Good morning. This is my second progress presentation for IntelliSignal — an
> AI-based adaptive traffic signal management system built using SUMO and Deep
> Reinforcement Learning. Today I will show the working simulation foundation I have
> built, the fixed-time baseline controller, the traffic metrics I am collecting, and
> the initial results. I will also be clear about what is still to come: the DQN
> controller is the next stage, and it is not implemented yet."

*Tip: saying this in the first 30 seconds prevents any misunderstanding later.*

---

## Slide 2 — Problem Statement

> "A normal traffic signal works on a fixed timer. Whoever installed it decided, in
> advance, how many seconds of green each direction gets, and that number never changes.
> But traffic does change — by time of day, by direction, and sometimes unexpectedly.
> So you get the familiar situation where one road is jammed while the crossing road is
> empty but still holding green. That costs waiting time, queue length and throughput.
> My project asks whether the signal can instead decide its own timing by looking at the
> traffic, using reinforcement learning."

---

## Slide 3 — Project Objective

> "There are seven objectives. The first five are complete: simulate traffic in SUMO,
> model a four-way intersection, create different traffic conditions, build a fixed-time
> baseline, and collect traffic metrics. The last two — build the DQN controller and
> compare it against fixed-time — are the next stage. I deliberately did not start the
> DQN before the baseline was measurable, because without a baseline there is nothing to
> compare a learned controller against."

---

## Slide 4 — Technology Stack

> "On the left is what I actually used: Python, SUMO, TraCI, NumPy, Pandas, Matplotlib
> and PyYAML. On the right is what I will add later: SUMO-RL, Gymnasium,
> Stable-Baselines3, PyTorch, the DQN itself and a Streamlit dashboard. I have kept
> these two lists separate on purpose — nothing on the right is in my code today.
> SUMO is the traffic simulator, TraCI is the interface that lets my Python program
> control it, and PyYAML keeps every tunable number in configuration files instead of
> scattered through the code."

---

## Slide 5 — Current Architecture

> "The flow is: traffic demand goes into SUMO, SUMO runs the four-way intersection with
> a fixed-time signal and simulates every vehicle individually. My Python program talks
> to SUMO through TraCI one simulation second at a time, reads the metrics, writes them
> to CSV, and Matplotlib turns those CSV files into graphs.
> The important point for the next stage is this: the RL agent replaces exactly one box
> in this diagram — the fixed-time signal. Everything else stays identical, which is what
> makes the later comparison fair."

---

## Slide 6 — Current Implementation

> "These are the items that are actually working. The network has four approaches of two
> hundred metres, two lanes in each direction, and sixteen signal-controlled turning
> movements at the junction. The signal runs North-South green, yellow, East-West green,
> yellow, and all of those durations come from a YAML configuration file — they are not
> hard-coded anywhere. Both traffic scenarios run, the metrics are collected from the
> live simulation, the CSV files and the graphs are generated automatically, and I have
> twenty-four automated tests that all pass."

*If asked to prove a point, open `config/moderate.yaml` — it is short and readable.*

---

## Slide 7 — Live Simulation / Screenshots

> "These screenshots are taken from the simulation while it is running. You can see the
> intersection, vehicles approaching, a queue being held at red, the same queue
> discharging when the signal turns green, and the difference between the moderate and
> heavy scenarios."

Then run the live demo:

```bash
python simulation/run_simulation.py --scenario moderate --gui
```

While it runs, point out: the four approaches, the signal changing colour, vehicles
stopping at red, the queue building up and then clearing on green. Then run the heavy
scenario and point out the visibly longer queues.

*Tip: `--duration 300` makes the live run finish in a few seconds if time is short.*

---

## Slide 8 — Initial Results

> "Both scenarios use the same road network and the same signal timings. The only thing
> I changed is the traffic demand — 1400 vehicles per hour for moderate, 3000 for heavy.
> Under moderate demand the average vehicle waits about ten seconds and the average
> queue is under four vehicles. Under heavy demand the same signal plan gives about
> fifty-two seconds of waiting and an average queue of forty vehicles. So roughly double
> the demand produces about five times the waiting time and eleven times the queue.
> Throughput does go up in absolute terms, but it stops keeping up with demand — ninety-four
> vehicles are still waiting when the simulation window ends, against nine in the
> moderate case.
> These numbers come straight out of the simulation. They are in `results/csv/`, and the
> graphs are drawn from those same files."

*If a result looks surprising, say so honestly rather than explaining it away.*

---

## Slide 9 — Current Limitations

> "To be clear about the boundaries of this stage: the control is fixed-time only, the
> traffic is simulated rather than measured on a real road, the intersection is a single
> simplified junction, and I have studied two demand levels. There is no AI control, no
> DQN, no real-world data and no hardware integration yet. That is the plan, not a
> shortfall — this stage exists to produce a baseline I can measure the AI against."

---

## Slide 10 — Next Stage

> "For the next stage I will wrap this same scenario as a reinforcement-learning
> environment using SUMO-RL and Gymnasium. The state will be the queue length and
> waiting time on each approach plus the current phase; the action will be which phase
> to run next; and the reward will penalise waiting time, so that reducing delay
> increases the reward. I will then train a DQN agent with experience replay and a
> target network, and evaluate it on the same network and the same two scenarios, using
> the same metrics you saw on slide eight. That gives a direct Fixed-Time versus DQN
> comparison."

---

# Expected faculty questions and answers

**Q. Why SUMO?**
Because it provides a traffic-simulation environment where traffic flow and signal
behaviour can be studied safely without physical hardware. It is open source, it models
every vehicle individually, and it is the standard simulator used in traffic-signal
reinforcement-learning research.

**Q. What is TraCI?**
The Traffic Control Interface. It is the interface used to communicate with and control
SUMO programmatically. My Python program advances the simulation one step at a time,
reads the state of every lane and vehicle, and can change the signal — all while the
simulation is running.

**Q. Why did you start with a fixed-time controller?**
It establishes a baseline against which the future adaptive DQN controller can be
compared. Without a measured baseline, there is no way to claim that a learned
controller is an improvement.

**Q. Why two traffic scenarios?**
To observe system behaviour under different traffic demands. A fixed-time plan can look
perfectly acceptable at moderate demand; the weakness only appears when demand rises.
The two scenarios make that visible, and they also give the later DQN two different
conditions to be tested on.

**Q. What will the DQN do later?**
The DQN will observe the traffic state — queue lengths, waiting times and the current
signal phase — and select signal-control actions based on learned Q-values. It learns by
receiving a reward that becomes less negative as waiting time and queues go down.

**Q. What is the current AI implementation?**
There is none yet, and I want to be straightforward about that. The current stage
establishes the simulation and baseline environment. The DQN-based AI controller is the
next development stage.

**Q. How do you know the results are real and not made up?**
Every number is read out of the running simulation through TraCI and written to a CSV
file, and the graphs are drawn only from those CSV files. Nothing is typed into the
plotting code. The runs use a fixed random seed, so re-running the command reproduces the
same numbers — I can do that live.

**Q. How is the average waiting time calculated?**
For every vehicle, SUMO accumulates the time it spends below 0.1 m/s. I read that value
each step and average it over every vehicle that entered the network, including the ones
still queueing at the end of the run.
*(If asked why it differs from SUMO's own end-of-run statistic: SUMO averages only over
vehicles that finished their trip, so it reports a lower value in the heavy scenario. I
include the vehicles that are still waiting, which is the more honest figure for a
congested case.)*

**Q. How is the queue length measured?**
Every simulation second I ask SUMO how many halting vehicles are on each of the eight
incoming lanes, add them up, and at the end I average over all steps. "Halting" is
SUMO's definition: speed below 0.1 m/s.

**Q. What exactly is throughput here?**
The number of vehicles that completed their route — that is, actually drove through the
intersection and left the network — within the simulation window.

**Q. Why is the signal state a sixteen-character string?**
The junction has sixteen signal-controlled turning movements: four approaches with four
movements each. Each character controls one movement: `G` protected green, `g` permissive
green for the left turn, `y` yellow, `r` red.

**Q. What is the `g` for the left turn?**
A permissive green. The vehicle may go, but it must give way to oncoming traffic. That is
how an ordinary intersection without a separate left-turn arrow behaves.

**Q. Why is there no all-red phase?**
The configuration supports one — `signal.all_red` in the YAML file — but it is set to
zero in these runs, so the results reflect the simplest four-phase plan. Turning it on
only requires changing that one value.

**Q. Why 1800 seconds?**
Half an hour of simulated traffic is long enough for queues to reach a steady pattern —
you can see that in the queue-over-time graph — and short enough to run in a few seconds
on a laptop, which matters for a live demonstration.

**Q. Are the vehicles real or scripted?**
Neither is hand-scripted. I define the demand — how many vehicles per hour enter from
each direction and how they split between left, straight and right. SUMO then generates
the individual vehicles and moves them using its own car-following and lane-changing
models.

**Q. Would the heavy scenario gridlock?**
It does not. The teleport count is zero in both runs, which means SUMO never had to
remove a permanently blocked vehicle. The heavy scenario is congested but still flowing,
which is what makes it a useful test case.

**Q. What happens if SUMO crashes during a run?**
The TraCI connection is closed in a `finally` block, so SUMO is shut down even when an
error occurs and no orphan process is left running.

**Q. How long did the simulation take to run?**
A few seconds per scenario on a normal laptop — the moderate scenario runs about 490
times faster than real time. The simulation was kept deliberately small so it can be
demonstrated live.

**Q. What is the next milestone?**
Wrapping this scenario as a Gymnasium environment and defining the state, action and
reward. Training the DQN comes after that interface is working and tested.
