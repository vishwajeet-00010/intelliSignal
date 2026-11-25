# The dashboard

The dashboard is the product: a Streamlit application over one junction,
one engine and one set of timing rules. Every number on it was produced
by a SUMO run that either happened while the page was open or had already
been measured and kept.

```bash
streamlit run dashboard.py
```

It opens in a browser; `.streamlit/config.toml` holds the theme. Nothing
needs to be trained or generated first: without a promoted model the DQN
option says so and Hybrid decides instead, and without SUMO the sidebar
names the check that failed rather than showing a blank page.

## 1. One engine behind everything

Every page goes through `simulation/engine.py`, and that goes through
`evaluation/harness.py` - the same harness the training, selection and
promotion scripts use. There is no second code path for the picture. A
controller the dashboard runs is run by the same function, on the same
vehicle list, under the same timing profile and measured the same two
ways as a controller the evaluation scripts run, so a number on screen
and a number in a report are the same number.

`run_controller()` is the whole interface: a workload and one of
`fixed_time`, `actuated`, `max_pressure` or `dqn` go in, and a `RunResult`
comes back. It carries `metrics` (from the live collector and from SUMO's
own trip records), `compliance` (what the controller says it did), `audit`
(what the signal actually showed, checked independently), `decisions` (one
row per green: how long, and what the network thought), `greens` (the
green timeline, run-length encoded from the recorded signal), `series`
(per-second rows) and `replay` (where every vehicle was, second by
second).

The headline measurement is **total delay per requested vehicle**: time
lost on the road plus time spent unable to enter, over every vehicle the
traffic asked for, including those still on the network when the run ends.
It cannot be improved by refusing traffic. **Mean waiting time** is
reported beside it because it is what the earlier stages reported.

A long run would freeze the page, so `dashboard/jobs.py` puts each run on
a worker thread and the page polls a snapshot. Jobs are keyed by workload,
controller set and seed, so leaving a page and coming back finds the job
still running rather than starting a second one.

## 2. The five pages

The sidebar chooses the page and the traffic, and shows two badges: whether
a validated model is in charge, and whether the environment (SUMO, PyTorch,
the built network) is complete. `?page=Compare` in the URL opens that page
directly, which is how the pages are screenshotted.

### Control room

Choose a controller - DQN (trained), Fixed-time, Actuated or Max-pressure -
press **Run the junction**, and watch what came out. For a fixed plan two
select-sliders set the north-south and east-west green from the profile's
menu, and a button offers the best plan found when the model was selected
rather than making the user sweep for it.

Below the run: the headline tiles, the junction replay, the signal timeline
and the halted-vehicle curve, the green lengths against the profile's
limits, the checks panel, every decision the controller made with the
Q-values behind it, and a list of every run the dashboard has kept. A saved
run is reloaded when the traffic, the controller and the seed all match,
which is why a run can never appear beside traffic it was not measured
on.

### Traffic

What is on the road, either as one of five ready-made patterns - off-peak,
typical day, evening peak, kerb-lane rush, sudden surge - or described lane
by lane. Both end in the same object: an explicit list of vehicles, each
with its class, its lane, its turn and the second it arrives. Nothing is a
rate SUMO will sample differently next time, so two controllers meet
identical traffic and a comparison is a measurement rather than an
estimate. The preset seeds are 50000 and above, disjoint from the
training, validation and test splits, so a demonstration can never be run
on the evidence.

The page also shows the fleet every run uses - 62 % motorcycles, 22 % cars,
11 % auto-rickshaws, 5 % buses, with each class's length, startup delay and
road space in PCU - and how much of a green of a given length actually
moves traffic once the queue's startup delay is paid.

Hand-written traffic goes through `dashboard/custom_form.py` over
`custom_traffic/`: the form holds what was typed, the resolver decides
everything left blank and tags each value `user`, `derived` or `auto`, and
the preview is rendered from the same resolved workload the route file is
written from. Contradictions are reported with the field to change rather
than silently fixed. `docs/custom_traffic_and_hybrid.md` has the rules.

### Compare

Several controllers, one set of vehicles, one table. Fixed-time is always
included because it is the baseline every other row is measured against,
and `engine.compare()` refuses to put two different workloads, or two
different seeds, into one table: that is the mistake the table exists to
prevent. Below it are a delay bar chart, the halted-vehicle curves
overlaid, and one tab per controller with its tiles, checks, timeline and
decisions. If any run failed a check, the page says so above the table.

The lower half is **Hybrid**, which is not another controller but the rule
that decides between a tuned fixed plan and the model, with a wall-clock
stopwatch running. It tunes a plan on this exact traffic, tests whatever
model exists against it, and keeps whichever wins; a tie goes to fixed
time. The job lives in `simulation/hybrid_job.py`, so a browser refresh
reconnects to the running search instead of restarting it.

### The model

Which model is in charge, why it is allowed to be, and what it cost to find
out, in that order.

First the registry card: what is registered, how many networks it averages,
the greens it is allowed, when it was promoted, and the checks that decide
whether it may take the signal at all. A model is used only if it is
registered, its file still matches the SHA-256 recorded at promotion, its
greens lie inside the profile, its promotion record shows it beating a
tuned fixed plan, and it was selected on traffic it was not reported on.
Each check is shown with its answer, so the page says why the model is not
in charge rather than only that it is not.

Then what it was measured at on the test traffic, then how it was chosen:
every trained run re-scored on the validation traffic, every averaged
ensemble tried, every fixed plan on the menu measured, and the test split
read once at the end. That is read straight out of the selection report
written by `evaluation/select_profile_model.py`, so the claim on screen and
the evidence behind it are the same file. Last, the validation curve of the
promoted run, if its history is on disk. An expander lists the older
Stage 1-3 checkpoints, kept so their published numbers reproduce and not
offered as controllers because they were trained on a single generic car
with no startup delay under different timing rules.

### Method

How the junction is modelled, how a controller is measured, and what the
project does not claim: the timing rules and the arithmetic showing the
maximum-red rule can never bind under them, why short greens lose once
startup delay is modelled, the two measurements and the cross-check between
them, the separation of training, validation, test and demonstration
traffic, the junction and fleet parameters, and the limitations. The
limitations are on the page rather than in a footnote because every one of
them has at some point been the thing that made a number wrong.

## 3. The junction replay

`dashboard/junction.py` draws the junction as one self-contained HTML
component - no plotting library and no network access, so it works on a
projector in a room with no internet. The geometry comes from
`sumo/network/intersection.net.xml`: two lanes in and two out on each arm,
3.2 m wide, with the stop lines where SUMO puts them.

The vehicles are not a cartoon. `simulation/recorder.py` rides along with
the run as a harness observer and writes down, once per simulation step,
every vehicle's position and heading, the number halted and present on each
approach, and the signal state SUMO displayed. Positions are kept as whole
decimetres and headings as whole degrees, and vehicles are followed by
TraCI subscription, so recording costs one round trip per step rather than
two calls per vehicle per step. Every vehicle on the canvas was therefore
in the simulation whose numbers sit beside it, and the signal shown is the
state SUMO displayed rather than a re-derivation of what the controller
meant to do. The picture can disagree with the controller, which is
exactly when it is worth looking at.

## 4. The independent signal audit

A controller reports its own compliance, and a controller with a bug would
report its own bug as compliant, so every run is checked a second time.
`audit_signal()` in `simulation/recorder.py` reads the recorded signal
states - what SUMO actually displayed - and tests the timing rules against
them: every completed green inside the profile's limits, an amber and an
all-red of the right length after every green, the other direction served
next, and no approach held on red longer than the profile allows. Red is
measured as a driver sees it, from the end of an approach's own amber to
the start of its next green, which is the stricter of the two readings. The
audit shares no code with any controller.

The checks panel on every run reports that audit plus two measurement
checks: that SUMO's trip records and the live collector agree on how many
vehicles finished, and that there were no collisions and no teleports. A
run measured without recording the signal is reported as unexamined, not as
passing - `RunResult.trustworthy` requires the audit to exist and to have
passed.

## 5. Where runs are saved

Finished runs are written to `results/runs/` as
`<workload_id>_<controller>_<seed>.json`, replay included. The workload id
is a hash of every vehicle, its lane, its turn and its departure second, so
two runs with the same id met exactly the same traffic. That naming lets a
page reload a run instead of repeating it, and makes it impossible to show
one beside traffic it was not measured on. Saving is a convenience: if the
write fails the run is still shown, and `tools/clean_workspace.py` clears
the directory.

## 6. Session state

Streamlit re-runs the whole script on every interaction, so anything that
must survive a click lives in `st.session_state`. Every key and its default
is declared in `dashboard/state.py`, so one page cannot invent a key
another then fails to find.

| key | default | what it holds |
|---|---|---|
| `page` | `Control room` | which page is open |
| `traffic_kind` | `preset` | `preset` or `custom` |
| `preset` | `typical` | which ready-made pattern |
| `duration_s` | `900.0` | simulated seconds |
| `seed` | `6101` | SUMO's own seed |
| `custom_workload` | `None` | the resolved workload, as a dict |
| `custom_spec` | `None` | the form's own state, as a dict |
| `custom_issues` | `[]` | what the resolver said about it |
| `custom_regen_token` | `0` | bumped to redraw the generated values |
| `controller` | `dqn` | the controller on the Control room page |
| `plan_ns`, `plan_ew` | `None` | the chosen fixed plan; `None` uses the configured one |
| `compare_with` | `fixed_time`, `actuated` | the controllers on the Compare page |
| `job_key` | `""` | the single run in flight or last finished |
| `compare_job_key` | `""` | the comparison in flight or last finished |
| `hybrid_workload_id` | `""` | which traffic Hybrid was last asked about |
| `last_error` | `""` | the last failure worth showing |
| `profile` | `standard` | the operating timing profile |

The workload is stored as a dictionary rather than as the object, because
session state is pickled between reruns and a plain dict reloads
identically every time. `state.reset_traffic()` clears the job keys
whenever the traffic changes, which stops a run measured on one pattern
being shown beside the name of another, and `state.current_workload()` is
the one function every page asks for the current traffic, so two pages
cannot disagree about what is on the road.

## 7. The modules

`dashboard.py` is the entry point, holding the page registry, the sidebar
and the masthead. `dashboard/state.py` holds the session keys,
`dashboard/jobs.py` the worker threads, `dashboard/ui.py` the stylesheet
and every shared piece of interface, `dashboard/junction.py` the replay
component, `dashboard/charts.py` the Altair charts,
`dashboard/custom_form.py` the hand-written traffic form and
`dashboard/hybrid_panel.py` Hybrid's progress and decision. Each page is
one module under `dashboard/views/`, with `common.py` holding whatever
more than one of them shows.

The Stage 1-3 dashboard's own modules - `components.py`, `controls.py`,
`theme.py` and `traffic_panel.py` - were deleted when this one replaced
them. Nothing imported them any more. `simulation/controller.py` and
`simulation/custom_runs.py` remain, because the Stage 1-3 tests and two
frozen verification scripts still run against them, but the dashboard
does not touch either.

`dashboard/ui.py` enforces three rules so no page has to: a number is never
shown without its unit and its meaning, good and bad have fixed colours
everywhere, and nothing claims more than it measured - `verdict` renders
confirmed, unverified and failed as three distinct states and will not show
a plain tick for something that was never checked.
