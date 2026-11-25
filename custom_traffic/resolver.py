"""
IntelliSignal
custom_traffic/resolver.py

Turns a partly-filled form into a workload that can actually be run.

The rules, in the order they are applied:

1. **At least one real value is required.** An empty form is a mistake, not a
   request to invent a scenario. "Generate example" fills the form first.
2. **Every explicit value survives, including zero.** A lane the user set to
   0 stays empty; a lane the user did not select stays at zero demand and is
   never filled in for them.
3. **Missing independent values are filled** from the base scenario and the
   network's own limits, with any randomness drawn from the workload's seed
   so the same form always resolves the same way.
4. **Dependent values are derived.** If a lane has 200 vehicles and the user
   assigns 50 to turn right, the other 150 go to that lane's remaining
   permitted movements in the base scenario's proportions.
5. **Contradictions are reported, not silently fixed**, and each one names
   the field that has to change.

Everything it decides is tagged `user`, `derived` or `auto`, so the preview
can show which numbers came from the person at the keyboard.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from custom_traffic.network import (LANE_SPEED_KMH, LANES_PER_APPROACH,
                                    MOVEMENT_ORDER, movements_for_lane)
from custom_traffic.spec import (CustomSpec, LaneInput, TURNS_AS_COUNTS,
                                 TURNS_AS_PERCENT)
from custom_traffic.workload import (FROM_AUTO, FROM_DERIVED, FROM_USER,
                                     ResolvedLane, ResolvedWorkload, Vehicle)
from utils.config_loader import load_config

#: A lane cannot physically launch vehicles faster than this. Used only to
#: warn - SUMO still holds the surplus at the entry, and the run reports them
#: as "waiting to enter" rather than pretending they arrived.
APPROX_LANE_CAPACITY_VEH_PER_HOUR = 1800.0

#: Bounds the form accepts, so a typo cannot start an hour-long run.
MAX_VEHICLES_PER_LANE = 5000
MIN_SPEED_KMH = 5.0
MAX_SPEED_KMH = 130.0


@dataclass
class Issue:
    """Something the user has to fix, or should know about."""

    field: str                 # "count", "window", "speed", "turns", "form"
    message: str
    lane_key: str = ""         # "north_0", or "" for whole-form problems
    severity: str = "error"    # "error" | "warning"

    @property
    def is_error(self) -> bool:
        return self.severity == "error"

    def __str__(self) -> str:
        where = "{}: ".format(self.lane_key) if self.lane_key else ""
        return "{}{}".format(where, self.message)


@dataclass
class Resolution:
    """The outcome of resolving a spec."""

    workload: Optional[ResolvedWorkload]
    issues: List[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.workload is not None and not self.errors

    @property
    def errors(self) -> List[Issue]:
        return [issue for issue in self.issues if issue.is_error]

    @property
    def warnings(self) -> List[Issue]:
        return [issue for issue in self.issues if not issue.is_error]


# ---------------------------------------------------------------------------
def _stream(seed: int, key: str) -> random.Random:
    """A generator that depends on the seed and the lane, nothing else.

    Per-lane streams mean that editing one lane does not reshuffle the
    automatically filled values of every other lane.
    """
    mixed = (int(seed) * 1000003) ^ (sum(ord(c) * (i + 7)
                                         for i, c in enumerate(key)) * 31)
    return random.Random(mixed & 0x7FFFFFFF)


def _largest_remainder(total: int, weights: Dict[str, float]) -> Dict[str, int]:
    """Split `total` by `weights` so the parts are integers that still sum.

    Plain rounding loses or invents vehicles; the largest-remainder method
    does not, which matters because the preview promises an exact count.
    """
    usable = {k: max(float(v), 0.0) for k, v in weights.items()}
    mass = sum(usable.values())
    if total <= 0 or mass <= 0:
        return {k: 0 for k in weights}

    exact = {k: total * v / mass for k, v in usable.items()}
    out = {k: int(math.floor(v)) for k, v in exact.items()}
    remaining = total - sum(out.values())
    order = sorted(exact, key=lambda k: (-(exact[k] - math.floor(exact[k])), k))
    for index in range(remaining):
        out[order[index % len(order)]] += 1
    return out


def _lane_turn_weights(config: Dict[str, Any], lane: int) -> Dict[str, float]:
    """The base scenario's turn ratios, restricted to what this lane can do."""
    ratios = config["demand"]["turn_ratios"]
    allowed = movements_for_lane(lane)
    weights = {movement: float(ratios.get(movement, 0.0)) for movement in allowed}
    if sum(weights.values()) <= 0:                 # pathological config
        weights = {movement: 1.0 for movement in allowed}
    return weights


def _default_lane_count(config: Dict[str, Any], approach: str,
                        duration_s: float, rng: random.Random) -> int:
    """A plausible number of vehicles for a lane the user left blank."""
    per_hour = float(config["demand"][approach]) / LANES_PER_APPROACH
    expected = per_hour * duration_s / 3600.0
    jittered = expected * rng.uniform(0.85, 1.15)
    return max(int(round(jittered)), 1)


# ---------------------------------------------------------------------------
def resolve(spec: CustomSpec,
            regenerate_token: int = 0) -> Resolution:
    """Resolve `spec` into a runnable workload, or report what is wrong.

    `regenerate_token` changes only the automatically generated values. The
    user's own entries are untouched by it, which is what "Regenerate" means
    on the form.
    """
    issues: List[Issue] = []

    try:
        config = load_config(spec.base_scenario)
    except Exception as exc:
        return Resolution(None, [Issue("form", str(exc))])

    duration = float(spec.duration_s if spec.duration_s is not None
                     else config["simulation"]["duration"])
    step_length = float(config["simulation"]["step_length"])
    if duration <= 0:
        issues.append(Issue("duration", "Duration must be more than 0 seconds."))
        return Resolution(None, issues)

    if not spec.has_user_input():
        issues.append(Issue(
            "form",
            "Enter traffic on at least one lane before running. Use "
            "“Generate example” to fill the form with a worked "
            "example you can then edit."))
        return Resolution(None, issues)

    base_speed_kmh = round(
        float(config["vehicle"]["max_speed"]) * 3.6, 1)

    resolved_lanes: List[ResolvedLane] = []
    for entry in spec.lanes:
        lane, lane_issues = _resolve_lane(
            entry, config, duration, base_speed_kmh,
            seed=int(spec.seed) + int(regenerate_token))
        issues.extend(lane_issues)
        resolved_lanes.append(lane)

    if any(issue.is_error for issue in issues):
        return Resolution(None, issues)

    total = sum(lane.count for lane in resolved_lanes)
    if total <= 0:
        issues.append(Issue(
            "form", "Every selected lane resolved to zero vehicles, so there "
                    "would be nothing to simulate. Give at least one lane a "
                    "vehicle count above zero."))
        return Resolution(None, issues)

    traffic_model = str((config.get("vehicle") or {}).get(
        "traffic_model", "generic"))
    vehicles = _build_vehicle_list(resolved_lanes, step_length, traffic_model)

    workload = ResolvedWorkload(
        name=spec.name or "custom",
        base_scenario=spec.base_scenario,
        duration_s=duration,
        step_length=step_length,
        seed=int(spec.seed),
        lanes=resolved_lanes,
        vehicles=vehicles,
        signal={key: float(config["signal"][key])
                for key in ("ns_green", "ew_green", "yellow", "all_red")},
        vehicle_type={key: float(config["vehicle"][key])
                      for key in ("max_speed", "accel", "decel", "length",
                                  "min_gap", "sigma")},
        traffic_model=traffic_model,
        notes=[str(issue) for issue in issues if not issue.is_error],
    )
    return Resolution(workload, issues)


# ---------------------------------------------------------------------------
def _resolve_lane(entry: LaneInput, config: Dict[str, Any], duration: float,
                  base_speed_kmh: float, seed: int
                  ) -> Tuple[ResolvedLane, List[Issue]]:
    """Decide every value for one lane, or say why it cannot be decided."""
    issues: List[Issue] = []
    key = entry.key
    provenance: Dict[str, str] = {}
    allowed = movements_for_lane(entry.lane)
    rng = _stream(seed, key)

    # --- a lane the user did not select carries no traffic ----------------
    if not entry.enabled:
        return ResolvedLane(
            approach=entry.approach, lane=entry.lane, enabled=False,
            count=0, window_start_s=0.0, window_end_s=duration,
            speed_kmh=base_speed_kmh,
            turns={movement: 0 for movement in allowed},
            provenance={"count": FROM_USER, "window": FROM_AUTO,
                        "speed": FROM_AUTO, "turns": FROM_AUTO},
        ), issues

    stated_turns = entry.stated_turns()

    # --- vehicle count ----------------------------------------------------
    count: Optional[int] = None
    if entry.count is not None:
        if entry.count < 0:
            issues.append(Issue("count", "Vehicle count cannot be negative.", key))
        elif entry.count > MAX_VEHICLES_PER_LANE:
            issues.append(Issue(
                "count", "Vehicle count is above the {} limit for one lane."
                .format(MAX_VEHICLES_PER_LANE), key))
        count = int(entry.count)
        provenance["count"] = FROM_USER
    elif entry.count_min is not None or entry.count_max is not None:
        low = entry.count_min if entry.count_min is not None else entry.count_max
        high = entry.count_max if entry.count_max is not None else entry.count_min
        low, high = int(low), int(high)
        if low > high:
            issues.append(Issue(
                "count", "Minimum vehicles ({}) is above the maximum ({}). "
                         "Swap them or clear one.".format(low, high), key))
        elif low < 0:
            issues.append(Issue("count", "Vehicle count cannot be negative.", key))
        else:
            count = rng.randint(low, high)
            provenance["count"] = FROM_AUTO
    elif entry.turns_mode == TURNS_AS_COUNTS and stated_turns:
        # The turning counts ARE the demand when no total was given.
        count = int(sum(stated_turns.values()))
        provenance["count"] = FROM_DERIVED
    else:
        count = _default_lane_count(config, entry.approach, duration, rng)
        provenance["count"] = FROM_AUTO

    if count is None:
        count = 0

    # --- arrival window ---------------------------------------------------
    start = entry.window_start_s
    end = entry.window_end_s
    if start is None and end is None:
        start, end = 0.0, duration
        provenance["window"] = FROM_AUTO
    else:
        if start is None:
            start = 0.0
        if end is None:
            end = duration
        provenance["window"] = FROM_USER
    start, end = float(start), float(end)

    if start < 0:
        issues.append(Issue("window", "The window cannot start before 0 s.", key))
    if end <= start and count > 0:
        issues.append(Issue(
            "window", "The window ends at {:.0f} s, which is not after its "
                      "start of {:.0f} s.".format(end, start), key))
    if end > duration:
        issues.append(Issue(
            "window", "The window ends at {:.0f} s but the run is only "
                      "{:.0f} s long. Shorten the window or lengthen the run."
            .format(end, duration), key))

    # --- desired speed ----------------------------------------------------
    if entry.speed_kmh is not None:
        speed = float(entry.speed_kmh)
        provenance["speed"] = FROM_USER
        if not MIN_SPEED_KMH <= speed <= MAX_SPEED_KMH:
            issues.append(Issue(
                "speed", "Desired speed must be between {:.0f} and {:.0f} "
                         "km/h.".format(MIN_SPEED_KMH, MAX_SPEED_KMH), key))
    else:
        speed = base_speed_kmh
        provenance["speed"] = FROM_AUTO

    if speed > LANE_SPEED_KMH:
        issues.append(Issue(
            "speed", "{:.0f} km/h is above the {:.0f} km/h lane limit, so "
                     "vehicles will travel at the limit."
            .format(speed, LANE_SPEED_KMH), key, severity="warning"))

    # --- turning movements ------------------------------------------------
    turns, turn_issues = _resolve_turns(entry, count, config, allowed, key)
    issues.extend(turn_issues)
    provenance["turns"] = (FROM_USER if len(stated_turns) == len(allowed)
                           else FROM_DERIVED if stated_turns else FROM_AUTO)

    # --- a note when the lane is asked for more than it can take ----------
    window = max(end - start, 0.0)
    if count > 0 and window > 0:
        rate = count * 3600.0 / window
        if rate > APPROX_LANE_CAPACITY_VEH_PER_HOUR:
            issues.append(Issue(
                "count",
                "{} vehicles in {:.0f} s is about {:.0f} veh/h, above what "
                "one lane can discharge (~{:.0f} veh/h). SUMO will hold the "
                "surplus at the entry; the run reports them as waiting to "
                "enter.".format(count, window, rate,
                                APPROX_LANE_CAPACITY_VEH_PER_HOUR),
                key, severity="warning"))

    return ResolvedLane(
        approach=entry.approach, lane=entry.lane, enabled=True,
        count=int(count), window_start_s=start, window_end_s=end,
        speed_kmh=round(speed, 1), turns=turns, provenance=provenance,
    ), issues


# ---------------------------------------------------------------------------
def _resolve_turns(entry: LaneInput, count: int, config: Dict[str, Any],
                   allowed: Tuple[str, ...], key: str
                   ) -> Tuple[Dict[str, int], List[Issue]]:
    """Split a lane's vehicles across the movements it is allowed to make."""
    issues: List[Issue] = []
    stated = entry.stated_turns()
    turns = {movement: 0 for movement in allowed}

    if not stated:
        return _largest_remainder(
            count, _lane_turn_weights(config, entry.lane)), issues

    if entry.turns_mode == TURNS_AS_PERCENT:
        total_percent = sum(stated.values())
        if total_percent > 100.0 + 1e-6:
            issues.append(Issue(
                "turns", "The turn percentages add up to {:.0f} %, which is "
                         "more than 100 %.".format(total_percent), key))
            return turns, issues
        assigned = {movement: int(round(count * value / 100.0))
                    for movement, value in stated.items()}
        remainder = count - sum(assigned.values())
        missing = [m for m in allowed if m not in stated]
        if missing:
            weights = {m: _lane_turn_weights(config, entry.lane)[m]
                       for m in missing}
            assigned.update(_largest_remainder(max(remainder, 0), weights))
        elif remainder:
            # Percentages that did not reach 100 %: give the rest to the
            # largest stated movement rather than quietly losing vehicles.
            biggest = max(stated, key=lambda m: stated[m])
            assigned[biggest] = assigned.get(biggest, 0) + remainder
        turns.update({m: int(v) for m, v in assigned.items()})
        return turns, issues

    # --- exact counts -----------------------------------------------------
    stated_total = int(sum(stated.values()))
    missing = [m for m in allowed if m not in stated]

    if not missing:
        if stated_total != count:
            issues.append(Issue(
                "turns",
                "The turn counts add up to {} but the lane has {} vehicles. "
                "Change the vehicle count, or clear one movement so it can "
                "be worked out from the others.".format(stated_total, count),
                key))
            return turns, issues
        turns.update({m: int(v) for m, v in stated.items()})
        return turns, issues

    if stated_total > count:
        issues.append(Issue(
            "turns",
            "The turn counts already add up to {}, which is more than the "
            "lane's {} vehicles.".format(stated_total, count), key))
        return turns, issues

    turns.update({m: int(v) for m, v in stated.items()})
    leftover = count - stated_total
    weights = {m: _lane_turn_weights(config, entry.lane)[m] for m in missing}
    turns.update(_largest_remainder(leftover, weights))
    return turns, issues


# ---------------------------------------------------------------------------
def _build_vehicle_list(lanes: List[ResolvedLane], step_length: float,
                        traffic_model: str = "generic") -> List[Vehicle]:
    """Expand the lane totals into one entry per vehicle.

    Departures are spread evenly across each lane's window, which is what
    makes "200 cars over 30 minutes" and "200 cars in the first 5 minutes"
    visibly different runs: the first releases one car every 9 s, the second
    one every 1.5 s. Movements are interleaved rather than served in blocks,
    so a lane's right-turners are not all at the end.
    """
    from custom_traffic.fleet import assign, get_fleet

    fleet = get_fleet(traffic_model)
    vehicles: List[Vehicle] = []
    for lane in lanes:
        if lane.count <= 0:
            continue
        order = _interleave(lane.turns)
        classes = assign(len(order), fleet)
        window = max(lane.window_end_s - lane.window_start_s, 0.0)
        gap = (window / lane.count) if lane.count > 1 and window > 0 else 0.0
        for index, movement in enumerate(order):
            depart = lane.window_start_s + gap * index
            depart = round(depart / step_length) * step_length
            vehicles.append(Vehicle(
                vehicle_id="{}{}_{:04d}".format(
                    lane.approach[0].upper(), lane.lane, index),
                approach=lane.approach, lane=lane.lane, movement=movement,
                depart=round(float(depart), 2),
                speed_kmh=float(lane.speed_kmh),
                vehicle_class=classes[index].name,
            ))
    vehicles.sort(key=lambda v: (v.depart, v.vehicle_id))
    return vehicles


def _interleave(turns: Dict[str, int]) -> List[str]:
    """Spread the movements as evenly as the counts allow.

    Largest-remainder again, but over positions: each movement is placed at
    the positions where its running share is furthest behind, which gives
    "s r s s l s r ..." rather than "r r s s s s l l".
    """
    total = sum(turns.values())
    if total <= 0:
        return []
    remaining = {m: int(c) for m, c in turns.items() if c > 0}
    placed = {m: 0 for m in remaining}
    order: List[str] = []
    for position in range(1, total + 1):
        best, best_debt = None, None
        for movement in sorted(remaining):
            if placed[movement] >= remaining[movement]:
                continue
            target = remaining[movement] * position / float(total)
            debt = target - placed[movement]
            if best_debt is None or debt > best_debt:
                best, best_debt = movement, debt
        if best is None:
            break
        order.append(best)
        placed[best] += 1
    return order


# ---------------------------------------------------------------------------
def workload_from_scenario(scenario: str, duration_s: Optional[float] = None,
                           seed: int = 6101) -> "ResolvedWorkload":
    """Re-express a built-in scenario's demand as an explicit vehicle list.

    Moderate and Heavy describe demand as SUMO `<flow vehsPerHour=...>`,
    which is fine for a fixed scenario but means two controllers get the
    same traffic only because SUMO happens to be deterministic. Hybrid needs
    to hand both controllers the *same vehicles*, so it works on a workload.

    The counts are the scenario's own demand spread evenly over its two
    lanes and over the run, and the turn split is the scenario's own ratios.
    It is therefore the same traffic in intent - but it is NOT byte-identical
    to the flow-based scenario, so anything measured this way is labelled as
    a Hybrid run rather than compared with the published baselines.
    """
    config = load_config(scenario)
    duration = float(duration_s if duration_s is not None
                     else config["simulation"]["duration"])

    spec = CustomSpec(name=scenario, base_scenario=scenario,
                      duration_s=duration, seed=int(seed), lanes=[])
    for approach in ("north", "east", "south", "west"):
        per_hour = float(config["demand"][approach]) / LANES_PER_APPROACH
        per_lane = int(round(per_hour * duration / 3600.0))
        for lane in range(LANES_PER_APPROACH):
            spec.lanes.append(LaneInput(
                approach=approach, lane=lane, enabled=True,
                count=per_lane, window_start_s=0.0, window_end_s=duration,
                speed_kmh=round(float(config["vehicle"]["max_speed"]) * 3.6, 1),
            ))
    resolution = resolve(spec)
    if resolution.workload is None:                # pragma: no cover
        raise ValueError("; ".join(str(issue) for issue in resolution.errors))
    return resolution.workload


def example_spec(base_scenario: str = "indian_arterial", seed: int = 6101
                 ) -> CustomSpec:
    """A worked example for the “Generate example” button.

    It is the example from the brief: 200 cars on one lane, 300-350 on
    another, with speed, turns and arrival windows deliberately left blank so
    the resolver has something to fill in and the preview has something to
    mark as generated.
    """
    spec = CustomSpec.empty(base_scenario=base_scenario, seed=seed)
    spec.name = "example"
    north = spec.lane("north", 0)
    north.enabled = True
    north.count = 200

    east = spec.lane("east", 0)
    east.enabled = True
    east.count_min = 300
    east.count_max = 350

    south = spec.lane("south", 1)
    south.enabled = True
    south.count = 120
    south.speed_kmh = 40.0
    return spec
