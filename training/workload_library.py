"""
IntelliSignal
training/workload_library.py

The traffic a controller is trained, selected and finally judged on.

Why this exists
---------------
Every model before 24 Nov 2026 was trained on exactly two demand
patterns - `moderate` and `heavy` - both balanced, both steady for the
whole run. A custom workload is nothing like either, and the measured
consequence was severe: on 528 vehicles concentrated on two of the eight
lanes, the moderate model's observation pinned at its clipping limit in
up to 93 % of decisions and it emitted the shortest green on its menu 88 %
of the time.

A controller cannot be expected to generalise to traffic it has never
been shown. This module builds that variety:

    balanced        even demand on all four approaches
    directional     one axis much busier than the other
    lane_skew       one lane of an approach much busier than its partner
    burst           a quiet run with a heavy pulse in the middle
    ramp            demand that builds through the run
    turn_heavy      an unusual turning split
    light/heavy     the same shapes at different intensities

Three disjoint splits
---------------------
    train       what the agent sees while learning
    validation  what `best.pt` is chosen on
    test        what the final number is reported on, touched once

The three use different workload seeds AND different SUMO seeds, so a
model cannot be selected on the traffic it was fitted to, nor reported on
the traffic it was selected on.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from custom_traffic.network import APPROACHES, LANES_PER_APPROACH
from custom_traffic.resolver import resolve
from custom_traffic.spec import CustomSpec, LaneInput
from custom_traffic.workload import ResolvedWorkload

#: Workload-seed bases. Disjoint, so no shape is ever repeated verbatim
#: across two splits.
SEED_BASE = {"train": 11000, "validation": 22000, "test": 33000}

#: SUMO seeds, also disjoint per split.
SUMO_SEED_BASE = {"train": 8000, "validation": 7000, "test": 6100}

#: How many distinct workloads each split holds. Training covers every
#: shape at every intensity; validation and test take a rotated subset so
#: a split is never a prefix of another.
SPLIT_SIZE = {"train": 18, "validation": 6, "test": 6}

#: Where each split starts in the (shape, intensity) enumeration, and how
#: far it strides. The pairs are ordered intensity-major, so a contiguous
#: block of six would be all one intensity; striding by three gives
#: validation and test two workloads at each of light, moderate and heavy,
#: with no pair shared between them.
SPLIT_WALK = {"train": (0, 1), "validation": (1, 3), "test": (2, 3)}

SHAPES = ("balanced", "directional", "lane_skew", "burst", "ramp",
          "turn_heavy")

#: Vehicles per hour per approach, before any shaping.
#:
#: Calibrated against the Indian fleet's real capacity. With startup lost
#: time and safe headways this junction serves about 93 % of 700 veh/h per
#: approach and saturates above roughly 900, so "heavy" sits just inside
#: the serviceable range rather than beyond it - an oversaturated training
#: set teaches a controller nothing except that it cannot win.
INTENSITY = {"light": 280.0, "moderate": 480.0, "heavy": 700.0}

#: The scenario every generated workload inherits its fleet, clearance and
#: step length from. `indian_arterial` carries the mixed fleet with real
#: startup delay; `moderate` and `heavy` keep the single generic car and
#: are left alone so their recorded results still reproduce.
BASE_SCENARIO = "indian_arterial"


# ---------------------------------------------------------------------------
def _lane(spec: CustomSpec, approach: str, index: int, count: int,
          window: Tuple[float, float] = None, speed: float = None,
          turns: Dict[str, float] = None, mode: str = "counts") -> None:
    entry = spec.lane(approach, index)
    entry.enabled = True
    entry.count = int(max(count, 0))
    if window:
        entry.window_start_s, entry.window_end_s = float(window[0]), float(window[1])
    if speed is not None:
        entry.speed_kmh = float(speed)
    if turns:
        entry.turns_mode = mode
        entry.turns = dict(turns)


def _per_lane(vph: float, duration: float) -> int:
    return int(round(vph * duration / 3600.0 / LANES_PER_APPROACH))


# ---------------------------------------------------------------------------
def build(shape: str, intensity: str, duration_s: float, seed: int
          ) -> ResolvedWorkload:
    """One workload of a given shape and intensity."""
    base = INTENSITY[intensity]
    spec = CustomSpec(name="{}_{}".format(shape, intensity),
                      base_scenario=BASE_SCENARIO,
                      duration_s=float(duration_s), seed=int(seed),
                      lanes=[LaneInput(approach=a, lane=l)
                             for a in APPROACHES
                             for l in range(LANES_PER_APPROACH)])
    full = (0.0, float(duration_s))
    n = _per_lane(base, duration_s)

    if shape == "balanced":
        for approach in APPROACHES:
            for index in range(LANES_PER_APPROACH):
                _lane(spec, approach, index, n, full)

    elif shape == "directional":
        # One axis carries roughly three times the other.
        for approach in APPROACHES:
            heavy_axis = approach in ("north", "south")
            count = int(n * (1.5 if heavy_axis else 0.5))
            for index in range(LANES_PER_APPROACH):
                _lane(spec, approach, index, count, full)

    elif shape == "lane_skew":
        # The kerb-side lane of each approach carries most of the demand,
        # which the per-direction totals in the observation cannot see.
        for approach in APPROACHES:
            _lane(spec, approach, 0, int(n * 1.7), full)
            _lane(spec, approach, 1, int(n * 0.3), full)

    elif shape == "burst":
        # Quiet, then a pulse through the middle third on one axis.
        third = duration_s / 3.0
        for approach in APPROACHES:
            for index in range(LANES_PER_APPROACH):
                _lane(spec, approach, index, int(n * 0.5), full)
        for approach in ("east", "west"):
            _lane(spec, approach, 0, int(n * 1.4), (third, 2 * third))

    elif shape == "ramp":
        # Demand arrives in the second half only, so the controller meets
        # a rising load rather than a steady one.
        half = duration_s / 2.0
        for approach in APPROACHES:
            for index in range(LANES_PER_APPROACH):
                _lane(spec, approach, index, int(n * 0.4), (0.0, half))
        for approach in APPROACHES:
            _lane(spec, approach, 0, int(n * 1.1), (half, duration_s))

    elif shape == "turn_heavy":
        # Unusual turning splits: the kerb lane mostly turns right, the
        # centre lane mostly turns left.
        for approach in APPROACHES:
            _lane(spec, approach, 0, n, full,
                  turns={"right": 70.0}, mode="percent")
            _lane(spec, approach, 1, n, full,
                  turns={"left": 70.0}, mode="percent")
    else:
        raise ValueError("Unknown shape {!r}. Use one of {}.".format(
            shape, ", ".join(SHAPES)))

    resolution = resolve(spec)
    if resolution.workload is None:
        raise ValueError("{}/{} did not resolve: {}".format(
            shape, intensity, "; ".join(str(i) for i in resolution.errors)))
    return resolution.workload


# ---------------------------------------------------------------------------
def split(name: str, duration_s: float = 900.0) -> List[ResolvedWorkload]:
    """The workloads of one split, in a fixed order.

    Deterministic: the same call always returns the same traffic, so a
    training run is reproducible and two candidates can be compared on
    identical workloads.
    """
    if name not in SEED_BASE:
        raise ValueError("Unknown split {!r}. Use train, validation or test."
                         .format(name))
    size = SPLIT_SIZE[name]
    base = SEED_BASE[name]
    offset, stride = SPLIT_WALK[name]

    # Every (shape, intensity) pair, in a fixed order.
    pairs = [(shape, intensity)
             for intensity in INTENSITY
             for shape in SHAPES]

    out: List[ResolvedWorkload] = []
    for index in range(size):
        shape, intensity = pairs[(offset + index * stride) % len(pairs)]
        out.append(build(shape, intensity, duration_s, base + index * 17))
    return out


def sumo_seeds(name: str, count: int) -> List[int]:
    """SUMO seeds for a split. Disjoint across splits by construction."""
    base = SUMO_SEED_BASE[name]
    return [base + index for index in range(count)]


# ---------------------------------------------------------------------------
def describe(workloads: List[ResolvedWorkload]) -> List[Dict[str, object]]:
    """A short table of what a split actually contains."""
    rows = []
    for workload in workloads:
        per_approach = workload.demand_per_approach()
        lanes = workload.active_lanes
        busiest = max(lanes, key=lambda l: l.vehicles_per_hour) if lanes else None
        rows.append({
            "name": workload.name,
            "id": workload.workload_id[:8],
            "vehicles": workload.total_vehicles,
            "duration_s": workload.duration_s,
            "ns": per_approach.get("north", 0) + per_approach.get("south", 0),
            "ew": per_approach.get("east", 0) + per_approach.get("west", 0),
            "busiest_lane_vph": round(busiest.vehicles_per_hour, 0) if busiest else 0,
            "lanes_used": len(lanes),
        })
    return rows
