"""
IntelliSignal
custom_traffic/workload.py

The resolved workload: every value decided, and a vehicle list to prove it.

This is what the preview shows and what SUMO runs. There is no second
interpretation step between them - `custom_traffic/builder.py` writes one
`<vehicle>` element per entry of `ResolvedWorkload.vehicles`, so "what the
preview said" and "what was simulated" cannot disagree.

Why an explicit vehicle list instead of `<flow>`
------------------------------------------------
The Moderate and Heavy scenarios use `<flow vehsPerHour=...>`, which asks
SUMO to insert vehicles on its own schedule. That is fine for a fixed demand
written in a YAML file, but it cannot express "200 cars in the first five
minutes", it does not give an exact count, and two controllers compared on
the same seed get the same schedule only because SUMO happens to be
deterministic. Writing the vehicles out removes all three problems: the
count is exact, the arrival window is exact, and both controllers are handed
the same file.

`workload_id` is a hash of everything that affects the simulation. A saved
Hybrid decision is reused only when that id matches, so a workload that was
edited is never answered with the previous workload's result.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from custom_traffic.network import lane_label, mps_to_kmh
from utils.config_loader import PROJECT_ROOT

WORKLOAD_DIR = os.path.join(PROJECT_ROOT, "results", "custom_workloads")

#: How a value came to be. Shown in the preview so a user can see what they
#: chose and what was chosen for them.
FROM_USER = "user"
FROM_AUTO = "auto"
FROM_DERIVED = "derived"


@dataclass
class Vehicle:
    """One vehicle, fully specified."""

    vehicle_id: str
    approach: str
    lane: int
    movement: str
    depart: float            # seconds from the start of the simulation
    speed_kmh: float         # desired maximum speed
    #: Which class of vehicle this is - motorcycle, car, auto, bus. The
    #: mix is decided once, when the workload is resolved, so the same
    #: workload always puts the same vehicle in the same place and both
    #: controllers in a comparison meet identical traffic.
    vehicle_class: str = "car"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Vehicle":
        return cls(**data)


@dataclass
class ResolvedLane:
    """One lane after resolution: no None left anywhere."""

    approach: str
    lane: int
    enabled: bool
    count: int
    window_start_s: float
    window_end_s: float
    speed_kmh: float
    turns: Dict[str, int]
    #: field name -> FROM_USER / FROM_AUTO / FROM_DERIVED
    provenance: Dict[str, str] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return lane_label(self.approach, self.lane)

    @property
    def window_seconds(self) -> float:
        return max(self.window_end_s - self.window_start_s, 0.0)

    @property
    def vehicles_per_hour(self) -> float:
        window = self.window_seconds
        return (self.count * 3600.0 / window) if window > 0 else 0.0

    def source_of(self, field_name: str) -> str:
        return self.provenance.get(field_name, FROM_AUTO)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResolvedLane":
        return cls(
            approach=data["approach"], lane=int(data["lane"]),
            enabled=bool(data["enabled"]), count=int(data["count"]),
            window_start_s=float(data["window_start_s"]),
            window_end_s=float(data["window_end_s"]),
            speed_kmh=float(data["speed_kmh"]),
            turns={k: int(v) for k, v in (data.get("turns") or {}).items()},
            provenance=dict(data.get("provenance") or {}),
        )


@dataclass
class ResolvedWorkload:
    """A complete, runnable custom scenario."""

    name: str
    base_scenario: str
    duration_s: float
    step_length: float
    seed: int
    lanes: List[ResolvedLane]
    vehicles: List[Vehicle]
    #: The signal plan, vehicle type and other non-demand settings inherited
    #: from the base scenario, copied in so the workload is self-contained.
    signal: Dict[str, float] = field(default_factory=dict)
    vehicle_type: Dict[str, float] = field(default_factory=dict)
    #: "generic" (the single car every earlier result used) or
    #: "indian_urban" (the two-wheeler-dominant mix with startup delay).
    traffic_model: str = "generic"
    notes: List[str] = field(default_factory=list)
    created_at: str = ""

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now().isoformat(timespec="seconds")

    # ------------------------------------------------------------------
    @property
    def total_vehicles(self) -> int:
        return len(self.vehicles)

    @property
    def active_lanes(self) -> List[ResolvedLane]:
        return [lane for lane in self.lanes if lane.enabled and lane.count > 0]

    def lane(self, approach: str, index: int) -> ResolvedLane:
        for entry in self.lanes:
            if entry.approach == approach and entry.lane == index:
                return entry
        raise KeyError("{} lane {} is not in this workload.".format(
            approach, index))

    def demand_per_approach(self) -> Dict[str, int]:
        totals: Dict[str, int] = {}
        for entry in self.lanes:
            totals[entry.approach] = totals.get(entry.approach, 0) + entry.count
        return totals

    def speeds_used(self) -> List[float]:
        """Distinct desired speeds, so the builder can write one vType each."""
        return sorted({round(lane.speed_kmh, 1)
                       for lane in self.lanes if lane.count > 0})

    def fleet_mix(self) -> Dict[str, int]:
        """How many of each vehicle class the workload actually contains."""
        counts: Dict[str, int] = {}
        for vehicle in self.vehicles:
            counts[vehicle.vehicle_class] = counts.get(
                vehicle.vehicle_class, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def total_pcu(self) -> float:
        """The demand in passenger car units, which is how Indian signal
        design measures capacity. Sixty motorcycles are not sixty cars."""
        from custom_traffic.fleet import get_fleet
        pcu = {v.name: v.pcu for v in get_fleet(self.traffic_model)}
        return round(sum(pcu.get(v.vehicle_class, 1.0)
                         for v in self.vehicles), 1)

    # ------------------------------------------------------------------
    def signature(self) -> Dict[str, Any]:
        """Everything that changes the simulation, and nothing that does not.

        `created_at` and `notes` are excluded on purpose: re-resolving the
        same inputs a minute later must produce the same id, or a saved
        Hybrid decision could never be reused.
        """
        return {
            "base_scenario": self.base_scenario,
            "duration_s": round(float(self.duration_s), 3),
            "step_length": round(float(self.step_length), 3),
            "seed": int(self.seed),
            "signal": {k: round(float(v), 3) for k, v in sorted(self.signal.items())},
            "vehicle_type": {k: round(float(v), 4)
                             for k, v in sorted(self.vehicle_type.items())},
            "traffic_model": self.traffic_model,
            "vehicles": [
                [v.vehicle_id, v.approach, v.lane, v.movement,
                 round(float(v.depart), 3), round(float(v.speed_kmh), 2),
                 v.vehicle_class]
                for v in self.vehicles],
        }

    @property
    def workload_id(self) -> str:
        payload = json.dumps(self.signature(), sort_keys=True,
                             separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "name": self.name,
            "base_scenario": self.base_scenario,
            "duration_s": self.duration_s,
            "step_length": self.step_length,
            "seed": self.seed,
            "signal": dict(self.signal),
            "vehicle_type": dict(self.vehicle_type),
            "traffic_model": self.traffic_model,
            "lanes": [lane.to_dict() for lane in self.lanes],
            "vehicles": [v.to_dict() for v in self.vehicles],
            "notes": list(self.notes),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResolvedWorkload":
        return cls(
            name=data["name"], base_scenario=data["base_scenario"],
            duration_s=float(data["duration_s"]),
            step_length=float(data["step_length"]),
            seed=int(data["seed"]),
            lanes=[ResolvedLane.from_dict(entry) for entry in data["lanes"]],
            vehicles=[Vehicle.from_dict(entry) for entry in data["vehicles"]],
            signal=dict(data.get("signal") or {}),
            vehicle_type=dict(data.get("vehicle_type") or {}),
            traffic_model=data.get("traffic_model", "generic"),
            notes=list(data.get("notes") or []),
            created_at=data.get("created_at", ""),
        )

    # ------------------------------------------------------------------
    def save(self, directory: str = WORKLOAD_DIR) -> str:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "{}.json".format(self.workload_id))
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        return path

    @staticmethod
    def load(path_or_id: str, directory: str = WORKLOAD_DIR
             ) -> "ResolvedWorkload":
        path = path_or_id
        if not os.path.isfile(path):
            path = os.path.join(directory, "{}.json".format(path_or_id))
        with open(path, "r", encoding="utf-8") as handle:
            return ResolvedWorkload.from_dict(json.load(handle))

    @staticmethod
    def saved_workloads(directory: str = WORKLOAD_DIR) -> List[Dict[str, Any]]:
        """Short descriptions of every saved workload, newest first."""
        if not os.path.isdir(directory):
            return []
        out = []
        for name in os.listdir(directory):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(directory, name), "r",
                          encoding="utf-8") as handle:
                    data = json.load(handle)
            except (OSError, ValueError):
                continue
            out.append({
                "workload_id": data.get("workload_id", name[:-5]),
                "name": data.get("name", ""),
                "vehicles": len(data.get("vehicles") or []),
                "duration_s": data.get("duration_s"),
                "created_at": data.get("created_at", ""),
                "path": os.path.join(directory, name),
            })
        return sorted(out, key=lambda row: row["created_at"], reverse=True)

    # ------------------------------------------------------------------
    def preview_rows(self) -> List[Dict[str, Any]]:
        """One row per lane for the preview table, with units spelled out."""
        rows = []
        for entry in self.lanes:
            if not entry.enabled and entry.count == 0:
                continue
            turns = ", ".join(
                "{} {}".format(count, movement)
                for movement, count in entry.turns.items() if count)
            rows.append({
                "Lane": entry.label,
                "Vehicles": entry.count,
                "Arrive between": "{:.0f}-{:.0f} s".format(
                    entry.window_start_s, entry.window_end_s),
                "Rate": "{:.0f} veh/h".format(entry.vehicles_per_hour),
                "Desired speed": "{:.0f} km/h".format(entry.speed_kmh),
                "Turns": turns or "-",
                "count_source": entry.source_of("count"),
                "window_source": entry.source_of("window"),
                "speed_source": entry.source_of("speed"),
                "turns_source": entry.source_of("turns"),
            })
        return rows


# ---------------------------------------------------------------------------
def describe_speed(speed_kmh: float, limit_kmh: float) -> str:
    """One sentence about what a desired speed will actually do."""
    if speed_kmh >= limit_kmh:
        return ("{:.0f} km/h is at or above the {:.0f} km/h lane limit, so "
                "vehicles will travel at the limit.".format(speed_kmh,
                                                            limit_kmh))
    return ("{:.0f} km/h desired; SUMO sets the actual speed from the traffic "
            "ahead.".format(speed_kmh))


def mps_speed(speed_kmh: float) -> float:
    return float(speed_kmh) / 3.6


__all__ = ["Vehicle", "ResolvedLane", "ResolvedWorkload", "WORKLOAD_DIR",
           "FROM_USER", "FROM_AUTO", "FROM_DERIVED", "describe_speed",
           "mps_speed", "mps_to_kmh"]
