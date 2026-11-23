"""
IntelliSignal
custom_traffic/spec.py

What the user typed, before anything is filled in for them.

A `CustomSpec` is deliberately full of `None`. `None` means "the user did not
say", which is NOT the same as 0 - a lane with `count = 0` is a lane the user
explicitly emptied, and it must stay empty. Keeping the two apart is the whole
reason this is a separate type from the resolved workload: once the resolver
has run, every field has a value and the distinction is gone, so it has to be
carried here.

    CustomSpec            what the user typed, mostly None
         |
         |  custom_traffic/resolver.py
         v
    ResolvedWorkload      every field decided, each marked user or auto
         |
         |  custom_traffic/builder.py
         v
    SUMO .rou.xml         one explicit <vehicle> per vehicle
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from custom_traffic.network import (APPROACHES, LANES_PER_APPROACH,
                                    MOVEMENT_ORDER, lane_label,
                                    movements_for_lane)

#: How a lane's turning split was given.
TURNS_AS_COUNTS = "counts"
TURNS_AS_PERCENT = "percent"


@dataclass
class LaneInput:
    """One incoming lane, as the form presents it.

    Every traffic field may be None. `enabled` is the user's choice of whether
    this lane carries traffic at all; a disabled lane resolves to zero demand
    and is never filled in for them.
    """

    approach: str
    lane: int
    enabled: bool = False

    #: Exactly this many vehicles.
    count: Optional[int] = None
    #: ...or somewhere in this range, drawn once with the workload's seed.
    count_min: Optional[int] = None
    count_max: Optional[int] = None

    #: The window during which the vehicles arrive, in seconds from the start.
    window_start_s: Optional[float] = None
    window_end_s: Optional[float] = None

    #: Desired speed. SUMO decides the actual speed as traffic interacts.
    speed_kmh: Optional[float] = None

    #: Turning split. Values are vehicle counts or percentages depending on
    #: `turns_mode`; a movement left as None is derived from the others.
    turns_mode: str = TURNS_AS_COUNTS
    turns: Dict[str, Optional[float]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.approach not in APPROACHES:
            raise ValueError("Unknown approach {!r}.".format(self.approach))
        if self.lane not in range(LANES_PER_APPROACH):
            raise ValueError("Unknown lane {!r}.".format(self.lane))
        allowed = movements_for_lane(self.lane)
        for movement in list(self.turns):
            if movement not in MOVEMENT_ORDER:
                raise ValueError("Unknown movement {!r}.".format(movement))
            if movement not in allowed:
                raise ValueError(
                    "{} cannot turn {}. That lane serves {}.".format(
                        lane_label(self.approach, self.lane), movement,
                        " and ".join(allowed)))

    # ------------------------------------------------------------------
    @property
    def key(self) -> str:
        return "{}_{}".format(self.approach, self.lane)

    @property
    def label(self) -> str:
        return lane_label(self.approach, self.lane)

    @property
    def allowed_movements(self):
        return movements_for_lane(self.lane)

    # ------------------------------------------------------------------
    def stated_turns(self) -> Dict[str, float]:
        """Only the turning values the user actually gave."""
        return {movement: float(value)
                for movement, value in self.turns.items()
                if value is not None}

    def has_user_input(self) -> bool:
        """True when the user put a traffic number on this lane.

        `enabled` on its own is not input - ticking a lane and typing nothing
        is a request for the resolver to fill it, not a value.
        """
        if any(value is not None for value in
               (self.count, self.count_min, self.count_max,
                self.window_start_s, self.window_end_s, self.speed_kmh)):
            return True
        return bool(self.stated_turns())

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "approach": self.approach, "lane": self.lane,
            "enabled": self.enabled, "count": self.count,
            "count_min": self.count_min, "count_max": self.count_max,
            "window_start_s": self.window_start_s,
            "window_end_s": self.window_end_s,
            "speed_kmh": self.speed_kmh,
            "turns_mode": self.turns_mode,
            "turns": dict(self.turns),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LaneInput":
        return cls(
            approach=data["approach"], lane=int(data["lane"]),
            enabled=bool(data.get("enabled", False)),
            count=data.get("count"), count_min=data.get("count_min"),
            count_max=data.get("count_max"),
            window_start_s=data.get("window_start_s"),
            window_end_s=data.get("window_end_s"),
            speed_kmh=data.get("speed_kmh"),
            turns_mode=data.get("turns_mode", TURNS_AS_COUNTS),
            turns=dict(data.get("turns") or {}),
        )


# ---------------------------------------------------------------------------
@dataclass
class CustomSpec:
    """A whole custom workload as typed, plus the few settings around it."""

    name: str = "custom"
    #: Which scenario's vehicle type, signal plan and step length to start
    #: from. Everything not about demand is inherited from it.
    base_scenario: str = "moderate"
    #: Simulated seconds. None means "take the base scenario's duration".
    duration_s: Optional[float] = None
    #: Seed for the generated values AND for SUMO, so a workload is
    #: reproducible from the spec alone.
    seed: int = 6101
    lanes: List[LaneInput] = field(default_factory=list)

    # ------------------------------------------------------------------
    @classmethod
    def empty(cls, base_scenario: str = "moderate", seed: int = 6101
              ) -> "CustomSpec":
        """A blank form: every lane present, nothing selected, nothing typed."""
        return cls(base_scenario=base_scenario, seed=seed, lanes=[
            LaneInput(approach=approach, lane=lane)
            for approach in APPROACHES
            for lane in range(LANES_PER_APPROACH)])

    # ------------------------------------------------------------------
    def lane(self, approach: str, lane: int) -> LaneInput:
        for entry in self.lanes:
            if entry.approach == approach and entry.lane == lane:
                return entry
        raise KeyError("No lane {} {} in this spec.".format(approach, lane))

    def enabled_lanes(self) -> List[LaneInput]:
        return [entry for entry in self.lanes if entry.enabled]

    def has_user_input(self) -> bool:
        """At least one real traffic value somewhere.

        The resolver refuses to invent a whole scenario from nothing: an
        empty form is a mistake, not a request. "Generate example" fills the
        form first and then this is true.
        """
        return any(entry.enabled and entry.has_user_input()
                   for entry in self.lanes)

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "base_scenario": self.base_scenario,
            "duration_s": self.duration_s, "seed": int(self.seed),
            "lanes": [entry.to_dict() for entry in self.lanes],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CustomSpec":
        return cls(
            name=data.get("name", "custom"),
            base_scenario=data.get("base_scenario", "moderate"),
            duration_s=data.get("duration_s"),
            seed=int(data.get("seed", 6101)),
            lanes=[LaneInput.from_dict(entry) for entry in data.get("lanes", [])],
        )
