"""
IntelliSignal
rl/phase_state.py

WHAT THE AGENT SEES when it is choosing a green duration.

The Stage 2 observation lists the four approaches in a fixed compass order
(north, east, south, west) and adds a flag saying which pair currently has
green. That works, but it makes the network learn the same thing twice: the
rule "give more time to the direction with the longer queue" has to be
learned once for north-south greens and again, with the inputs permuted, for
east-west greens.

Here the observation is written from the point of view of the green that is
about to be served, so that rule is one rule:

    index  name                meaning
    -----  ------------------  -------------------------------------------
      0    served_queue        halting vehicles on the two approaches that
                               are about to get green
      1    other_queue         halting vehicles on the two approaches that
                               will stay red
      2    served_approaching  vehicles present but NOT halting on the
                               served approaches - traffic that will arrive
                               during this green
      3    other_approaching   the same for the direction kept at red
      4    served_waiting      summed current waiting time, served direction
      5    other_waiting       summed current waiting time, other direction
      6    is_north_south      1.0 when the green about to be served is
                               north-south, 0.0 when it is east-west
      7    worst_approach      the longest queue on any SINGLE approach
      8    last_served_green   the green this direction got last time
      9    last_other_green    the green the other direction got last time

Index 6 is kept deliberately. The demand is not symmetric - 850 veh/h on
each of north and south against 650 on each of east and west - so the best
plan is not symmetric either, and a policy that could not tell the two
directions apart could not express 7 s against 6 s.

Index 7 is the only entry that is not a per-direction aggregate. North and
south are summed into one number, so a reading of ten halting vehicles can
mean five waiting on each or ten waiting on one. The second case is the one
that spills back, and it is invisible in indices 0 and 1.

Indices 2 and 3 are the ones the Stage 2 observation has no equivalent of.
Vehicles are released into this network by SUMO `<flow vehsPerHour=...>`
without `random="true"`, which inserts them on an equidistant schedule. The
traffic therefore arrives in a repeating pattern, and knowing how much of it
is already on the approach but not yet stopped is what lets a controller end
a green just as the last vehicle of a platoon clears.

Everything is normalised into [0, 1] and clipped, so the observation always
sits inside the declared Box space.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from rl.phase_action import PHASE_EW, PHASE_NS
from utils.scenario_builder import INCOMING_LANES

#: The approaches that move together under each green, and the lanes of each.
PHASE_APPROACHES: Dict[int, List[str]] = {
    PHASE_NS: ["north", "south"],
    PHASE_EW: ["east", "west"],
}

APPROACH_LANES: Dict[str, List[str]] = {
    "north": ["N2C_0", "N2C_1"],
    "east":  ["E2C_0", "E2C_1"],
    "south": ["S2C_0", "S2C_1"],
    "west":  ["W2C_0", "W2C_1"],
}

# The two directions together must describe exactly the Stage 1 incoming
# lanes, or this observation would be measuring a different road.
_flat = [lane for lanes in APPROACH_LANES.values() for lane in lanes]
assert sorted(_flat) == sorted(INCOMING_LANES), (
    "rl/phase_state.py and utils/scenario_builder.py disagree about the "
    "incoming lanes")

OBSERVATION_LABELS = (
    "served_queue", "other_queue",
    "served_approaching", "other_approaching",
    "served_waiting", "other_waiting",
    "is_north_south",
    "worst_approach",
    "last_served_green", "last_other_green",
)

OBSERVATION_SIZE = len(OBSERVATION_LABELS)      # 10


#: How an unbounded reading is squeezed into [0, 1].
#:
#: ``clip``  value / scale, then clipped. Everything at or above the scale
#:           reads exactly 1.0. This is what every model up to 23 Nov 2026
#:           was trained with, so it stays the default and those checkpoints
#:           keep behaving as they did.
#: ``soft``  value / (value + scale). Strictly increasing everywhere, so no
#:           two different congestion levels ever produce the same number.
#:           At the scale it reads 0.5, at twice the scale 0.67, at five
#:           times 0.83, and it approaches 1.0 without reaching it.
SATURATION_CLIP = "clip"
SATURATION_SOFT = "soft"

#: Bumped when the meaning of the observation changes. A checkpoint records
#: it, and `extractor_from_contract` rebuilds the extractor the policy was
#: trained with rather than today's default.
OBSERVATION_VERSION_CLIP = 1
OBSERVATION_VERSION_SOFT = 2


class PhaseStateExtractor:
    """Reads the traffic state from SUMO, from the served direction's view.

    Why the saturation mode exists
    ------------------------------
    The scales are measured from one scenario, and a custom workload can
    be far outside it. Measured on a 528-vehicle custom workload with
    1312 veh/h on a single lane, against the moderate model's own scales:

        worst_approach      93.2 % of observations pinned at exactly 1.0
        served_waiting      74.8 %
        served_queue        46.6 %

    A pinned input carries no information. The policy was shown a nearly
    constant vector and returned a nearly constant action - 88 % of its
    greens were the shortest on the menu. That is not a timing fault and
    not a reward fault; the controller simply could not see the difference
    between busy and very busy.

    ``saturation="soft"`` removes the pinning without changing what the
    numbers mean near the calibrated range.
    """

    def __init__(self, max_queue_per_direction: float = 20.0,
                 max_waiting_per_direction: float = 200.0,
                 max_vehicles_per_direction: float = 30.0,
                 max_queue_per_approach: float = 12.0,
                 max_green_seconds: float = 20.0,
                 saturation: str = SATURATION_CLIP):
        if saturation not in (SATURATION_CLIP, SATURATION_SOFT):
            raise ValueError(
                "saturation must be {!r} or {!r}".format(SATURATION_CLIP,
                                                         SATURATION_SOFT))
        self.saturation = saturation
        self.max_queue = float(max_queue_per_direction)
        self.max_waiting = float(max_waiting_per_direction)
        self.max_vehicles = float(max_vehicles_per_direction)
        self.max_approach_queue = float(max_queue_per_approach)
        self.max_green = float(max_green_seconds)
        self._subscribed = False
        self._subscription_vars = ()
        for name, value in (("max_queue_per_direction", self.max_queue),
                            ("max_waiting_per_direction", self.max_waiting),
                            ("max_vehicles_per_direction", self.max_vehicles),
                            ("max_queue_per_approach", self.max_approach_queue),
                            ("max_green_seconds", self.max_green)):
            if value <= 0:
                raise ValueError("{} must be positive".format(name))

    # ------------------------------------------------------------------
    def _scale(self, value: float, scale: float) -> float:
        """Squeeze an unbounded reading into [0, 1).

        `soft` is `v / (v + k)`: monotone for every v >= 0, so ten halting
        vehicles and forty halting vehicles are still different numbers
        even when the scale was calibrated at four.
        """
        value = max(float(value), 0.0)
        if self.saturation == SATURATION_SOFT:
            return value / (value + float(scale))
        return value / float(scale)

    @property
    def observation_version(self) -> int:
        return (OBSERVATION_VERSION_SOFT
                if self.saturation == SATURATION_SOFT
                else OBSERVATION_VERSION_CLIP)

    # ------------------------------------------------------------------
    def subscribe(self, conn) -> None:
        """Ask SUMO to push the eight lanes' readings once per step.

        Without this, reading the state costs 24 TraCI round-trips for every
        simulated second - 43,000 per episode - and a round-trip, not the
        traffic simulation, is what a training run spends its time on. A
        subscription collapses all of them into one call to
        `getAllSubscriptionResults()`, and the values are identical: SUMO
        fills them from the same variables the direct getters read.
        """
        import traci.constants as tc
        self._subscription_vars = (tc.LAST_STEP_VEHICLE_HALTING_NUMBER,
                                   tc.VAR_WAITING_TIME,
                                   tc.LAST_STEP_VEHICLE_NUMBER)
        for lanes in APPROACH_LANES.values():
            for lane in lanes:
                conn.lane.subscribe(lane, self._subscription_vars)
        self._subscribed = True

    # ------------------------------------------------------------------
    def raw(self, conn) -> Dict[str, Any]:
        """Unnormalised readings, per approach and per direction."""
        per_approach_queue: Dict[str, float] = {}
        per_approach_waiting: Dict[str, float] = {}
        per_approach_vehicles: Dict[str, float] = {}

        pushed = None
        if getattr(self, "_subscribed", False):
            pushed = conn.lane.getAllSubscriptionResults()

        if pushed:
            halting, waiting_var, vehicles_var = self._subscription_vars
            for approach, lanes in APPROACH_LANES.items():
                per_approach_queue[approach] = float(sum(
                    pushed[lane][halting] for lane in lanes))
                per_approach_waiting[approach] = float(sum(
                    pushed[lane][waiting_var] for lane in lanes))
                per_approach_vehicles[approach] = float(sum(
                    pushed[lane][vehicles_var] for lane in lanes))
        else:
            for approach, lanes in APPROACH_LANES.items():
                per_approach_queue[approach] = float(sum(
                    conn.lane.getLastStepHaltingNumber(lane) for lane in lanes))
                per_approach_waiting[approach] = float(sum(
                    conn.lane.getWaitingTime(lane) for lane in lanes))
                per_approach_vehicles[approach] = float(sum(
                    conn.lane.getLastStepVehicleNumber(lane) for lane in lanes))

        out: Dict[str, Any] = {
            "approach_queues": per_approach_queue,
            "approach_waiting": per_approach_waiting,
            "queues": {}, "waiting": {}, "vehicles": {},
        }
        for phase, approaches in PHASE_APPROACHES.items():
            out["queues"][phase] = sum(per_approach_queue[a] for a in approaches)
            out["waiting"][phase] = sum(per_approach_waiting[a] for a in approaches)
            out["vehicles"][phase] = sum(per_approach_vehicles[a] for a in approaches)
        return out

    # ------------------------------------------------------------------
    def observe(self, raw: Dict[str, Any], phase: int,
                last_served_green: float,
                last_other_green: float) -> np.ndarray:
        """Turn a raw reading into the normalised observation vector."""
        other = PHASE_EW if phase == PHASE_NS else PHASE_NS

        served_queue = raw["queues"][phase]
        other_queue = raw["queues"][other]
        served_approaching = max(raw["vehicles"][phase] - served_queue, 0.0)
        other_approaching = max(raw["vehicles"][other] - other_queue, 0.0)
        worst_approach = max(raw["approach_queues"].values())

        scale = self._scale
        values = [
            scale(served_queue, self.max_queue),
            scale(other_queue, self.max_queue),
            scale(served_approaching, self.max_vehicles),
            scale(other_approaching, self.max_vehicles),
            scale(raw["waiting"][phase], self.max_waiting),
            scale(raw["waiting"][other], self.max_waiting),
            1.0 if phase == PHASE_NS else 0.0,
            scale(worst_approach, self.max_approach_queue),
            float(last_served_green) / self.max_green,
            float(last_other_green) / self.max_green,
        ]
        observation = np.asarray(values, dtype=np.float32)
        return np.clip(observation, 0.0, 1.0, dtype=np.float32)

    # ------------------------------------------------------------------
    @staticmethod
    def zeros() -> np.ndarray:
        return np.zeros(OBSERVATION_SIZE, dtype=np.float32)

    # ------------------------------------------------------------------
    def contract(self) -> Dict[str, float]:
        """The normalisation constants, stored with every checkpoint.

        A policy is only meaningful against the scales it was trained with,
        so they travel with the weights rather than living in a config file
        that can be edited afterwards.
        """
        return {
            "observation_size": float(OBSERVATION_SIZE),
            "observation_version": float(self.observation_version),
            "saturation": self.saturation,
            "max_queue_per_direction": self.max_queue,
            "max_waiting_per_direction": self.max_waiting,
            "max_vehicles_per_direction": self.max_vehicles,
            "max_queue_per_approach": self.max_approach_queue,
            "max_green_seconds": self.max_green,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def describe(observation: np.ndarray) -> str:
        return ", ".join("{}={:.3f}".format(name, value)
                         for name, value in zip(OBSERVATION_LABELS, observation))


# ---------------------------------------------------------------------------
KNOWN_CONTRACT_KEYS = ("max_queue_per_direction", "max_waiting_per_direction",
                       "max_vehicles_per_direction", "max_queue_per_approach",
                       "max_green_seconds")


def extractor_from_contract(contract: Optional[Dict[str, Any]]
                            ) -> PhaseStateExtractor:
    """Rebuild the extractor a checkpoint was trained with.

    A contract with no `saturation` key predates the soft mode, so it gets
    the clipping it was trained with. Reading an old policy with the new
    scaling would feed it inputs it has never seen.
    """
    if not contract:
        return PhaseStateExtractor()
    scales = {key: float(contract[key]) for key in KNOWN_CONTRACT_KEYS
              if key in contract}
    return PhaseStateExtractor(
        saturation=str(contract.get("saturation", SATURATION_CLIP)), **scales)
