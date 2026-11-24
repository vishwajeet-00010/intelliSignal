"""
IntelliSignal
custom_traffic/fleet.py

The vehicles on the road, and why a four-second green is not a plan.

The problem this file exists to fix
-----------------------------------
Until now every vehicle in this project was one generic car:

    accel 2.6 m/s2, length 5.0 m, no startup delay, 50 km/h

That car begins moving the instant the signal turns green. Real ones do
not. A driver takes about a second to react, the driver behind reacts to
*them*, and the wave takes several seconds to reach the back of the
queue. Those seconds are **startup lost time**, and on an Indian urban
approach they are worth roughly 2-4 seconds of every green.

With no startup lost time in the model, a four-second green looked
efficient: four seconds of green produced nearly four seconds of
discharge. With it modelled, a four-second green produces almost
nothing - the queue has barely begun to move when the light changes.
That is why the measured optimum used to be 7 s and why practitioners
use 15 s and more.

So the fix for "4 s greens are not practical" is not only to forbid them.
It is to make the simulation agree.

The Indian urban fleet
----------------------
Shares and parameters are set for a typical Indian city arterial:
two-wheeler dominant, a meaningful share of three-wheelers, and buses
that accelerate slowly and occupy a lot of stop line.

    motorcycle  62 %   2.2 m   quick away, filters into small gaps
    car         22 %   4.5 m
    auto        11 %   3.2 m   three-wheeler, slower and lower top speed
    bus          5 %  11.0 m   slow to start, long

`tau` is the desired time headway and `startupDelay` is the pause before
a stopped vehicle moves off; together they are what produce realistic
saturation flow and startup lost time. The mix is expressed in PCU
(passenger car units) as well as vehicle counts, because Indian capacity
work is normally done in PCU and a lane that is fine for 600
motorcycles is not fine for 600 buses.

Nothing here changes `moderate` or `heavy`. Those scenarios keep their
original single `car` type so every recorded number still reproduces;
the fleet is opt-in through `traffic_model: indian_urban`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

#: The name used in a scenario file to ask for this fleet.
INDIAN_URBAN = "indian_urban"
GENERIC = "generic"


@dataclass(frozen=True)
class VehicleClass:
    """One kind of vehicle, with the parameters SUMO needs."""

    name: str
    share: float             # fraction of vehicles, not of PCU
    length: float            # m
    accel: float             # m/s2
    decel: float             # m/s2
    max_speed_kmh: float
    min_gap: float           # m
    sigma: float             # driver imperfection, 0 = perfect
    tau: float               # desired time headway, s
    startup_delay: float     # s before a stopped vehicle moves off
    pcu: float               # passenger car units
    sumo_class: str
    gui_shape: str

    @property
    def max_speed_mps(self) -> float:
        return self.max_speed_kmh / 3.6


#: Shares from typical Indian urban arterial counts; the dynamics are set
#: so that saturation headway lands near 2.1-2.4 s/PCU, which is the range
#: Indian signal-design practice uses (against 1.9 s in Western manuals).
#: `tau` is never below the 1.0 s simulation step. The Krauss car-following
#: model is collision-free only while the desired headway is at least one
#: step; at tau 0.8 a first attempt at this fleet produced 400-565
#: rear-end collisions per run, which would have made every number below
#: meaningless. Indian headways really are short, but the place to express
#: that is `minGap` and `sigma`, not an unsafe tau.
INDIAN_URBAN_FLEET: Tuple[VehicleClass, ...] = (
    VehicleClass("motorcycle", 0.62, 2.2, 2.8, 5.0, 50.0, 1.2, 0.50, 1.0,
                 0.6, 0.25, "motorcycle", "motorcycle"),
    VehicleClass("car", 0.22, 4.5, 2.0, 4.5, 50.0, 2.0, 0.45, 1.2,
                 1.2, 1.00, "passenger", "passenger"),
    VehicleClass("auto", 0.11, 3.2, 1.6, 4.0, 40.0, 1.8, 0.55, 1.1,
                 1.0, 0.50, "taxi", "passenger/hatchback"),
    VehicleClass("bus", 0.05, 11.0, 1.0, 3.5, 40.0, 2.5, 0.45, 1.6,
                 2.0, 3.00, "bus", "bus"),
)

#: The single generic car every earlier result was measured with. Kept so
#: `moderate` and `heavy` still reproduce exactly.
GENERIC_FLEET: Tuple[VehicleClass, ...] = (
    VehicleClass("car", 1.0, 5.0, 2.6, 4.5, 50.0, 2.5, 0.5, 1.0,
                 0.0, 1.0, "passenger", "passenger"),
)

FLEETS: Dict[str, Tuple[VehicleClass, ...]] = {
    INDIAN_URBAN: INDIAN_URBAN_FLEET,
    GENERIC: GENERIC_FLEET,
}


# ---------------------------------------------------------------------------
def get_fleet(name: str) -> Tuple[VehicleClass, ...]:
    try:
        return FLEETS[str(name or GENERIC)]
    except KeyError:
        raise ValueError("Unknown traffic model {!r}. Available: {}.".format(
            name, ", ".join(sorted(FLEETS))))


def fleet_of(config: Dict[str, Any]) -> Tuple[VehicleClass, ...]:
    """The fleet a scenario asks for, defaulting to the historical car."""
    return get_fleet((config.get("vehicle") or {}).get("traffic_model",
                                                       GENERIC))


# ---------------------------------------------------------------------------
def vtype_xml(vehicle: VehicleClass, speed_kmh: float = None,
              suffix: str = "") -> str:
    """One `<vType>` line.

    `speed_kmh` lets a lane's desired speed override the class's own top
    speed, which is how the Custom Traffic form's speed field works. SUMO
    still caps everything at the lane limit.
    """
    speed = (vehicle.max_speed_kmh if speed_kmh is None
             else min(float(speed_kmh), vehicle.max_speed_kmh))
    return ('    <vType id="{name}{suffix}" vClass="{cls}" accel="{accel}" '
            'decel="{decel}" sigma="{sigma}" length="{length}" '
            'minGap="{gap}" maxSpeed="{speed:.2f}" tau="{tau}" '
            'startupDelay="{startup}" guiShape="{shape}"/>'.format(
                name=vehicle.name, suffix=suffix, cls=vehicle.sumo_class,
                accel=vehicle.accel, decel=vehicle.decel,
                sigma=vehicle.sigma, length=vehicle.length,
                gap=vehicle.min_gap, speed=speed / 3.6, tau=vehicle.tau,
                startup=vehicle.startup_delay, shape=vehicle.gui_shape))


def vtype_id(vehicle: VehicleClass, suffix: str = "") -> str:
    return "{}{}".format(vehicle.name, suffix)


# ---------------------------------------------------------------------------
def assign(count: int, fleet: Sequence[VehicleClass]) -> List[VehicleClass]:
    """Pick the vehicle class for each of `count` vehicles, deterministically.

    Largest-remainder on the shares, then interleaved rather than served
    in blocks, so a queue is a realistic mixture instead of sixty
    motorcycles followed by five buses.
    """
    if count <= 0:
        return []
    if len(fleet) == 1:
        return [fleet[0]] * count

    exact = {v.name: count * v.share for v in fleet}
    whole = {name: int(value) for name, value in exact.items()}
    remaining = count - sum(whole.values())
    order = sorted(exact, key=lambda n: (-(exact[n] - int(exact[n])), n))
    for index in range(remaining):
        whole[order[index % len(order)]] += 1

    by_name = {v.name: v for v in fleet}
    placed = {name: 0 for name in whole}
    out: List[VehicleClass] = []
    for position in range(1, count + 1):
        best, best_debt = None, None
        for name in sorted(whole):
            if placed[name] >= whole[name]:
                continue
            debt = whole[name] * position / float(count) - placed[name]
            if best_debt is None or debt > best_debt:
                best, best_debt = name, debt
        if best is None:
            break
        out.append(by_name[best])
        placed[best] += 1
    return out


# ---------------------------------------------------------------------------
def pcu_of(vehicles: Sequence[VehicleClass]) -> float:
    return sum(v.pcu for v in vehicles)


def describe(fleet: Sequence[VehicleClass]) -> List[Dict[str, Any]]:
    """A table for the interface: what is actually on the road."""
    return [{
        "Vehicle": v.name,
        "Share": "{:.0f} %".format(100 * v.share),
        "Length": "{:.1f} m".format(v.length),
        "Top speed": "{:.0f} km/h".format(v.max_speed_kmh),
        "Starts after": "{:.1f} s".format(v.startup_delay),
        "PCU": "{:.2f}".format(v.pcu),
    } for v in fleet]


def mean_pcu(fleet: Sequence[VehicleClass]) -> float:
    return sum(v.share * v.pcu for v in fleet)


def mean_startup_delay(fleet: Sequence[VehicleClass]) -> float:
    """Average pause before a stopped vehicle moves off.

    Not the whole startup lost time - the queue's reaction cascades, so
    the measured loss per green is larger than this. It is the per-vehicle
    part of it, and it is the number the interface quotes when it explains
    why a short green wastes most of itself.
    """
    return sum(v.share * v.startup_delay for v in fleet)


def effective_green_note(green_seconds: float,
                         fleet: Sequence[VehicleClass]) -> str:
    """One sentence on how much of a green actually moves traffic."""
    lost = 2.0 + mean_startup_delay(fleet)       # queue cascade plus reaction
    usable = max(green_seconds - lost, 0.0)
    share = (usable / green_seconds * 100.0) if green_seconds > 0 else 0.0
    return ("A {:.0f} s green loses about {:.1f} s to vehicles starting "
            "off, leaving roughly {:.1f} s ({:.0f} %) that actually moves "
            "traffic.".format(green_seconds, lost, usable, share))
