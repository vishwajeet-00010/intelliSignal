"""
IntelliSignal
utils/delay_accounting.py

Honest delay accounting, from SUMO's own trip information.

Why the existing metric is not enough
-------------------------------------
`utils/metrics.py` reports

    average_waiting_time = sum of accumulated waiting / vehicles SEEN

A vehicle that SUMO never managed to insert is in neither the numerator
nor the denominator. It simply is not there. So a controller that blocks
the entrance of a saturated lane is rewarded twice: the vehicles it keeps
out contribute no waiting, and they do not dilute the average either.

Measured on a 528-vehicle custom workload, the learned controller left 41
vehicles un-inserted and none of their delay appeared anywhere in its
score. The result did not flip - it lost on waiting as well - but the hole
is structural, and it grows exactly when a controller is behaving worst.

What this module adds
---------------------
It does **not** change `average_waiting_time`. That field keeps its
meaning so every historical number stays comparable. It adds a second,
versioned set of fields computed from SUMO's `tripinfo` output:

    requested                 vehicles the demand asked for
    entered / completed       what SUMO managed
    waiting_to_enter          requested but never inserted
    mean_stopped_waiting      per entered vehicle (the historical measure)
    mean_time_loss            per entered vehicle - time lost against a
                              free-flow trip. A SUPERSET of stopped
                              waiting, so the two are never added together
    mean_insertion_delay      per requested vehicle, including vehicles
                              that never entered at all
    mean_total_delay          per requested vehicle: time loss plus
                              insertion delay, which are disjoint

The last one is the number a comparison should lead with, because it is
the only one a controller cannot improve by refusing vehicles.

Not double-counting
-------------------
SUMO's `timeLoss` already contains the time a vehicle spent stopped, so
`timeLoss + waitingTime` would charge standing still twice. `departDelay`
is disjoint from both: it is time before the vehicle existed on the
network. Only those two are summed.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

#: Bumped if the meaning of any field below changes.
METRICS_VERSION = 3


def enable_tripinfo(sumocfg_path: str, output_path: str) -> str:
    """Add a `tripinfo-output` to an already-written .sumocfg.

    Done by rewriting the generated file rather than by changing
    `utils/scenario_builder.build_sumocfg`, so a run that does not ask for
    trip information produces byte-identical inputs to every historical
    run.
    """
    with open(sumocfg_path, "r", encoding="utf-8") as handle:
        text = handle.read()
    if "tripinfo-output" in text:
        return output_path

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    relative = os.path.relpath(
        output_path, os.path.dirname(os.path.abspath(sumocfg_path)))
    # `write-unfinished` is what makes the accounting honest. Without it
    # SUMO reports only the vehicles that reached their destination, so a
    # controller that leaves traffic stuck inside the junction when the
    # clock runs out is charged nothing for it - and the worse the
    # controller, the more of those there are.
    block = ('    <output>\n'
             '        <tripinfo-output value="{}"/>\n'
             '        <tripinfo-output.write-unfinished value="true"/>\n'
             '    </output>\n\n'
             '</configuration>').format(relative.replace(os.sep, "/"))
    text = text.replace("</configuration>", block)
    with open(sumocfg_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return output_path


# ---------------------------------------------------------------------------
def read_tripinfo(path: str) -> List[Dict[str, float]]:
    """Every trip SUMO recorded, finished or not, as plain dictionaries.

    A vehicle still on the network when the run ended has an `arrival`
    of -1 and a `timeLoss` counted up to that moment.
    """
    if not os.path.isfile(path):
        return []
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        # SUMO writes the closing tag only on a clean shutdown; a run that
        # was cut short still has usable rows, so salvage what is there.
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        if "</tripinfos>" not in text:
            text += "</tripinfos>"
        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            return []

    trips = []
    for element in root.iter("tripinfo"):
        get = element.get
        trips.append({
            "id": get("id", ""),
            "depart": float(get("depart", 0.0)),
            "departDelay": float(get("departDelay", 0.0)),
            "duration": float(get("duration", 0.0)),
            "waitingTime": float(get("waitingTime", 0.0)),
            "timeLoss": float(get("timeLoss", 0.0)),
            "arrival": float(get("arrival", -1.0)),
        })
    return trips


# ---------------------------------------------------------------------------
def account(trips: List[Dict[str, float]], scheduled_departures: List[float],
            entered_ids: Optional[List[str]] = None,
            run_end_s: float = 0.0,
            still_on_network: int = 0) -> Dict[str, Any]:
    """Build the versioned delay accounting.

    `scheduled_departures` is every vehicle the demand asked for, with the
    time it was due to depart. For a custom workload that list is exact.
    Vehicles that never entered are charged `run_end - scheduled`, which
    is the time they actually spent waiting to get in.
    """
    requested = len(scheduled_departures)
    # SUMO is asked for the unfinished trips as well and marks them with
    # an arrival of -1. They entered and they were delayed, so they count
    # towards the delay; they did not finish, so they do not count towards
    # completion. Leaving them out was worth about a second a vehicle to
    # whichever controller stranded the most traffic.
    arrived = [trip for trip in trips if trip.get("arrival", -1.0) >= 0.0]
    unfinished = [trip for trip in trips if trip.get("arrival", -1.0) < 0.0]
    completed = len(arrived)
    # A vehicle that entered either finished (it is in `arrived`), or was
    # still inside at the end. SUMO reports the second kind when asked;
    # an older file does not, and then the caller's count is all there
    # is - so the larger of the two is the honest answer.
    entered = completed + max(len(unfinished), max(int(still_on_network), 0))
    if entered_ids is not None:
        entered = len(entered_ids)
    never_entered = max(requested - entered, 0)

    total_time_loss = sum(trip["timeLoss"] for trip in trips)
    total_stopped = sum(trip["waitingTime"] for trip in trips)
    inserted_delay = sum(trip["departDelay"] for trip in trips)

    # The vehicles that never got in are the last ones scheduled, because
    # a queue at the entry is served in order. Charging them from their
    # scheduled time to the end of the run is the honest lower bound: they
    # waited at least that long and were still waiting when time ran out.
    unserved_delay = 0.0
    if never_entered:
        for scheduled in sorted(scheduled_departures)[-never_entered:]:
            unserved_delay += max(float(run_end_s) - float(scheduled), 0.0)

    insertion_delay = inserted_delay + unserved_delay
    total_delay = total_time_loss + insertion_delay

    def per(value, count):
        return (value / count) if count else 0.0

    return {
        "metrics_version": METRICS_VERSION,
        "requested": requested,
        "entered": entered,
        "completed": completed,
        "still_on_network": max(entered - completed, 0),
        "unfinished_charged": len(unfinished),
        "waiting_to_enter": never_entered,
        "total_time_loss_s": round(total_time_loss, 2),
        "total_stopped_waiting_s": round(total_stopped, 2),
        "total_insertion_delay_s": round(insertion_delay, 2),
        "unserved_insertion_delay_s": round(unserved_delay, 2),
        "total_delay_s": round(total_delay, 2),
        "mean_stopped_waiting_per_entered": round(per(total_stopped, entered), 3),
        "mean_time_loss_per_entered": round(per(total_time_loss, entered), 3),
        "mean_insertion_delay_per_requested": round(
            per(insertion_delay, requested), 3),
        "mean_total_delay_per_requested": round(per(total_delay, requested), 3),
        "completion_rate": round(per(completed, requested), 4),
    }


# ---------------------------------------------------------------------------
def cross_check(accounting: Dict[str, Any], collector_summary: Dict[str, Any]
                ) -> Dict[str, Any]:
    """Compare the trip-info accounting with the live collector.

    They measure the same run two different ways, so a large disagreement
    means one of them is wrong. Reported rather than reconciled silently.
    """
    live_completed = int(collector_summary.get("throughput", 0))
    trip_completed = int(accounting["completed"])
    live_waiting = float(collector_summary.get("average_waiting_time", 0.0))
    trip_waiting = float(accounting["mean_stopped_waiting_per_entered"])
    return {
        "completed_live": live_completed,
        "completed_tripinfo": trip_completed,
        "completed_agree": live_completed == trip_completed,
        "mean_waiting_live": round(live_waiting, 3),
        "mean_stopped_waiting_tripinfo": round(trip_waiting, 3),
        "waiting_difference": round(trip_waiting - live_waiting, 3),
        "note": (
            "The live figure is SUMO's accumulated waiting time sampled "
            "every second and averaged over vehicles seen. The trip-info "
            "figure is SUMO's own per-trip waitingTime averaged over "
            "vehicles that entered. They are close but not identical by "
            "construction: the live one includes vehicles still on the "
            "network at the end, the trip-info one does not."),
    }
