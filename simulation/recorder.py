"""
IntelliSignal
simulation/recorder.py

What the junction looked like, second by second - and an audit of the
signal that does not take the controller's word for anything.

Two things live here.

`ReplayRecorder` rides along with a run and writes down, once a second,
where every vehicle is, how many are halted on each approach, and what the
signal is showing. The dashboard replays that on a drawing of the junction,
so what is on screen is the simulation that was measured rather than a
cartoon of it.

`audit_signal` then reads the recorded signal states - what SUMO actually
displayed - and checks the timing rules against them: every completed green
inside the profile's limits, an amber and an all-red of the right length
after every green, and no approach held on red for longer than the profile
allows. A controller reports its own compliance too (`timing_compliance`),
but a controller with a bug would report its own bug as compliant. This
audit shares no code with any controller.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.scenario_builder import (ALL_RED, EW_GREEN, EW_YELLOW, NS_GREEN,
                                    NS_YELLOW, TLS_ID)

REPLAY_VERSION = 1

#: Approach edges, in the order every per-approach list uses.
APPROACHES = ("N", "S", "E", "W")
APPROACH_LANES = {
    "N": ("N2C_0", "N2C_1"), "S": ("S2C_0", "S2C_1"),
    "E": ("E2C_0", "E2C_1"), "W": ("W2C_0", "W2C_1"),
}

#: Short codes for the five things the signal can show.
NS_G, NS_Y, EW_G, EW_Y, RED, OTHER = "NG", "NY", "EG", "EY", "RR", "??"
_STATE_CODE = {NS_GREEN: NS_G, NS_YELLOW: NS_Y, EW_GREEN: EW_G,
               EW_YELLOW: EW_Y, ALL_RED: RED}
STATE_LABEL = {
    NS_G: "North-South green", NS_Y: "North-South amber",
    EW_G: "East-West green", EW_Y: "East-West amber",
    RED: "All red", OTHER: "Unrecognised state",
}
GROUP_OF = {NS_G: "NS", NS_Y: "NS", EW_G: "EW", EW_Y: "EW"}

#: Positions are stored as whole decimetres and headings as whole degrees.
POSITION_SCALE = 10


def state_code(state: str) -> str:
    """The short code for a 16-character SUMO signal state."""
    return _STATE_CODE.get(str(state), OTHER)


# ===========================================================================
class ReplayRecorder:
    """Records vehicles, queues and the signal once per simulation step.

    Used as a harness observer: `attach(connection)` once SUMO is up, then
    `on_step()` after every `simulationStep()`.
    """

    def __init__(self, workload=None, step_length: float = 1.0,
                 record_vehicles: bool = True):
        self.step_length = float(
            workload.step_length if workload is not None else step_length)
        self.record_vehicles = bool(record_vehicles)
        self.conn = None

        #: vehicle id -> class name, known up front from the workload
        self._class_of: Dict[str, str] = {}
        if workload is not None:
            self._class_of = {v.vehicle_id: v.vehicle_class
                              for v in workload.vehicles}
        self.classes: List[str] = sorted(set(self._class_of.values())) or ["car"]
        self._class_index = {name: i for i, name in enumerate(self.classes)}

        self._index: Dict[str, int] = {}       # vehicle id -> small integer
        self.vehicle_class: List[int] = []     # by that integer
        self.frames: List[List[int]] = []      # flat [i, x, y, heading, ...]
        self.times: List[float] = []
        self.signal: List[str] = []            # one short code per step
        self.queues: Dict[str, List[int]] = {a: [] for a in APPROACHES}
        self.present: Dict[str, List[int]] = {a: [] for a in APPROACHES}
        self.collisions = 0
        self.emergency_stops = 0

    # ------------------------------------------------------------------
    def attach(self, connection) -> None:
        self.conn = connection

    # ------------------------------------------------------------------
    def on_step(self) -> None:
        import traci.constants as tc

        conn = self.conn
        if conn is None:
            return
        self.times.append(float(conn.simulation.getTime()))
        self.signal.append(state_code(
            conn.trafficlight.getRedYellowGreenState(TLS_ID)))

        for approach in APPROACHES:
            halted = 0
            present = 0
            for lane in APPROACH_LANES[approach]:
                halted += int(conn.lane.getLastStepHaltingNumber(lane))
                present += int(conn.lane.getLastStepVehicleNumber(lane))
            self.queues[approach].append(halted)
            self.present[approach].append(present)

        self.collisions += int(conn.simulation.getCollidingVehiclesNumber())
        self.emergency_stops += int(
            conn.simulation.getEmergencyStoppingVehiclesNumber())

        if not self.record_vehicles:
            return
        # Vehicles are followed by subscription: one round-trip per step
        # for all of them, instead of two calls per vehicle per step.
        for vehicle_id in conn.simulation.getDepartedIDList():
            conn.vehicle.subscribe(vehicle_id, (tc.VAR_POSITION, tc.VAR_ANGLE))
        frame: List[int] = []
        for vehicle_id, values in conn.vehicle.getAllSubscriptionResults().items():
            position = values.get(tc.VAR_POSITION)
            if position is None:
                continue
            index = self._index.get(vehicle_id)
            if index is None:
                index = len(self._index)
                self._index[vehicle_id] = index
                self.vehicle_class.append(self._class_index.get(
                    self._class_of.get(vehicle_id, self.classes[0]), 0))
            frame += [index,
                      int(round(position[0] * POSITION_SCALE)),
                      int(round(position[1] * POSITION_SCALE)),
                      int(round(float(values.get(tc.VAR_ANGLE, 0.0)))) % 360]
        self.frames.append(frame)

    # ------------------------------------------------------------------
    def replay(self) -> Dict[str, Any]:
        """Everything the junction replay needs, as plain JSON types."""
        return {
            "version": REPLAY_VERSION,
            "step_length": self.step_length,
            "steps": len(self.times),
            "position_scale": POSITION_SCALE,
            "classes": list(self.classes),
            "vehicle_class": list(self.vehicle_class),
            "frames": self.frames if self.record_vehicles else [],
            "signal": list(self.signal),
            "queues": {a: list(v) for a, v in self.queues.items()},
            "present": {a: list(v) for a, v in self.present.items()},
        }

    def series(self) -> List[Dict[str, Any]]:
        """Per-second rows for charts: time, signal and per-approach queues."""
        rows = []
        for index, moment in enumerate(self.times):
            row = {"time": moment, "signal": self.signal[index]}
            total = 0
            for approach in APPROACHES:
                row["queue_" + approach] = self.queues[approach][index]
                total += self.queues[approach][index]
            row["queue_NS"] = row["queue_N"] + row["queue_S"]
            row["queue_EW"] = row["queue_E"] + row["queue_W"]
            row["queue_total"] = total
            rows.append(row)
        return rows


# ===========================================================================
def signal_segments(codes: Sequence[str], step_length: float = 1.0,
                    start_time: float = 0.0) -> List[Dict[str, Any]]:
    """Run-length encode the per-step signal codes into timed segments."""
    segments: List[Dict[str, Any]] = []
    for index, code in enumerate(codes):
        begin = start_time + index * step_length
        if segments and segments[-1]["code"] == code:
            segments[-1]["end"] = begin + step_length
            segments[-1]["seconds"] += step_length
            continue
        segments.append({"code": code, "label": STATE_LABEL.get(code, code),
                         "start": begin, "end": begin + step_length,
                         "seconds": step_length})
    for segment in segments:
        segment["seconds"] = round(segment["seconds"], 3)
        segment["start"] = round(segment["start"], 3)
        segment["end"] = round(segment["end"], 3)
    return segments


def green_timeline(segments: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One row per green that was shown, in order.

    `completed` is False for a green that the end of the run cut off: it
    was never ended by the controller, so its length says nothing about
    whether the controller obeys the minimum.
    """
    greens = []
    for index, segment in enumerate(segments):
        if segment["code"] not in (NS_G, EW_G):
            continue
        greens.append({
            "group": GROUP_OF[segment["code"]],
            "phase": ("North-South" if segment["code"] == NS_G
                      else "East-West"),
            "start": segment["start"], "end": segment["end"],
            "green_seconds": segment["seconds"],
            "completed": index < len(segments) - 1,
        })
    return greens


def audit_signal(codes: Sequence[str], profile, yellow: float,
                 all_red: float, step_length: float = 1.0) -> Dict[str, Any]:
    """Check the timing rules against what the signal actually showed.

    Returns the findings as a dict with a `violations` list; an empty list
    means every rule held. Nothing here calls into a controller.

    Red is measured as a driver sees it: from the moment an approach's
    amber ends to the moment its next green begins. That includes the
    all-red after its own amber, so it is `all_red` seconds longer than
    the controller's internal bookkeeping (the next green plus one
    clearance) - the stricter of the two readings.
    """
    segments = signal_segments(codes, step_length)
    tolerance = 1e-6
    violations: List[Dict[str, Any]] = []

    def violation(rule: str, at, detail: str) -> None:
        violations.append({"rule": rule, "at": at, "detail": detail})

    completed_greens: List[float] = []
    truncated = 0
    last_green = {"NS": None, "EW": None}     # last COMPLETED green length
    red_since = {"NS": 0.0, "EW": 0.0}        # when the current red began
    longest_red = {"NS": 0.0, "EW": 0.0}
    green_code = {"NS": NS_G, "EW": EW_G}
    amber_code = {"NS": NS_Y, "EW": EW_Y}
    other_of = {"NS": "EW", "EW": "NS"}

    unknown = sum(1 for code in codes if code == OTHER)
    if unknown:
        violation("known states", None,
                  "{} second(s) showed a signal state that is none of the "
                  "five this junction uses.".format(unknown))

    for index, segment in enumerate(segments):
        code = segment["code"]
        is_last = index == len(segments) - 1
        following = None if is_last else segments[index + 1]
        previous = segments[index - 1] if index else None

        if code in (NS_G, EW_G):
            group = GROUP_OF[code]
            red_seconds = segment["start"] - red_since[group]
            longest_red[group] = max(longest_red[group], red_seconds)
            before = last_green[group]
            limit = (profile.max_red_seconds(before) if before is not None
                     else profile.initial_max_red_seconds())
            if red_seconds > limit + tolerance:
                violation("maximum red", segment["start"],
                          "{} was red for {:g} s; the limit after its "
                          "previous green was {:g} s.".format(
                              group, red_seconds, limit))
            if is_last:
                # Cut off by the end of the run, not ended by the
                # controller, so its length is not evidence of anything.
                truncated += 1
                continue
            seconds = segment["seconds"]
            completed_greens.append(seconds)
            last_green[group] = seconds
            if seconds < profile.min_green - tolerance:
                violation("minimum green", segment["start"],
                          "{} green ran {:g} s; the minimum is {:g} s."
                          .format(group, seconds, profile.min_green))
            if seconds > profile.max_green + tolerance:
                violation("maximum green", segment["start"],
                          "{} green ran {:g} s; the maximum is {:g} s."
                          .format(group, seconds, profile.max_green))
            if following["code"] != amber_code[group]:
                red_since[group] = segment["end"]
                violation("amber after green", segment["end"],
                          "{} green was followed by {} instead of its "
                          "amber.".format(group, following["label"]))

        elif code in (NS_Y, EW_Y):
            group = GROUP_OF[code]
            red_since[group] = segment["end"]
            if previous is not None and previous["code"] != green_code[group]:
                violation("amber after green", segment["start"],
                          "{} amber appeared without its green before it."
                          .format(group))
            if is_last:
                continue
            if abs(segment["seconds"] - yellow) > tolerance:
                violation("amber length", segment["start"],
                          "{} amber ran {:g} s; it should be {:g} s.".format(
                              group, segment["seconds"], yellow))
            if all_red > tolerance:
                if following["code"] != RED:
                    violation("all-red after amber", following["start"],
                              "{} amber was followed by {} instead of "
                              "all-red.".format(group, following["label"]))
            elif following["code"] != green_code[other_of[group]]:
                violation("alternation", following["start"],
                          "After {} the signal showed {}.".format(
                              group, following["label"]))

        elif code == RED:
            if previous is None or is_last:
                continue
            if previous["code"] not in (NS_Y, EW_Y):
                violation("all-red after amber", segment["start"],
                          "All-red appeared after {}.".format(
                              previous["label"]))
                continue
            if abs(segment["seconds"] - all_red) > tolerance:
                violation("all-red length", segment["start"],
                          "All-red ran {:g} s; it should be {:g} s.".format(
                              segment["seconds"], all_red))
            expected = green_code[other_of[GROUP_OF[previous["code"]]]]
            if following["code"] != expected:
                violation("alternation", following["start"],
                          "After {} and all-red the signal showed {}."
                          .format(previous["label"], following["label"]))

    # An approach still on red when the run ends has been red since then.
    end = segments[-1]["end"] if segments else 0.0
    showing = segments[-1]["code"] if segments else RED
    for group in ("NS", "EW"):
        if showing in (green_code[group], amber_code[group]):
            continue
        longest_red[group] = max(longest_red[group], end - red_since[group])

    return {
        "profile": profile.name,
        "min_green": profile.min_green, "max_green": profile.max_green,
        "yellow": float(yellow), "all_red": float(all_red),
        "greens_completed": len(completed_greens),
        "greens_truncated": truncated,
        "shortest_green": min(completed_greens) if completed_greens else None,
        "longest_green": max(completed_greens) if completed_greens else None,
        "mean_green": (round(sum(completed_greens) / len(completed_greens), 3)
                       if completed_greens else None),
        "longest_red": {g: round(v, 3) for g, v in longest_red.items()},
        "violations": violations,
        "compliant": not violations,
        "segments": len(segments),
    }
