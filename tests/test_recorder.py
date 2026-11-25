"""
Tests for `simulation/recorder.py`.

The audit in that module is the project's second opinion: it reads the
signal states SUMO displayed and checks the timing rules against them,
sharing no code with any controller. These tests give it signal
sequences with known faults and insist it finds them - because an audit
that cannot fail is worth nothing.
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl.timing_profile import get_profile                       # noqa: E402
from simulation.recorder import (APPROACHES, EW_G, EW_Y, NS_G,  # noqa: E402
                                 NS_Y, OTHER, RED, ReplayRecorder,
                                 audit_signal, green_timeline,
                                 signal_segments, state_code)
from utils.scenario_builder import (ALL_RED, EW_GREEN, NS_GREEN,  # noqa: E402
                                    NS_YELLOW)

PROFILE = get_profile("standard")
YELLOW, CLEAR = 3.0, 2.0


def sequence(*blocks):
    """Build a per-second list of signal codes from (code, seconds) pairs."""
    out = []
    for code, seconds in blocks:
        out += [code] * int(seconds)
    return out


def good_run(cycles=3):
    blocks = []
    for index in range(cycles):
        blocks += [(NS_G, 22), (NS_Y, 3), (RED, 2),
                   (EW_G, 15), (EW_Y, 3), (RED, 2)]
    return sequence(*blocks)


def rules(codes):
    return {v["rule"] for v in audit_signal(codes, PROFILE, YELLOW,
                                            CLEAR)["violations"]}


# ---------------------------------------------------------------------------
# The states themselves
# ---------------------------------------------------------------------------
def test_the_five_states_this_junction_uses_are_recognised():
    assert state_code(NS_GREEN) == NS_G
    assert state_code(NS_YELLOW) == NS_Y
    assert state_code(EW_GREEN) == EW_G
    assert state_code(ALL_RED) == RED


def test_an_unknown_state_is_reported_rather_than_guessed():
    assert state_code("GGGGGGGGGGGGGGGG") == OTHER
    assert "known states" in rules([OTHER] * 10)


# ---------------------------------------------------------------------------
# Segments and the green timeline
# ---------------------------------------------------------------------------
def test_segments_are_the_runs_of_one_state():
    segments = signal_segments(sequence((NS_G, 15), (NS_Y, 3), (RED, 2)))
    assert [s["code"] for s in segments] == [NS_G, NS_Y, RED]
    assert [s["seconds"] for s in segments] == [15.0, 3.0, 2.0]
    assert segments[0]["start"] == 0.0 and segments[0]["end"] == 15.0
    assert segments[2]["end"] == 20.0


def test_the_timeline_has_one_row_per_green_in_order():
    timeline = green_timeline(signal_segments(good_run(2)))
    assert [row["phase"] for row in timeline] == [
        "North-South", "East-West", "North-South", "East-West"]
    assert [row["green_seconds"] for row in timeline] == [22, 15, 22, 15]


def test_a_green_cut_off_by_the_end_of_the_run_is_marked_unfinished():
    timeline = green_timeline(signal_segments(
        sequence((NS_G, 22), (NS_Y, 3), (RED, 2), (EW_G, 7))))
    assert timeline[0]["completed"] is True
    assert timeline[-1]["completed"] is False


# ---------------------------------------------------------------------------
# What a clean run looks like
# ---------------------------------------------------------------------------
def test_a_correct_run_passes_every_rule():
    report = audit_signal(good_run(), PROFILE, YELLOW, CLEAR)
    assert report["compliant"] is True
    assert report["violations"] == []
    assert report["greens_completed"] == 6
    assert report["shortest_green"] == 15.0
    assert report["longest_green"] == 22.0


def test_the_last_green_is_not_counted_as_a_short_one():
    """It was cut off by the end of the run, not ended by the controller."""
    codes = good_run(2) + sequence((NS_G, 4))
    report = audit_signal(codes, PROFILE, YELLOW, CLEAR)
    assert report["compliant"] is True
    assert report["greens_truncated"] == 1
    assert report["greens_completed"] == 4


def test_red_is_measured_as_a_driver_experiences_it():
    """From the end of this group's amber to the start of its next green."""
    report = audit_signal(good_run(2), PROFILE, YELLOW, CLEAR)
    # north-south: its amber ends, then 2 s all-red, 15 s east-west green,
    # 3 s amber, 2 s all-red = 22 s before it is green again.
    assert report["longest_red"]["NS"] == pytest.approx(22.0)
    assert report["longest_red"]["EW"] == pytest.approx(29.0)


# ---------------------------------------------------------------------------
# Every rule must be able to fail
# ---------------------------------------------------------------------------
def test_a_green_below_the_minimum_is_caught():
    codes = sequence((NS_G, 10), (NS_Y, 3), (RED, 2),
                     (EW_G, 15), (EW_Y, 3), (RED, 2), (NS_G, 15))
    assert "minimum green" in rules(codes)


def test_a_green_above_the_maximum_is_caught():
    codes = sequence((NS_G, 31), (NS_Y, 3), (RED, 2),
                     (EW_G, 15), (EW_Y, 3), (RED, 2), (NS_G, 15))
    assert "maximum green" in rules(codes)


def test_a_short_amber_is_caught():
    codes = sequence((NS_G, 20), (NS_Y, 1), (RED, 2),
                     (EW_G, 15), (EW_Y, 3), (RED, 2), (NS_G, 15))
    assert "amber length" in rules(codes)


def test_a_missing_all_red_is_caught():
    codes = sequence((NS_G, 20), (NS_Y, 3),
                     (EW_G, 15), (EW_Y, 3), (RED, 2), (NS_G, 15))
    assert "all-red after amber" in rules(codes)


def test_a_green_that_follows_its_own_green_is_caught():
    """Both directions green one after the other, with no clearance."""
    codes = sequence((NS_G, 20), (NS_Y, 3), (RED, 2), (EW_G, 15),
                     (NS_G, 15), (NS_Y, 3), (RED, 2), (EW_G, 15))
    assert rules(codes) & {"amber after green", "alternation"}


def test_an_approach_starved_of_green_is_caught():
    codes = (sequence((NS_G, 15), (NS_Y, 3), (RED, 2))
             + sequence((EW_G, 30), (EW_Y, 3), (RED, 42))
             + sequence((NS_G, 15), (NS_Y, 3), (RED, 2), (EW_G, 15)))
    found = rules(codes)
    assert "maximum red" in found
    report = audit_signal(codes, PROFILE, YELLOW, CLEAR)
    assert report["longest_red"]["NS"] > 60.0


def test_a_clean_report_states_what_it_checked():
    report = audit_signal(good_run(), PROFILE, YELLOW, CLEAR)
    assert report["profile"] == "standard"
    assert report["min_green"] == 15.0 and report["max_green"] == 30.0
    assert report["yellow"] == YELLOW and report["all_red"] == CLEAR


# ---------------------------------------------------------------------------
# The recorder itself
# ---------------------------------------------------------------------------
class FakeLane:
    def __init__(self, halting, present):
        self._halting, self._present = halting, present

    def getLastStepHaltingNumber(self, lane):
        return self._halting

    def getLastStepVehicleNumber(self, lane):
        return self._present


class FakeConnection:
    """Enough of a TraCI connection for the recorder to read a step."""

    def __init__(self, state=NS_GREEN, halting=2, present=5):
        self.lane = FakeLane(halting, present)
        self._state = state
        self._time = 0.0
        outer = self

        class Simulation:
            def getTime(self):
                return outer._time

            def getDepartedIDList(self):
                return []

            def getCollidingVehiclesNumber(self):
                return 0

            def getEmergencyStoppingVehiclesNumber(self):
                return 0

        class TrafficLight:
            def getRedYellowGreenState(self, tls):
                return outer._state

        class Vehicle:
            def subscribe(self, *args, **kwargs):
                pass

            def getAllSubscriptionResults(self):
                return {}

        self.simulation = Simulation()
        self.trafficlight = TrafficLight()
        self.vehicle = Vehicle()

    def step(self, seconds=1.0):
        self._time += seconds


def test_the_recorder_writes_one_row_per_step():
    recorder = ReplayRecorder(step_length=1.0)
    connection = FakeConnection()
    recorder.attach(connection)
    for _ in range(5):
        connection.step()
        recorder.on_step()

    replay = recorder.replay()
    assert replay["steps"] == 5
    assert replay["signal"] == [NS_G] * 5
    # An approach is two lanes, so its queue is the sum of both of them.
    for approach in APPROACHES:
        assert replay["queues"][approach] == [4] * 5
        assert replay["present"][approach] == [10] * 5


def test_the_recorder_series_adds_the_two_axis_totals():
    recorder = ReplayRecorder(step_length=1.0)
    connection = FakeConnection(halting=3)
    recorder.attach(connection)
    connection.step()
    recorder.on_step()

    row = recorder.series()[0]
    # three halted on each of eight lanes: six an approach, twelve an axis
    assert row["queue_N"] == 6 and row["queue_E"] == 6
    assert row["queue_NS"] == 12 and row["queue_EW"] == 12
    assert row["queue_total"] == 24
    assert row["signal"] == NS_G


def test_a_recorder_with_no_connection_does_nothing():
    recorder = ReplayRecorder(step_length=1.0)
    recorder.on_step()
    assert recorder.replay()["steps"] == 0
