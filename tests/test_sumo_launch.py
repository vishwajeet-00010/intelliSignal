"""
Tests for `utils/sumo_launch.py`.

Starting SUMO used to be `traci.start`, which asks the operating system
for a free port, closes it, and hands the number to a new process. With
a dozen simulations starting at once that window is wide enough for two
of them to be given the same port, and the loser either dies or - worse -
connects to the other one's simulation and measures the wrong traffic.

These tests cover the three defences: a per-process port window, a port
that is probed before it is used, and a connection that is dropped
rather than used when the SUMO it launched is not the one that answered.
"""

import os
import socket
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import sumo_launch                                   # noqa: E402
from utils.sumo_launch import (PORT_FLOOR, WINDOW, WINDOWS,     # noqa: E402
                               SumoStartError, discard_connection,
                               next_port, port_is_free, start_sumo,
                               window_base)


# ---------------------------------------------------------------------------
# Ports
# ---------------------------------------------------------------------------
def test_a_process_draws_from_its_own_window():
    base = window_base()
    assert base == window_base(), "the window must not move during a run"
    assert PORT_FLOOR <= base < PORT_FLOOR + WINDOWS * WINDOW


def test_two_different_processes_usually_get_different_windows():
    bases = {window_base(pid) for pid in range(1000, 1000 + 40 * 4, 4)}
    # Windows hands out process ids in multiples of four; the window count
    # is prime so those still spread out instead of landing on a quarter
    # of the windows.
    assert len(bases) >= 35


def test_the_window_is_below_the_ephemeral_range():
    assert PORT_FLOOR >= 20000
    assert PORT_FLOOR + WINDOWS * WINDOW < 49152


def test_a_port_in_use_is_not_offered():
    holder = socket.socket()
    holder.bind(("", 0))
    holder.listen(1)
    taken = holder.getsockname()[1]
    try:
        assert port_is_free(taken) is False
    finally:
        holder.close()


def test_next_port_returns_something_free_from_this_window():
    port = next_port()
    assert port_is_free(port)
    base = window_base()
    assert base <= port < base + WINDOW or port >= 1024


def test_next_port_falls_back_when_the_whole_window_is_busy(monkeypatch):
    monkeypatch.setattr(sumo_launch, "port_is_free", lambda port: False)
    monkeypatch.setattr("sumolib.miscutils.getFreeSocketPort", lambda: 45678)
    assert next_port() == 45678


# ---------------------------------------------------------------------------
# Connections left behind
# ---------------------------------------------------------------------------
class FakeSocket:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeProcess:
    def __init__(self, alive=True):
        self.alive = alive
        self.killed = False

    def poll(self):
        return None if self.alive else 1

    def kill(self):
        self.killed = True
        self.alive = False


class FakeConnection:
    def __init__(self, alive=True):
        self._socket = FakeSocket()
        self._process = FakeProcess(alive)
        self.closed_politely = False

    def close(self, wait=True):
        self.closed_politely = True


def test_discarding_an_unknown_label_is_harmless():
    assert discard_connection("no_such_label_here") is False


def test_a_stale_connection_is_closed_and_forgotten():
    import traci.connection as registry

    connection = FakeConnection()
    registry._connections["pytest_stale"] = connection
    try:
        assert discard_connection("pytest_stale") is True
        assert connection.closed_politely is True
        assert connection._socket.closed is True
        assert connection._process.killed is True
        assert "pytest_stale" not in registry._connections
    finally:
        registry._connections.pop("pytest_stale", None)


def test_a_connection_that_is_not_ours_is_dropped_without_closing_it():
    """Closing it politely would end somebody else's simulation."""
    import traci.connection as registry

    connection = FakeConnection()
    registry._connections["pytest_other"] = connection
    try:
        assert discard_connection("pytest_other", polite=False) is True
        assert connection.closed_politely is False
        assert connection._process.killed is False
        assert connection._socket.closed is True
        assert "pytest_other" not in registry._connections
    finally:
        registry._connections.pop("pytest_other", None)


# ---------------------------------------------------------------------------
# Starting
# ---------------------------------------------------------------------------
def test_a_start_that_keeps_failing_says_so_and_names_the_command(monkeypatch):
    from traci.exceptions import FatalTraCIError

    attempts = []

    def refuse(command, port=None, label=None, numRetries=0, stdout=None):
        attempts.append(port)
        raise FatalTraCIError("no")

    monkeypatch.setattr("traci.start", refuse)
    with pytest.raises(SumoStartError) as error:
        start_sumo(["sumo", "-c", "nothing.sumocfg"], label="pytest_fail",
                   attempts=3)
    assert len(attempts) == 3, "every attempt must be tried"
    assert len(set(attempts)) == 3, "and each on a different port"
    assert "nothing.sumocfg" in str(error.value)


def test_a_transient_failure_is_retried_and_then_succeeds(monkeypatch):
    from traci.exceptions import FatalTraCIError

    state = {"calls": 0}
    connection = FakeConnection()

    def flaky(command, port=None, label=None, numRetries=0, stdout=None):
        state["calls"] += 1
        if state["calls"] == 1:
            raise FatalTraCIError("port taken")

    monkeypatch.setattr("traci.start", flaky)
    monkeypatch.setattr("traci.getConnection", lambda label: connection)
    assert start_sumo(["sumo"], label="pytest_flaky") is connection
    assert state["calls"] == 2


def test_a_port_answered_by_somebody_elses_simulation_is_refused(monkeypatch):
    """The SUMO this call launched is already dead, so the simulation on
    the other end of the socket belongs to another process."""
    dead = FakeConnection(alive=False)

    monkeypatch.setattr(
        "traci.start",
        lambda command, port=None, label=None, numRetries=0, stdout=None: None)
    monkeypatch.setattr("traci.getConnection", lambda label: dead)
    with pytest.raises(SumoStartError) as error:
        start_sumo(["sumo"], label="pytest_hijack", attempts=2)
    assert "another simulation" in str(error.value)
    assert dead.closed_politely is False, \
        "it is not ours to close"


def test_a_label_left_behind_by_a_crashed_run_is_cleared_first(monkeypatch):
    import traci.connection as registry

    stale = FakeConnection()
    fresh = FakeConnection()
    registry._connections["pytest_reuse"] = stale
    monkeypatch.setattr(
        "traci.start",
        lambda command, port=None, label=None, numRetries=0, stdout=None: None)
    monkeypatch.setattr("traci.getConnection", lambda label: fresh)
    try:
        assert start_sumo(["sumo"], label="pytest_reuse") is fresh
        assert stale.closed_politely is True
    finally:
        registry._connections.pop("pytest_reuse", None)


def test_sumo_chatter_is_discarded_unless_it_is_asked_for():
    import subprocess

    assert sumo_launch.QUIET == subprocess.DEVNULL


# ---------------------------------------------------------------------------
# What this module exists for
# ---------------------------------------------------------------------------
def test_a_port_clash_is_absorbed_rather_than_ending_the_run():
    """The failure this module was written for, reproduced.

    Under a parallel sweep SUMO occasionally prints "Address already in
    use" and exits: two processes were handed the same free port between
    the probe and the bind. Before the retry, that killed whichever
    client was asking - and in a 1,700-run evaluation that meant losing
    the sweep an hour in.
    """
    from traci.exceptions import FatalTraCIError

    attempts = []
    connection = FakeConnection()

    def clash_once(command, port=None, label=None, numRetries=0, stdout=None):
        attempts.append(port)
        if len(attempts) == 1:
            raise FatalTraCIError(
                "tcpip::Socket::accept() Unable to create listening socket: "
                "Address already in use")

    import traci

    original_start = traci.start
    original_get = traci.getConnection
    traci.start = clash_once
    traci.getConnection = lambda label: connection
    try:
        assert start_sumo(["sumo"], label="pytest_clash") is connection
    finally:
        traci.start = original_start
        traci.getConnection = original_get
    assert len(attempts) == 2, "it must try again"
    assert attempts[0] != attempts[1], "and on a different port"
