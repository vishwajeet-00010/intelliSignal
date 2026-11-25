"""
IntelliSignal
utils/sumo_launch.py

Starting SUMO so that the simulation a caller talks to is its own.

`traci.start` finds a port by asking the operating system for a free one,
closing it, and handing the number to a new SUMO process. Between the close
and SUMO's own bind the port belongs to nobody, and on a machine running a
dozen simulations at once - six training runs, a validation sweep and the
test suite is an ordinary afternoon here - two processes are sometimes
handed the same number. One SUMO then fails to bind and exits, and its
client either dies with "Connection closed by SUMO" or, worse, connects to
the OTHER process's simulation and quietly measures the wrong traffic.

Three things make that go away:

  * each process draws ports from its own small window, chosen from its
    process id and placed below the operating system's dynamic range, so
    two processes are not competing for the same numbers at all;
  * a port is probed before it is used, and a failed start is retried on
    the next port instead of being fatal;
  * after connecting, the SUMO process this call launched must still be
    alive. If it is not, the connection is to somebody else's simulation
    and is dropped without being used.

A connection left behind under the same label by a run that crashed is
closed first. In a long-lived process such as the dashboard, that stale
label would otherwise refuse every later run until the server restarted.
"""

from __future__ import annotations

import itertools
import os
import socket
import subprocess
import time
from typing import Optional, Sequence

#: Ports are taken from here, below Windows' and Linux's dynamic ranges, so
#: nothing the operating system hands out on its own can land on them.
PORT_FLOOR = 20000
#: How many ports one process rotates through.
WINDOW = 16
#: A prime, so process ids - multiples of four on Windows - still spread
#: over every window instead of a quarter of them.
WINDOWS = 1753

_sequence = itertools.count()

#: Where SUMO's standard output goes unless a caller asks to see it.
QUIET = subprocess.DEVNULL


class SumoStartError(RuntimeError):
    """SUMO could not be started and connected to."""


# ---------------------------------------------------------------------------
def window_base(pid: Optional[int] = None) -> int:
    """The first port of the window a process draws from."""
    pid = os.getpid() if pid is None else int(pid)
    return PORT_FLOOR + (pid % WINDOWS) * WINDOW


def port_is_free(port: int) -> bool:
    """Can a server bind this port right now?"""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("", int(port)))
    except OSError:
        return False
    finally:
        probe.close()
    return True


def next_port() -> int:
    """A free port from this process's window, or any free port at all."""
    base = window_base()
    for _ in range(WINDOW):
        port = base + next(_sequence) % WINDOW
        if port_is_free(port):
            return port
    # Every port of the window is busy - a process holding sixteen live
    # simulations. Fall back to asking the operating system.
    from sumolib.miscutils import getFreeSocketPort
    return int(getFreeSocketPort())


# ---------------------------------------------------------------------------
def discard_connection(label: str, polite: bool = True) -> bool:
    """Drop whatever connection is registered under `label`.

    `polite` sends SUMO the close command first. It must be False when the
    connection turned out to belong to another process's simulation:
    closing that politely would end somebody else's run.
    """
    import traci.connection as registry

    connection = registry._connections.get(label)
    if connection is None:
        return False
    if polite:
        try:
            connection.close(False)
        except Exception:
            pass
    sock = getattr(connection, "_socket", None)
    if sock is not None:
        try:
            sock.close()
        except Exception:
            pass
    if polite:
        process = getattr(connection, "_process", None)
        try:
            if process is not None and process.poll() is None:
                process.kill()
        except Exception:
            pass
    for key in [key for key, value in list(registry._connections.items())
                if value is connection]:
        registry._connections.pop(key, None)
    return True


# ---------------------------------------------------------------------------
def start_sumo(command: Sequence[str], label: str, attempts: int = 4,
               connect_retries: int = 60, stdout=QUIET):
    """Start SUMO with `command` and return this process's connection to it.

    A drop-in for `traci.start(command, label=label)` followed by
    `traci.getConnection(label)`. The label also becomes TraCI's current
    connection, exactly as `traci.start` leaves it, so code written against
    the module-level functions keeps working.

    SUMO's own progress chatter ("Loading net-file ... done") is discarded
    by default; pass `stdout=None` to see it. Errors go to stderr and are
    always shown.
    """
    import traci
    from traci.exceptions import FatalTraCIError, TraCIException

    discard_connection(label)
    failure: Optional[BaseException] = None
    for attempt in range(1, max(int(attempts), 1) + 1):
        port = next_port()
        try:
            traci.start(list(command), port=port, label=label,
                        numRetries=int(connect_retries), stdout=stdout)
            connection = traci.getConnection(label)
        except (FatalTraCIError, TraCIException, OSError) as exc:
            failure = exc
            discard_connection(label, polite=False)
            time.sleep(0.05 * attempt)
            continue

        process = getattr(connection, "_process", None)
        if process is not None and process.poll() is not None:
            # Something answered on the port, but the SUMO launched here
            # has already exited - so that something is not ours.
            failure = SumoStartError(
                "port {} was answered by another simulation".format(port))
            discard_connection(label, polite=False)
            time.sleep(0.05 * attempt)
            continue
        return connection

    raise SumoStartError(
        "SUMO could not be started after {} attempts (last error: {}). "
        "Command: {}".format(attempts, failure, " ".join(map(str, command)))
    ) from failure
