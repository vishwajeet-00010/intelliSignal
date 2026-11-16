"""
IntelliSignal - Stage 1
metrics.py

Collects traffic metrics from a running SUMO simulation through TraCI.

Every value produced here is read from SUMO while the simulation runs.
Nothing is estimated, assumed or hard-coded.

Metrics collected in Stage 1
----------------------------
vehicles              number of vehicles that entered the simulation
average_waiting_time  mean accumulated waiting time per vehicle  [s]
average_queue_length  mean number of halting vehicles on the 8 incoming
                      lanes, averaged over every simulation step  [vehicles]
throughput            vehicles that completed their route, i.e. that drove
                      through the intersection and left the network

Stage 3 addition: an explicit TraCI connection
----------------------------------------------
Stage 1 drives SUMO through the module-level `traci` functions, which talk to
the default connection. The Stage 2 RL environment uses a *labelled*
connection object instead, so that several simulations can never be confused
with one another.

`MetricsCollector(conn=...)` therefore accepts that connection object. The
default is the `traci` module itself, so every Stage 1 call site is unchanged
and the Stage 1 numbers are produced by exactly the same code. This is what
makes the Fixed-Time vs DQN comparison fair: both controllers are measured by
this one collector, not by two implementations that might drift apart.
"""

from __future__ import annotations

from typing import Any, Dict, List

import traci

from utils.scenario_builder import INCOMING_EDGES, INCOMING_LANES, TLS_ID


class MetricsCollector:
    """Accumulates SUMO/TraCI measurements over a whole simulation run."""

    def __init__(self, step_length: float = 1.0,
                 lanes: List[str] = None, tls_id: str = TLS_ID,
                 conn=None):
        self.step_length = float(step_length)
        self.lanes = list(lanes) if lanes else list(INCOMING_LANES)
        self.tls_id = tls_id

        # The TraCI entry point to read from: the module (Stage 1 default) or
        # a labelled connection object (the Stage 2/3 RL environment).
        self.conn = conn if conn is not None else traci

        # Per-vehicle accumulated waiting time, keyed by SUMO vehicle id.
        self._waiting_time: Dict[str, float] = {}

        self._steps = 0
        self._queue_sum = 0.0          # sum of halting vehicles over all steps
        self._max_queue = 0            # largest queue seen in any single step
        self._departed = 0             # vehicles inserted into the network
        self._arrived = 0              # vehicles that finished their route
        self._teleports = 0            # SUMO teleports (sign of heavy gridlock)
        self._sim_time = 0.0

        # Per-step record, used for the time-series CSV and extra plots.
        self.timeseries: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    def collect_step(self) -> None:
        """Read one simulation step from SUMO. Call after simulationStep()."""
        conn = self.conn
        self._steps += 1
        self._sim_time = conn.simulation.getTime()

        self._departed += conn.simulation.getDepartedNumber()
        self._arrived += conn.simulation.getArrivedNumber()
        self._teleports += conn.simulation.getStartingTeleportNumber()

        # Queue: halting vehicles (speed < 0.1 m/s) on each incoming lane.
        queue = 0
        for lane in self.lanes:
            queue += conn.lane.getLastStepHaltingNumber(lane)
        self._queue_sum += queue
        self._max_queue = max(self._max_queue, queue)

        # Waiting time: keep the latest accumulated value for every vehicle.
        for vid in conn.vehicle.getIDList():
            self._waiting_time[vid] = conn.vehicle.getAccumulatedWaitingTime(vid)

        self.timeseries.append(
            {
                "time": round(self._sim_time, 1),
                "running_vehicles": conn.vehicle.getIDCount(),
                "queue_length": queue,
                "arrived_cumulative": self._arrived,
                "signal_state": conn.trafficlight.getRedYellowGreenState(self.tls_id),
                "signal_phase": conn.trafficlight.getPhase(self.tls_id),
            }
        )

    # ------------------------------------------------------------------
    def summary(self, scenario_name: str) -> Dict[str, Any]:
        """Return the summary metrics of the run as a plain dictionary."""
        n_vehicles = len(self._waiting_time)
        avg_wait = (
            sum(self._waiting_time.values()) / n_vehicles if n_vehicles else 0.0
        )
        avg_queue = self._queue_sum / self._steps if self._steps else 0.0

        return {
            "scenario": scenario_name,
            "vehicles": n_vehicles,
            "average_waiting_time": round(avg_wait, 2),
            "average_queue_length": round(avg_queue, 2),
            "throughput": self._arrived,
            "simulation_time": round(self._sim_time, 1),
            "max_queue_length": self._max_queue,
            "vehicles_departed": self._departed,
            "vehicles_still_in_network": max(n_vehicles - self._arrived, 0),
            "teleports": self._teleports,
        }

    # ------------------------------------------------------------------
    @property
    def steps(self) -> int:
        return self._steps

    @property
    def monitored_lanes(self) -> List[str]:
        return list(self.lanes)

    @property
    def monitored_edges(self) -> List[str]:
        return list(INCOMING_EDGES)
