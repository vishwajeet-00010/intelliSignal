"""
IntelliSignal - Stage 2
rl/state.py

Defines WHAT THE AGENT SEES.

The observation is a 10-number vector. It is deliberately small and
deliberately readable: every entry can be pointed at and explained.

    index  name             meaning
    -----  ---------------  ------------------------------------------------
      0    north_queue      halting vehicles on the North approach
      1    east_queue       halting vehicles on the East approach
      2    south_queue      halting vehicles on the South approach
      3    west_queue       halting vehicles on the West approach
      4    north_waiting    summed current waiting time, North approach
      5    east_waiting     summed current waiting time, East approach
      6    south_waiting    summed current waiting time, South approach
      7    west_waiting     summed current waiting time, West approach
      8    current_phase    0.0 = North-South green, 1.0 = East-West green
      9    phase_elapsed    how long the current green has been running

Every entry is normalised into [0, 1] so the observation space is well
conditioned for a neural network later. Queue and waiting values use measured
scenario-specific scales, and the observation is clipped to [0, 1].

"Halting" is SUMO's own definition: a vehicle moving slower than 0.1 m/s.
"Current waiting time" is SUMO's per-vehicle waiting time, which resets when
the vehicle starts moving again - unlike accumulated waiting time it stays in
a bounded range, which makes it a much better input for a learning agent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from utils.scenario_builder import INCOMING_LANES

# The order used everywhere in Stage 2: North, East, South, West.
APPROACH_ORDER = ("north", "east", "south", "west")

# The two incoming lanes that make up each approach.
#   lane 0 = kerb-side lane (right turn + straight ahead)
#   lane 1 = centre lane    (left turn)
APPROACH_LANES: Dict[str, List[str]] = {
    "north": ["N2C_0", "N2C_1"],
    "east":  ["E2C_0", "E2C_1"],
    "south": ["S2C_0", "S2C_1"],
    "west":  ["W2C_0", "W2C_1"],
}

# Safety check against the Stage 1 definition - the two must describe the
# same eight lanes, or the RL environment would be measuring the wrong road.
_flat = [lane for lanes in APPROACH_LANES.values() for lane in lanes]
assert sorted(_flat) == sorted(INCOMING_LANES), (
    "rl/state.py and utils/scenario_builder.py disagree about the incoming lanes")

OBSERVATION_LABELS = (
    [a + "_queue" for a in APPROACH_ORDER]
    + [a + "_waiting" for a in APPROACH_ORDER]
    + ["current_phase", "phase_elapsed"]
)

OBSERVATION_SIZE = len(OBSERVATION_LABELS)   # 10


class StateExtractor:
    """Reads the raw traffic state from SUMO and normalises it."""

    def __init__(self, max_queue_per_approach: float,
                 max_waiting_per_approach: float,
                 max_green: float, min_green: float = 10.0,
                 phase_elapsed_scale: Optional[float] = None):
        self.max_queue = float(max_queue_per_approach)
        self.max_waiting = float(max_waiting_per_approach)
        self.max_green = float(max_green)
        self.min_green = float(min_green)
        self.phase_elapsed_scale = float(
            phase_elapsed_scale if phase_elapsed_scale is not None else
            min(self.max_green, 4.0 * self.min_green))
        if self.phase_elapsed_scale <= 0:
            raise ValueError("phase_elapsed_scale must be positive")

    # ------------------------------------------------------------------
    def raw(self, conn) -> Dict[str, Any]:
        """Read the unnormalised traffic state from a live SUMO connection.

        Returned in real units, which is what the reward and the info
        dictionary use:
            queues   - vehicles per approach
            waiting  - vehicle-seconds per approach
        """
        queues = {}
        waiting = {}
        for approach in APPROACH_ORDER:
            lanes = APPROACH_LANES[approach]
            queues[approach] = sum(
                conn.lane.getLastStepHaltingNumber(lane) for lane in lanes)
            waiting[approach] = sum(
                conn.lane.getWaitingTime(lane) for lane in lanes)
        return {"queues": queues, "waiting": waiting}

    # ------------------------------------------------------------------
    def observe(self, raw: Dict[str, Any], phase_index: int,
                green_elapsed: float) -> np.ndarray:
        """Turn a raw reading into the normalised 10-number observation."""
        values = []
        for approach in APPROACH_ORDER:
            values.append(raw["queues"][approach] / self.max_queue)
        for approach in APPROACH_ORDER:
            values.append(raw["waiting"][approach] / self.max_waiting)
        values.append(float(phase_index))                 # already 0.0 or 1.0
        values.append(green_elapsed / self.phase_elapsed_scale)

        observation = np.asarray(values, dtype=np.float32)
        # Clip so the observation is always inside the declared space, even
        # when a queue or a delay is larger than the normalisation constant.
        return np.clip(observation, 0.0, 1.0, dtype=np.float32)

    # ------------------------------------------------------------------
    @staticmethod
    def zeros() -> np.ndarray:
        """An all-zero observation, used before SUMO has been stepped."""
        return np.zeros(OBSERVATION_SIZE, dtype=np.float32)

    # ------------------------------------------------------------------
    @staticmethod
    def describe(observation: np.ndarray) -> str:
        """One readable line per observation - used by the manual demo."""
        return ", ".join(
            "{}={:.3f}".format(name, value)
            for name, value in zip(OBSERVATION_LABELS, observation))
