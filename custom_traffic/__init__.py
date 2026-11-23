"""
IntelliSignal
custom_traffic

User-defined traffic for one intersection.

    CustomSpec        what was typed, with None meaning "not said"
         |  resolver.resolve()
         v
    ResolvedWorkload  every value decided, each tagged user / derived / auto
         |  builder.build_workload()
         v
    SUMO inputs       one explicit <vehicle> per vehicle

The point of the middle step is that the preview and the simulation are the
same object. The preview is rendered from `ResolvedWorkload`, the route file
is written from the same `ResolvedWorkload`, and both controllers in a
comparison are handed that one file - so "what I was shown" and "what was
run" cannot drift apart, and a run can be reproduced from its workload id.
"""

from custom_traffic.builder import (build_workload, cleanup_workload_inputs,
                                    scenario_name)
from custom_traffic.network import (APPROACHES, LANE_SPEED_KMH,
                                    LANES_PER_APPROACH, all_lanes, lane_id,
                                    lane_label, movements_for_lane)
from custom_traffic.resolver import Issue, Resolution, example_spec, resolve
from custom_traffic.spec import (CustomSpec, LaneInput, TURNS_AS_COUNTS,
                                 TURNS_AS_PERCENT)
from custom_traffic.workload import (FROM_AUTO, FROM_DERIVED, FROM_USER,
                                     ResolvedLane, ResolvedWorkload, Vehicle,
                                     WORKLOAD_DIR)

__all__ = [
    "APPROACHES", "LANES_PER_APPROACH", "LANE_SPEED_KMH",
    "all_lanes", "lane_id", "lane_label", "movements_for_lane",
    "CustomSpec", "LaneInput", "TURNS_AS_COUNTS", "TURNS_AS_PERCENT",
    "Issue", "Resolution", "resolve", "example_spec",
    "ResolvedWorkload", "ResolvedLane", "Vehicle", "WORKLOAD_DIR",
    "FROM_USER", "FROM_AUTO", "FROM_DERIVED",
    "build_workload", "cleanup_workload_inputs", "scenario_name",
]
