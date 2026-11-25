"""
IntelliSignal
custom_traffic/builder.py

Writes the SUMO input files for a resolved custom workload.

It reuses the Stage 1 `build_tls` and `build_sumocfg` unchanged, so a custom
run gets exactly the same signal program format and the same simulation
settings as Moderate and Heavy. Only the route file is different, and only
because the demand is given as individual vehicles rather than as flows:

    <vType id="car_50" maxSpeed="13.89" .../>
    <route id="north_right" edges="N2C C2W"/>
    <vehicle id="N0_0000" type="car_50" route="north_right"
             depart="0.00" departLane="0" departSpeed="max"/>

`departLane` is the exact lane the user configured, and the route is one the
lane can serve - `custom_traffic/resolver.py` only ever assigns a movement to
a lane that has a connection for it. Getting that wrong does not fail
loudly: SUMO holds the vehicle and eventually teleports it, which would show
up as mysteriously bad traffic rather than as an error.

One vType per distinct desired speed. SUMO caps a vehicle at the lane's own
limit, so a desired speed above 50 km/h simply means "as fast as allowed" -
which is why the preview says the speed is desired, not actual.
"""

from __future__ import annotations

import copy
import glob
import os
from typing import Any, Dict, List

from custom_traffic.fleet import get_fleet, vtype_id as fleet_vtype_id, vtype_xml
from custom_traffic.network import route_edges, route_id
from custom_traffic.workload import ResolvedWorkload, mps_speed
from utils.config_loader import load_config
from utils.scenario_builder import (NETWORK_FILE, ROUTES_DIR,
                                    build_sumocfg, build_tls)

#: Generated custom scenarios are named like this, so they can never collide
#: with `moderate`, `heavy`, the `_rl` environment files or a sweep's files.
CUSTOM_PREFIX = "custom_"


def scenario_name(workload: ResolvedWorkload, tag: str = "") -> str:
    """The SUMO scenario name for this workload.

    The workload id is part of it, so two different workloads never write
    over each other's route file, and re-running the same workload reuses
    the same name.
    """
    return "{}{}{}".format(CUSTOM_PREFIX, workload.workload_id, tag)


# ---------------------------------------------------------------------------
def _speed_suffix(speed_kmh: float) -> str:
    return "_{:.0f}".format(round(float(speed_kmh)))


def vtype_id(speed_kmh: float) -> str:
    """Kept for the generic single-car fleet and for older callers."""
    return "car{}".format(_speed_suffix(speed_kmh))


def build_route_file(workload: ResolvedWorkload, name: str) -> str:
    """Write the explicit vehicle list and return the file path."""
    os.makedirs(ROUTES_DIR, exist_ok=True)
    veh = workload.vehicle_type

    lines: List[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<!-- IntelliSignal custom workload {} -->".format(workload.workload_id),
        "<!-- Generated from the resolved preview. One element per vehicle, so",
        "     the run and the preview cannot disagree. Do not edit by hand. -->",
        '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
        ' xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">',
        "",
        "    <!-- Vehicle types: one per (class, desired speed) in use.",
        "         `startupDelay` is the pause before a stopped vehicle moves",
        "         off, and `tau` the desired headway. Together they are what",
        "         makes a short green mostly useless: the queue has barely",
        "         begun to move before the light changes. -->",
    ]

    fleet = {v.name: v for v in get_fleet(workload.traffic_model)}
    speeds = workload.speeds_used()
    used_classes = sorted({v.vehicle_class for v in workload.vehicles})
    for class_name in used_classes:          # NOT `name`: that is the file
        vehicle = fleet.get(class_name)
        if vehicle is None:
            continue
        for speed_kmh in speeds:
            lines.append(vtype_xml(vehicle, speed_kmh=speed_kmh,
                                   suffix=_speed_suffix(speed_kmh)))

    lines += ["", "    <!-- Routes: one per approach and turning movement -->"]
    used_routes = sorted({(v.approach, v.movement) for v in workload.vehicles})
    for approach, movement in used_routes:
        lines.append('    <route id="{rid}" edges="{edges}"/>'.format(
            rid=route_id(approach, movement),
            edges=route_edges(approach, movement)))

    lines += ["", "    <!-- Vehicles, in departure order -->"]
    for vehicle in workload.vehicles:
        klass = fleet.get(vehicle.vehicle_class)
        suffix = _speed_suffix(vehicle.speed_kmh)
        type_id = (fleet_vtype_id(klass, suffix) if klass is not None
                   else "car" + suffix)
        lines.append(
            '    <vehicle id="{vid}" type="{vtype}" route="{route}"'
            ' depart="{depart:.2f}" departLane="{lane}" departSpeed="max"/>'
            .format(vid=vehicle.vehicle_id, vtype=type_id,
                    route=route_id(vehicle.approach, vehicle.movement),
                    depart=vehicle.depart, lane=vehicle.lane))

    lines += ["", "</routes>"]

    path = os.path.join(ROUTES_DIR, "{}.rou.xml".format(name))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
def workload_config(workload: ResolvedWorkload, name: str,
                    signal: Dict[str, float] = None,
                    duration: float = None) -> Dict[str, Any]:
    """A Stage 1 style config dict for this workload.

    Used for the traffic-light program and the .sumocfg, both of which are
    written by the Stage 1 builders. `signal` overrides the inherited plan,
    which is how the Hybrid controller tries a tuned plan on the same
    traffic.
    """
    config = copy.deepcopy(load_config(workload.base_scenario))
    config["scenario"]["name"] = name
    config["scenario"]["description"] = (
        "Custom workload {} - {} vehicles".format(
            workload.workload_id, workload.total_vehicles))
    config["simulation"]["duration"] = float(
        duration if duration is not None else workload.duration_s)
    config["simulation"]["step_length"] = float(workload.step_length)
    config["simulation"]["seed"] = int(workload.seed)
    config["signal"].update(workload.signal)
    if signal:
        config["signal"].update({k: float(v) for k, v in signal.items()})
    return config


def build_workload(workload: ResolvedWorkload, tag: str = "",
                   signal: Dict[str, float] = None,
                   duration: float = None) -> Dict[str, str]:
    """Write every SUMO input for this workload and return the paths."""
    if not os.path.isfile(NETWORK_FILE):
        raise FileNotFoundError(
            "Network file not found: {}\n"
            "Build it first with:  python sumo/network/build_network.py"
            .format(NETWORK_FILE))

    name = scenario_name(workload, tag)
    config = workload_config(workload, name, signal=signal, duration=duration)
    route_file = build_route_file(workload, name)
    tls_file = build_tls(config)
    sumocfg = build_sumocfg(config, route_file, tls_file)
    return {"net": NETWORK_FILE, "routes": route_file, "tls": tls_file,
            "sumocfg": sumocfg, "name": name, "config": config}


# ---------------------------------------------------------------------------
def cleanup_workload_inputs(workload: ResolvedWorkload = None,
                            tag: str = "", prefix: bool = False) -> int:
    """Delete generated SUMO inputs. Returns how many files were removed.

    Three scopes, narrowest first:

      * `workload` and `tag`  - exactly the files `build_workload` wrote
        for that tag, and nothing else. This is what every runner calls
        when it finishes.
      * `prefix=True`         - every tag of that workload that STARTS with
        `tag`. The Hybrid job uses it to sweep up all of its own runs.
      * no workload           - every custom workload's files. Repository
        cleanup only; never call it while anything is running.

    The first scope used to be a prefix match as well, which meant a runner
    tagged `_val7001_p1236` also deleted the inputs of the process tagged
    `_val7001_p12364` - while that process's SUMO was still reading them.
    """
    from utils.scenario_builder import ADDITIONAL_DIR, SUMOCFG_DIR

    if workload is None:
        stem = CUSTOM_PREFIX + "*"
    else:
        stem = glob.escape(scenario_name(workload, tag)) + ("*" if prefix else "")
    removed = 0
    for directory, pattern in ((ROUTES_DIR, "{}.rou.xml"),
                               (ADDITIONAL_DIR, "{}_tls.add.xml"),
                               (SUMOCFG_DIR, "{}.sumocfg")):
        for path in glob.glob(os.path.join(directory, pattern.format(stem))):
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
    return removed
