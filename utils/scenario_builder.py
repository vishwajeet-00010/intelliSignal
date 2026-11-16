"""
IntelliSignal - Stage 1
scenario_builder.py

Turns a scenario configuration (YAML) into the real SUMO input files:

    config/<scenario>.yaml
            |
            +--> sumo/routes/<scenario>.rou.xml      (traffic demand)
            +--> sumo/additional/<scenario>_tls.add.xml (fixed-time signal)
            +--> sumo/config/<scenario>.sumocfg      (SUMO configuration)

Nothing about the traffic demand or the signal timing is hard-coded here:
every number comes from the YAML file. The files that are written are normal
SUMO XML files, so they can also be opened directly with sumo-gui.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from utils.config_loader import PROJECT_ROOT

SUMO_DIR = os.path.join(PROJECT_ROOT, "sumo")
NETWORK_FILE = os.path.join(SUMO_DIR, "network", "intersection.net.xml")
ROUTES_DIR = os.path.join(SUMO_DIR, "routes")
ADDITIONAL_DIR = os.path.join(SUMO_DIR, "additional")
SUMOCFG_DIR = os.path.join(SUMO_DIR, "config")

TLS_ID = "C"

# netconvert writes a default traffic-light program with programID "0" into the
# network file. SUMO does not allow a second program with the same id, so the
# IntelliSignal fixed-time program is written with programID "fixed" and made
# active at time 0 with a WAUT switch (SUMO's standard program-switch element).
# This way the scenario uses our configured timings even when the .sumocfg is
# opened directly in sumo-gui, without any Python code running.
TLS_PROGRAM_ID = "fixed"
WAUT_ID = "intelliSignal_waut"

# --------------------------------------------------------------------------
# Geometry of the intersection.
#
# For every approach: which edge the vehicle arrives on, and which edge it
# leaves on for each turning movement.  A route is simply "<in edge> <out edge>".
# --------------------------------------------------------------------------
MOVEMENTS = {
    #            incoming   right    straight   left
    "north": ("N2C", {"right": "C2W", "straight": "C2S", "left": "C2E"}),
    "south": ("S2C", {"right": "C2E", "straight": "C2N", "left": "C2W"}),
    "east":  ("E2C", {"right": "C2N", "straight": "C2W", "left": "C2S"}),
    "west":  ("W2C", {"right": "C2S", "straight": "C2E", "left": "C2N"}),
}

# --------------------------------------------------------------------------
# Traffic-light link indices produced by netconvert for this network.
#
# The junction has 16 controlled links, grouped by approach in this order:
#     index  0- 3 : NORTH approach (right, through, through, left)
#     index  4- 7 : EAST  approach (right, through, through, left)
#     index  8-11 : SOUTH approach (right, through, through, left)
#     index 12-15 : WEST  approach (right, through, through, left)
#
# "G" = protected green, "g" = permissive green (left turn must give way),
# "y" = yellow, "r" = red.  One 4-character block per approach.
# --------------------------------------------------------------------------
_GREEN_BLOCK = "GGGg"
_YELLOW_BLOCK = "yyyy"
_RED_BLOCK = "rrrr"


def _state(north: str, east: str, south: str, west: str) -> str:
    """Build the 16-character signal state string from four 4-char blocks."""
    state = north + east + south + west
    assert len(state) == 16, "Signal state must control 16 links"
    return state


NS_GREEN = _state(_GREEN_BLOCK, _RED_BLOCK, _GREEN_BLOCK, _RED_BLOCK)
NS_YELLOW = _state(_YELLOW_BLOCK, _RED_BLOCK, _YELLOW_BLOCK, _RED_BLOCK)
EW_GREEN = _state(_RED_BLOCK, _GREEN_BLOCK, _RED_BLOCK, _GREEN_BLOCK)
EW_YELLOW = _state(_RED_BLOCK, _YELLOW_BLOCK, _RED_BLOCK, _YELLOW_BLOCK)
ALL_RED = _state(_RED_BLOCK, _RED_BLOCK, _RED_BLOCK, _RED_BLOCK)

# The eight lanes that feed the intersection - queues are measured here.
INCOMING_LANES = [
    "N2C_0", "N2C_1",
    "S2C_0", "S2C_1",
    "E2C_0", "E2C_1",
    "W2C_0", "W2C_1",
]
INCOMING_EDGES = ["N2C", "S2C", "E2C", "W2C"]


def _ensure_dirs() -> None:
    for directory in (ROUTES_DIR, ADDITIONAL_DIR, SUMOCFG_DIR):
        os.makedirs(directory, exist_ok=True)


# --------------------------------------------------------------------------
# Traffic demand -> SUMO route file
# --------------------------------------------------------------------------
def build_routes(config: Dict[str, Any]) -> str:
    """Write sumo/routes/<scenario>.rou.xml and return its path."""
    _ensure_dirs()
    name = config["scenario"]["name"]
    veh = config["vehicle"]
    dem = config["demand"]
    ratios = dem["turn_ratios"]
    end = float(config["simulation"]["duration"])

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<!-- IntelliSignal Stage 1 - generated from config/{}.yaml -->".format(name),
        "<!-- Do not edit by hand: change the YAML file and re-run the scenario. -->",
        '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
        ' xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">',
        "",
        '    <vType id="car" vClass="passenger"'
        ' accel="{accel}" decel="{decel}" sigma="{sigma}"'
        ' length="{length}" minGap="{min_gap}" maxSpeed="{max_speed}"'
        ' guiShape="passenger"/>'.format(**veh),
        "",
    ]

    # One route per (approach, turning movement)
    lines.append("    <!-- Routes: one per approach and turning movement -->")
    for approach, (edge_in, outs) in MOVEMENTS.items():
        for turn, edge_out in outs.items():
            lines.append(
                '    <route id="{a}_{t}" edges="{i} {o}"/>'.format(
                    a=approach, t=turn, i=edge_in, o=edge_out
                )
            )
    lines.append("")

    # One flow per route, scaled by the demand of its approach
    lines.append("    <!-- Flows: vehicles per hour, taken from the YAML demand -->")
    for approach, (_, outs) in MOVEMENTS.items():
        approach_vph = float(dem[approach])
        for turn in ("right", "straight", "left"):
            vph = approach_vph * float(ratios[turn])
            lines.append(
                '    <flow id="f_{a}_{t}" type="car" route="{a}_{t}"'
                ' begin="0" end="{end:.0f}" vehsPerHour="{vph:.1f}"'
                ' departLane="best" departSpeed="max"/>'.format(
                    a=approach, t=turn, end=end, vph=vph
                )
            )
    lines.append("")
    lines.append("</routes>")

    path = os.path.join(ROUTES_DIR, "{}.rou.xml".format(name))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------
# Signal timings -> SUMO additional file (fixed-time controller)
# --------------------------------------------------------------------------
def build_tls(config: Dict[str, Any]) -> str:
    """Write the fixed-time traffic-light program and return its path.

    The phase order is:
        North-South green -> yellow -> (all red) ->
        East-West  green -> yellow -> (all red) -> repeat
    All durations come from the 'signal' section of the YAML file.
    """
    _ensure_dirs()
    name = config["scenario"]["name"]
    sig = config["signal"]

    phases = [
        (float(sig["ns_green"]), NS_GREEN, "North-South green"),
        (float(sig["yellow"]), NS_YELLOW, "North-South yellow"),
    ]
    if float(sig["all_red"]) > 0:
        phases.append((float(sig["all_red"]), ALL_RED, "All red"))
    phases.append((float(sig["ew_green"]), EW_GREEN, "East-West green"))
    phases.append((float(sig["yellow"]), EW_YELLOW, "East-West yellow"))
    if float(sig["all_red"]) > 0:
        phases.append((float(sig["all_red"]), ALL_RED, "All red"))

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<!-- IntelliSignal Stage 1 - fixed-time signal program -->",
        "<!-- Generated from config/{}.yaml - do not edit by hand. -->".format(name),
        '<additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
        ' xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">',
        "",
        '    <tlLogic id="{}" type="static" programID="{}" offset="0">'.format(
            TLS_ID, TLS_PROGRAM_ID),
    ]
    for duration, state, comment in phases:
        lines.append(
            '        <phase duration="{d:.0f}" state="{s}"/>  <!-- {c} -->'.format(
                d=duration, s=state, c=comment
            )
        )
    lines.append("    </tlLogic>")
    lines.append("")
    lines.append("    <!-- Activate the program above at time 0 -->")
    lines.append(
        '    <WAUT refTime="0" id="{w}" startProg="{p}">'.format(
            w=WAUT_ID, p=TLS_PROGRAM_ID))
    lines.append('        <wautSwitch time="0" to="{}"/>'.format(TLS_PROGRAM_ID))
    lines.append("    </WAUT>")
    lines.append(
        '    <wautJunction wautID="{w}" junctionID="{t}"/>'.format(
            w=WAUT_ID, t=TLS_ID))
    lines.append("")
    lines.append("</additional>")

    path = os.path.join(ADDITIONAL_DIR, "{}_tls.add.xml".format(name))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------
# SUMO configuration file
# --------------------------------------------------------------------------
def build_sumocfg(config: Dict[str, Any], route_file: str, tls_file: str) -> str:
    """Write sumo/config/<scenario>.sumocfg and return its path."""
    _ensure_dirs()
    name = config["scenario"]["name"]
    sim = config["simulation"]
    duration = float(sim["duration"])

    def rel(path: str) -> str:
        return os.path.relpath(path, SUMOCFG_DIR).replace(os.sep, "/")

    content = """<?xml version="1.0" encoding="UTF-8"?>
<!-- IntelliSignal Stage 1 - generated from config/{name}.yaml -->
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
               xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">

    <input>
        <net-file value="{net}"/>
        <route-files value="{rou}"/>
        <additional-files value="{add}"/>
    </input>

    <time>
        <begin value="0"/>
        <end value="{end:.0f}"/>
        <step-length value="{step}"/>
    </time>

    <processing>
        <!-- Remember the whole simulation when accumulating waiting time -->
        <waiting-time-memory value="{end:.0f}"/>
        <time-to-teleport value="{teleport:.0f}"/>
    </processing>

    <report>
        <no-step-log value="true"/>
        <duration-log.statistics value="true"/>
    </report>

    <random_number>
        <seed value="{seed}"/>
    </random_number>

</configuration>
""".format(
        name=name,
        net=rel(NETWORK_FILE),
        rou=rel(route_file),
        add=rel(tls_file),
        end=duration,
        step=sim["step_length"],
        teleport=float(sim.get("time_to_teleport", 300)),
        seed=int(sim["seed"]),
    )

    path = os.path.join(SUMOCFG_DIR, "{}.sumocfg".format(name))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return path


def build_scenario(config: Dict[str, Any]) -> Dict[str, str]:
    """Build every SUMO input file for a scenario. Returns the file paths."""
    if not os.path.isfile(NETWORK_FILE):
        raise FileNotFoundError(
            "Network file not found: {}\n"
            "Build it first with:  python sumo/network/build_network.py".format(NETWORK_FILE)
        )
    route_file = build_routes(config)
    tls_file = build_tls(config)
    sumocfg = build_sumocfg(config, route_file, tls_file)
    return {
        "net": NETWORK_FILE,
        "routes": route_file,
        "tls": tls_file,
        "sumocfg": sumocfg,
    }
