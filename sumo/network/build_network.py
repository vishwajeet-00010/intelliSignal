"""
IntelliSignal - Stage 1
build_network.py

Rebuilds sumo/network/intersection.net.xml from the three SUMO source files
(nodes, edges, connections) using netconvert.

The generated network file is committed with the project, so this script only
has to be run when the intersection layout itself is changed.

Usage
-----
    python sumo/network/build_network.py
"""

from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def netconvert_binary() -> str:
    """Locate netconvert through sumolib (SUMO_HOME or the sumo pip package)."""
    if "SUMO_HOME" in os.environ:
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        if os.path.isdir(tools) and tools not in sys.path:
            sys.path.append(tools)
    try:
        import sumolib
    except ImportError:
        raise SystemExit(
            "sumolib is not available. Set SUMO_HOME or run:\n"
            "    pip install eclipse-sumo traci sumolib")
    return sumolib.checkBinary("netconvert")


def main() -> int:
    output = os.path.join(HERE, "intersection.net.xml")
    cmd = [
        netconvert_binary(),
        "--node-files", os.path.join(HERE, "intersection.nod.xml"),
        "--edge-files", os.path.join(HERE, "intersection.edg.xml"),
        "--connection-files", os.path.join(HERE, "intersection.con.xml"),
        "--output-file", output,
        "--no-turnarounds", "true",
        "--tls.default-type", "static",
    ]
    print("Running netconvert ...")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("netconvert failed.", file=sys.stderr)
        return result.returncode
    print("Network written to:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
