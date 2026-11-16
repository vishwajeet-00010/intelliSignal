"""
IntelliSignal - Stage 1
capture_screenshots.py

Captures real screenshots of the running SUMO simulation for the progress
presentation.

Every image is produced by sumo-gui itself, through the TraCI gui.screenshot
command, while the scenario is running. The pictures therefore show actual
simulation states - nothing is drawn, edited or faked. The only post-processing
is cropping the image to the part of the picture that contains the road
network, which needs Pillow; without Pillow the full screenshots are kept.

Images written to presentation/screenshots/:
    01_intersection_network.png   the four-way intersection network
    02_vehicles_approaching.png   vehicles driving towards the intersection
    03_vehicles_waiting_red.png   a queue held at a red signal
    04_vehicles_moving_green.png  the same approach discharging on green
    05_moderate_traffic.png       moderate traffic scenario
    06_heavy_traffic.png          heavy traffic scenario

Usage
-----
    python presentation/capture_screenshots.py
    python presentation/capture_screenshots.py --window-size 1920,1200

On a machine without a display, run it inside a virtual display:
    xvfb-run -a python presentation/capture_screenshots.py
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from simulation.run_simulation import prepare_sumo_imports, sumo_binary  # noqa: E402
from utils.config_loader import load_config  # noqa: E402
from utils.scenario_builder import TLS_ID, build_scenario  # noqa: E402

SHOT_DIR = os.path.join(PROJECT_ROOT, "presentation", "screenshots")
VIEW = "View #0"

# Phase numbers of the fixed-time program (see utils/scenario_builder.py)
PHASE_NS_GREEN = 0
PHASE_EW_GREEN = 2

NS_LANES = ["N2C_0", "N2C_1", "S2C_0", "S2C_1"]

# The road cross of the network spans this square, in metres.
NETWORK_SPAN = 400.0


# --------------------------------------------------------------------------
def _raw_screenshot(traci, path: str) -> bool:
    """Ask sumo-gui for a screenshot; SUMO writes it on the next step."""
    traci.gui.screenshot(VIEW, path)
    traci.simulationStep()
    for _ in range(30):
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            return True
        time.sleep(0.1)
    return os.path.isfile(path)


class ViewFitter:
    """Finds where the network is drawn, so pictures can be framed properly.

    sumo-gui is asked to show a boundary that is comfortably larger than the
    network. One reference screenshot is then measured to find, in pixels,
    where the road network sits and how many pixels one metre is worth.
    Every later screenshot is cropped using those measurements, either to the
    whole network or to a square around the intersection.

    Measuring needs Pillow. Without Pillow the uncropped screenshots are kept,
    which is still perfectly usable.
    """

    def __init__(self, scratch_dir: str):
        self.scratch = scratch_dir
        self.enabled = False
        self.bbox = None            # (left, top, right, bottom) of the network
        self.center = None          # pixel position of the intersection
        self.px_per_m = None
        self.image_size = None

    # ------------------------------------------------------------------
    # How much of the shorter image side the network should fill.
    TARGET_FILL = 0.85

    def measure(self, traci, half: float = 260.0, attempts: int = 6) -> bool:
        try:
            from PIL import Image  # noqa: F401
        except ImportError:
            print("  (Pillow not installed - screenshots will not be cropped)")
            return False

        probe = os.path.join(self.scratch, "calibration.png")
        for attempt in range(attempts):
            traci.gui.setBoundary(VIEW, -half, -half, half, half)
            # Give sumo-gui a couple of steps to redraw with the new view
            # before the reference screenshot is taken.
            traci.simulationStep()
            traci.simulationStep()
            if os.path.exists(probe):
                os.remove(probe)
            if not _raw_screenshot(traci, probe):
                return False

            bbox, size = self._content_bbox(probe)
            if bbox is None:
                half *= 1.6          # nothing visible - zoom out
                continue

            left, top, right, bottom = bbox
            width, height = size
            box_w = right - left
            box_h = bottom - top
            # The network is a square cross, so a bounding box that is far
            # from square means part of it is outside the picture.
            cut_off = (left <= 4 or top <= 4
                       or right >= width - 4 or bottom >= height - 54
                       or min(box_w, box_h) < 0.85 * max(box_w, box_h))
            last_try = attempt == attempts - 1

            if cut_off and not last_try:
                half *= 1.6          # network is cut off - zoom out
                continue

            # If the network is drawn much smaller than the window, zoom in
            # so the screenshots use the available resolution.
            fill = max(box_w, box_h) / min(width, height)
            if not cut_off and fill < 0.70 and not last_try:
                half *= fill / self.TARGET_FILL
                continue

            self.bbox = bbox
            self.image_size = size
            self.center = ((left + right) / 2.0, (top + bottom) / 2.0)
            self.px_per_m = max(box_w, box_h) / NETWORK_SPAN
            self.enabled = True
            print("  view measured: image {}x{} px, {:.2f} px per metre, "
                  "intersection at ({:.0f}, {:.0f})".format(
                      width, height, self.px_per_m, *self.center))
            return True
        return False

    # ------------------------------------------------------------------
    @staticmethod
    def _content_bbox(path: str):
        """Bounding box of everything that is not background in `path`."""
        from PIL import Image

        with Image.open(path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            # The scale bar is drawn in the bottom-left corner - ignore it.
            usable_height = max(height - 50, 1)
            pixels = rgb.load()
            background = pixels[1, 1]
            xs, ys = [], []
            for y in range(0, usable_height, 2):
                for x in range(0, width, 2):
                    if pixels[x, y] != background:
                        xs.append(x)
                        ys.append(y)
        if not xs:
            return None, (width, height)
        return (min(xs), min(ys), max(xs), max(ys)), (width, height)

    # ------------------------------------------------------------------
    def save(self, traci, filename: str, world_half: float = None) -> str:
        """Take a screenshot and crop it. `world_half` = metres around the
        intersection; None means "show the whole network"."""
        os.makedirs(SHOT_DIR, exist_ok=True)
        target = os.path.join(SHOT_DIR, filename)

        if not self.enabled:
            return target if _raw_screenshot(traci, target) else ""

        raw = os.path.join(self.scratch, "raw.png")
        if os.path.exists(raw):
            os.remove(raw)
        if not _raw_screenshot(traci, raw):
            return ""

        from PIL import Image

        with Image.open(raw) as image:
            width, height = image.size
            if world_half is None:
                left, top, right, bottom = self.bbox
                margin = int(0.04 * max(right - left, bottom - top))
                box = (left - margin, top - margin, right + margin, bottom + margin)
            else:
                half_px = world_half * self.px_per_m
                cx, cy = self.center
                box = (cx - half_px, cy - half_px, cx + half_px, cy + half_px)
            box = (max(int(box[0]), 0), max(int(box[1]), 0),
                   min(int(box[2]), width), min(int(box[3]), height))
            image.crop(box).save(target)
        os.remove(raw)
        return target


# --------------------------------------------------------------------------
def _ns_queue(traci) -> int:
    return sum(traci.lane.getLastStepHaltingNumber(lane) for lane in NS_LANES)


def _run_and_capture(scenario: str, shots: list, window_size: str,
                     max_steps: int = 2500) -> list:
    """Open sumo-gui for `scenario` and take the requested screenshots.

    `shots` is a list of (filename, condition, world_half). `condition` is a
    function of the traci module that returns True at the right moment;
    `world_half` is the half-width in metres of the area to show, or None for
    the whole network.
    """
    prepare_sumo_imports()
    import traci

    config = load_config(scenario)
    files = build_scenario(config)

    cmd = [
        sumo_binary(True),
        "-c", files["sumocfg"],
        "--start", "--quit-on-end",
        "--window-size", window_size,
        "--delay", "0",
    ]

    os.makedirs(SHOT_DIR, exist_ok=True)
    scratch = tempfile.mkdtemp(prefix="intellisignal_shots_")
    fitter = ViewFitter(scratch)
    created = []

    traci.start(cmd, label="shots_{}".format(scenario))
    try:
        traci.simulationStep()
        traci.gui.setSchema(VIEW, "real world")
        fitter.measure(traci)

        pending = list(shots)
        step = 0
        while pending and step < max_steps:
            filename, condition, world_half = pending[0]
            if condition(traci):
                path = fitter.save(traci, filename, world_half)
                if path and os.path.isfile(path):
                    created.append(path)
                    print("  captured {:<32} at t = {:.0f}s".format(
                        filename, traci.simulation.getTime()))
                else:
                    print("  FAILED   {}".format(filename))
                pending.pop(0)
            else:
                traci.simulationStep()
            step += 1

        for filename, _, _ in pending:
            print("  NOT captured (condition never met): {}".format(filename))
    finally:
        try:
            traci.close()
        except Exception:
            pass
        for name in ("calibration.png", "raw.png"):
            try:
                os.remove(os.path.join(scratch, name))
            except OSError:
                pass
        try:
            os.rmdir(scratch)
        except OSError:
            pass
    return created


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Capture real SUMO screenshots for the Stage 1 presentation.")
    parser.add_argument("--window-size", default="1600,1000",
                        help="sumo-gui window size, e.g. 1600,1000")
    args = parser.parse_args(argv)

    print("Capturing screenshots from the live SUMO simulation ...")

    print("\n[moderate traffic]")
    moderate_shots = [
        # the network itself, right at the start of the run
        ("01_intersection_network.png",
         lambda t: t.simulation.getTime() >= 2, None),
        # first vehicles on their way to the junction
        ("02_vehicles_approaching.png",
         lambda t: t.vehicle.getIDCount() >= 8, None),
        # North-South queue held at red (East-West has green)
        ("03_vehicles_waiting_red.png",
         lambda t: t.trafficlight.getPhase(TLS_ID) == PHASE_EW_GREEN
         and _ns_queue(t) >= 4, 110.0),
        # the same approach discharging on green
        ("04_vehicles_moving_green.png",
         lambda t: t.trafficlight.getPhase(TLS_ID) == PHASE_NS_GREEN
         and t.simulation.getTime() > 200, 110.0),
        # a general view of the moderate scenario
        ("05_moderate_traffic.png",
         lambda t: t.simulation.getTime() >= 600, None),
    ]
    created = _run_and_capture("moderate", moderate_shots, args.window_size)

    print("\n[heavy traffic]")
    heavy_shots = [
        ("06_heavy_traffic.png",
         lambda t: t.simulation.getTime() >= 900, None),
    ]
    created += _run_and_capture("heavy", heavy_shots, args.window_size,
                                max_steps=1500)

    print("\n{} screenshot(s) saved in presentation/screenshots/".format(len(created)))
    return 0 if len(created) == 6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
