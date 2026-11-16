"""
IntelliSignal - Stage 1
tests/test_simulation.py

Tests for the Stage 1 simulation foundation. They check the things the
project actually depends on:

  * the scenario configuration files load and are validated
  * invalid configurations are rejected
  * the SUMO network file exists and contains the traffic light
  * the route file and the traffic-light program are generated from the YAML
  * SUMO and TraCI are installed and a connection can be opened
  * a short simulation runs, produces metrics and shuts SUMO down
  * a CSV file with the expected columns is written
  * the plotting code turns the CSV files into image files

Run with:
    pytest -v tests/
"""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import config_loader                                    # noqa: E402
from utils.config_loader import ConfigError, cycle_time, load_config  # noqa: E402
from utils.scenario_builder import (INCOMING_LANES, NETWORK_FILE,   # noqa: E402
                                    TLS_ID, TLS_PROGRAM_ID, build_scenario)

SCENARIOS = ["moderate", "heavy"]

# A short run is enough to prove the pipeline works end to end.
SMOKE_DURATION = 120.0


# ---------------------------------------------------------------- configuration
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_config_loads(scenario):
    config = load_config(scenario)
    assert config["scenario"]["name"] == scenario
    assert config["simulation"]["duration"] > 0
    assert config["signal"]["ns_green"] > 0
    assert cycle_time(config) > 0


def test_both_scenarios_are_available():
    available = config_loader.available_scenarios()
    for scenario in SCENARIOS:
        assert scenario in available


def test_unknown_scenario_is_rejected():
    with pytest.raises(ConfigError):
        load_config("this_scenario_does_not_exist")


def test_invalid_config_is_rejected():
    broken = {
        "scenario": {"name": "broken"},
        "simulation": {"duration": 100, "step_length": 1.0, "seed": 1},
        "signal": {"ns_green": 30, "ew_green": 30, "yellow": 4, "all_red": 0},
        # turn ratios do not add up to 1.0
        "demand": {"north": 1, "south": 1, "east": 1, "west": 1,
                   "turn_ratios": {"right": 0.5, "straight": 0.5, "left": 0.5}},
        "vehicle": {"max_speed": 10, "accel": 2, "decel": 4,
                    "length": 5, "min_gap": 2, "sigma": 0.5},
    }
    with pytest.raises(ConfigError):
        config_loader.validate_config(broken)


def test_heavy_demand_is_larger_than_moderate():
    moderate = config_loader.total_demand(load_config("moderate"))
    heavy = config_loader.total_demand(load_config("heavy"))
    assert heavy > moderate


def test_both_scenarios_use_the_same_signal_plan():
    """The two scenarios must differ only in demand, so that they compare."""
    moderate = load_config("moderate")["signal"]
    heavy = load_config("heavy")["signal"]
    assert moderate == heavy


# ---------------------------------------------------------------- SUMO inputs
def test_network_file_exists():
    assert os.path.isfile(NETWORK_FILE), (
        "Network missing - build it with: python sumo/network/build_network.py")
    root = ET.parse(NETWORK_FILE).getroot()
    junction_ids = [j.get("id") for j in root.findall("junction")]
    assert TLS_ID in junction_ids


def test_network_has_all_four_approaches():
    root = ET.parse(NETWORK_FILE).getroot()
    edge_ids = {e.get("id") for e in root.findall("edge")}
    for edge in ("N2C", "S2C", "E2C", "W2C", "C2N", "C2S", "C2E", "C2W"):
        assert edge in edge_ids


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_scenario_files_are_generated(scenario):
    config = load_config(scenario)
    files = build_scenario(config)
    for key in ("net", "routes", "tls", "sumocfg"):
        assert os.path.isfile(files[key]), "missing generated file: {}".format(key)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_route_file_matches_the_configured_demand(scenario):
    config = load_config(scenario)
    files = build_scenario(config)
    root = ET.parse(files["routes"]).getroot()
    flows = root.findall("flow")
    # 4 approaches x 3 turning movements
    assert len(flows) == 12
    total = sum(float(f.get("vehsPerHour")) for f in flows)
    assert total == pytest.approx(config_loader.total_demand(config), rel=1e-6)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_traffic_light_program_matches_the_config(scenario):
    config = load_config(scenario)
    files = build_scenario(config)
    root = ET.parse(files["tls"]).getroot()
    logic = root.find("tlLogic")
    assert logic.get("id") == TLS_ID
    assert logic.get("programID") == TLS_PROGRAM_ID

    phases = logic.findall("phase")
    signal = config["signal"]
    expected = [signal["ns_green"], signal["yellow"],
                signal["ew_green"], signal["yellow"]]
    assert len(phases) == len(expected)
    for phase, duration in zip(phases, expected):
        assert float(phase.get("duration")) == pytest.approx(float(duration))
        # every phase controls all 16 links of the junction
        assert len(phase.get("state")) == 16

    # green must alternate between the two directions
    assert "G" in phases[0].get("state")[:4]     # North gets green first
    assert phases[0].get("state")[4:8] == "rrrr"  # East is red at that moment
    assert phases[2].get("state")[:4] == "rrrr"   # then North is red
    assert "G" in phases[2].get("state")[4:8]     # and East gets green


# ---------------------------------------------------------------- SUMO / TraCI
def test_sumo_and_traci_are_installed():
    from simulation.run_simulation import prepare_sumo_imports, sumo_binary

    prepare_sumo_imports()
    import traci  # noqa: F401

    binary = sumo_binary(False)
    assert os.path.isfile(binary) or binary  # sumolib returned a usable command


def test_traci_can_connect_and_close():
    """Open SUMO, step a few times, read the signal, and close cleanly."""
    from simulation.run_simulation import prepare_sumo_imports, sumo_binary

    prepare_sumo_imports()
    import traci

    config = load_config("moderate")
    config["simulation"]["duration"] = 60
    files = build_scenario(config)

    traci.start([sumo_binary(False), "-c", files["sumocfg"]], label="pytest_connect")
    try:
        assert traci.trafficlight.getProgram(TLS_ID) == TLS_PROGRAM_ID
        for _ in range(30):
            traci.simulationStep()
        assert traci.simulation.getTime() == pytest.approx(30.0)
        state = traci.trafficlight.getRedYellowGreenState(TLS_ID)
        assert len(state) == 16
        for lane in INCOMING_LANES:
            assert traci.lane.getLastStepHaltingNumber(lane) >= 0
    finally:
        traci.close()

    # After closing, a new connection must be possible again.
    traci.start([sumo_binary(False), "-c", files["sumocfg"]], label="pytest_reconnect")
    traci.simulationStep()
    traci.close()


# ---------------------------------------------------------------- full pipeline
@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    """Run a short moderate scenario into a temporary results folder."""
    from simulation.run_simulation import run_scenario

    out = tmp_path_factory.mktemp("results")
    summary = run_scenario("moderate", quiet=True,
                           duration=SMOKE_DURATION, results_dir=str(out))
    return summary, out


def test_simulation_runs_and_terminates(smoke_run):
    summary, _ = smoke_run
    assert summary["simulation_time"] == pytest.approx(SMOKE_DURATION)


def test_vehicles_are_generated_and_move(smoke_run):
    summary, _ = smoke_run
    assert summary["vehicles"] > 0, "no vehicle entered the simulation"
    assert summary["throughput"] > 0, "no vehicle got through the intersection"
    assert summary["throughput"] <= summary["vehicles"]


def test_vehicles_queue_at_the_signal(smoke_run):
    summary, _ = smoke_run
    # With a red phase on every cycle at least one vehicle has to wait.
    assert summary["average_waiting_time"] > 0
    assert summary["max_queue_length"] > 0


def test_metrics_have_the_expected_fields(smoke_run):
    summary, _ = smoke_run
    for key in ("scenario", "vehicles", "average_waiting_time",
                "average_queue_length", "throughput", "simulation_time"):
        assert key in summary
    assert summary["average_queue_length"] >= 0


def test_csv_is_written_with_the_expected_columns(smoke_run):
    import pandas as pd

    _, out = smoke_run
    path = out / "moderate_results.csv"
    assert path.is_file()
    frame = pd.read_csv(path)
    assert len(frame) == 1
    for column in ("scenario", "vehicles", "average_waiting_time",
                   "average_queue_length", "throughput", "simulation_time"):
        assert column in frame.columns


def test_timeseries_csv_is_written(smoke_run):
    import pandas as pd

    _, out = smoke_run
    path = out / "moderate_timeseries.csv"
    assert path.is_file()
    frame = pd.read_csv(path)
    assert len(frame) == int(SMOKE_DURATION)
    assert frame["queue_length"].min() >= 0


# ---------------------------------------------------------------- plots
def test_plots_are_generated_from_the_csv_files():
    """Uses the real results in results/csv/ produced by the two scenarios."""
    from simulation import plot_results

    csv_dir = os.path.join(PROJECT_ROOT, "results", "csv")
    for scenario in SCENARIOS:
        path = os.path.join(csv_dir, "{}_results.csv".format(scenario))
        if not os.path.isfile(path):
            pytest.skip("run both scenarios first: {} is missing".format(path))

    created = plot_results.generate_plots()
    assert len(created) >= 3
    for path in created:
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0
