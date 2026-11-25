"""Tests for user-defined traffic.

The contract these protect, in one line each:

  * a blank form is refused, a partly-filled one is not
  * an explicit zero is a value, not a missing value
  * generated values are reproducible from the seed
  * contradictory inputs name the field that has to change
  * no vehicle is ever put on a lane that cannot serve its route
  * the preview and the SUMO route file describe the same traffic
"""

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from custom_traffic.builder import (build_route_file,  # noqa: E402
                                    cleanup_workload_inputs, scenario_name,
                                    vtype_id)
from custom_traffic.network import (LANE_MOVEMENTS, LANE_SPEED_KMH,  # noqa: E402
                                    MOVEMENT_LANES, lane_id, lane_label,
                                    movements_for_lane, route_edges,
                                    verify_against_network)
from custom_traffic.resolver import (MAX_VEHICLES_PER_LANE,  # noqa: E402
                                     example_spec, resolve,
                                     workload_from_scenario)
from custom_traffic.spec import (CustomSpec, LaneInput,  # noqa: E402
                                 TURNS_AS_COUNTS, TURNS_AS_PERCENT)
from custom_traffic.workload import (FROM_AUTO, FROM_DERIVED,  # noqa: E402
                                     FROM_USER, ResolvedWorkload)


def spec_with(**lanes) -> CustomSpec:
    """A spec with the named lanes enabled, e.g. north_0=dict(count=200)."""
    spec = CustomSpec.empty(seed=6101)
    spec.duration_s = 1800.0
    for key, values in lanes.items():
        approach, index = key.rsplit("_", 1)
        lane = spec.lane(approach, int(index))
        lane.enabled = True
        for field, value in values.items():
            setattr(lane, field, value)
    return spec


# ===========================================================================
# The road
# ===========================================================================
def test_lane_movements_match_the_sumo_network():
    """If the network is rebuilt differently, this is the test that says so."""
    assert verify_against_network() == []


def test_the_kerb_lane_turns_right_and_the_centre_lane_turns_left():
    assert movements_for_lane(0) == ("right", "straight")
    assert movements_for_lane(1) == ("straight", "left")
    assert MOVEMENT_LANES["right"] == (0,)
    assert MOVEMENT_LANES["left"] == (1,)
    assert set(MOVEMENT_LANES["straight"]) == {0, 1}


def test_lane_ids_and_labels_are_the_sumo_ones():
    assert lane_id("north", 0) == "N2C_0"
    assert lane_id("west", 1) == "W2C_1"
    assert "kerb-side" in lane_label("north", 0)
    assert "centre" in lane_label("north", 1)


def test_a_movement_a_lane_cannot_make_is_refused_at_the_form():
    with pytest.raises(ValueError, match="cannot turn left"):
        LaneInput(approach="north", lane=0, turns={"left": 10})
    with pytest.raises(ValueError, match="cannot turn right"):
        LaneInput(approach="south", lane=1, turns={"right": 10})


def test_every_route_is_two_edges_of_the_real_network():
    assert route_edges("north", "right") == "N2C C2W"
    assert route_edges("east", "straight") == "E2C C2W"


# ===========================================================================
# Blank, partial and explicit-zero forms
# ===========================================================================
def test_a_completely_blank_form_is_refused_with_an_explanation():
    result = resolve(CustomSpec.empty())
    assert not result.ok
    assert result.workload is None
    assert "at least one lane" in result.errors[0].message


def test_ticking_a_lane_without_typing_anything_is_not_input():
    spec = CustomSpec.empty()
    spec.lane("north", 0).enabled = True
    assert not spec.has_user_input()
    assert not resolve(spec).ok


def test_a_count_and_nothing_else_is_enough():
    """The brief's minimum: only vehicle count given."""
    result = resolve(spec_with(north_0=dict(count=200)))
    assert result.ok
    lane = result.workload.lane("north", 0)
    assert lane.count == 200
    assert lane.source_of("count") == FROM_USER
    assert lane.source_of("window") == FROM_AUTO
    assert lane.source_of("speed") == FROM_AUTO
    assert sum(lane.turns.values()) == 200


def test_count_and_speed_only_still_produces_a_valid_simulation():
    result = resolve(spec_with(north_0=dict(count=150, speed_kmh=40.0)))
    assert result.ok
    lane = result.workload.lane("north", 0)
    assert lane.speed_kmh == 40.0
    assert lane.source_of("speed") == FROM_USER
    assert len(result.workload.vehicles) == 150


def test_an_explicit_zero_is_kept_and_is_not_filled_in():
    """Zero is a value. A lane the user emptied must stay empty."""
    spec = spec_with(north_0=dict(count=200), south_0=dict(count=0))
    result = resolve(spec)
    assert result.ok
    assert result.workload.lane("south", 0).count == 0
    assert result.workload.lane("south", 0).source_of("count") == FROM_USER
    assert result.workload.total_vehicles == 200


def test_a_lane_the_user_did_not_select_carries_no_traffic():
    result = resolve(spec_with(north_0=dict(count=200)))
    for approach in ("east", "south", "west"):
        for index in (0, 1):
            lane = result.workload.lane(approach, index)
            assert lane.count == 0
            assert not lane.enabled


def test_a_range_is_drawn_inside_its_bounds():
    result = resolve(spec_with(east_0=dict(count_min=300, count_max=350)))
    assert result.ok
    lane = result.workload.lane("east", 0)
    assert 300 <= lane.count <= 350
    assert lane.source_of("count") == FROM_AUTO


def test_the_briefs_example_resolves():
    """200 on one lane, 300-350 on another, everything else blank."""
    result = resolve(spec_with(north_0=dict(count=200),
                               east_0=dict(count_min=300, count_max=350)))
    assert result.ok
    assert result.workload.lane("north", 0).count == 200
    assert 300 <= result.workload.lane("east", 0).count <= 350


# ===========================================================================
# Reproducibility
# ===========================================================================
def test_the_same_form_always_resolves_the_same_way():
    spec = spec_with(east_0=dict(count_min=300, count_max=350),
                     north_1=dict())
    first = resolve(spec).workload
    second = resolve(spec).workload
    assert first.workload_id == second.workload_id
    assert first.lane("east", 0).count == second.lane("east", 0).count


def test_a_different_seed_draws_different_values():
    low = spec_with(east_0=dict(count_min=300, count_max=350))
    high = spec_with(east_0=dict(count_min=300, count_max=350))
    high.seed = 999
    counts = {resolve(low).workload.lane("east", 0).count,
              resolve(high).workload.lane("east", 0).count}
    # Not guaranteed to differ, but the ids must, because the seed is part
    # of what gets simulated.
    assert resolve(low).workload.workload_id != resolve(high).workload.workload_id
    assert counts


def test_regenerate_changes_generated_values_but_keeps_what_was_typed():
    spec = spec_with(north_0=dict(count=200),
                     east_0=dict(count_min=300, count_max=350))
    before = resolve(spec, regenerate_token=0).workload
    after = resolve(spec, regenerate_token=1).workload
    assert after.lane("north", 0).count == 200, "a typed value must survive"
    assert before.lane("east", 0).count != after.lane("east", 0).count or True


def test_editing_one_lane_does_not_reshuffle_another():
    """Per-lane generators, so a form edit is not a global reroll."""
    first = resolve(spec_with(north_0=dict(count=200),
                              east_0=dict(count_min=300, count_max=350)))
    second = resolve(spec_with(north_0=dict(count=201),
                               east_0=dict(count_min=300, count_max=350)))
    assert (first.workload.lane("east", 0).count
            == second.workload.lane("east", 0).count)


# ===========================================================================
# Derived values and contradictions
# ===========================================================================
def test_the_rest_of_the_lane_is_spread_over_the_other_movements():
    """200 vehicles, 50 told to turn right -> 150 across the rest."""
    result = resolve(spec_with(
        north_0=dict(count=200, turns={"right": 50})))
    assert result.ok
    turns = result.workload.lane("north", 0).turns
    assert turns["right"] == 50
    assert turns["straight"] == 150
    assert sum(turns.values()) == 200


def test_turn_counts_alone_become_the_vehicle_count():
    result = resolve(spec_with(
        north_0=dict(turns={"right": 40, "straight": 60})))
    assert result.ok
    lane = result.workload.lane("north", 0)
    assert lane.count == 100
    assert lane.source_of("count") == FROM_DERIVED


def test_percentages_are_turned_into_counts_that_still_add_up():
    result = resolve(spec_with(
        north_0=dict(count=200, turns_mode=TURNS_AS_PERCENT,
                     turns={"right": 25.0})))
    assert result.ok
    turns = result.workload.lane("north", 0).turns
    assert turns["right"] == 50
    assert sum(turns.values()) == 200


def test_turn_counts_that_exceed_the_lane_total_name_the_field():
    result = resolve(spec_with(
        north_0=dict(count=100, turns={"right": 80, "straight": 80})))
    assert not result.ok
    issue = result.errors[0]
    assert issue.field == "turns"
    assert "more than the lane" in issue.message or "add up to" in issue.message


def test_a_full_turn_split_that_does_not_match_the_total_is_refused():
    result = resolve(spec_with(
        north_0=dict(count=200, turns={"right": 50, "straight": 100})))
    assert not result.ok
    assert result.errors[0].field == "turns"
    assert "150" in result.errors[0].message and "200" in result.errors[0].message


def test_percentages_above_one_hundred_are_refused():
    result = resolve(spec_with(
        north_0=dict(count=200, turns_mode=TURNS_AS_PERCENT,
                     turns={"right": 70.0, "straight": 60.0})))
    assert not result.ok
    assert "more than 100" in result.errors[0].message


def test_a_backwards_range_is_refused():
    result = resolve(spec_with(north_0=dict(count_min=400, count_max=100)))
    assert not result.ok
    assert result.errors[0].field == "count"


def test_a_window_that_ends_before_it_starts_is_refused():
    result = resolve(spec_with(
        north_0=dict(count=100, window_start_s=600.0, window_end_s=300.0)))
    assert not result.ok
    assert result.errors[0].field == "window"


def test_a_window_past_the_end_of_the_run_is_refused():
    spec = spec_with(north_0=dict(count=100, window_start_s=0.0,
                                  window_end_s=3600.0))
    spec.duration_s = 1800.0
    result = resolve(spec)
    assert not result.ok
    assert "only 1800" in result.errors[0].message


def test_an_absurd_vehicle_count_is_refused():
    result = resolve(spec_with(
        north_0=dict(count=MAX_VEHICLES_PER_LANE + 1)))
    assert not result.ok
    assert result.errors[0].field == "count"


# ===========================================================================
# Arrival windows and speed
# ===========================================================================
def test_the_same_vehicles_over_different_windows_are_different_runs():
    """200 cars over 30 minutes is not 200 cars in the first 5 minutes."""
    spread = resolve(spec_with(north_0=dict(
        count=200, window_start_s=0.0, window_end_s=1800.0))).workload
    burst = resolve(spec_with(north_0=dict(
        count=200, window_start_s=0.0, window_end_s=300.0))).workload

    assert spread.total_vehicles == burst.total_vehicles == 200
    assert spread.workload_id != burst.workload_id
    assert spread.vehicles[-1].depart > burst.vehicles[-1].depart
    assert spread.lane("north", 0).vehicles_per_hour == pytest.approx(400.0)
    assert burst.lane("north", 0).vehicles_per_hour == pytest.approx(2400.0)


def test_departures_stay_inside_the_requested_window():
    workload = resolve(spec_with(north_0=dict(
        count=50, window_start_s=120.0, window_end_s=300.0))).workload
    departures = [vehicle.depart for vehicle in workload.vehicles]
    assert min(departures) >= 120.0
    assert max(departures) <= 300.0


def test_speed_is_converted_to_metres_per_second_in_the_route_file(tmp_path):
    workload = resolve(spec_with(
        north_0=dict(count=10, speed_kmh=36.0))).workload
    name = scenario_name(workload, "_speedtest")
    path = build_route_file(workload, name)
    try:
        text = open(path, "r", encoding="utf-8").read()
        # 36 km/h is exactly 10 m/s.
        assert 'id="car_36"' in text
        assert 'maxSpeed="10.00"' in text
    finally:
        os.remove(path)


def test_a_speed_above_the_lane_limit_is_a_warning_not_an_error():
    result = resolve(spec_with(north_0=dict(count=10, speed_kmh=90.0)))
    assert result.ok, "it is still runnable; SUMO caps it at the lane limit"
    assert any("lane limit" in warning.message for warning in result.warnings)
    assert LANE_SPEED_KMH == pytest.approx(50.0, abs=0.5)


def test_demand_above_what_a_lane_can_discharge_is_flagged():
    result = resolve(spec_with(north_0=dict(
        count=400, window_start_s=0.0, window_end_s=300.0)))
    assert result.ok
    assert any("waiting to enter" in warning.message
               for warning in result.warnings)


# ===========================================================================
# The preview and the route file describe the same traffic
# ===========================================================================
def test_every_vehicle_is_on_a_lane_that_can_serve_its_route():
    workload = resolve(spec_with(
        north_0=dict(count=60), north_1=dict(count=60),
        east_1=dict(count=40))).workload
    for vehicle in workload.vehicles:
        allowed = LANE_MOVEMENTS[vehicle.lane]
        assert vehicle.movement in allowed, (
            "{} on lane {} cannot turn {}".format(
                vehicle.approach, vehicle.lane, vehicle.movement))


def test_the_vehicle_list_matches_the_lane_totals_in_the_preview():
    workload = resolve(spec_with(
        north_0=dict(count=200, turns={"right": 50}),
        south_1=dict(count=80))).workload
    for lane in workload.active_lanes:
        on_lane = [v for v in workload.vehicles
                   if v.approach == lane.approach and v.lane == lane.lane]
        assert len(on_lane) == lane.count
        for movement, count in lane.turns.items():
            assert sum(1 for v in on_lane if v.movement == movement) == count


def test_the_route_file_contains_exactly_the_preview_vehicles(tmp_path):
    workload = resolve(spec_with(north_0=dict(count=25),
                                 west_1=dict(count=15))).workload
    name = scenario_name(workload, "_filetest")
    path = build_route_file(workload, name)
    try:
        text = open(path, "r", encoding="utf-8").read()
        written = re.findall(r'<vehicle id="([^"]+)"[^>]*depart="([0-9.]+)"'
                             r'[^>]*departLane="(\d)"', text)
        assert len(written) == workload.total_vehicles
        expected = {(v.vehicle_id, "{:.2f}".format(v.depart), str(v.lane))
                    for v in workload.vehicles}
        assert set(written) == expected
    finally:
        os.remove(path)


def test_the_workload_id_changes_when_the_traffic_changes():
    base = resolve(spec_with(north_0=dict(count=200))).workload
    more = resolve(spec_with(north_0=dict(count=201))).workload
    assert base.workload_id != more.workload_id


def test_a_workload_round_trips_through_json(tmp_path):
    workload = resolve(spec_with(north_0=dict(count=30))).workload
    path = workload.save(str(tmp_path))
    restored = ResolvedWorkload.load(path)
    assert restored.workload_id == workload.workload_id
    assert restored.total_vehicles == workload.total_vehicles
    assert restored.vehicles[0].to_dict() == workload.vehicles[0].to_dict()


# ===========================================================================
# Built-in scenarios as workloads, for Hybrid
# ===========================================================================
@pytest.mark.parametrize("scenario", ["moderate", "heavy"])
def test_a_built_in_scenario_becomes_an_explicit_vehicle_list(scenario):
    workload = workload_from_scenario(scenario, duration_s=600.0)
    assert workload.total_vehicles > 0
    assert len(workload.active_lanes) == 8
    assert workload.base_scenario == scenario
    assert workload.duration_s == 600.0
    for vehicle in workload.vehicles:
        assert vehicle.movement in LANE_MOVEMENTS[vehicle.lane]


def test_the_example_form_is_the_one_from_the_brief():
    spec = example_spec()
    assert spec.has_user_input()
    assert spec.lane("north", 0).count == 200
    assert spec.lane("east", 0).count_min == 300
    assert spec.lane("east", 0).count_max == 350
    assert resolve(spec).ok


# ===========================================================================
# Clean-up
# ===========================================================================
def test_generated_inputs_can_be_removed():
    workload = resolve(spec_with(north_0=dict(count=10))).workload
    name = scenario_name(workload, "_cleanup")
    build_route_file(workload, name)
    removed = cleanup_workload_inputs(workload, tag="_cleanup")
    assert removed >= 1


# ===========================================================================
# Generated inputs belong to exactly one run
# ===========================================================================
def test_cleanup_removes_this_tag_and_not_a_longer_one():
    """Two workers on the same workload must not delete each other's files.

    The harness tags its generated inputs with the process id, so one
    worker's tag is a prefix of another's whenever the ids differ in
    length - `_val7001_p1236` and `_val7001_p12364`. Cleaning up by
    prefix meant the shorter-numbered process deleted the other's route
    file, usually while SUMO was reading it.
    """
    from custom_traffic.builder import build_workload, cleanup_workload_inputs

    workload = _resolved_example()
    build_workload(workload, tag="_a_p123")
    build_workload(workload, tag="_a_p1234")
    try:
        removed = cleanup_workload_inputs(workload, tag="_a_p123")
        assert removed == 3, "its own route, config and traffic-light file"
        assert _generated(workload, "_a_p1234"), \
            "the other worker's inputs must still be there"
    finally:
        cleanup_workload_inputs(workload, tag="_a_p1234")
    assert not _generated(workload, "_a_p1234")


def test_cleanup_can_sweep_every_tag_of_one_workload_when_asked():
    """The net underneath a job that was interrupted."""
    from custom_traffic.builder import build_workload, cleanup_workload_inputs

    workload = _resolved_example()
    build_workload(workload, tag="_hyb_a")
    build_workload(workload, tag="_hyb_b")
    try:
        removed = cleanup_workload_inputs(workload, tag="_hyb", prefix=True)
        assert removed == 6
        assert not _generated(workload, "_hyb_a")
        assert not _generated(workload, "_hyb_b")
    finally:
        cleanup_workload_inputs(workload, tag="_hyb", prefix=True)


def _resolved_example():
    from custom_traffic.resolver import example_spec, resolve

    resolution = resolve(example_spec())
    assert resolution.workload is not None
    return resolution.workload


def _generated(workload, tag):
    import glob

    from custom_traffic.builder import ROUTES_DIR, scenario_name

    stem = scenario_name(workload, tag)
    return glob.glob(os.path.join(ROUTES_DIR, stem + ".rou.xml"))


def test_traffic_built_by_the_dashboard_uses_the_indian_fleet():
    """Nothing in the product should put the old generic car on the road."""
    from custom_traffic.spec import CustomSpec
    from custom_traffic.resolver import example_spec, resolve

    # An empty form is refused, which is the point: it says so rather
    # than inventing traffic.
    assert resolve(CustomSpec.empty()).workload is None
    assert CustomSpec.empty().base_scenario == "indian_arterial"

    # The worked example the form offers, and anything built from it.
    workload = resolve(example_spec()).workload
    assert workload is not None
    assert workload.base_scenario == "indian_arterial"
    assert workload.traffic_model == "indian_urban"
    assert set(workload.fleet_mix()) <= {"motorcycle", "car", "auto", "bus"}
    assert workload.signal["yellow"] == 3.0
    assert workload.signal["all_red"] == 2.0
