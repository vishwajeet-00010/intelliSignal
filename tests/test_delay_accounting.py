"""Tests for the versioned delay accounting.

The hole these close: `average_waiting_time` divides the waiting of
vehicles SEEN by the number of vehicles SEEN. A vehicle SUMO never managed
to insert is in neither term, so a controller that blocks a saturated
entry is rewarded for it twice.

What is pinned here:

  * a vehicle that never entered is charged for the time it waited
  * `timeLoss` already contains stopped time, so the two are never added
  * the historical field keeps its meaning and its name
  * the two independent measurements are cross-checked rather than
    assumed to agree
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.delay_accounting import (METRICS_VERSION, account,  # noqa: E402
                                    cross_check, enable_tripinfo,
                                    read_tripinfo)

TRIPINFO = """<?xml version="1.0" encoding="UTF-8"?>
<tripinfos>
    <tripinfo id="a" depart="0.00" departDelay="0.00" duration="40.00"
              waitingTime="5.00" timeLoss="12.00" arrival="40.00"/>
    <tripinfo id="b" depart="10.00" departDelay="2.00" duration="60.00"
              waitingTime="25.00" timeLoss="38.00" arrival="70.00"/>
</tripinfos>
"""

#: SUMO is asked for the unfinished trips too (`write-unfinished`), and
#: marks a vehicle still on the network when the run ended with an
#: arrival of -1. Its delay counts; its journey does not.
TRIPINFO_WITH_UNFINISHED = """<?xml version="1.0" encoding="UTF-8"?>
<tripinfos>
    <tripinfo id="a" depart="0.00" departDelay="0.00" duration="40.00"
              waitingTime="5.00" timeLoss="12.00" arrival="40.00"/>
    <tripinfo id="b" depart="10.00" departDelay="2.00" duration="60.00"
              waitingTime="25.00" timeLoss="38.00" arrival="70.00"/>
    <tripinfo id="c" depart="20.00" departDelay="0.00" duration="180.00"
              waitingTime="90.00" timeLoss="140.00" arrival="-1.00"/>
</tripinfos>
"""


def write(tmp_path, text=TRIPINFO):
    path = tmp_path / "trips.xml"
    path.write_text(text, encoding="utf-8")
    return str(path)


# ===========================================================================
def test_tripinfo_is_parsed(tmp_path):
    trips = read_tripinfo(write(tmp_path))
    assert len(trips) == 2
    assert trips[0]["timeLoss"] == 12.0
    assert trips[1]["departDelay"] == 2.0


def test_a_missing_file_is_not_an_error(tmp_path):
    assert read_tripinfo(str(tmp_path / "nothing.xml")) == []


def test_a_file_cut_off_mid_write_is_salvaged(tmp_path):
    """SUMO writes the closing tag only on a clean shutdown."""
    truncated = TRIPINFO.replace("</tripinfos>", "")
    trips = read_tripinfo(write(tmp_path, truncated))
    assert len(trips) == 2


# ===========================================================================
def test_vehicles_that_never_entered_are_charged(tmp_path):
    """The whole point: 10 requested, 2 completed, 8 never got in."""
    trips = read_tripinfo(write(tmp_path))
    scheduled = [float(10 * i) for i in range(10)]     # 0, 10, ... 90
    result = account(trips, scheduled, run_end_s=200.0, still_on_network=0)

    assert result["requested"] == 10
    assert result["completed"] == 2
    assert result["waiting_to_enter"] == 8
    # The eight never-inserted are the eight latest scheduled: 20..90.
    # Each waited from its scheduled time to the end of the run.
    expected = sum(200.0 - t for t in scheduled[2:])
    assert result["unserved_insertion_delay_s"] == pytest.approx(expected)
    assert result["mean_total_delay_per_requested"] > 0


def test_blocking_the_entry_cannot_improve_the_honest_metric(tmp_path):
    """Two runs, same traffic. One lets everyone in, one blocks most."""
    everyone = read_tripinfo(write(tmp_path, _ten_trips()))
    only_two = read_tripinfo(write(tmp_path, TRIPINFO))
    scheduled = [float(10 * i) for i in range(10)]

    lets_them_in = account(everyone, scheduled, run_end_s=200.0)
    blocks_them = account(only_two, scheduled, run_end_s=200.0)

    assert lets_them_in["waiting_to_enter"] == 0
    assert blocks_them["waiting_to_enter"] == 8
    assert (blocks_them["mean_total_delay_per_requested"]
            > lets_them_in["mean_total_delay_per_requested"]), (
        "refusing vehicles must make the honest metric worse, not better")


def _ten_trips() -> str:
    """Ten vehicles that all got in and all finished."""
    rows = [
        '    <tripinfo id="v{i}" depart="{d:.2f}" departDelay="0.00" '
        'duration="40.00" waitingTime="5.00" timeLoss="12.00" '
        'arrival="{a:.2f}"/>'.format(i=i, d=10.0 * i, a=10.0 * i + 40.0)
        for i in range(10)]
    return "\n".join(['<?xml version="1.0" encoding="UTF-8"?>', "<tripinfos>"]
                     + rows + ["</tripinfos>", ""])


def test_a_vehicle_still_inside_at_the_end_is_charged_for_its_delay(tmp_path):
    """Otherwise the worse controller gets the delay it caused for free.

    A controller that strands traffic inside the junction leaves more
    unfinished trips than one that clears it. While those trips were
    left out of the sum, stranding traffic lowered the reported delay.
    """
    scheduled = [0.0, 10.0, 20.0]
    with_stranded = read_tripinfo(write(tmp_path, TRIPINFO_WITH_UNFINISHED))

    # The same run as SUMO used to report it: the stranded vehicle is
    # known to be on the network, but its trip is not in the file, so
    # none of the delay it suffered is counted anywhere.
    unreported = account(read_tripinfo(write(tmp_path, TRIPINFO)), scheduled,
                         run_end_s=200.0, still_on_network=1)
    counted = account(with_stranded, scheduled, run_end_s=200.0)

    assert counted["unfinished_charged"] == 1
    assert counted["completed"] == 2, "an unfinished trip is not completed"
    assert counted["entered"] == 3, "but it did get onto the network"
    assert counted["still_on_network"] == 1
    assert counted["waiting_to_enter"] == 0

    # The two describe the same run; only one of them charges for it.
    assert unreported["entered"] == counted["entered"]
    assert unreported["completed"] == counted["completed"]
    assert counted["total_time_loss_s"] - unreported["total_time_loss_s"] \
        == pytest.approx(140.0)
    assert (counted["mean_total_delay_per_requested"]
            > unreported["mean_total_delay_per_requested"]), (
        "a vehicle the signal left stuck inside must cost the controller "
        "something, or stranding traffic improves the score")


def test_stopped_waiting_and_time_loss_are_never_added(tmp_path):
    """timeLoss already contains the time spent stopped."""
    trips = read_tripinfo(write(tmp_path))
    result = account(trips, [0.0, 10.0], run_end_s=200.0)

    total_loss = 12.0 + 38.0
    total_stopped = 5.0 + 25.0
    insertion = 0.0 + 2.0
    assert result["total_time_loss_s"] == pytest.approx(total_loss)
    assert result["total_stopped_waiting_s"] == pytest.approx(total_stopped)
    # The total is time loss plus insertion delay. Stopped time is inside
    # time loss, so adding it too would charge standing still twice.
    assert result["total_delay_s"] == pytest.approx(total_loss + insertion)
    assert result["total_delay_s"] != pytest.approx(
        total_loss + total_stopped + insertion)


def test_the_accounting_is_versioned(tmp_path):
    result = account(read_tripinfo(write(tmp_path)), [0.0, 10.0],
                     run_end_s=100.0)
    assert result["metrics_version"] == METRICS_VERSION >= 2


def test_completion_rate_is_against_what_was_requested(tmp_path):
    trips = read_tripinfo(write(tmp_path))
    result = account(trips, [0.0] * 8, run_end_s=100.0)
    assert result["completion_rate"] == pytest.approx(2 / 8)


def test_nothing_requested_does_not_divide_by_zero():
    result = account([], [], run_end_s=100.0)
    assert result["mean_total_delay_per_requested"] == 0.0
    assert result["completion_rate"] == 0.0


# ===========================================================================
def test_the_cross_check_notices_a_disagreement(tmp_path):
    trips = read_tripinfo(write(tmp_path))
    result = account(trips, [0.0, 10.0], run_end_s=100.0)

    agreeing = cross_check(result, {"throughput": 2, "average_waiting_time": 15.0})
    assert agreeing["completed_agree"] is True

    disagreeing = cross_check(result, {"throughput": 450,
                                       "average_waiting_time": 15.0})
    assert disagreeing["completed_agree"] is False, (
        "a silent disagreement between two measurements of the same run "
        "is exactly what this is here to catch")


# ===========================================================================
def test_enabling_tripinfo_rewrites_only_the_generated_config(tmp_path):
    config = tmp_path / "x.sumocfg"
    config.write_text(
        '<?xml version="1.0"?>\n<configuration>\n'
        '    <input><net-file value="n.xml"/></input>\n'
        '</configuration>\n', encoding="utf-8")
    output = str(tmp_path / "out" / "trips.xml")

    enable_tripinfo(str(config), output)
    text = config.read_text(encoding="utf-8")
    assert "<tripinfo-output " in text
    assert "<tripinfo-output.write-unfinished " in text, (
        "the unfinished trips are what stop a stranded vehicle being free")
    assert text.count("</configuration>") == 1

    # Applying it twice must not nest a second output block.
    enable_tripinfo(str(config), output)
    assert config.read_text(encoding="utf-8").count("<output>") == 1


def test_the_historical_field_is_not_touched(tmp_path):
    """`average_waiting_time` keeps its meaning so old numbers compare."""
    import inspect

    from utils import metrics
    source = inspect.getsource(metrics.MetricsCollector.summary)
    assert "average_waiting_time" in source
    assert "sum(self._waiting_time.values()) / n_vehicles" in source, (
        "the historical definition must stay exactly as it was; the "
        "corrected measurement lives in utils/delay_accounting.py")
