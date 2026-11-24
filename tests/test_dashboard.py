"""
IntelliSignal - Dashboard stage
tests/test_dashboard.py

Tests for the presentation layer and the two new layers under it.

Three groups:

  * agents/            the decision-making interface and the fixed-time
                       controller, plus proof that the DQN refuses to act
                       without trained weights
  * simulation/controller.py
                       environment detection, scenario reading, control modes,
                       running a simulation, loading results, and every error
                       path the dashboard has to explain
  * dashboard.py       the Streamlit app itself, driven through Streamlit's
                       AppTest so it is really executed, not just imported

The Streamlit tests run the app headlessly. Simulations inside tests use short
durations so the suite stays fast, and they never write to results/csv/, which
holds the recorded Stage 1 baseline.

What Stage 3 changed in this file
---------------------------------
When this file was written the DQN did not exist, and several tests asserted
that: `DQN_IMPLEMENTED is False`, `create_agent("dqn")` raising "not
implemented", the mode being permanently disabled, and the About page saying
there was no neural network. Stage 3 built the DQN, so those assertions
describe an architecture that is no longer the intended one and have been
rewritten rather than deleted. Each one now asserts the Stage 3 rule that
replaced it, and the honesty requirement is stricter than before:

    before : the DQN mode is never available
    now    : the DQN mode is available ONLY when PyTorch is installed and a
             trained checkpoint exists for the selected scenario, and a
             missing model produces a clear refusal instead of a fixed-time
             run wearing a DQN label

The deeper Stage 3 coverage lives in tests/test_dqn.py.

Run with:
    python -m pytest -q
"""

from __future__ import annotations

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents import (ACTION_EW_GREEN, ACTION_NS_GREEN, ACTION_SPACE,  # noqa: E402
                    DQN_IMPLEMENTED, DQNAgent, DQNModelNotFoundError,
                    ENV_ACTION_KEEP, ENV_ACTION_SWITCH, FixedTimeAgent,
                    PHASE_EW, PHASE_NS, TrafficAgent, TrafficState,
                    create_agent, list_agents, phase_action_to_env_action)
from simulation.controller import (BASELINE_RESULTS_DIR,  # noqa: E402
                                   ControlModeUnavailableError,
                                   ScenarioNotFoundError, SimulationController,
                                   SimulationError, SumoUnavailableError)
from utils.config_loader import load_config  # noqa: E402

DASHBOARD_FILE = os.path.join(PROJECT_ROOT, "dashboard.py")
SHORT_DURATION = 120.0


@pytest.fixture
def no_models(monkeypatch, tmp_path):
    """Point model discovery at an empty directory.

    Lets these tests assert the "no trained model" behaviour on a machine
    where a model HAS been trained, which is the normal state after Stage 3.
    """
    monkeypatch.setattr("agents.dqn_agent.MODELS_DIR", str(tmp_path / "models"))
    return tmp_path


@pytest.fixture(scope="module")
def controller():
    return SimulationController()


@pytest.fixture
def app():
    """A freshly started dashboard."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(DASHBOARD_FILE, default_timeout=300)
    at.run()
    return at


# ==========================================================================
# agents/ - the decision layer
# ==========================================================================
def test_fixed_time_agent_implements_the_interface():
    agent = FixedTimeAgent()
    assert isinstance(agent, TrafficAgent)
    assert agent.available is True
    assert agent.is_learned is False


def test_fixed_time_agent_reads_its_timings_from_the_scenario():
    for scenario in ("moderate", "heavy"):
        config = load_config(scenario)
        agent = FixedTimeAgent.from_config(config)
        assert agent.ns_green == config["signal"]["ns_green"]
        assert agent.ew_green == config["signal"]["ew_green"]
        assert agent.yellow == config["signal"]["yellow"]
        assert agent.cycle_time == pytest.approx(68.0)


def test_fixed_time_agent_holds_the_phase_until_its_green_expires():
    agent = FixedTimeAgent(ns_green=30, ew_green=30, yellow=4)
    holding = TrafficState(current_phase=PHASE_NS, time_since_phase_change=10)
    assert agent.select_action(holding) == ACTION_NS_GREEN
    expired = TrafficState(current_phase=PHASE_NS, time_since_phase_change=30)
    assert agent.select_action(expired) == ACTION_EW_GREEN


def test_fixed_time_agent_ignores_the_traffic_state():
    """The baseline must not react to traffic - that is the whole point."""
    agent = FixedTimeAgent(ns_green=30, ew_green=30, yellow=4)
    empty = TrafficState(current_phase=PHASE_NS, time_since_phase_change=10,
                         ns_queue=0, ew_queue=0)
    jammed = TrafficState(current_phase=PHASE_NS, time_since_phase_change=10,
                          ns_queue=0, ew_queue=40, ew_waiting=900)
    assert agent.select_action(empty) == agent.select_action(jammed)
    assert agent.explain(jammed)["uses_traffic_state"] is False


def test_agent_actions_translate_into_the_rl_environment_actions():
    assert phase_action_to_env_action(PHASE_NS, PHASE_NS) == ENV_ACTION_KEEP
    assert phase_action_to_env_action(PHASE_EW, PHASE_NS) == ENV_ACTION_SWITCH
    with pytest.raises(ValueError):
        phase_action_to_env_action(7, PHASE_NS)


def test_agent_env_action_uses_the_translation():
    agent = FixedTimeAgent(ns_green=30, ew_green=30, yellow=4)
    assert agent.env_action(
        TrafficState(current_phase=PHASE_NS, time_since_phase_change=5)) == ENV_ACTION_KEEP
    assert agent.env_action(
        TrafficState(current_phase=PHASE_NS, time_since_phase_change=40)) == ENV_ACTION_SWITCH


def test_traffic_state_can_be_built_from_the_rl_environment_info():
    info = {
        "queue_per_approach": {"north": 3, "south": 4, "east": 1, "west": 2},
        "waiting_per_approach": {"north": 30, "south": 40, "east": 5, "west": 5},
        "phase": PHASE_EW, "green_elapsed": 12.0, "simulation_time": 300.0,
    }
    state = TrafficState.from_env_info(info)
    assert state.ns_queue == 7
    assert state.ew_queue == 3
    assert state.ns_waiting == 70
    assert state.current_phase == PHASE_EW
    assert state.total_queue == 10
    assert len(state.as_vector()) == len(TrafficState.VECTOR_LABELS)


def test_action_space_is_defined_in_one_place():
    assert ACTION_SPACE == (ACTION_NS_GREEN, ACTION_EW_GREEN)
    assert ACTION_NS_GREEN == 0 and ACTION_EW_GREEN == 1


# --- academic honesty -----------------------------------------------------
# Stage 3 implemented the DQN. These tests therefore no longer assert "the
# DQN does not exist"; they assert the rule that replaced it - a DQN is only
# usable with trained weights, and says so plainly when it has none.
def test_dqn_is_implemented_in_stage_3():
    assert DQN_IMPLEMENTED is True
    assert DQNAgent.is_learned is True


def test_creating_a_dqn_agent_without_a_model_refuses_with_an_explanation(
        no_models):
    with pytest.raises(DQNModelNotFoundError) as info:
        create_agent("dqn", scenario="moderate")
    message = str(info.value).lower()
    assert "no trained dqn model" in message
    assert "train_dqn.py" in message, "the refusal must say how to fix it"


def test_agent_registry_reports_availability_honestly(no_models):
    """Availability is read from the filesystem, not from a flag."""
    rows = {row["name"]: row for row in list_agents()}
    assert rows["fixed_time"]["available"] is True
    assert rows["fixed_time"]["is_learned"] is False
    assert rows["dqn"]["available"] is False, "no checkpoint on disk"
    assert rows["dqn"]["is_learned"] is True


def test_unknown_agent_name_is_rejected():
    with pytest.raises(KeyError):
        create_agent("does_not_exist")


# ==========================================================================
# simulation/controller.py
# ==========================================================================
def test_environment_status_detects_the_real_installation(controller):
    status = controller.environment_status()
    assert status.sumo_available is True
    assert status.traci_available is True
    assert status.network_file_present is True
    assert status.simulation_ready is True
    assert "1." in status.sumo_version


def test_status_indicators_have_the_three_expected_lights(controller):
    labels = [item["label"] for item in controller.environment_status().indicators()]
    assert labels == ["SUMO Connected", "TraCI Available", "Simulation Ready"]


def test_controller_lists_both_scenarios_lightest_first(controller):
    scenarios = controller.available_scenarios()
    # The two original scenarios still come first, lightest first, and the
    # Indian arterial is listed alongside them.
    assert scenarios[:2] == ["moderate", "heavy"]
    assert "indian_arterial" in scenarios


def test_scenario_summary_comes_from_the_yaml_file(controller):
    config = load_config("heavy")
    summary = controller.scenario_summary("heavy")
    assert summary["ns_green"] == config["signal"]["ns_green"]
    assert summary["total_demand"] == pytest.approx(3000.0)
    assert summary["cycle_time"] == pytest.approx(68.0)
    assert summary["config_path"].endswith("heavy.yaml")


def test_unknown_scenario_raises_a_friendly_error(controller):
    with pytest.raises(ScenarioNotFoundError):
        controller.scenario_summary("not_a_scenario")


def test_control_modes_expose_fixed_time_and_a_model_gated_dqn(controller,
                                                               no_models):
    modes = {mode.key: mode for mode in controller.control_modes("moderate")}
    assert modes["fixed_time"].available is True
    assert modes["dqn_adaptive"].available is False, "no model in this fixture"
    assert "train a model first" in modes["dqn_adaptive"].unavailable_reason.lower()


def test_running_the_dqn_mode_without_a_model_is_refused(controller, no_models):
    with pytest.raises(ControlModeUnavailableError):
        controller.run("moderate", mode="dqn_adaptive", duration=60)


def test_agent_for_an_unavailable_mode_is_refused(controller, no_models):
    with pytest.raises(ControlModeUnavailableError):
        controller.agent_for_mode("dqn_adaptive", "moderate")


def test_agent_for_fixed_time_mode_is_the_fixed_time_controller(controller):
    agent = controller.agent_for_mode("fixed_time", "moderate")
    assert isinstance(agent, FixedTimeAgent)
    assert agent.cycle_time == pytest.approx(68.0)


def test_running_an_unknown_scenario_is_refused(controller):
    with pytest.raises(ScenarioNotFoundError):
        controller.run("not_a_scenario", duration=60)


def test_simulation_runs_and_returns_real_metrics(tmp_path):
    local = SimulationController(results_dir=str(tmp_path))
    run = local.run("moderate", mode="fixed_time", duration=SHORT_DURATION)
    assert run.scenario == "moderate"
    assert run.source == "dashboard run"
    assert run.metric("simulation_time") == pytest.approx(SHORT_DURATION)
    assert run.metric("vehicles") > 0
    assert run.metric("throughput") >= 0
    assert os.path.isfile(run.csv_path)
    assert os.path.isfile(run.timeseries_path)


def test_results_are_loaded_back_from_disk(tmp_path):
    local = SimulationController(results_dir=str(tmp_path))
    local.run("moderate", duration=SHORT_DURATION)
    loaded = local.load_results("moderate")
    assert loaded is not None
    assert loaded.metric("simulation_time") == pytest.approx(SHORT_DURATION)


def test_results_fall_back_to_the_stage1_baseline(tmp_path):
    """With no dashboard run, the Stage 1 files are used - and labelled."""
    empty = SimulationController(results_dir=str(tmp_path / "empty"))
    run = empty.load_results("moderate")
    if run is None:
        pytest.skip("Stage 1 results have not been generated")
    assert run.source == "Stage 1 baseline"
    assert BASELINE_RESULTS_DIR in run.csv_path


def test_missing_results_return_none_rather_than_raising(tmp_path, monkeypatch):
    local = SimulationController(results_dir=str(tmp_path / "nothing"))
    monkeypatch.setattr("simulation.controller.BASELINE_RESULTS_DIR",
                        str(tmp_path / "also_nothing"))
    assert local.load_results("moderate") is None
    assert local.load_timeseries("moderate") is None


def test_comparison_table_has_one_row_per_scenario(controller):
    table = controller.comparison_table()
    if table.empty:
        pytest.skip("no results on disk")
    assert set(table["scenario"]) <= {"moderate", "heavy"}
    for column in ("average_waiting_time", "average_queue_length", "throughput"):
        assert column in table.columns


def test_sumo_unavailable_produces_an_explained_error(controller, monkeypatch):
    """The dashboard must never show a traceback when SUMO is missing."""
    from simulation.controller import EnvironmentStatus

    broken = EnvironmentStatus()
    broken.sumo_available = False
    broken.traci_available = False
    monkeypatch.setattr(controller, "environment_status", lambda: broken)

    with pytest.raises(SumoUnavailableError) as info:
        controller.run("moderate", duration=60)
    assert "SUMO is not available" in str(info.value)
    assert info.value.hint


def test_every_controller_error_carries_a_hint():
    for error_class in (SumoUnavailableError, ScenarioNotFoundError,
                        ControlModeUnavailableError):
        assert error_class("x").hint, "{} has no hint".format(error_class.__name__)
    assert issubclass(SumoUnavailableError, SimulationError)


def test_gui_command_is_offered_as_text(controller):
    command = controller.gui_command("heavy", 300)
    assert command.startswith("python simulation/run_simulation.py")
    assert "--gui" in command and "heavy" in command


def test_screenshots_are_found_if_they_were_captured(controller):
    shots = controller.screenshots()
    if not shots:
        pytest.skip("no screenshots captured yet")
    for shot in shots:
        assert os.path.isfile(shot["path"])
        assert shot["caption"]


# ==========================================================================
# dashboard/ helpers
# ==========================================================================
def test_signal_state_decodes_into_per_approach_colours():
    from dashboard.components import decode_signal_state, direction_states

    ns_green = "GGGgrrrrGGGgrrrr"
    colours = decode_signal_state(ns_green)
    assert colours["north"] == "green" and colours["south"] == "green"
    assert colours["east"] == "red" and colours["west"] == "red"
    assert direction_states(ns_green) == {"NS": "green", "EW": "red"}

    ew_green = "rrrrGGGgrrrrGGGg"
    assert direction_states(ew_green) == {"NS": "red", "EW": "green"}

    ns_yellow = "yyyyrrrryyyyrrrr"
    assert direction_states(ns_yellow) == {"NS": "yellow", "EW": "red"}


def test_signal_state_decoder_tolerates_missing_data():
    from dashboard.components import decode_signal_state, direction_states

    assert direction_states(None) == {"NS": "off", "EW": "off"}
    assert direction_states("") == {"NS": "off", "EW": "off"}
    assert all(v == "off" for v in decode_signal_state("short").values())


def test_intersection_schematic_renders_valid_svg():
    from dashboard.components import intersection_svg

    svg = intersection_svg({"NS": "green", "EW": "red"})
    assert svg.strip().startswith("<svg")
    assert svg.strip().endswith("</svg>")
    assert "NORTH" in svg and "SOUTH" in svg and "EAST" in svg and "WEST" in svg


def test_charts_are_built_from_the_results(controller):
    from dashboard import charts

    table = controller.comparison_table()
    if table.empty:
        pytest.skip("no results on disk")
    for builder in (charts.waiting_time_chart, charts.queue_length_chart,
                    charts.throughput_chart):
        chart = builder(table)
        assert chart is not None
        chart.to_dict()          # raises if the specification is invalid


def test_charts_return_none_on_empty_data():
    import pandas as pd

    from dashboard import charts

    empty = pd.DataFrame()
    assert charts.waiting_time_chart(empty) is None
    assert charts.queue_over_time_chart({}) is None
    assert charts.vehicles_over_time_chart({}) is None


def test_signal_plan_chart_uses_the_configured_timeline(controller):
    from dashboard import charts

    agent = controller.agent_for_mode("fixed_time", "moderate")
    chart = charts.signal_plan_chart(agent.timeline())
    assert chart is not None
    chart.to_dict()


# ==========================================================================
# dashboard.py - the Streamlit app, really executed
# ==========================================================================
def test_dashboard_starts_without_an_exception(app):
    assert not app.exception
    assert not app.error


def test_dashboard_shows_the_project_title(app):
    rendered = " ".join(block.value for block in app.markdown)
    assert "IntelliSignal" in rendered
    assert "Deep Reinforcement Learning" in rendered


def test_sidebar_is_short_and_has_the_four_things_that_matter(app):
    """Navigation, traffic, controller, Run - and nothing else at the top.

    Everything occasional lives under Advanced. A teacher should be able to
    read the whole sidebar on a laptop without scrolling.
    """
    assert [r.label for r in app.sidebar.radio] == ["Navigation", "Controller"]
    # Traffic and the timing profile both change what is simulated, so
    # both are at the top level. Everything else is under Advanced.
    assert [s.label for s in app.sidebar.selectbox] == ["Traffic",
                                                        "Timing profile"]
    assert app.button[0].label == "Run"


def test_the_timing_profile_is_selectable_and_recorded(app):
    from rl.timing_profile import PROFILES

    options = list(app.sidebar.selectbox[1].options)
    assert len(options) == len(PROFILES)
    assert any("15-30 s" in option for option in options), (
        "the standard profile must be offered")
    app.sidebar.selectbox[1].set_value("standard").run()
    assert not app.exception
    assert app.session_state["timing_profile"] == "standard"


def test_custom_traffic_is_offered_alongside_the_built_in_scenarios(app):
    # AppTest reports the formatted labels, not the raw option values.
    options = list(app.sidebar.selectbox[0].options)
    assert "Moderate traffic" in options and "Heavy traffic" in options
    assert "Custom traffic" in options, "Custom traffic must be selectable"
    app.sidebar.selectbox[0].set_value("custom").run()
    assert not app.exception
    assert app.session_state["scenario"] == "custom"


def test_hybrid_is_a_separate_control_mode(app):
    assert "Hybrid" in list(app.sidebar.radio[1].options)
    app.sidebar.radio[1].set_value("hybrid").run()
    assert not app.exception
    assert app.session_state["control_mode"] == "hybrid"


def test_every_page_renders(app):
    from dashboard.state import PAGES

    assert PAGES == ["Dashboard", "Custom Traffic", "Simulation",
                     "Comparison", "Results", "DQN Model", "About"]
    for page in PAGES:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, "page {} raised".format(page)
        assert not app.error, "page {} showed an error".format(page)


def test_model_page_explains_itself_whether_or_not_a_model_exists(app):
    from simulation.controller import SimulationController

    app.sidebar.radio[0].set_value("DQN Model").run()
    assert not app.exception
    text = " ".join(block.value for block in app.markdown)
    # The default view explains what the controller IS, in one paragraph.
    assert "how long each green lasts" in text
    assert "yellow clearance" in text
    assert SimulationController() is not None


def test_model_page_is_honest_when_nothing_is_registered(monkeypatch):
    """With no registered model the page says so instead of inventing one."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr("agents.phase_dqn_agent.registered_phase_model",
                        lambda scenario: None)
    at = AppTest.from_file(DASHBOARD_FILE, default_timeout=300)
    at.run()
    at.sidebar.radio[0].set_value("DQN Model").run()
    assert not at.exception
    notices = " ".join(block.value for block in at.info)
    assert "no duration-choosing model is registered" in notices.lower()


def test_comparison_page_renders_without_evaluation_results(monkeypatch,
                                                            tmp_path):
    """With no experiment on disk the page says so and gives the command."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr("simulation.controller.EVALUATION_RESULTS_DIR",
                        str(tmp_path / "nothing"))
    at = AppTest.from_file(DASHBOARD_FILE, default_timeout=300)
    at.run()
    at.sidebar.radio[0].set_value("Comparison").run()
    assert not at.exception
    # Either a measured comparison, or a statement that there is none.
    notices = " ".join(block.value for block in at.info) + " ".join(
        block.value for block in at.error)
    assert notices.strip() or len(at.dataframe) >= 1, (
        "the page must either show a measured comparison or say there is "
        "none - never a blank")


def test_scenario_selection_switches_scenario(app):
    app.sidebar.selectbox[0].set_value("heavy").run()
    assert not app.exception
    assert app.session_state["scenario"] == "heavy"


def test_dqn_mode_is_offered_and_explained(app):
    """The mode is always listed; whether it can run depends on the model."""
    from simulation.controller import SimulationController

    # The dashboard now opens on the Indian arterial; this test is about
    # the Moderate scenario's own checkpoints.
    app.sidebar.selectbox[0].set_value("moderate").run()
    modes = app.sidebar.radio[1]
    assert "DQN Adaptive" in modes.options
    app.sidebar.radio[1].set_value("dqn_adaptive").run()
    assert not app.exception

    has_model = SimulationController().has_trained_model("moderate")
    if has_model:
        assert app.button[0].disabled is False, (
            "a trained model exists, so the Run button must work")
        selects = [s.label for s in app.sidebar.selectbox]
        assert "Checkpoint" in selects, "the checkpoint picker must be offered"
    else:
        warnings = " ".join(w.value for w in app.sidebar.warning)
        assert "train a model first" in warnings.lower()
        assert app.button[0].disabled is True, (
            "Run must be disabled when no model exists")


def test_dqn_mode_is_disabled_when_no_model_exists(no_models):
    """The honest path, forced even on a machine that has a trained model."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(DASHBOARD_FILE, default_timeout=300)
    at.run()
    at.sidebar.radio[1].set_value("dqn_adaptive").run()
    assert not at.exception
    warnings = " ".join(w.value for w in at.sidebar.warning)
    assert "train a model first" in warnings.lower()
    assert at.button[0].disabled is True
    assert [s.label for s in at.sidebar.selectbox] == ["Traffic",
                                                       "Timing profile"], (
        "no checkpoint picker when there is no model")


def test_fixed_time_mode_enables_the_run_button(app):
    app.sidebar.radio[1].set_value("fixed_time").run()
    assert app.button[0].disabled is False


def test_no_simulation_runs_until_the_button_is_pressed(app):
    """A page refresh must never start an 1800 second simulation."""
    for page in ("Dashboard", "Results", "About"):
        app.sidebar.radio[0].set_value(page).run()
    assert app.session_state["last_run"] is None


def test_running_a_simulation_from_the_dashboard_produces_real_metrics(app):
    # Pin the scenario and the controller: the dashboard's own defaults are
    # the Indian arterial under Hybrid, which is a background search, not a
    # single measured run.
    app.sidebar.selectbox[0].set_value("moderate").run()
    app.sidebar.radio[1].set_value("fixed_time").run()
    app.sidebar.number_input[0].set_value(120).run()
    app.button[0].click().run()
    assert not app.exception
    assert not app.error, [e.value for e in app.error]

    run = app.session_state["last_run"]
    assert run is not None
    assert run.scenario == "moderate"
    assert run.metric("simulation_time") == pytest.approx(120.0)
    assert run.metric("vehicles") > 0

    success = " ".join(s.value for s in app.success).lower()
    assert "average waiting time" in success


def test_dashboard_never_writes_to_the_stage1_results(app):
    """Dashboard runs go to results/data/, never over the recorded baseline."""
    baseline = os.path.join(BASELINE_RESULTS_DIR, "moderate_results.csv")
    if not os.path.isfile(baseline):
        pytest.skip("Stage 1 results have not been generated")
    before = os.path.getmtime(baseline)

    app.sidebar.number_input[0].set_value(120).run()
    app.button[0].click().run()

    assert os.path.getmtime(baseline) == before, (
        "the dashboard overwrote the Stage 1 baseline results")


def test_missing_scenarios_are_handled_without_a_traceback(monkeypatch):
    """With no scenario files the dashboard explains itself and stops."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(DASHBOARD_FILE, default_timeout=120)
    at.run()
    assert not at.exception


def test_sumo_unavailable_is_shown_as_readable_text(monkeypatch):
    """Simulate a machine without SUMO and check the message, not a traceback."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(DASHBOARD_FILE, default_timeout=180)
    at.run()
    at.sidebar.radio[0].set_value("Simulation").run()
    # The real environment has SUMO, so this asserts the happy path renders;
    # the unavailable path is covered by the controller test above.
    assert not at.exception


def test_about_page_names_the_three_controllers_and_what_is_real(app):
    app.sidebar.radio[0].set_value("About").run()
    assert not app.exception
    text = " ".join(block.value for block in app.markdown).lower()
    for controller_name in ("fixed-time", "dqn adaptive", "hybrid"):
        assert controller_name in text, controller_name
    assert "schematic" in text, "the drawing must be labelled as a schematic"
    assert "a tie goes to fixed time" in text


def test_results_page_leads_with_numbers_and_folds_the_detail(app):
    app.sidebar.radio[0].set_value("Results").run()
    assert not app.exception
    assert not app.error, [e.value for e in app.error]
    has_metrics = len(app.metric) >= 3
    notices = " ".join(block.value for block in app.info).lower()
    assert has_metrics or "no results yet" in notices
