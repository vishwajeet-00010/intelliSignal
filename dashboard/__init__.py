"""
IntelliSignal - the presentation layer.

The Streamlit dashboard. It talks to `simulation.controller.SimulationController`
and to nothing else: it never imports SUMO, TraCI or an agent directly. That
separation is what allows a DQN controller to be added later without touching
any of this code.

    dashboard.py            the entry point - `streamlit run dashboard.py`
    dashboard/theme.py      colours, CSS and shared formatting
    dashboard/state.py      Streamlit session state
    dashboard/controls.py   the sidebar
    dashboard/components.py status lights, KPI cards, intersection, signal
    dashboard/charts.py     the Altair charts
"""
