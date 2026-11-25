"""
The IntelliSignal dashboard.

**The rule this package holds to: it talks to `simulation/engine.py` and
to nothing else.** Not to SUMO, not to the harness, not to a registry, not
to a checkpoint. The engine runs every controller through
`evaluation/harness.py`, which is what makes the numbers on these pages
the same numbers as the ones in the reports - and keeping the rule is
what stops a second, slightly different measurement path growing inside
the interface, which is exactly what happened to the dashboard this one
replaced.

    dashboard.py          the shell: the sidebar, the page list, the
                          standing facts every page shows
    views/                one module per page, each with `render(ctx)`
    views/common.py       what more than one page shows: a run's
                          headline numbers, its checks, its decisions
    ui.py                 the stylesheet and the small pieces - tiles,
                          badges, callouts, the check list
    junction.py           the junction drawn from a recorded run
    charts.py             every chart
    jobs.py               running simulations on a thread so the page
                          does not freeze
    state.py              what survives a click
    custom_form.py        describing traffic lane by lane
    hybrid_panel.py       Hybrid's progress and its decision

The two exceptions, both deliberate: `custom_form.py` builds a workload
through `custom_traffic/`, and `hybrid_panel.py` renders a decision made
by `simulation/hybrid_job.py`. Neither measures anything.
"""
