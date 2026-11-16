"""
IntelliSignal - Dashboard
dashboard/theme.py

Colours, styling and the small formatting helpers the pages share.

The two-colour categorical palette is fixed: one colour per scenario, assigned
by identity and never by rank, so "moderate" is the same blue on every chart
and every page. The pair was checked for colour-vision deficiency separation
(worst-case deltaE 25.0 for protanopia, 30.6 for normal vision against a white
surface), and every bar carries a printed value so the chart never depends on
colour alone.
"""

from __future__ import annotations

from typing import Dict

# --- categorical palette: one colour per scenario --------------------------
SCENARIO_COLORS: Dict[str, str] = {
    "moderate": "#3b6fb6",
    "heavy": "#d98324",
}
SCENARIO_LABELS: Dict[str, str] = {
    "moderate": "Moderate traffic",
    "heavy": "Heavy traffic",
}
FALLBACK_COLOR = "#7a7f85"

# --- signal colours (status, not categorical) ------------------------------
SIGNAL_COLORS = {
    "green": "#2f8f4e",
    "yellow": "#e0a800",
    "red": "#c1392b",
    "off": "#9aa0a6",
}

# --- ink and surfaces ------------------------------------------------------
INK = "#1f2326"
INK_MUTED = "#6b7075"
SURFACE = "#ffffff"
SURFACE_ALT = "#f4f6f8"
BORDER = "#dcdfe3"
OK = "#2f8f4e"
BAD = "#c1392b"
WARN = "#e0a800"


def scenario_color(scenario: str) -> str:
    return SCENARIO_COLORS.get(scenario, FALLBACK_COLOR)


def scenario_label(scenario: str) -> str:
    return SCENARIO_LABELS.get(scenario, scenario.title())


# --------------------------------------------------------------------------
CSS = """
<style>
/* ---- layout ---------------------------------------------------------- */
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }
#MainMenu, footer { visibility: hidden; }

/* ---- project header -------------------------------------------------- */
.is-header { border-bottom: 1px solid #dcdfe3; padding-bottom: 1rem; margin-bottom: 1.4rem; }
.is-header h1 { font-size: 2.0rem; font-weight: 650; letter-spacing: -0.02em;
                margin: 0 0 .25rem 0; color: #1f2326; }
.is-header p  { margin: 0; color: #6b7075; font-size: .95rem; line-height: 1.45; }

/* ---- status lights --------------------------------------------------- */
.is-status-row { display: flex; gap: 1.5rem; flex-wrap: wrap; margin: .2rem 0 .4rem 0; }
.is-status { display: flex; align-items: center; gap: .5rem; font-size: .86rem; color: #1f2326; }
.is-dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; flex: none; }
.is-status .detail { color: #6b7075; font-size: .8rem; }

/* ---- KPI cards ------------------------------------------------------- */
.is-kpi { background: #ffffff; border: 1px solid #dcdfe3; border-radius: 10px;
          padding: 1rem 1.1rem; height: 100%; }
.is-kpi .label { font-size: .74rem; letter-spacing: .06em; text-transform: uppercase;
                 color: #6b7075; margin-bottom: .45rem; }
.is-kpi .value { font-size: 1.85rem; font-weight: 650; color: #1f2326; line-height: 1.1; }
.is-kpi .unit  { font-size: .95rem; font-weight: 500; color: #6b7075; margin-left: .25rem; }
.is-kpi .note  { font-size: .76rem; color: #6b7075; margin-top: .4rem; }

/* ---- section headings ------------------------------------------------ */
.is-section { font-size: 1.08rem; font-weight: 620; color: #1f2326;
              margin: 1.9rem 0 .2rem 0; }
.is-section-sub { font-size: .84rem; color: #6b7075; margin: 0 0 .9rem 0; }

/* ---- badges ---------------------------------------------------------- */
.is-badge { display: inline-block; font-size: .68rem; letter-spacing: .07em;
            text-transform: uppercase; padding: .16rem .5rem; border-radius: 4px;
            font-weight: 600; vertical-align: middle; }
.is-badge.implemented { background: #e6f2ea; color: #1f6b3b; border: 1px solid #bfdfcb; }
.is-badge.planned     { background: #f2f3f5; color: #5b6166; border: 1px solid #dcdfe3; }
.is-badge.baseline    { background: #eaf0f9; color: #2b5285; border: 1px solid #c6d6ec; }

/* ---- panels ---------------------------------------------------------- */
.is-panel { background: #f4f6f8; border: 1px solid #dcdfe3; border-radius: 10px;
            padding: 1rem 1.15rem; }
.is-panel h4 { margin: 0 0 .5rem 0; font-size: .95rem; font-weight: 620; color: #1f2326; }
.is-panel p  { margin: 0; font-size: .86rem; color: #4a4f54; line-height: 1.5; }

/* ---- signal readout -------------------------------------------------- */
.is-signal { display: flex; gap: 1.6rem; align-items: center; }
.is-signal .lamp { display: flex; align-items: center; gap: .55rem;
                   font-size: .95rem; font-weight: 560; color: #1f2326; }
.is-signal .bulb { width: 15px; height: 15px; border-radius: 50%;
                   border: 2px solid rgba(0,0,0,.12); }

/* ---- sidebar --------------------------------------------------------- */
[data-testid="stSidebar"] { background: #f4f6f8; }
.is-brand { font-size: 1.12rem; font-weight: 700; letter-spacing: .02em;
            color: #1f2326; margin-bottom: .1rem; }
.is-brand-sub { font-size: .72rem; color: #6b7075; margin-bottom: 1.1rem;
                line-height: 1.35; }
.is-side-label { font-size: .72rem; letter-spacing: .07em; text-transform: uppercase;
                 color: #6b7075; margin: 1.1rem 0 .3rem 0; font-weight: 600; }

/* ---- misc ------------------------------------------------------------ */
.is-caption { font-size: .78rem; color: #6b7075; margin-top: .35rem; }
hr.is-rule { border: none; border-top: 1px solid #dcdfe3; margin: 1.6rem 0 0 0; }
</style>
"""


# --------------------------------------------------------------------------
def fmt(value, digits: int = 2, dash: str = "—") -> str:
    """Format a number for display, tolerating None and non-numbers."""
    if value is None:
        return dash
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if digits == 0:
        return "{:,.0f}".format(number)
    return "{:,.{d}f}".format(number, d=digits)
