"""
IntelliSignal
dashboard/ui.py

The look of the dashboard, and the small pieces every page is built from.

One stylesheet, one set of colours, one way of showing a number. A page
asks for a metric tile or a verdict and gets the same thing everywhere,
so the interface reads as one instrument rather than six screens that
happen to be in the same app.

Three rules are enforced here rather than left to each page:

  * **A number is never shown without its unit and its meaning.** Every
    tile takes a help string, and the ones that matter are written out in
    `METRIC_HELP` so that the same metric is explained the same way on
    every page.
  * **Good and bad have fixed colours.** Lower delay is better, so an
    improvement is green whether it appears as a percentage, a bar or a
    sentence.
  * **Nothing claims more than it measured.** `verdict` renders the three
    states - confirmed, unverified, failed - and refuses to show a plain
    tick for something that was not checked.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import streamlit as st

# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------
INK = "#10151c"
MUTED = "#5b6775"
FAINT = "#8b97a5"
LINE = "#dde3ea"
PANEL = "#ffffff"
CANVAS = "#f3f5f8"

BRAND = "#1f5fa9"
BRAND_SOFT = "#e8f0fa"
GOOD = "#13795b"
GOOD_SOFT = "#e2f3ec"
WARN = "#9a6508"
WARN_SOFT = "#fdf1dc"
BAD = "#b0342c"
BAD_SOFT = "#fbe9e7"

#: North-south and east-west keep the same two colours on every chart,
#: every table and the junction itself.
NS_COLOR = "#2f6fb5"
EW_COLOR = "#c2763a"

SIGNAL_COLORS = {"NG": "#1f9d55", "NY": "#d9a211", "EG": "#1f9d55",
                 "EY": "#d9a211", "RR": "#c0392b", "??": "#8b97a5"}

CONTROLLER_COLORS = {
    "dqn": BRAND, "fixed_time": "#7a8795", "actuated": "#8a5fb0",
    "max_pressure": "#b07f3f", "hybrid": "#157f74",
}

VEHICLE_COLORS = {"motorcycle": "#e0923c", "car": "#3d7bbf",
                  "auto": "#49a66f", "bus": "#b5484a"}


# ---------------------------------------------------------------------------
METRIC_HELP = {
    "delay_per_requested": (
        "Total delay per requested vehicle, in seconds. Time lost on the "
        "road plus time spent unable to enter, divided by every vehicle "
        "the traffic asked for. Lower is better. This is the headline "
        "number because a signal cannot improve it by refusing traffic."),
    "waiting_per_vehicle": (
        "Mean waiting time per vehicle, in seconds - how long the average "
        "vehicle spent stopped. Lower is better. Kept because it is the "
        "measure the earlier stages of this project reported."),
    "completed": (
        "Vehicles that finished their journey through the junction before "
        "the run ended. Higher is better."),
    "waiting_to_enter": (
        "Vehicles the traffic asked for that never got onto the network, "
        "because the approach was full. Zero is what you want: anything "
        "above it means the junction turned traffic away."),
    "queue_mean": (
        "Average number of vehicles halted on the eight approach lanes, "
        "over every second of the run."),
    "mean_green": "Average length of the greens that were served, in seconds.",
    "pcu": (
        "Passenger car units. A motorcycle is 0.25, a car 1.0, an "
        "auto-rickshaw 0.5 and a bus 3.0, so sixty motorcycles are not "
        "sixty cars' worth of road space."),
}


# ---------------------------------------------------------------------------
def inject_css() -> None:
    """The one stylesheet. Called once per page render."""
    st.markdown(
        """
        <style>
          :root {
            --ink: %(ink)s; --muted: %(muted)s; --faint: %(faint)s;
            --line: %(line)s; --panel: %(panel)s; --canvas: %(canvas)s;
            --brand: %(brand)s; --brand-soft: %(brand_soft)s;
            --good: %(good)s; --good-soft: %(good_soft)s;
            --warn: %(warn)s; --warn-soft: %(warn_soft)s;
            --bad: %(bad)s; --bad-soft: %(bad_soft)s;
          }
          .stApp { background: var(--canvas); }
          .block-container { padding-top: 3.2rem; padding-bottom: 3rem;
                             max-width: 1480px; }
          h1, h2, h3, h4 { color: var(--ink); letter-spacing: -0.01em; }
          h1 { font-size: 1.65rem; font-weight: 650; }
          h2 { font-size: 1.22rem; font-weight: 620; margin-top: .4rem; }
          h3 { font-size: 1.02rem; font-weight: 600; }
          p, li, label, .stMarkdown { color: var(--ink); }

          /* ---- masthead ---- */
          .is-mast { display:flex; align-items:center; gap:.85rem;
                     padding:.55rem .95rem; background:var(--panel);
                     border:1px solid var(--line); border-radius:12px;
                     margin-bottom:.9rem; }
          .is-mast-mark { width:34px; height:34px; border-radius:9px;
                          background:var(--brand); color:#fff; flex:0 0 auto;
                          display:flex; align-items:center; justify-content:center;
                          font-weight:700; font-size:.95rem; letter-spacing:-.02em; }
          .is-mast-title { font-weight:660; font-size:1.02rem; line-height:1.2; }
          .is-mast-sub { color:var(--muted); font-size:.78rem; }
          .is-mast-spacer { flex:1 1 auto; }

          /* ---- panels ---- */
          .is-panel { background:var(--panel); border:1px solid var(--line);
                      border-radius:12px; padding:1rem 1.1rem; }
          .is-panel + .is-panel { margin-top:.7rem; }
          .is-panel-title { font-weight:620; font-size:.95rem; margin-bottom:.1rem; }
          .is-panel-sub { color:var(--muted); font-size:.8rem; }

          /* ---- metric tiles ---- */
          .is-tiles { display:flex; gap:.6rem; flex-wrap:wrap; }
          .is-tile { flex:1 1 150px; background:var(--panel);
                     border:1px solid var(--line); border-radius:11px;
                     padding:.6rem .8rem .65rem; min-width:140px; }
          .is-tile-k { color:var(--muted); font-size:.72rem; font-weight:560;
                       text-transform:uppercase; letter-spacing:.04em; }
          .is-tile-v { font-size:1.5rem; font-weight:660; line-height:1.15;
                       color:var(--ink); font-variant-numeric:tabular-nums; }
          .is-tile-u { font-size:.8rem; font-weight:500; color:var(--muted);
                       margin-left:.12rem; }
          .is-tile-n { font-size:.75rem; color:var(--muted); margin-top:.1rem; }
          .is-tile.good { border-left:3px solid var(--good); }
          .is-tile.bad  { border-left:3px solid var(--bad); }
          .is-tile.lead { border-left:3px solid var(--brand); }

          /* ---- badges ---- */
          .is-badge { display:inline-flex; align-items:center; gap:.3rem;
                      padding:.12rem .5rem; border-radius:99px;
                      font-size:.74rem; font-weight:570; white-space:nowrap; }
          .is-badge.ok   { background:var(--good-soft); color:var(--good); }
          .is-badge.warn { background:var(--warn-soft); color:var(--warn); }
          .is-badge.bad  { background:var(--bad-soft);  color:var(--bad); }
          .is-badge.info { background:var(--brand-soft);color:var(--brand); }
          .is-badge.flat { background:#eef1f5; color:var(--muted); }

          /* ---- rule strip ---- */
          .is-rules { display:flex; gap:.4rem; flex-wrap:wrap;
                      margin:.1rem 0 .85rem; }
          .is-rule { background:var(--panel); border:1px solid var(--line);
                     border-radius:8px; padding:.3rem .6rem; font-size:.76rem;
                     color:var(--muted); }
          .is-rule b { color:var(--ink); font-weight:600; }

          /* ---- callouts ---- */
          .is-note { border-radius:10px; padding:.65rem .85rem; font-size:.86rem;
                     border:1px solid transparent; }
          .is-note.ok   { background:var(--good-soft); border-color:#bfe3d4;
                          color:#0d5f47; }
          .is-note.warn { background:var(--warn-soft); border-color:#f0dcb4;
                          color:#7d5207; }
          .is-note.bad  { background:var(--bad-soft);  border-color:#f2c8c3;
                          color:#8f2a23; }
          .is-note.info { background:var(--brand-soft);border-color:#c9ddf3;
                          color:#174c88; }
          .is-note b { font-weight:640; }

          /* ---- check list ---- */
          .is-check { display:flex; gap:.55rem; align-items:flex-start;
                      padding:.34rem 0; border-bottom:1px dashed var(--line);
                      font-size:.84rem; }
          .is-check:last-child { border-bottom:none; }
          .is-check-m { flex:0 0 auto; width:1.1rem; text-align:center;
                        font-weight:700; }
          .is-check-m.ok { color:var(--good); }
          .is-check-m.bad { color:var(--bad); }
          .is-check-m.wait { color:var(--faint); }
          .is-check-t b { font-weight:600; }
          .is-check-t span { color:var(--muted); }

          /* ---- misc ---- */
          .is-kv { font-size:.84rem; color:var(--muted); }
          .is-kv b { color:var(--ink); font-weight:600; }
          .is-foot { color:var(--faint); font-size:.74rem; margin-top:.35rem; }
          div[data-testid="stMetricValue"] { font-variant-numeric:tabular-nums; }
          section[data-testid="stSidebar"] { background:var(--panel);
                                             border-right:1px solid var(--line); }
          section[data-testid="stSidebar"] .block-container { padding-top:1.1rem; }
          .stTabs [data-baseweb="tab-list"] { gap:.15rem; }
          .stTabs [data-baseweb="tab"] { font-size:.88rem; }
        </style>
        """ % {"ink": INK, "muted": MUTED, "faint": FAINT, "line": LINE,
               "panel": PANEL, "canvas": CANVAS, "brand": BRAND,
               "brand_soft": BRAND_SOFT, "good": GOOD, "good_soft": GOOD_SOFT,
               "warn": WARN, "warn_soft": WARN_SOFT, "bad": BAD,
               "bad_soft": BAD_SOFT},
        unsafe_allow_html=True)


# ---------------------------------------------------------------------------
def esc(text: Any) -> str:
    """Markup-safe text. Everything user-supplied goes through this."""
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _BOLD(match) -> str:
    return "<b>" + match.group(1) + "</b>"


def rich(text: Any) -> str:
    """Markup-safe text in which **this** is still bold.

    The notes on these pages are sentences, and a sentence that needs one
    word emphasised should not have to be built out of HTML by its caller.
    Everything else is escaped first, so the emphasis is the only markup
    that survives.
    """
    import re

    return re.sub("[*][*](.+?)[*][*]", _BOLD, esc(text))


def masthead(title: str, subtitle: str, right: str = "") -> None:
    st.markdown(
        '<div class="is-mast"><div class="is-mast-mark">IS</div>'
        '<div><div class="is-mast-title">{}</div>'
        '<div class="is-mast-sub">{}</div></div>'
        '<div class="is-mast-spacer"></div><div>{}</div></div>'.format(
            esc(title), esc(subtitle), right),
        unsafe_allow_html=True)


def badge(text: str, tone: str = "flat") -> str:
    """Markup for an inline badge. Returned, not written."""
    return '<span class="is-badge {}">{}</span>'.format(tone, esc(text))


def rules_strip(items: Sequence[Sequence[str]]) -> None:
    """The small strip of standing facts: timing rules, fleet, model."""
    chips = "".join('<div class="is-rule"><b>{}</b> {}</div>'.format(
        esc(name), esc(value)) for name, value in items)
    st.markdown('<div class="is-rules">{}</div>'.format(chips),
                unsafe_allow_html=True)


def note(text: str, tone: str = "info", title: str = "") -> None:
    head = "<b>{}</b><br>".format(esc(title)) if title else ""
    st.markdown('<div class="is-note {}">{}{}</div>'.format(
        tone, head, rich(text)), unsafe_allow_html=True)


def fmt(value: Any, digits: int = 2, dash: str = "--") -> str:
    if value is None:
        return dash
    try:
        number = float(value)
    except (TypeError, ValueError):
        return esc(value)
    if number != number:                                  # NaN
        return dash
    return "{:,.{}f}".format(number, digits)


# ---------------------------------------------------------------------------
def tile(label: str, value: Any, unit: str = "", note_text: str = "",
         tone: str = "", digits: int = 2) -> str:
    """Markup for one metric tile."""
    shown = fmt(value, digits) if isinstance(value, (int, float)) else esc(value)
    return (
        '<div class="is-tile {tone}"><div class="is-tile-k">{k}</div>'
        '<div class="is-tile-v">{v}<span class="is-tile-u">{u}</span></div>'
        '<div class="is-tile-n">{n}</div></div>'.format(
            tone=tone, k=esc(label), v=shown, u=esc(unit), n=esc(note_text)))


def tiles(items: Sequence[Dict[str, Any]]) -> None:
    """A row of metric tiles. Each item is the keyword arguments of `tile`."""
    st.markdown('<div class="is-tiles">{}</div>'.format(
        "".join(tile(**item) for item in items)), unsafe_allow_html=True)


def checks(rows: Sequence[Dict[str, Any]]) -> None:
    """The vigilance list: what was checked, and what the answer was.

    A row with `ok` set to None was not checked, and says so rather than
    showing a tick.
    """
    out = []
    for row in rows:
        state = row.get("ok")
        mark, klass = ("?", "wait") if state is None else (
            ("+", "ok") if state else ("!", "bad"))
        out.append(
            '<div class="is-check"><div class="is-check-m {k}">{m}</div>'
            '<div class="is-check-t"><b>{t}</b><br><span>{d}</span></div>'
            '</div>'.format(k=klass, m=mark, t=esc(row.get("check", "")),
                            d=esc(row.get("detail", ""))))
    st.markdown("".join(out), unsafe_allow_html=True)


def panel_open(title: str = "", subtitle: str = "") -> None:
    head = ""
    if title:
        head += '<div class="is-panel-title">{}</div>'.format(esc(title))
    if subtitle:
        head += '<div class="is-panel-sub">{}</div>'.format(esc(subtitle))
    st.markdown('<div class="is-panel">{}'.format(head), unsafe_allow_html=True)


def panel_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def kv(label: str, value: Any) -> None:
    st.markdown('<div class="is-kv">{} <b>{}</b></div>'.format(
        esc(label), esc(value)), unsafe_allow_html=True)


def footnote(text: str) -> None:
    st.markdown('<div class="is-foot">{}</div>'.format(esc(text)),
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
def change_badge(change_percent: Optional[float], lower_is_better: bool = True
                 ) -> str:
    """A signed change, coloured by whether it is an improvement."""
    if change_percent is None:
        return badge("baseline", "flat")
    better = change_percent < 0 if lower_is_better else change_percent > 0
    if abs(change_percent) < 0.05:
        return badge("no change", "flat")
    return badge("{:+.1f} %".format(change_percent), "ok" if better else "bad")


def verdict_badge(result) -> str:
    """One badge summarising whether a run can be believed."""
    if not result.audit:
        return badge("not audited", "flat")
    if not result.audit.get("compliant", True):
        return badge("timing violated", "bad")
    if not result.metrics.get("measurements_agree", True):
        return badge("measurements disagree", "warn")
    if result.metrics.get("collisions", 0):
        return badge("collisions", "bad")
    return badge("audited clean", "ok")
