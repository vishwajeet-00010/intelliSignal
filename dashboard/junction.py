"""
IntelliSignal
dashboard/junction.py

The junction, drawn from the run that was actually measured.

Every vehicle on this canvas was in the simulation whose numbers are on
the screen beside it: `simulation/recorder.py` wrote down where each one
was, once a second, while `evaluation/harness.py` measured the run. The
signal shown is the state SUMO displayed, not a re-derivation of what the
controller meant to do - which is why the picture can disagree with the
controller, and is worth looking at when it does.

The road itself is read from `sumo/network/intersection.net.xml`: two
lanes in and two out on each arm, 3.2 m wide, with the stop lines where
SUMO puts them.

It renders as one self-contained HTML component - no plotting library, no
network access - so it works on a projector in a room with no internet.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import streamlit as st
import streamlit.components.v1 as components

from dashboard.ui import EW_COLOR, NS_COLOR, VEHICLE_COLORS

# --- geometry, from the network file --------------------------------------
CENTRE = 200.0
LANE_WIDTH = 3.2
HALF_ROAD = 2 * LANE_WIDTH          # two lanes each way
JUNCTION_HALF = 10.4                # the stop lines sit here
WORLD = 400.0

#: Lane centre lines, as the network has them.
LANE_OFFSETS = {
    "N_in": (195.2, 198.4), "S_in": (204.8, 201.6),
    "E_in": (204.8, 201.6), "W_in": (195.2, 198.4),
}

PHASE_LABEL = {"NG": "North-South green", "NY": "North-South amber",
               "EG": "East-West green", "EY": "East-West amber",
               "RR": "All red", "??": "Unknown"}


def _palette(classes) -> Dict[str, str]:
    return {name: VEHICLE_COLORS.get(name, "#6f7b8a") for name in classes}


# ---------------------------------------------------------------------------
def replay_view(replay: Dict[str, Any], height: int = 620,
                key: str = "junction", title: str = "",
                autoplay: bool = True) -> None:
    """Draw the replay of one run, with its own transport controls."""
    if not replay or not replay.get("steps"):
        st.info("This run was measured without recording the junction, so "
                "there is nothing to replay.")
        return

    payload = {
        "replay": replay,
        "colors": _palette(replay.get("classes") or ["car"]),
        "geometry": {
            "centre": CENTRE, "laneWidth": LANE_WIDTH,
            "halfRoad": HALF_ROAD, "junctionHalf": JUNCTION_HALF,
            "world": WORLD, "lanes": LANE_OFFSETS,
        },
        "phaseLabel": PHASE_LABEL,
        "ns": NS_COLOR, "ew": EW_COLOR,
        "title": title, "autoplay": bool(autoplay),
    }
    chrome = 70 if _is_still(replay) else 150
    html = (_HTML.replace("__PAYLOAD__",
                          json.dumps(payload, separators=(",", ":")))
            .replace("__STAGE__", str(max(int(height) - chrome, 220)))
            .replace("__UID__", "j" + str(abs(hash(key)) % 10 ** 8)))
    if _is_still(replay):
        # One frame has nothing to play, scrub or read a timeline from.
        html = html.replace('<div class="ribbon">', '<div hidden>')
        html = html.replace('<div class="ctrl">', '<div hidden>')
    # In its own frame. `st.html` renders the markup but does not run the
    # script - a canvas needs the script - so this stays a component.
    components.html(_DOCUMENT.replace("__FRAGMENT__", html),
                    height=height, scrolling=False)


# ---------------------------------------------------------------------------
def still(signal_code: str = "NG", queues: Optional[Dict[str, int]] = None,
          height: int = 300, label: str = "") -> None:
    """The junction with no traffic on it - for a page with no run yet."""
    queues = queues or {"N": 0, "S": 0, "E": 0, "W": 0}
    replay = {
        "version": 1, "step_length": 1.0, "steps": 1, "position_scale": 10,
        "classes": ["car"], "vehicle_class": [], "frames": [[]],
        "signal": [signal_code],
        "queues": {k: [int(v)] for k, v in queues.items()},
        "present": {k: [int(v)] for k, v in queues.items()},
    }
    replay_view(replay, height=height, key="still", title=label,
                autoplay=False)


def _is_still(replay) -> bool:
    return int(replay.get("steps") or 0) <= 1


# ---------------------------------------------------------------------------
#: The fragment is written so that it would also work inline: every rule
#: is scoped to one generated id and the script is an IIFE that touches
#: nothing outside it. It is served in a frame because Streamlit renders
#: injected markup without executing its scripts.
_DOCUMENT = """<!doctype html>
<html><head><meta charset="utf-8"></head>
<body style="margin:0;background:transparent">
__FRAGMENT__
</body></html>
"""

_HTML = r"""
<div id="__UID__" class="is-junction">
<style>
  #__UID__{ --ink:#10151c; --muted:#5b6775; --line:#dde3ea; --panel:#ffffff; }
  #__UID__ *{ box-sizing:border-box; }
  #__UID__{ margin:0; font-family:"Segoe UI",system-ui,-apple-system,sans-serif;
        color:var(--ink); background:transparent; }
  #__UID__ > .wrap{ background:var(--panel); border:1px solid var(--line);
         border-radius:12px; padding:.7rem .8rem .55rem; }
  #__UID__ .top{ display:flex; align-items:center; gap:.6rem; margin-bottom:.45rem; }
  #__UID__ .ttl{ font-weight:620; font-size:.92rem; }
  #__UID__ .sp{ flex:1 1 auto; }
  #__UID__ .state{ font-size:.78rem; font-weight:600; padding:.14rem .55rem;
          border-radius:99px; background:#eef1f5; color:var(--muted); }
  #__UID__ .stage{ display:flex; gap:.7rem; align-items:stretch; height:__STAGE__px; }
  #__UID__ .board{ flex:1 1 auto; min-width:0; }
  #__UID__ #__UID__-cv{ border-radius:10px; background:#eceff3; display:block; }
  #__UID__ .side{ flex:0 0 164px; display:flex; flex-direction:column; gap:.4rem;
         overflow:auto; }
  #__UID__ .card{ border:1px solid var(--line); border-radius:9px; padding:.4rem .55rem; }
  #__UID__ .card h4{ margin:0 0 .18rem; font-size:.68rem; text-transform:uppercase;
            letter-spacing:.04em; color:var(--muted); font-weight:600; }
  #__UID__ .row{ display:flex; justify-content:space-between; font-size:.79rem;
        padding:.08rem 0; font-variant-numeric:tabular-nums; }
  #__UID__ .row b{ font-weight:620; }
  #__UID__ .bar{ height:5px; border-radius:3px; background:#e4e8ee; overflow:hidden;
        margin-top:.1rem; }
  #__UID__ .bar i{ display:block; height:100%; background:#2f6fb5; }
  #__UID__ .legend{ display:flex; flex-wrap:wrap; gap:.3rem; }
  #__UID__ .lg{ display:flex; align-items:center; gap:.22rem; font-size:.71rem;
       color:var(--muted); }
  #__UID__ .sw{ width:9px; height:9px; border-radius:2px; }
  #__UID__ .ribbon{ margin-top:.5rem; }
  #__UID__ .ribbon canvas{ background:transparent; border-radius:4px; display:block; }
  #__UID__ .ctrl{ display:flex; align-items:center; gap:.45rem; margin-top:.4rem; }
  #__UID__ button{ font:inherit; font-size:.78rem; font-weight:560; cursor:pointer;
          border:1px solid var(--line); background:#fff; color:var(--ink);
          border-radius:7px; padding:.22rem .6rem; }
  #__UID__ button:hover{ background:#f3f6fa; }
  #__UID__ button.p{ background:#1f5fa9; border-color:#1f5fa9; color:#fff; min-width:62px; }
  #__UID__ input[type=range]{ flex:1 1 auto; accent-color:#1f5fa9; }
  #__UID__ .clock{ font-size:.78rem; color:var(--muted); font-variant-numeric:tabular-nums;
          min-width:92px; text-align:right; }
</style>

<div class="wrap">
  <div class="top">
    <div class="ttl" id="__UID__-title"></div><div class="sp"></div>
    <div class="state" id="__UID__-state">--</div>
  </div>
  <div class="stage">
    <div class="board"><canvas id="__UID__-cv" width="1120" height="700"></canvas></div>
    <div class="side">
      <div class="card"><h4>Queueing now</h4><div id="__UID__-queues"></div></div>
      <div class="card"><h4>On the junction</h4>
        <div class="row"><span>vehicles</span><b id="__UID__-nveh">0</b></div>
        <div class="row"><span>halted</span><b id="__UID__-nhalt">0</b></div>
      </div>
      <div class="card"><h4>Fleet</h4><div class="legend" id="__UID__-legend"></div></div>
      <div class="card"><h4>View</h4>
        <input type="range" id="__UID__-zoom" min="60" max="200" step="10" value="110"
               style="width:100%">
        <div class="row"><span>metres each way</span><b id="__UID__-zoomv">110</b></div>
      </div>
    </div>
  </div>
  <div class="ribbon"><canvas id="__UID__-rb" width="1200" height="26"></canvas></div>
  <div class="ctrl">
    <button class="p" id="__UID__-play">Play</button>
    <button id="__UID__-back">&#8592;</button><button id="__UID__-fwd">&#8594;</button>
    <input type="range" id="__UID__-scrub" min="0" value="0">
    <button id="__UID__-speed">2&times;</button>
    <div class="clock" id="__UID__-clock">0 s</div>
  </div>
</div>
<script>
(function(){
const ROOT = document.getElementById("__UID__");
if (!ROOT || ROOT.dataset.ready) return;
ROOT.dataset.ready = '1';
try {

const P = __PAYLOAD__;
const R = P.replay, G = P.geometry;
const SCALE = R.position_scale || 10;
const STEP = R.step_length || 1;
const N = R.steps || (R.signal || []).length;
const APPR = ["N","S","E","W"];
const APPR_NAME = {N:"North", S:"South", E:"East", W:"West"};
/* The ribbon uses the two direction colours for green, so a glance shows
   which way the junction was serving, not just that something was green. */
const SIG_FILL = {NG:P.ns, NY:"#e0b43c", EG:P.ew, EY:"#e0b43c",
                  RR:"#c0392b", "??":"#8b97a5"};

const cv = ROOT.querySelector("#__UID__-cv"), ctx = cv.getContext("2d");
const rb = ROOT.querySelector("#__UID__-rb"), rctx = rb.getContext("2d");
const board = ROOT.querySelector(".board");
const ribbon = ROOT.querySelector(".ribbon");
const scrub = ROOT.querySelector("#__UID__-scrub");
ROOT.querySelector("#__UID__-title").textContent = P.title || "";
scrub.max = Math.max(N - 1, 0);

/* The default view is chosen from the run itself: far enough out that
   the longest queue of the whole run fits, and no further. */
function defaultSpan(){
  /* The longest queue of the whole run, turned into metres: a stopped
     vehicle and its gap take about six metres on this fleet. */
  let longest = 0;
  ["N","S","E","W"].forEach(a => {
    (R.queues[a] || []).forEach(q => { if (q > longest) longest = q; });
  });
  const metres = G.junctionHalf + longest * 6 + 25;
  return Math.min(190, Math.max(60, Math.ceil(metres / 10) * 10));
}

let frame = 0, playing = false, speed = 2, span = defaultSpan(), last = 0;

/* ---- legend ---------------------------------------------------------- */
(function legend(){
  const host = ROOT.querySelector("#__UID__-legend");
  (R.classes || []).forEach(name => {
    const d = document.createElement("div"); d.className = "lg";
    d.innerHTML = '<span class="sw" style="background:' +
      (P.colors[name] || "#6f7b8a") + '"></span>' + name;
    host.appendChild(d);
  });
})();

/* ---- which group each approach belongs to ---------------------------- */
function approachSignal(code, appr){
  const ns = (appr === "N" || appr === "S");
  if (code === "NG") return ns ? "g" : "r";
  if (code === "NY") return ns ? "y" : "r";
  if (code === "EG") return ns ? "r" : "g";
  if (code === "EY") return ns ? "r" : "y";
  return "r";
}
const LIGHT = {g:"#1f9d55", y:"#d9a211", r:"#c0392b"};

/* ---- world -> screen --------------------------------------------------
   The canvas is wider than it is tall, so the vertical span is the one
   the zoom control sets and the horizontal span follows the aspect. A
   metre is the same length in both directions; nothing is stretched. */
function tx(){
  const px = cv.height / (2 * span);
  const spanX = cv.width / (2 * px);
  return { px, spanX, spanY: span,
           map:(wx, wy) => [ (wx - (G.centre - spanX)) * px,
                             ((G.centre + span) - wy) * px ] };
}

function drawRoads(T){
  const m = T.map, px = T.px, C = G.centre, H = G.halfRoad, J = G.junctionHalf;
  const SX = T.spanX, SY = T.spanY;
  ctx.fillStyle = "#eceff3"; ctx.fillRect(0, 0, cv.width, cv.height);
  /* carriageways */
  ctx.fillStyle = "#4c555f";
  let a = m(C - SX, C + H), b = m(C + SX, C - H);
  ctx.fillRect(a[0], a[1], b[0]-a[0], b[1]-a[1]);
  a = m(C - H, C + SY); b = m(C + H, C - SY);
  ctx.fillRect(a[0], a[1], b[0]-a[0], b[1]-a[1]);

  /* centre lines */
  ctx.strokeStyle = "#f2d675"; ctx.lineWidth = Math.max(1.4, 0.35*px);
  ctx.beginPath();
  let p1 = m(C - SX, C), p2 = m(C - J, C); ctx.moveTo(p1[0],p1[1]); ctx.lineTo(p2[0],p2[1]);
  p1 = m(C + J, C); p2 = m(C + SX, C);     ctx.moveTo(p1[0],p1[1]); ctx.lineTo(p2[0],p2[1]);
  p1 = m(C, C - SY); p2 = m(C, C - J);     ctx.moveTo(p1[0],p1[1]); ctx.lineTo(p2[0],p2[1]);
  p1 = m(C, C + J); p2 = m(C, C + SY);     ctx.moveTo(p1[0],p1[1]); ctx.lineTo(p2[0],p2[1]);
  ctx.stroke();

  /* lane divider dashes, between the two lanes of each direction */
  ctx.strokeStyle = "#cfd6dd"; ctx.lineWidth = Math.max(1, 0.2*px);
  ctx.setLineDash([6, 7]);
  [C - G.laneWidth, C + G.laneWidth].forEach(off => {
    ctx.beginPath();
    let q1 = m(C - SX, off), q2 = m(C - J, off);
    ctx.moveTo(q1[0],q1[1]); ctx.lineTo(q2[0],q2[1]);
    q1 = m(C + J, off); q2 = m(C + SX, off);
    ctx.moveTo(q1[0],q1[1]); ctx.lineTo(q2[0],q2[1]);
    q1 = m(off, C - SY); q2 = m(off, C - J);
    ctx.moveTo(q1[0],q1[1]); ctx.lineTo(q2[0],q2[1]);
    q1 = m(off, C + J); q2 = m(off, C + SY);
    ctx.moveTo(q1[0],q1[1]); ctx.lineTo(q2[0],q2[1]);
    ctx.stroke();
  });
  ctx.setLineDash([]);

  /* junction box */
  a = m(C - J, C + J); b = m(C + J, C - J);
  ctx.fillStyle = "#555f69"; ctx.fillRect(a[0],a[1],b[0]-a[0],b[1]-a[1]);

  /* which way is which */
  ctx.fillStyle = "#97a2ae";
  ctx.font = "600 " + Math.max(11, 3.2*px) + "px Segoe UI, sans-serif";
  ctx.textAlign = "center"; ctx.textBaseline = "middle";
  let t = m(C - H - 8, C + SY - 6); ctx.fillText("NORTH", t[0], t[1]);
  t = m(C + H + 8, C - SY + 6);     ctx.fillText("SOUTH", t[0], t[1]);
  ctx.save();
  t = m(C + SX - 6, C + H + 8); ctx.translate(t[0], t[1]); ctx.rotate(Math.PI/2);
  ctx.fillText("EAST", 0, 0); ctx.restore();
  ctx.save();
  t = m(C - SX + 6, C - H - 8); ctx.translate(t[0], t[1]); ctx.rotate(-Math.PI/2);
  ctx.fillText("WEST", 0, 0); ctx.restore();
}

function drawStopLines(T, code){
  const m = T.map, px = T.px, C = G.centre, J = G.junctionHalf, L = G.laneWidth;
  const bars = {
    N: [[C - 2*L, C + J], [C, C + J]],
    S: [[C, C - J], [C + 2*L, C - J]],
    E: [[C + J, C], [C + J, C + 2*L]],
    W: [[C - J, C - 2*L], [C - J, C]],
  };
  /* the pole stands on the kerb to the right of each approach, where
     the driver stopped at the line is looking. */
  const heads = {
    N: [C - 2*L - 3.4, C + J + 3.4], S: [C + 2*L + 3.4, C - J - 3.4],
    E: [C + J + 3.4, C + 2*L + 3.4], W: [C - J - 3.4, C - 2*L - 3.4],
  };
  APPR.forEach(ap => {
    const st = approachSignal(code, ap);
    const [p, q] = bars[ap];
    const s = m(p[0], p[1]), e = m(q[0], q[1]);
    ctx.strokeStyle = LIGHT[st];
    ctx.lineWidth = Math.max(3, 0.9*px);
    ctx.beginPath(); ctx.moveTo(s[0], s[1]); ctx.lineTo(e[0], e[1]); ctx.stroke();

    const h = m(heads[ap][0], heads[ap][1]);
    const r = Math.max(3.4, 1.15*px);
    ctx.beginPath();
    ctx.roundRect ? ctx.roundRect(h[0]-r*1.05, h[1]-r*2.9, r*2.1, r*5.8, r*0.6)
                  : ctx.rect(h[0]-r*1.05, h[1]-r*2.9, r*2.1, r*5.8);
    ctx.fillStyle = "#2b3138"; ctx.fill();
    ["g","y","r"].forEach((lamp, i) => {
      ctx.beginPath();
      ctx.arc(h[0], h[1] + (i-1)*r*1.75, r*0.72, 0, 6.2832);
      ctx.fillStyle = lamp === st ? LIGHT[lamp] : "#3e464f";
      ctx.fill();
    });
  });
}

function drawVehicles(T, f){
  const m = T.map, px = T.px;
  const flat = (R.frames && R.frames[f]) || [];
  const classes = R.classes || ["car"];
  for (let i = 0; i < flat.length; i += 4){
    const id = flat[i], wx = flat[i+1]/SCALE, wy = flat[i+2]/SCALE,
          ang = flat[i+3];
    if (Math.abs(wx - G.centre) > T.spanX + 8 ||
        Math.abs(wy - G.centre) > T.spanY + 8) continue;
    const cls = classes[(R.vehicle_class || [])[id] || 0] || "car";
    const col = P.colors[cls] || "#6f7b8a";
    const len = cls === "bus" ? 11 : cls === "motorcycle" ? 2.2
              : cls === "auto" ? 3.2 : 4.5;
    const wid = cls === "bus" ? 2.6 : cls === "motorcycle" ? 0.9
              : cls === "auto" ? 1.5 : 1.9;
    const s = m(wx, wy);
    ctx.save(); ctx.translate(s[0], s[1]);
    /* SUMO angle: 0 = north, clockwise. Canvas: rotate clockwise from +x. */
    ctx.rotate((ang) * Math.PI / 180);
    const L = Math.max(len * px, 2.2), W = Math.max(wid * px, 1.6);
    ctx.fillStyle = col;
    ctx.beginPath();
    const rr = Math.min(W/2, 2.2);
    ctx.roundRect ? ctx.roundRect(-W/2, -L/2, W, L, rr)
                  : ctx.rect(-W/2, -L/2, W, L);
    ctx.fill();
    if (L > 7){ ctx.fillStyle = "rgba(255,255,255,.5)";
                ctx.fillRect(-W/2+W*0.22, -L/2+L*0.12, W*0.56, L*0.16); }
    ctx.restore();
  }
}

function drawRibbon(f){
  const w = rb.width, h = rb.height, sig = R.signal || [];
  rctx.clearRect(0,0,w,h);
  const unit = w / Math.max(sig.length, 1);
  for (let i = 0; i < sig.length; i++){
    rctx.fillStyle = SIG_FILL[sig[i]] || "#8b97a5";
    rctx.fillRect(i*unit, 4, Math.ceil(unit)+0.6, h-12);
  }
  rctx.fillStyle = "#10151c";
  rctx.fillRect(Math.min(f*unit, w-2), 0, 2, h);
}

function render(){
  const T = tx(), sig = (R.signal || [])[frame] || "??";
  drawRoads(T); drawStopLines(T, sig); drawVehicles(T, frame); drawRibbon(frame);

  ROOT.querySelector("#__UID__-state").textContent =
    (P.phaseLabel[sig] || sig);
  ROOT.querySelector("#__UID__-state").style.background =
    sig === "RR" ? "#fbe9e7" : (sig === "NY" || sig === "EY") ? "#fdf1dc" : "#e2f3ec";
  ROOT.querySelector("#__UID__-state").style.color =
    sig === "RR" ? "#b0342c" : (sig === "NY" || sig === "EY") ? "#9a6508" : "#13795b";

  let total = 0, halted = 0, maxq = 1;
  APPR.forEach(a => { maxq = Math.max(maxq, ...(R.queues[a] || [0])); });
  const rows = APPR.map(a => {
    const q = (R.queues[a] || [])[frame] || 0;
    const p = (R.present[a] || [])[frame] || 0;
    total += p; halted += q;
    return '<div class="row"><span>' + APPR_NAME[a] + '</span><b>' + q + '</b></div>' +
           '<div class="bar"><i style="width:' + (100*q/maxq) + '%;background:' +
           ((a==="N"||a==="S") ? P.ns : P.ew) + '"></i></div>';
  }).join("");
  ROOT.querySelector("#__UID__-queues").innerHTML = rows;
  ROOT.querySelector("#__UID__-nveh").textContent = total;
  ROOT.querySelector("#__UID__-nhalt").textContent = halted;
  ROOT.querySelector("#__UID__-clock").textContent =
    (frame*STEP).toFixed(0) + " / " + ((N-1)*STEP).toFixed(0) + " s";
  scrub.value = frame;
}

/* ---- transport -------------------------------------------------------- */
function tick(ts){
  if (playing){
    if (!last) last = ts;
    const want = (ts - last) / 1000 * speed / STEP;
    if (want >= 1){ frame = (frame + Math.floor(want)) % N; last = ts; render(); }
  }
  requestAnimationFrame(tick);
}
ROOT.querySelector("#__UID__-play").onclick = function(){
  playing = !playing; last = 0;
  this.textContent = playing ? "Pause" : "Play";
};
ROOT.querySelector("#__UID__-back").onclick = () => { frame = Math.max(0, frame-1); render(); };
ROOT.querySelector("#__UID__-fwd").onclick  = () => { frame = Math.min(N-1, frame+1); render(); };
ROOT.querySelector("#__UID__-speed").onclick = function(){
  speed = speed >= 16 ? 1 : speed * 2;
  this.innerHTML = speed + "&times;";
};
scrub.oninput = e => { frame = +e.target.value; render(); };
rb.onclick = e => {
  const r = rb.getBoundingClientRect();
  frame = Math.max(0, Math.min(N-1, Math.round((e.clientX-r.left)/r.width*(N-1))));
  render();
};
const zoom = ROOT.querySelector("#__UID__-zoom");
zoom.value = span;
ROOT.querySelector("#__UID__-zoomv").textContent = span;
zoom.oninput = e => { span = +e.target.value;
  ROOT.querySelector("#__UID__-zoomv").textContent = span; render(); };

/* ---- the canvas is sized to its box, in device pixels, so the drawing
        is crisp on a projector and fills whatever width it is given --- */
let fitted = "";
function fit(){
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(board.clientWidth, 280);
  const h = Math.max(board.clientHeight, 220);
  const rw = Math.max(ribbon.clientWidth, 280);
  const signature = w + "x" + h + "x" + rw + "@" + dpr;
  if (signature === fitted){ render(); return; }
  fitted = signature;
  cv.width = Math.round(w * dpr);  cv.height = Math.round(h * dpr);
  cv.style.width = w + "px";       cv.style.height = h + "px";
  rb.width = Math.round(rw * dpr); rb.height = Math.round(26 * dpr);
  rb.style.width = rw + "px";      rb.style.height = "26px";
  render();
}
if (window.ResizeObserver)
  new ResizeObserver(() => requestAnimationFrame(fit)).observe(board);
window.addEventListener("resize", fit);


ROOT.querySelector("#__UID__-speed").innerHTML = speed + "&times;";
fit();
if (P.autoplay && N > 1){
  playing = true; ROOT.querySelector("#__UID__-play").textContent = "Pause";
}
requestAnimationFrame(tick);

} catch (err) {
  ROOT.querySelector('#__UID__-title').textContent =
    'drawing error: ' + err;
}
})();
</script>
</div>
"""
