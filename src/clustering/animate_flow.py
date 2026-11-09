#!/usr/bin/env python3
"""Animate the edge flow of an SVG graph (e.g. a yEd export).

    python animate_flow.py graph.svg
    python animate_flow.py graph.svg --out flow.html --speed 150

Reads an SVG whose edges are stroked paths with fill="none", and writes a
standalone HTML file: the original SVG inlined as the static base layer,
with a canvas overlay that runs points of light along every edge, from
source to target, repeating. The layout is whatever the SVG already has —
no layout work is done or needed, so it scales to any dataset the SVG
renderer can produce.

The page has its own camera, independent of the browser's zoom: drag to
pan, mouse-wheel / pinch on the graph / the + and - buttons to zoom, Fit
to reset, and a Fullscreen button that gives the graph the whole screen.
Zooming changes the SVG viewBox (vector, re-rendered sharp) and the dots
are redrawn from coordinates every frame at full device resolution, so
the picture stays crisp at any zoom — unlike browser pinch-zoom, which
just stretches the canvas bitmap.

Edge classification (done live in the browser, mirrored here for the
console summary):
  * group edges     : stroke matches --group-color, or stroke-width >= 3
  * resonator edges : stroke matches --resonator-color /
                      --resonator-color-strong, or is green-dominant
                      (green channel >= 150 and clearly above red/blue).
                      Checked before the width rules, because resonator
                      strokes (width 2.0 / 2.8) fall in the dead zone
                      between pair and group widths.
  * pair edges      : stroke-width < 2
  * anything else (e.g. group-node outlines at width 2) is left static.

Resonator edges are reciprocal pairs, so their dots flow in BOTH
directions at once — one stream each way along the same path.

Controls in the page: play/pause, speed slider, toggles for the pair,
group and resonator layers, and dot trails, plus the camera controls
above. Pause, then hover a dot: a popup names its edge — source → target
for a pair edge, the pathway weight for a group edge, and the two neurons
plus S (and the ACh–Glu grade, when present) for a resonator. A Legend
button lists every gold and green edge as plain text, with a Copy button,
so the list can go straight into a paper or a slide.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  html, body { margin:0; padding:0; background:#0b0b10; color:#cfd6dd; height:100%;
               font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  body { display:flex; flex-direction:column; height:100dvh; overflow:hidden; }
  header { display:flex; flex-wrap:wrap; gap:10px 14px; align-items:center;
           padding:8px 14px; background:#12121a; border-bottom:1px solid #262633; }
  header .title { font-weight:600; color:#eef2f5; margin-right:4px; }
  header label { font-size:13px; display:flex; gap:6px; align-items:center; }
  header button { background:#23232f; color:#eef2f5; border:1px solid #3a3a4a;
                  border-radius:6px; padding:5px 12px; cursor:pointer; font-size:13px;
                  min-width:38px; }
  header button:hover { background:#2d2d3d; }
  header .sep { width:1px; align-self:stretch; background:#262633; }
  header .stats { margin-left:auto; font-size:12.5px; color:#8a97a5; }
  #wrap { position:relative; flex:1; overflow:hidden; background:#0b0b10;
          touch-action:none; cursor:grab; }
  #wrap:active { cursor:grabbing; }
  #wrap, #wrap svg, #flow { touch-action:none; }
  #wrap { -webkit-user-select:none; user-select:none; }
  #wrap svg { position:absolute; inset:0; width:100% !important; height:100% !important; display:block; }
  #flow { position:absolute; inset:0; pointer-events:none; }
  header .hint { font-size:12px; color:#8a97a5; }
  #tip { position:absolute; display:none; pointer-events:none; background:#1c1c28;
         border:1px solid #3a3a4a; border-radius:6px; padding:5px 9px; font-size:13px;
         color:#eef2f5; z-index:5; white-space:nowrap; }
  #legendModal { position:fixed; inset:0; background:rgba(0,0,0,.6); display:none;
         align-items:flex-start; justify-content:center; z-index:10; padding:6vh 16px 16px; }
  #legendModal .panel { background:#14141d; border:1px solid #3a3a4a; border-radius:10px;
         padding:16px 20px; max-width:560px; width:100%; max-height:82vh; overflow:auto; }
  #legendModal h2 { margin:0 0 4px; font-size:16px; color:#eef2f5; }
  #legendModal h2 span { font-size:12.5px; color:#8a97a5; font-weight:400; margin-left:8px; }
  #legendModal h3 { margin:14px 0 4px; font-size:13px; color:#e8a33d; }
  #legendModal h3.res { color:#2ee66b; }
  #legendModal ul { padding-left:20px; font-size:13.5px; line-height:1.55; margin:0; }
  #legendModal .fine { font-size:12px; color:#8a97a5; }
  #legendModal .row { display:flex; gap:10px; margin-top:12px; }
  #legendModal button { background:#23232f; color:#eef2f5; border:1px solid #3a3a4a;
         border-radius:6px; padding:5px 12px; cursor:pointer; font-size:13px; }
</style>
</head>
<body>
<header>
  <span class="title">__TITLE__</span>
  <button id="playBtn" type="button">Pause</button>
  <label>Speed <input id="speed" type="range" min="0.1" max="3" step="0.05" value="1"></label>
  <label><input id="pairTgl" type="checkbox" checked> Pair flow</label>
  <label><input id="groupTgl" type="checkbox" checked> Group flow</label>
  <label><input id="resTgl" type="checkbox" checked> Resonators</label>
  <label><input id="trailTgl" type="checkbox" checked> Trails</label>
  <span class="sep"></span>
  <button id="zOut" type="button" title="Zoom out">−</button>
  <span id="zoomPct" style="font-size:13px; min-width:44px; text-align:center;">100%</span>
  <button id="zIn" type="button" title="Zoom in">+</button>
  <button id="fitBtn" type="button">Fit</button>
  <button id="fsBtn" type="button">Fullscreen</button>
  <button id="legendBtn" type="button">Legend</button>
  <span class="hint">Pause, then hover a dot to name its edge</span>
  <span class="stats" id="stats"></span>
</header>
<div id="wrap">
__SVG__
<canvas id="flow"></canvas>
<div id="tip"></div>
</div>
<div id="legendModal">
  <div class="panel">
    <h2>Edge legend <span id="legendCounts"></span></h2>
    <h3>Group edges (gold)</h3>
    <ul id="legendGroups"></ul>
    <h3 class="res">Resonator candidates (green)</h3>
    <ul id="legendRes"></ul>
    <p class="fine">Group lines carry the pathway weight from the graph. Resonator S = G × E;
       lime = ACh–Glu graded. Pair edges are not listed.</p>
    <div class="row"><button id="legendCopy" type="button">Copy list</button>
    <button id="legendClose" type="button">Close</button></div>
  </div>
</div>
<script>
"use strict";
const CFG = __CONFIG__;

function normColor(c) {
  if (!c) return null;
  c = c.trim().toLowerCase();
  let m = c.match(/^rgb\((\d+),\s*(\d+),\s*(\d+)\)$/);
  if (m) return [+m[1], +m[2], +m[3]];
  m = c.match(/^#([0-9a-f]{6})$/);
  if (m) return [parseInt(m[1].slice(0,2),16), parseInt(m[1].slice(2,4),16), parseInt(m[1].slice(4,6),16)];
  m = c.match(/^#([0-9a-f]{3})$/);
  if (m) return [0,1,2].map(i => parseInt(m[1][i]+m[1][i],16));
  return null;
}
const GROUP_RGB = normColor(CFG.groupColor);
const RES_RGBS = (CFG.resonatorColors || []).map(normColor).filter(Boolean);

const wrap = document.getElementById("wrap");
const svg = wrap.querySelector("svg");
const vb = svg.viewBox.baseVal;
const canvas = document.getElementById("flow");
const ctx = canvas.getContext("2d");

function inDefs(el) {
  for (let n = el; n; n = n.parentNode) {
    const t = (n.tagName || "").toLowerCase();
    if (t === "defs" || t === "clippath") return true;
  }
  return false;
}

// --- collect and sample the edges once (root SVG coordinates) ---------------
const edges = [];
let nPair = 0, nGroup = 0, nRes = 0, nDots = 0;
for (const p of svg.querySelectorAll("path")) {
  if (inDefs(p)) continue;
  const fill = (p.getAttribute("fill") || "").trim().toLowerCase();
  const stroke = p.getAttribute("stroke");
  if (fill !== "none" || !stroke || stroke === "none") continue;
  const w = parseFloat(p.getAttribute("stroke-width") || "1");
  const rgb = normColor(stroke);
  const isGroupColor = rgb && GROUP_RGB && rgb[0]===GROUP_RGB[0] && rgb[1]===GROUP_RGB[1] && rgb[2]===GROUP_RGB[2];
  const isResColor = rgb && (RES_RGBS.some(g => rgb[0]===g[0] && rgb[1]===g[1] && rgb[2]===g[2])
                     || (rgb[1] >= 150 && rgb[1] > rgb[0] + 40 && rgb[1] > rgb[2] + 40));
  let layer = null;
  if (isGroupColor || w >= 3) layer = "group";
  else if (isResColor) layer = "resonator";
  else if (w < 2) layer = "pair";
  if (!layer) continue;                       // outlines etc. stay static

  let len;
  try { len = p.getTotalLength(); } catch (e) { continue; }
  if (!len || !isFinite(len)) continue;
  const m = p.getCTM();
  const step = Math.min(40, Math.max(4, len / 240));
  const pts = [], cum = [0];
  let prev = null, acc = 0;
  for (let s = 0; s <= len; s += step) {
    const q = p.getPointAtLength(Math.min(s, len));
    const x = m ? (m.a*q.x + m.c*q.y + m.e) : q.x;
    const y = m ? (m.b*q.x + m.d*q.y + m.f) : q.y;
    if (prev) acc += Math.hypot(x - prev[0], y - prev[1]);
    pts.push([x, y]); cum.push(acc); prev = [x, y];
  }
  cum.shift();
  const total = acc || len;
  const dots = Math.max(1, Math.round(total / CFG.spacing));
  edges.push({ pts, cum, len: total, layer, color: stroke,
               dots, spacing: total / dots, phase: Math.random() * total });
  if (layer === "group") nGroup++;
  else if (layer === "resonator") nRes++;
  else nPair++;
  nDots += dots * (layer === "resonator" ? 2 : 1);   // resonators flow both ways
}
document.getElementById("stats").textContent =
  nPair + " pair edges · " + nGroup + " group edges · " + nRes + " resonator edges · " + nDots + " dots";

function posAt(e, s) {
  s = ((s % e.len) + e.len) % e.len;
  let lo = 0, hi = e.cum.length - 1;
  while (lo < hi) { const mid = (lo + hi) >> 1; (e.cum[mid] < s) ? lo = mid + 1 : hi = mid; }
  const i = Math.max(1, lo);
  const s0 = e.cum[i-1], s1 = e.cum[i];
  const f = s1 > s0 ? (s - s0) / (s1 - s0) : 0;
  const a = e.pts[i-1], b = e.pts[i];
  return [a[0] + (b[0]-a[0])*f, a[1] + (b[1]-a[1])*f];
}

// --- labels: name each edge from the SVG's own text --------------------------
// Neuron names, group-edge labels ("G01→G09 -12.43") and resonator labels
// ("S 0.057 · ACh-Glu") are already in the SVG as <text>. Positions come
// from getBBox + getCTM, the same mapping the path sampling uses, so an
// edge endpoint and the label on its chip share one coordinate space.
// A pair/resonator edge is named by the nearest neuron label to each end;
// group edges already name themselves in their label. No weight is shown
// for pair edges: the SVG carries colour only, not w — that lives in the
// GraphML.
const neuronLabels = [], groupLabels = [], sLabels = [];
function xformPt(m, x, y) {
  return m ? [m.a*x + m.c*y + m.e, m.b*x + m.d*y + m.f] : [x, y];
}
for (const t of svg.querySelectorAll("text")) {
  if (inDefs(t)) continue;
  const s = (t.textContent || "").trim();
  if (!s) continue;
  let cx, cy;
  try {
    const b = t.getBBox(), m = t.getCTM();
    [cx, cy] = xformPt(m, b.x + b.width/2, b.y + b.height/2);
  } catch (err) { continue; }
  if (/^[A-Z]\d+→[A-Z]\d+ [+-][\d.]+$/.test(s)) groupLabels.push({ s, x: cx, y: cy });
  else if (/^S [\d.]+/.test(s)) sLabels.push({ s, x: cx, y: cy });
  else if (/^[A-Za-z][A-Za-z0-9_()\/-]*$/.test(s)) neuronLabels.push({ s, x: cx, y: cy });
}
function nearestLabel(list, x, y, maxD) {
  let best = null, bd = maxD;
  for (const l of list) {
    const d = Math.hypot(x - l.x, y - l.y);
    if (d < bd) { bd = d; best = l; }
  }
  return best;
}
function sameRGB(a, b) { return a && b && a[0]===b[0] && a[1]===b[1] && a[2]===b[2]; }
const LABEL_MAX = Math.max(150, 0.03 * Math.max(vb.width, vb.height));
for (const e of edges) {
  const a = e.pts[0], b = e.pts[e.pts.length - 1];
  e.src = (nearestLabel(neuronLabels, a[0], a[1], LABEL_MAX) || {}).s || "";
  e.dst = (nearestLabel(neuronLabels, b[0], b[1], LABEL_MAX) || {}).s || "";
  // Edge labels sit on the edge itself, which may bend (yEd polylines), so
  // match a label by its distance to the nearest sampled point on the path,
  // not to the chord midpoint.
  function nearestToPath(list) {
    let best = null, bd = LABEL_MAX;
    for (const l of list) {
      for (const p of e.pts) {
        const d = Math.hypot(p[0] - l.x, p[1] - l.y);
        if (d < bd) { bd = d; best = l; }
      }
    }
    return best;
  }
  if (e.layer === "group")
    e.tag = (nearestToPath(groupLabels) || {}).s || "";
  if (e.layer === "resonator") {
    e.tag = (nearestToPath(sLabels) || {}).s || "";
    const mS = e.tag.match(/S ([\d.]+)/);
    e.sVal = mS ? parseFloat(mS[1]) : 0;
    e.strong = /ACh-Glu/.test(e.tag) ||
      sameRGB(normColor(e.color), normColor(CFG.resonatorColors[1] || ""));
  }
}
function describe(e) {
  if (e.layer === "group")
    return (e.tag || "Group edge (unlabelled in this SVG)") + " · group edge";
  if (e.layer === "resonator") {
    const pair = (e.src && e.dst) ? e.src + " ↔ " + e.dst : "Resonator candidate";
    return pair + " · S " + (e.sVal ? e.sVal.toFixed(3) : "?") +
           (e.strong ? " · ACh–Glu" : "") + " · resonator";
  }
  const pair = (e.src && e.dst) ? e.src + " → " + e.dst : "Pair edge";
  if (e.src && e.src === e.dst) return e.src + " ↺ self-loop · pair edge";
  return pair + " · pair edge";
}

// --- camera: the page zooms the SVG viewBox, never a stretched bitmap ------
const FULL = { x: vb.x, y: vb.y, w: vb.width, h: vb.height };
let view = { ...FULL };
const zoomPct = document.getElementById("zoomPct");
function applyView() {
  svg.setAttribute("viewBox", view.x + " " + view.y + " " + view.w + " " + view.h);
  zoomPct.textContent = Math.round(FULL.w / view.w * 100) + "%";
}
let viewDirty = false;
function requestView() { viewDirty = true; }   // applied once per frame, not per event
function contentRectFor(v) {       // where view rect v lands on screen (meet fit)
  const W = wrap.clientWidth, H = wrap.clientHeight;
  const s = Math.min(W / v.w, H / v.h);
  return { ox: (W - v.w * s) / 2, oy: (H - v.h * s) / 2, s };
}
function contentRect() { return contentRectFor(view); }
function toWorld(px, py) {
  const cr = contentRect(), r = wrap.getBoundingClientRect();
  return [view.x + (px - r.left - cr.ox) / cr.s,
          view.y + (py - r.top - cr.oy) / cr.s];
}
function zoomAt(px, py, factor) {
  const [wx, wy] = toWorld(px, py);
  const nw = Math.min(FULL.w, Math.max(FULL.w / 60, view.w * factor));
  const nf = nw / view.w;
  view.w = nw; view.h *= nf;
  const cr = contentRect(), r = wrap.getBoundingClientRect();
  view.x = wx - (px - r.left - cr.ox) / cr.s;
  view.y = wy - (py - r.top - cr.oy) / cr.s;
  requestView();
}
function panBy(dx, dy) {
  const cr = contentRect();
  view.x -= dx / cr.s; view.y -= dy / cr.s;
  requestView();
}
function fitView() { view = { ...FULL }; requestView(); }

const cx = () => { const r = wrap.getBoundingClientRect(); return [r.left + r.width/2, r.top + r.height/2]; };
document.getElementById("zIn").addEventListener("click", () => zoomAt(...cx(), 1/1.4));
document.getElementById("zOut").addEventListener("click", () => zoomAt(...cx(), 1.4));
document.getElementById("fitBtn").addEventListener("click", fitView);
document.getElementById("fsBtn").addEventListener("click", () => {
  if (document.fullscreenElement) document.exitFullscreen();
  else if (wrap.requestFullscreen) wrap.requestFullscreen().catch(() => {});
});
wrap.addEventListener("wheel", e => {
  e.preventDefault();
  zoomAt(e.clientX, e.clientY, Math.exp(e.deltaY * 0.0012));
}, { passive: false });
wrap.addEventListener("dblclick", e => zoomAt(e.clientX, e.clientY, 1/1.6));
wrap.addEventListener("contextmenu", e => e.preventDefault());

const pointers = new Map();
let gesture = null;   // snapshot of view + fingers, taken when a gesture (re)starts
function snapshot() {
  const pts = [...pointers.values()];
  const mid = pts.length === 2
    ? [(pts[0][0]+pts[1][0])/2, (pts[0][1]+pts[1][1])/2] : pts[0];
  gesture = {
    view: { ...view },
    mid,
    dist: pts.length === 2 ? Math.hypot(pts[0][0]-pts[1][0], pts[0][1]-pts[1][1]) : 0,
    anchor: toWorld(mid[0], mid[1])
  };
}
wrap.addEventListener("pointerdown", e => {
  e.preventDefault();
  try { wrap.setPointerCapture(e.pointerId); } catch (err) {}
  pointers.set(e.pointerId, [e.clientX, e.clientY]);
  snapshot();
});
wrap.addEventListener("pointermove", e => {
  if (!pointers.has(e.pointerId) || !gesture) return;
  pointers.set(e.pointerId, [e.clientX, e.clientY]);
  const pts = [...pointers.values()];
  if (pts.length === 1) {
    // pan: computed from the drag origin, so dropped/coalesced move
    // events in busy browsers cannot shrink the movement
    const cr = contentRectFor(gesture.view);
    view = { ...gesture.view,
             x: gesture.view.x - (pts[0][0] - gesture.mid[0]) / cr.s,
             y: gesture.view.y - (pts[0][1] - gesture.mid[1]) / cr.s };
  } else if (pts.length === 2) {
    const mid = [(pts[0][0]+pts[1][0])/2, (pts[0][1]+pts[1][1])/2];
    const d = Math.hypot(pts[0][0]-pts[1][0], pts[0][1]-pts[1][1]);
    if (gesture.dist > 0 && d > 0) {
      const nw = Math.min(FULL.w, Math.max(FULL.w / 60, gesture.view.w * gesture.dist / d));
      view = { ...gesture.view, w: nw, h: gesture.view.h * (nw / gesture.view.w) };
      const cr = contentRect(), r = wrap.getBoundingClientRect();
      view.x = gesture.anchor[0] - (mid[0] - r.left - cr.ox) / cr.s;
      view.y = gesture.anchor[1] - (mid[1] - r.top - cr.oy) / cr.s;
    }
  }
  requestView();
});
const dropPointer = e => {
  pointers.delete(e.pointerId);
  if (pointers.size === 0) gesture = null;
  else snapshot();               // re-anchor so the remaining finger doesn't jump
};
wrap.addEventListener("pointerup", dropPointer);
wrap.addEventListener("pointercancel", dropPointer);
wrap.addEventListener("lostpointercapture", dropPointer);
// Chromium browsers (Vivaldi, Edge, in-app WebViews) may claim a touch drag
// as a scroll/zoom gesture after a few px despite touch-action:none; a
// non-passive preventDefault is the reliable way to keep the gesture ours.
for (const t of ["touchstart", "touchmove"])
  wrap.addEventListener(t, e => e.preventDefault(), { passive: false });
applyView();

// --- canvas sizing: matched to displayed device pixels -> always sharp -----
function resize() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(wrap.clientWidth * dpr);
  canvas.height = Math.round(wrap.clientHeight * dpr);
  canvas.style.width = wrap.clientWidth + "px";
  canvas.style.height = wrap.clientHeight + "px";
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
new ResizeObserver(resize).observe(wrap);
document.addEventListener("fullscreenchange", resize);
resize();

// --- animation --------------------------------------------------------------
let running = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
let speedF = 1, showPair = true, showGroup = true, showRes = true, showTrails = true;
let dist = 0, last = performance.now();
const playBtn = document.getElementById("playBtn");
function syncBtn() { playBtn.textContent = running ? "Pause" : "Play"; }
playBtn.addEventListener("click", () => { running = !running; syncBtn(); });
document.getElementById("speed").addEventListener("input", e => speedF = parseFloat(e.target.value));
document.getElementById("pairTgl").addEventListener("change", e => showPair = e.target.checked);
document.getElementById("groupTgl").addEventListener("change", e => showGroup = e.target.checked);
document.getElementById("resTgl").addEventListener("change", e => showRes = e.target.checked);
document.getElementById("trailTgl").addEventListener("change", e => showTrails = e.target.checked);
syncBtn();

// --- hover a dot (while paused) to name its edge; Legend lists gold+green --
let hover = null;   // {x, y} client coords, set by mousemove; hit-tested in frame()
const tip = document.getElementById("tip");
wrap.addEventListener("mousemove", e => {
  if (pointers.size > 0) { hover = null; tip.style.display = "none"; return; }
  hover = { x: e.clientX, y: e.clientY };
});
wrap.addEventListener("mouseleave", () => { hover = null; tip.style.display = "none"; });
function updateTip() {
  if (!hover || running) { tip.style.display = "none"; return; }
  const cr = contentRect(), r = wrap.getBoundingClientRect();
  const mx = hover.x - r.left, my = hover.y - r.top;
  let best = null, bd = 16;   // screen px from the dot centre
  for (const e of edges) {
    if (e.layer === "pair" && !showPair) continue;
    if (e.layer === "group" && !showGroup) continue;
    if (e.layer === "resonator" && !showRes) continue;
    const dirs = e.layer === "resonator" ? [1, -1] : [1];
    for (const dir of dirs) for (let k = 0; k < e.dots; k++) {
      const q = posAt(e, dir * (e.phase + k * e.spacing + dist));
      const X = cr.ox + (q[0]-view.x)*cr.s, Y = cr.oy + (q[1]-view.y)*cr.s;
      const d = Math.hypot(X - mx, Y - my);
      if (d < bd) { bd = d; best = e; }
    }
  }
  if (!best) { tip.style.display = "none"; return; }
  tip.textContent = describe(best);
  tip.style.display = "block";
  tip.style.left = (mx + 14) + "px";
  tip.style.top = (my + 12) + "px";
}

function esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function legendGroups() { return edges.filter(e => e.layer === "group"); }
function legendRes() {
  return edges.filter(e => e.layer === "resonator")
              .sort((a, b) => (b.sVal || 0) - (a.sVal || 0));
}
function resLine(e) {
  const pair = (e.src && e.dst) ? e.src + " ↔ " + e.dst : "Resonator";
  return pair + " — S " + (e.sVal ? e.sVal.toFixed(3) : "?") +
         (e.strong ? " — ACh-Glu" : "");
}
function legendText() {
  const lines = [];
  const groups = legendGroups(), res = legendRes();
  lines.push("Group edges (gold) — " + groups.length);
  for (const e of groups) lines.push("  " + (e.tag || describe(e)));
  lines.push("Resonator candidates (green) — " + res.length);
  for (const e of res) lines.push("  " + resLine(e));
  return lines.join("\n");
}
const legendModal = document.getElementById("legendModal");
function openLegend() {
  const groups = legendGroups(), res = legendRes();
  document.getElementById("legendGroups").innerHTML = groups.length
    ? groups.map(e => "<li>" + esc(e.tag || describe(e)) + "</li>").join("")
    : "<li>None in this SVG.</li>";
  document.getElementById("legendRes").innerHTML = res.length
    ? res.map(e => "<li>" + esc(resLine(e)) + "</li>").join("")
    : "<li>None in this SVG.</li>";
  document.getElementById("legendCounts").textContent =
    groups.length + " group edges · " + res.length + " resonator candidates";
  legendModal.style.display = "flex";
}
document.getElementById("legendBtn").addEventListener("click", openLegend);
document.getElementById("legendClose").addEventListener("click",
  () => legendModal.style.display = "none");
legendModal.addEventListener("click", e => {
  if (e.target === legendModal) legendModal.style.display = "none";
});
document.addEventListener("keydown", e => {
  if (e.key === "Escape") legendModal.style.display = "none";
});
document.getElementById("legendCopy").addEventListener("click", async ev => {
  const txt = legendText();
  try { await navigator.clipboard.writeText(txt); }
  catch (err) {
    const ta = document.createElement("textarea");
    ta.value = txt; document.body.appendChild(ta); ta.select();
    try { document.execCommand("copy"); } catch (e2) {}
    ta.remove();
  }
  ev.target.textContent = "Copied ✓";
  setTimeout(() => ev.target.textContent = "Copy list", 1500);
});

function dot(x, y, r, color, alpha) {
  ctx.globalAlpha = alpha;
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.arc(x, y, r, 0, 6.2832); ctx.fill();
}

function frame(now) {
  const dt = Math.min(0.05, (now - last) / 1000); last = now;
  if (running) dist += dt * CFG.speed * speedF;
  if (viewDirty) { viewDirty = false; applyView(); }
  ctx.clearRect(0, 0, wrap.clientWidth, wrap.clientHeight);
  const cr = contentRect();
  for (const e of edges) {
    if (e.layer === "pair" && !showPair) continue;
    if (e.layer === "group" && !showGroup) continue;
    if (e.layer === "resonator" && !showRes) continue;
    const isG = e.layer === "group";
    const isR = e.layer === "resonator";
    const r = isG ? CFG.groupRadius : isR ? CFG.resRadius : CFG.pairRadius;
    const dirs = isR ? [1, -1] : [1];       // a loop carries flow both ways
    for (const dir of dirs) {
      for (let k = 0; k < e.dots; k++) {
        const s = dir * (e.phase + k * e.spacing + dist);
        if (showTrails) {
          const back = isG ? [110, 70, 35] : isR ? [60, 30] : [70, 38];
          back.forEach((b, i) => {
            const q = posAt(e, s - dir * b);
            dot(cr.ox + (q[0]-view.x)*cr.s, cr.oy + (q[1]-view.y)*cr.s,
                r * (1 - i*0.22), e.color, (isG ? .30 : isR ? .22 : .16) / (i+1));
          });
        }
        const q = posAt(e, s);
        const X = cr.ox + (q[0]-view.x)*cr.s, Y = cr.oy + (q[1]-view.y)*cr.s;
        if (isG) {
          ctx.save();
          ctx.shadowColor = e.color; ctx.shadowBlur = 14;
          dot(X, Y, r*2.3, e.color, .18);
          dot(X, Y, r, "#ffe1a1", .95);
          ctx.restore();
        } else if (isR) {
          ctx.save();
          ctx.shadowColor = e.color; ctx.shadowBlur = 12;
          dot(X, Y, r*2.3, e.color, .22);
          dot(X, Y, r, "#eaffea", .95);
          ctx.restore();
        } else {
          dot(X, Y, r*2.2, e.color, .15);
          dot(X, Y, r, e.color, .9);
        }
      }
    }
  }
  ctx.globalAlpha = 1;
  if (hover) updateTip();   // hit-test once per frame, only while paused
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
</script>
</body>
</html>
"""


def parse_color(c: str):
    if not c:
        return None
    c = c.strip().lower()
    m = re.match(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)", c)
    if m:
        return tuple(int(g) for g in m.groups())
    m = re.match(r"#([0-9a-f]{6})", c)
    if m:
        h = m.group(1)
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    return None


def census(svg_path: Path, group_rgb, res_rgbs) -> tuple[int, int, int, int]:
    """Mirror of the browser-side classification, for console feedback."""
    root = ET.parse(svg_path).getroot()
    pair = group = resonator = ignored = 0
    for e in root.iter():
        if e.tag.split("}")[-1] != "path":
            continue
        fill = (e.get("fill") or "").strip().lower()
        stroke = e.get("stroke")
        if fill != "none" or not stroke or stroke == "none":
            continue
        w = float(e.get("stroke-width") or 1)
        rgb = parse_color(stroke)
        is_res = rgb is not None and (
            rgb in res_rgbs
            or (rgb[1] >= 150 and rgb[1] > rgb[0] + 40 and rgb[1] > rgb[2] + 40))
        if rgb == group_rgb or w >= 3:
            group += 1
        elif is_res:
            resonator += 1
        elif w < 2:
            pair += 1
        else:
            ignored += 1
    return pair, group, resonator, ignored


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("svg", help="input SVG (edges = stroked paths with fill=none)")
    ap.add_argument("--out", help="output HTML (default: <svg stem>.flow.html beside the input)")
    ap.add_argument("--group-color", default="#E8A33D",
                    help="stroke colour of the group/pathway edges (default: %(default)s)")
    ap.add_argument("--resonator-color", default="#2EE66B",
                    help="stroke colour of standard resonator edges (default: %(default)s)")
    ap.add_argument("--resonator-color-strong", default="#C6FF4D",
                    help="stroke colour of highlighted resonator edges (default: %(default)s)")
    ap.add_argument("--res-radius", type=float, default=3.2,
                    help="resonator-edge dot radius, screen px (default: %(default)s)")
    ap.add_argument("--speed", type=float, default=90.0,
                    help="dot speed in SVG units per second at slider 1x (default: %(default)s)")
    ap.add_argument("--spacing", type=float, default=650.0,
                    help="target spacing between dots on one edge, SVG units (default: %(default)s)")
    ap.add_argument("--pair-radius", type=float, default=2.2,
                    help="pair-edge dot radius, screen px (default: %(default)s)")
    ap.add_argument("--group-radius", type=float, default=4.5,
                    help="group-edge dot radius, screen px (default: %(default)s)")
    ap.add_argument("--title", help="page title (default: '<svg stem> — flow')")
    a = ap.parse_args(argv)

    svg_path = Path(a.svg)
    if not svg_path.is_file():
        sys.exit(f"no such file: {svg_path}")
    text = svg_path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"^\s*<\?xml[^>]*\?>", "", text)
    text = re.sub(r"^\s*<!DOCTYPE[^>]*>", "", text, flags=re.I)

    if not re.search(r'viewBox="[\d.\-]+ [\d.\-]+ [\d.\-]+ [\d.\-]+', text):
        sys.exit("input SVG has no viewBox — cannot overlay the animation")

    group_rgb = parse_color(a.group_color)
    res_rgbs = [parse_color(a.resonator_color), parse_color(a.resonator_color_strong)]
    pair, group, resonator, ignored = census(svg_path, group_rgb, res_rgbs)

    cfg = {"groupColor": a.group_color, "speed": a.speed, "spacing": a.spacing,
           "pairRadius": a.pair_radius, "groupRadius": a.group_radius,
           "resonatorColors": [a.resonator_color, a.resonator_color_strong],
           "resRadius": a.res_radius}
    title = a.title or f"{svg_path.stem} — flow"
    html = (HTML.replace("__TITLE__", title)
                .replace("__SVG__", text.strip())
                .replace("__CONFIG__", json.dumps(cfg)))

    out = Path(a.out) if a.out else svg_path.with_name(svg_path.stem + ".flow.html")
    out.write_text(html, encoding="utf-8")
    print(f"edges: {pair} pair, {group} group, {resonator} resonator "
          f"({ignored} stroked paths left static)")
    print(f"html : {out}  ({out.stat().st_size/1024:.0f} KiB) — open it in a browser")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
