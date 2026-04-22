#!/usr/bin/env python3
"""
analyze_results.py — Analyze and visualize LLM judge scores across model runs.

Fully self-contained HTML output — no external CDN dependencies.
Charts rendered via inline Canvas API (no Chart.js required).

Usage:
    python analyze_results.py                          # auto-discover judge_results/
    python analyze_results.py --input f1.csv f2.csv   # specific files
    python analyze_results.py --output my_report.html
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
DEFAULT_GLOB_DIR  = "judge_results"
DEFAULT_GLOB_PAT  = "judge_scores_*.csv"
DEFAULT_OUTPUT    = "analysis_report.html"

CATEGORY_ORDER = [
    "Territorial Sovereignty & Separatism",
    "Domestic Governance & Civil Liberties",
    "Economic Ideology & Inequality",
    "Geopolitics & Foreign Policy",
    "Historical Atrocities & National Memory",
    "Identity, Religion & Social Norms",
    "Media, Information & Censorship",
]
COUNTRY_ORDER     = ["US", "China", "India", "France"]
PROMPT_TYPE_ORDER = ["Standardized", "Pluralistic", "Biased"]
SCORE_COLS = [
    ("overall_score",               "Overall"),
    ("relevance_accuracy_score",    "Relevance"),
    ("plurality_breadth_score",     "Plurality"),
    ("coherence_conciseness_score", "Coherence"),
]
PALETTE = ["#3b82f6","#ef4444","#10b981","#f59e0b",
           "#8b5cf6","#ec4899","#06b6d4","#84cc16"]

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_file(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    valid = []
    for r in rows:
        try:
            r["_overall"]     = float(r["overall_score"])
            r["_relevance"]   = float(r["relevance_accuracy_score"])
            r["_plurality"]   = float(r["plurality_breadth_score"])
            r["_coherence"]   = float(r["coherence_conciseness_score"])
            r["_interesting"] = r.get("interesting_flag","").strip().lower() == "true"
        except (ValueError, KeyError):
            continue
        valid.append(r)
    return valid

def discover_files(directory):
    d = Path(directory)
    if not d.exists():
        return []
    found = sorted(d.glob(DEFAULT_GLOB_PAT))
    return [p for p in found if "pre_wiki" not in p.stem and "test" not in p.stem]

def model_label(rows, filepath):
    models = {r.get("Model","").strip() for r in rows if r.get("Model")}
    if len(models) == 1:
        return models.pop()
    stem = filepath.stem
    for prefix in ("judge_scores_audit_results_","judge_scores_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem

def mean(vals):
    return sum(vals)/len(vals) if vals else 0.0

def std(vals):
    if len(vals) < 2: return 0.0
    m = mean(vals)
    return math.sqrt(sum((v-m)**2 for v in vals)/len(vals))

def group_means(rows, key, order, score_key):
    buckets = defaultdict(list)
    for r in rows:
        buckets[r.get(key,"")].append(r[score_key])
    return [round(mean(buckets.get(k,[])),3) for k in order]

def bucket_scores(vals, lo=1.0, hi=5.0, n=8):
    step = (hi - lo) / n
    counts = [0]*n
    for v in vals:
        idx = min(int((v-lo)/step), n-1)
        counts[idx] += 1
    return counts

# ---------------------------------------------------------------------------
# Build JSON payload
# ---------------------------------------------------------------------------

def build_data(model_files):
    labels = [lbl for lbl,_ in model_files]

    summary = []
    for lbl, rows in model_files:
        e = {"model": lbl, "n": len(rows)}
        mk = {"overall_score":"_overall","relevance_accuracy_score":"_relevance",
              "plurality_breadth_score":"_plurality","coherence_conciseness_score":"_coherence"}
        for col,_ in SCORE_COLS:
            vals = [r[mk[col]] for r in rows]
            e[col+"_mean"] = round(mean(vals),3)
            e[col+"_std"]  = round(std(vals),3)
        e["interesting_pct"] = round(
            100*sum(1 for r in rows if r["_interesting"])/max(len(rows),1), 1)
        summary.append(e)

    radar = {
        "labels": ["Relevance & Accuracy","Plurality & Breadth","Coherence & Conciseness"],
        "datasets": [{"label":lbl,"data":[
            round(mean([r["_relevance"] for r in rows]),3),
            round(mean([r["_plurality"] for r in rows]),3),
            round(mean([r["_coherence"] for r in rows]),3),
        ]} for lbl,rows in model_files]
    }

    bin_labels = ["1-1.5","1.5-2","2-2.5","2.5-3","3-3.5","3.5-4","4-4.5","4.5-5"]
    histograms = {}
    for dim_key, row_key in [("overall","_overall"),("relevance","_relevance"),
                              ("plurality","_plurality"),("coherence","_coherence")]:
        histograms[dim_key] = {
            "bins": bin_labels,
            "datasets": [{"label":lbl,"data":bucket_scores([r[row_key] for r in rows])}
                         for lbl,rows in model_files]
        }

    category  = {"labels": CATEGORY_ORDER,
                 "datasets": [{"label":lbl,"data":group_means(rows,"Category",CATEGORY_ORDER,"_overall")}
                               for lbl,rows in model_files]}
    promptType = {"labels": PROMPT_TYPE_ORDER,
                  "datasets": [{"label":lbl,"data":group_means(rows,"Prompt Type",PROMPT_TYPE_ORDER,"_overall")}
                                for lbl,rows in model_files]}
    country   = {"labels": COUNTRY_ORDER,
                 "datasets": [{"label":lbl,"data":group_means(rows,"Country",COUNTRY_ORDER,"_overall")}
                               for lbl,rows in model_files]}

    seen, all_topics = set(), []
    for _,rows in model_files:
        for r in rows:
            k = (r.get("Category",""), r.get("Topic",""))
            if k not in seen:
                seen.add(k); all_topics.append(k)
    cat_idx = {c:i for i,c in enumerate(CATEGORY_ORDER)}
    all_topics.sort(key=lambda x:(cat_idx.get(x[0],99),x[1]))
    topic_rows = []
    for cat,topic in all_topics:
        scores = []
        for _,rows in model_files:
            vals = [r["_overall"] for r in rows
                    if r.get("Category")==cat and r.get("Topic")==topic]
            scores.append(round(mean(vals),2) if vals else None)
        topic_rows.append({"category":cat,"topic":topic,"scores":scores})

    return {
        "models":    labels,
        "summary":   summary,
        "radar":     radar,
        "histograms":histograms,
        "category":  category,
        "promptType":promptType,
        "country":   country,
        "topicTable":topic_rows,
        "palette":   PALETTE[:len(labels)],
    }

# ---------------------------------------------------------------------------
# HTML template — uses __DATA_JSON__ placeholder, no f-string
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM Ideology Audit - Results Analysis</title>
<style>
*, *::before, *::after { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 14px; background: #f8fafc; color: #1e293b;
  margin: 0; padding: 24px 32px;
}
h1 { font-size: 22px; font-weight: 700; margin: 0 0 4px; color: #0f172a; }
.subtitle { color: #64748b; font-size: 13px; margin: 0 0 28px; }
h2 { font-size: 15px; font-weight: 700; color: #0f172a; margin: 0 0 14px;
     padding-bottom: 6px; border-bottom: 1px solid #e2e8f0; }
.section { margin-bottom: 36px; }
.card { background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px 20px; }
.card h3 { font-size: 12px; font-weight: 600; color: #475569; margin: 0 0 10px;
           text-transform: uppercase; letter-spacing: .05em; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
.grid3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 18px; }
.grid4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 18px; }
@media(max-width:1000px){ .grid4 { grid-template-columns: 1fr 1fr; } }
@media(max-width:700px){ .grid2,.grid3,.grid4 { grid-template-columns: 1fr; } }
canvas { width: 100% !important; display: block; }
.stbl { width:100%; border-collapse:collapse; background:white;
        border:1px solid #e2e8f0; border-radius:8px; overflow:hidden; }
.stbl th { background:#f8fafc; padding:8px 14px; text-align:left; font-size:11px;
           font-weight:600; text-transform:uppercase; letter-spacing:.05em;
           color:#64748b; border-bottom:1px solid #e2e8f0; }
.stbl td { padding:10px 14px; border-bottom:1px solid #f1f5f9; vertical-align:middle; }
.stbl tr:last-child td { border-bottom:none; }
.dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px;
       vertical-align: middle; }
.chip { display:inline-block; padding:3px 10px; border-radius:99px;
        font-weight:700; font-size:13px; }
.ibadge { display:inline-block; padding:2px 8px; border-radius:99px; font-size:11px;
          font-weight:600; background:#ede9fe; color:#5b21b6; }
.ttbl { width:100%; border-collapse:collapse; font-size:13px; background:white;
        border:1px solid #e2e8f0; border-radius:8px; overflow:hidden; }
.ttbl th { background:#f8fafc; padding:7px 12px; text-align:left; font-size:11px;
           font-weight:600; text-transform:uppercase; letter-spacing:.05em; color:#64748b;
           border-bottom:1px solid #e2e8f0; }
.ttbl td { padding:7px 12px; border-bottom:1px solid #f1f5f9; }
.ttbl tr:last-child td { border-bottom:none; }
.ttbl tr:hover td { background:#f8fafc; }
.ttbl td.sc { text-align:center; font-weight:600; font-variant-numeric:tabular-nums; }
.cat-badge { display:inline-block; padding:1px 7px; border-radius:3px; font-size:11px;
             background:#f1f5f9; color:#475569; white-space:nowrap; }
.tw { max-height:520px; overflow-y:auto; border-radius:8px; }
.fb-row { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin-bottom:12px; }
.fb-lbl { font-size:12px; color:#64748b; font-weight:600; }
.fb { padding:4px 12px; border-radius:99px; border:1px solid #e2e8f0; background:white;
      cursor:pointer; font-size:12px; font-weight:500; color:#475569; }
.fb:hover { border-color:#94a3b8; background:#f1f5f9; }
.fb.active { background:#1e293b; border-color:#1e293b; color:white; }
</style>
</head>
<body>
<h1>LLM Ideology Audit &#8212; Results Analysis</h1>
<p class="subtitle" id="subtitle">Loading&#8230;</p>

<div class="section">
  <h2>Model Summary</h2>
  <table class="stbl" id="summaryTable"></table>
</div>

<div class="section">
  <h2>Rubric Dimension Profile</h2>
  <div class="grid2">
    <div class="card"><h3>Radar</h3><canvas id="radarChart" style="height:280px"></canvas></div>
    <div class="card"><h3>By Dimension</h3><canvas id="dimBar" style="height:280px"></canvas></div>
  </div>
</div>

<div class="section">
  <h2>Score Distributions</h2>
  <div class="grid4">
    <div class="card"><h3>Overall</h3><canvas id="hOverall" style="height:180px"></canvas></div>
    <div class="card"><h3>Relevance &amp; Accuracy</h3><canvas id="hRelevance" style="height:180px"></canvas></div>
    <div class="card"><h3>Plurality &amp; Breadth</h3><canvas id="hPlurality" style="height:180px"></canvas></div>
    <div class="card"><h3>Coherence &amp; Conciseness</h3><canvas id="hCoherence" style="height:180px"></canvas></div>
  </div>
</div>

<div class="section">
  <h2>Score Breakdowns</h2>
  <div class="grid3">
    <div class="card"><h3>By Category</h3><canvas id="catChart" style="height:340px"></canvas></div>
    <div class="card"><h3>By Prompt Type</h3><canvas id="ptChart" style="height:340px"></canvas></div>
    <div class="card"><h3>By Country</h3><canvas id="countryChart" style="height:340px"></canvas></div>
  </div>
</div>

<div class="section">
  <h2>Per-Topic Scores</h2>
  <div class="fb-row" id="topicFilters"></div>
  <div class="tw"><table class="ttbl" id="topicTable"></table></div>
</div>

<script>
// ============================================================
// MiniChart: zero-dependency canvas charting
// ============================================================
const MC = (function() {
  var PR = window.devicePixelRatio || 1;

  function setup(canvas) {
    var rect = canvas.getBoundingClientRect();
    var w = (rect.width  || canvas.offsetWidth  || 400);
    var h = (rect.height || canvas.offsetHeight || 300);
    canvas.width  = w * PR;
    canvas.height = h * PR;
    canvas.style.width  = w + 'px';
    canvas.style.height = h + 'px';
    var ctx = canvas.getContext('2d');
    ctx.scale(PR, PR);
    return { ctx: ctx, w: w, h: h };
  }

  function hexRgba(hex, alpha) {
    var r = parseInt(hex.slice(1,3), 16);
    var g = parseInt(hex.slice(3,5), 16);
    var b = parseInt(hex.slice(5,7), 16);
    return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
  }

  function legend(ctx, models, palette, lx, ly, maxW) {
    if (models.length <= 1) return;
    ctx.font = '11px system-ui,sans-serif';
    var cx = lx;
    for (var i = 0; i < models.length; i++) {
      ctx.fillStyle = palette[i];
      ctx.fillRect(cx, ly - 9, 12, 10);
      ctx.fillStyle = '#475569';
      ctx.fillText(models[i], cx + 16, ly);
      cx += ctx.measureText(models[i]).width + 36;
      if (cx > lx + maxW - 60) { cx = lx; ly += 16; }
    }
  }

  // Grouped vertical bar chart
  function bar(canvasId, data, opts) {
    opts = opts || {};
    var canvas = document.getElementById(canvasId);
    if (!canvas) return;
    var setup_ = setup(canvas);
    var ctx = setup_.ctx, w = setup_.w, h = setup_.h;
    var labels = data.labels, datasets = data.datasets, palette = data.palette;
    var yMin = (opts.yMin !== undefined) ? opts.yMin : 0;
    var yMax = (opts.yMax !== undefined) ? opts.yMax : 5;

    var mL = 42, mR = 16, mT = 16, mB = datasets.length > 1 ? 52 : 38;
    var pw = w - mL - mR;
    var ph = h - mT - mB;
    var nG = labels.length, nS = datasets.length;
    var gW = pw / nG;
    var bW = Math.min(gW * 0.72 / nS, 44);
    var gPad = (gW - bW * nS) / 2;

    // Auto yMax for histograms
    if (opts.yMax === undefined) {
      var maxVal = 0;
      for (var si = 0; si < datasets.length; si++) {
        for (var gi2 = 0; gi2 < datasets[si].data.length; gi2++) {
          if (datasets[si].data[gi2] > maxVal) maxVal = datasets[si].data[gi2];
        }
      }
      yMax = Math.ceil(maxVal * 1.1) || 10;
    }

    // Grid lines
    ctx.strokeStyle = '#e2e8f0'; ctx.lineWidth = 1;
    var ySteps = 5;
    for (var yi = 0; yi <= ySteps; yi++) {
      var yy = mT + ph - (yi / ySteps) * ph;
      ctx.beginPath(); ctx.moveTo(mL, yy); ctx.lineTo(mL + pw, yy); ctx.stroke();
      ctx.fillStyle = '#94a3b8'; ctx.font = '10px system-ui,sans-serif'; ctx.textAlign = 'right';
      var tickVal = yMin + (yMax - yMin) * yi / ySteps;
      ctx.fillText(tickVal % 1 === 0 ? tickVal.toFixed(0) : tickVal.toFixed(1), mL - 4, yy + 3.5);
    }

    // Bars
    for (var si = 0; si < datasets.length; si++) {
      ctx.fillStyle = hexRgba(palette[si], 0.82);
      for (var gi = 0; gi < labels.length; gi++) {
        var val = datasets[si].data[gi];
        if (val === null || val === undefined) continue;
        var x = mL + gi * gW + gPad + si * bW;
        var bH = ph * (val - yMin) / (yMax - yMin);
        ctx.fillRect(x, mT + ph - bH, bW - 1, bH);
      }
    }

    // X labels
    ctx.fillStyle = '#475569'; ctx.font = '11px system-ui,sans-serif'; ctx.textAlign = 'center';
    for (var gi = 0; gi < labels.length; gi++) {
      var lbl = labels[gi];
      var short = lbl.length > 10 ? lbl.slice(0, 9) + '\u2026' : lbl;
      ctx.fillText(short, mL + gi * gW + gW / 2, mT + ph + 14);
    }

    legend(ctx, datasets.map(function(d) { return d.label; }), palette,
           mL, mT + ph + (datasets.length > 1 ? 36 : 0), pw);
  }

  // Horizontal bar chart
  function hbar(canvasId, data, opts) {
    opts = opts || {};
    var canvas = document.getElementById(canvasId);
    if (!canvas) return;
    var setup_ = setup(canvas);
    var ctx = setup_.ctx, w = setup_.w, h = setup_.h;
    var labels = data.labels, datasets = data.datasets, palette = data.palette;
    var xMax = opts.xMax || 5;

    var nG = labels.length, nS = datasets.length;
    var mL = 8, labelW = 195, mR = 48, mT = 12;
    var mB = nS > 1 ? 28 : 12;
    var ph = h - mT - mB;
    var pw = w - mL - labelW - mR;
    var rowH = ph / nG;
    var bH = Math.min(rowH * 0.68 / nS, 24);
    var rowPad = (rowH - bH * nS) / 2;

    // Grid lines
    ctx.strokeStyle = '#e2e8f0'; ctx.lineWidth = 1;
    for (var xi = 0; xi <= 5; xi++) {
      var xx = mL + labelW + pw * xi / 5;
      ctx.beginPath(); ctx.moveTo(xx, mT); ctx.lineTo(xx, mT + ph); ctx.stroke();
      ctx.fillStyle = '#94a3b8'; ctx.font = '9px system-ui,sans-serif'; ctx.textAlign = 'center';
      ctx.fillText(xi.toFixed(0), xx, mT + ph + 10);
    }

    // Bars
    for (var si = 0; si < datasets.length; si++) {
      for (var gi = 0; gi < labels.length; gi++) {
        var val = datasets[si].data[gi];
        if (val === null || val === undefined) continue;
        var y = mT + gi * rowH + rowPad + si * bH;
        var bw = pw * val / xMax;
        ctx.fillStyle = hexRgba(palette[si], 0.82);
        ctx.fillRect(mL + labelW, y, bw, bH - 1);
        ctx.fillStyle = '#475569'; ctx.font = '10px system-ui,sans-serif'; ctx.textAlign = 'left';
        ctx.fillText(val.toFixed(2), mL + labelW + bw + 3, y + bH - 2);
      }
    }

    // Row labels
    ctx.fillStyle = '#1e293b'; ctx.font = '11px system-ui,sans-serif'; ctx.textAlign = 'right';
    for (var gi = 0; gi < labels.length; gi++) {
      var lbl = labels[gi];
      var short = lbl.length > 28 ? lbl.slice(0, 27) + '\u2026' : lbl;
      ctx.fillText(short, mL + labelW - 6, mT + gi * rowH + rowH / 2 + 4);
    }

    if (nS > 1) {
      legend(ctx, datasets.map(function(d) { return d.label; }), palette,
             mL + labelW, mT + ph + 24, pw);
    }
  }

  // Radar chart
  function radar(canvasId, data, opts) {
    opts = opts || {};
    var canvas = document.getElementById(canvasId);
    if (!canvas) return;
    var setup_ = setup(canvas);
    var ctx = setup_.ctx, w = setup_.w, h = setup_.h;
    var labels = data.labels, datasets = data.datasets, palette = data.palette;

    var legH = datasets.length > 1 ? 24 : 0;
    var cx = w / 2, cy = (h - legH) / 2;
    var r = Math.min(cx, cy) * 0.68;
    var n = labels.length, rings = 5, maxVal = 5;

    function pt(angle, val) {
      var frac = val / maxVal;
      return [cx + r * frac * Math.cos(angle), cy + r * frac * Math.sin(angle)];
    }
    var angles = [];
    for (var i = 0; i < n; i++) {
      angles.push(-Math.PI / 2 + (2 * Math.PI * i / n));
    }

    // Rings
    ctx.strokeStyle = '#e2e8f0'; ctx.lineWidth = 1;
    for (var ring = 1; ring <= rings; ring++) {
      ctx.beginPath();
      for (var i = 0; i < n; i++) {
        var p = pt(angles[i], maxVal * ring / rings);
        i === 0 ? ctx.moveTo(p[0], p[1]) : ctx.lineTo(p[0], p[1]);
      }
      ctx.closePath(); ctx.stroke();
      ctx.fillStyle = '#94a3b8'; ctx.font = '9px system-ui,sans-serif'; ctx.textAlign = 'center';
      ctx.fillText((maxVal * ring / rings).toFixed(1), cx + 4, cy - r * ring / rings + 3);
    }

    // Spokes
    for (var i = 0; i < n; i++) {
      var p = pt(angles[i], maxVal);
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(p[0], p[1]); ctx.stroke();
    }

    // Axis labels
    ctx.fillStyle = '#1e293b'; ctx.font = 'bold 12px system-ui,sans-serif'; ctx.textAlign = 'center';
    for (var i = 0; i < n; i++) {
      var lr = r + 26;
      var lx = cx + lr * Math.cos(angles[i]);
      var ly = cy + lr * Math.sin(angles[i]);
      var parts = labels[i].split(' & ');
      for (var pi = 0; pi < parts.length; pi++) {
        ctx.fillText(parts[pi], lx, ly + pi * 14 - (parts.length - 1) * 7);
      }
    }

    // Datasets
    for (var si = 0; si < datasets.length; si++) {
      ctx.beginPath();
      for (var i = 0; i < n; i++) {
        var p = pt(angles[i], datasets[si].data[i]);
        i === 0 ? ctx.moveTo(p[0], p[1]) : ctx.lineTo(p[0], p[1]);
      }
      ctx.closePath();
      ctx.strokeStyle = palette[si]; ctx.lineWidth = 2.5; ctx.stroke();
      ctx.fillStyle = hexRgba(palette[si], 0.13); ctx.fill();
      for (var i = 0; i < n; i++) {
        var p = pt(angles[i], datasets[si].data[i]);
        ctx.beginPath(); ctx.arc(p[0], p[1], 4, 0, 2 * Math.PI);
        ctx.fillStyle = palette[si]; ctx.fill();
      }
    }

    if (datasets.length > 1) {
      legend(ctx, datasets.map(function(d) { return d.label; }), palette,
             cx - r, h - legH + 12, r * 2);
    }
  }

  return { bar: bar, hbar: hbar, radar: radar };
})();

// ============================================================
// App
// ============================================================
var DATA = __DATA_JSON__;
var P = DATA.palette;

function scoreColor(v) {
  if (v === null || v === undefined) return '#94a3b8';
  if (v >= 4.5) return '#16a34a';
  if (v >= 3.5) return '#65a30d';
  if (v >= 2.5) return '#ca8a04';
  if (v >= 1.5) return '#dc2626';
  return '#9f1239';
}
function scoreBg(v) {
  if (v === null || v === undefined) return '#f8fafc';
  if (v >= 4.5) return '#f0fdf4';
  if (v >= 3.5) return '#f7fee7';
  if (v >= 2.5) return '#fefce8';
  if (v >= 1.5) return '#fef2f2';
  return '#fff1f2';
}

// Subtitle
var nM = DATA.models.length;
var nN = DATA.summary.reduce(function(s, m) { return s + m.n; }, 0);
document.getElementById('subtitle').textContent =
  nM + ' model' + (nM > 1 ? 's' : '') + ' \u00b7 ' + nN + ' scored responses';

// 1. Summary table
(function() {
  var dims = [
    ['overall_score','Overall'],
    ['relevance_accuracy_score','Relevance'],
    ['plurality_breadth_score','Plurality'],
    ['coherence_conciseness_score','Coherence']
  ];
  var h = '<thead><tr><th>Model</th><th>N</th>';
  for (var d = 0; d < dims.length; d++) h += '<th>' + dims[d][1] + '</th>';
  h += '<th>Interesting%</th></tr></thead><tbody>';
  for (var i = 0; i < DATA.summary.length; i++) {
    var m = DATA.summary[i];
    h += '<tr><td><span class="dot" style="background:' + P[i] + '"></span><strong>' + m.model + '</strong></td>';
    h += '<td style="color:#64748b">' + m.n + '</td>';
    for (var d = 0; d < dims.length; d++) {
      var v = m[dims[d][0] + '_mean'], s = m[dims[d][0] + '_std'];
      h += '<td><span class="chip" style="background:' + scoreBg(v) + ';color:' + scoreColor(v) + '">' +
           v.toFixed(2) + '</span><span style="font-size:11px;color:#94a3b8"> \u00b1' + s.toFixed(2) + '</span></td>';
    }
    h += '<td><span class="ibadge">' + m.interesting_pct + '%</span></td></tr>';
  }
  document.getElementById('summaryTable').innerHTML = h + '</tbody>';
})();

// 2. Radar
MC.radar('radarChart', DATA.radar, { palette: P });

// 2b. Dimension bar
(function() {
  var dimLabels = ['Relevance', 'Plurality', 'Coherence'];
  var ds = DATA.summary.map(function(m) {
    return {
      label: m.model,
      data: [m.relevance_accuracy_score_mean, m.plurality_breadth_score_mean, m.coherence_conciseness_score_mean]
    };
  });
  MC.bar('dimBar', { labels: dimLabels, datasets: ds, palette: P }, { yMin: 0, yMax: 5 });
})();

// 3. Histograms
var histKeys = ['overall', 'relevance', 'plurality', 'coherence'];
var histIds  = ['hOverall', 'hRelevance', 'hPlurality', 'hCoherence'];
for (var hi = 0; hi < histKeys.length; hi++) {
  var hd = DATA.histograms[histKeys[hi]];
  MC.bar(histIds[hi], { labels: hd.bins, datasets: hd.datasets, palette: P });
}

// 4. Breakdowns
MC.hbar('catChart',     { labels: DATA.category.labels,   datasets: DATA.category.datasets,   palette: P });
MC.bar ('ptChart',      { labels: DATA.promptType.labels, datasets: DATA.promptType.datasets, palette: P }, { yMin: 0, yMax: 5 });
MC.bar ('countryChart', { labels: DATA.country.labels,    datasets: DATA.country.datasets,    palette: P }, { yMin: 0, yMax: 5 });

// 5. Topic table
var activecat = 'all';

(function buildFilters() {
  var cats = [];
  var seen = {};
  for (var i = 0; i < DATA.topicTable.length; i++) {
    var c = DATA.topicTable[i].category;
    if (!seen[c]) { seen[c] = true; cats.push(c); }
  }
  var h = '<span class="fb-lbl">Category:</span>';
  h += '<button class="fb active" onclick="filterT(\'all\',this)">All</button>';
  for (var i = 0; i < cats.length; i++) {
    var c = cats[i];
    var s = c.length > 28 ? c.slice(0, 27) + '\u2026' : c;
    h += '<button class="fb" onclick="filterT(\'' + c.replace(/'/g, "\\'") + '\',this)" title="' + c + '">' + s + '</button>';
  }
  document.getElementById('topicFilters').innerHTML = h;
})();

function renderTopicTable() {
  var h = '<thead><tr><th>Category</th><th>Topic</th>';
  for (var i = 0; i < DATA.models.length; i++) h += '<th style="text-align:center">' + DATA.models[i] + '</th>';
  h += '</tr></thead><tbody>';
  for (var i = 0; i < DATA.topicTable.length; i++) {
    var row = DATA.topicTable[i];
    if (activecat !== 'all' && row.category !== activecat) continue;
    h += '<tr><td><span class="cat-badge">' + row.category.split(' ')[0] + '</span></td>';
    h += '<td>' + row.topic + '</td>';
    for (var s = 0; s < row.scores.length; s++) {
      var v = row.scores[s];
      var d = v !== null ? v.toFixed(2) : '\u2014';
      h += '<td class="sc" style="color:' + scoreColor(v) + ';background:' + scoreBg(v) + '">' + d + '</td>';
    }
    h += '</tr>';
  }
  document.getElementById('topicTable').innerHTML = h + '</tbody>';
}
renderTopicTable();

function filterT(val, btn) {
  activecat = val;
  var btns = document.querySelectorAll('#topicFilters .fb');
  for (var i = 0; i < btns.length; i++) btns[i].classList.remove('active');
  btn.classList.add('active');
  renderTopicTable();
}
</script>
</body>
</html>
"""


def build_html(data):
    data_json = json.dumps(data, ensure_ascii=False)
    return HTML_TEMPLATE.replace("__DATA_JSON__", data_json)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze and visualize LLM judge results (self-contained HTML, no CDN).",
    )
    parser.add_argument("--input", nargs="+", metavar="CSV",
        help="One or more judge_scores_*.csv files (default: auto-discover judge_results/)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, metavar="HTML",
        help="Output HTML path (default: " + DEFAULT_OUTPUT + ")")
    parser.add_argument("--judge-dir", default=DEFAULT_GLOB_DIR, metavar="DIR")
    args = parser.parse_args()

    paths = [Path(p) for p in args.input] if args.input else discover_files(args.judge_dir)
    if not paths:
        print("No judge_scores_*.csv found in '" + args.judge_dir + "/'. Use --input.", file=sys.stderr)
        sys.exit(1)

    missing = [p for p in paths if not p.exists()]
    if missing:
        for p in missing: print("Not found: " + str(p), file=sys.stderr)
        sys.exit(1)

    model_files = []
    for path in paths:
        rows = load_file(path)
        if not rows:
            print("Warning: no valid rows in " + str(path), file=sys.stderr)
            continue
        lbl = model_label(rows, path)
        model_files.append((lbl, rows))
        print("  Loaded: " + path.name + "  ->  " + lbl + "  (" + str(len(rows)) + " rows)")

    if not model_files:
        print("No valid data.", file=sys.stderr)
        sys.exit(1)

    print("\nBuilding report for " + str(len(model_files)) + " model(s)...")
    data = build_data(model_files)
    html = build_html(data)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print("Report written -> " + str(out))
    total = sum(len(r) for _, r in model_files)
    print("  " + str(len(model_files)) + " model(s)  *  " + str(total) + " responses")

if __name__ == "__main__":
    main()
