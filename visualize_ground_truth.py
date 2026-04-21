#!/usr/bin/env python3
"""
visualize_ground_truth.py — Generate an interactive HTML visualization of
the Wikipedia ground truth articles sourced by build_ground_truth.py.

Opens ground_truth.csv and produces ground_truth_viz.html, which can be
opened in any browser.  Rows are grouped by country and color-coded by
quality (good / fallback / empty).  Article titles link directly to their
Wikipedia pages.

Usage:
    python visualize_ground_truth.py
    python visualize_ground_truth.py --input ground_truth.csv --output ground_truth_viz.html
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_INPUT  = "ground_truth.csv"
DEFAULT_OUTPUT = "ground_truth_viz.html"

COUNTRY_ORDER = ["US", "China", "India", "France"]

QUALITY_META = {
    "good":     {"label": "Good",     "color": "#16a34a", "bg": "#f0fdf4", "border": "#bbf7d0"},
    "fallback": {"label": "Fallback", "color": "#b45309", "bg": "#fffbeb", "border": "#fde68a"},
    "empty":    {"label": "Empty",    "color": "#dc2626", "bg": "#fef2f2", "border": "#fecaca"},
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ground_truth(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["article_titles"] = json.loads(row.get("article_titles") or "[]")
        row["article_urls"]   = json.loads(row.get("article_urls")   or "[]")
        row["char_counts"]    = json.loads(row.get("char_counts")     or "[]")
        row["article_count"]  = int(row.get("article_count") or 0)
        row["total_chars"]    = int(row.get("total_chars")   or 0)
    return rows


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def quality_badge(quality: str) -> str:
    m = QUALITY_META.get(quality, QUALITY_META["empty"])
    return (
        f'<span class="badge" style="'
        f'background:{m["bg"]};color:{m["color"]};'
        f'border:1px solid {m["border"]}">'
        f'{m["label"]}</span>'
    )


def article_pills(titles: list[str], urls: list[str], chars: list[int]) -> str:
    if not titles:
        return '<span class="no-article">No article found</span>'
    pills = []
    for i, (title, url, ch) in enumerate(zip(titles, urls, chars)):
        pills.append(
            f'<a class="article-pill" href="{url}" target="_blank" title="{ch:,} chars">'
            f'{title}</a>'
        )
    return "".join(pills)


def build_summary(rows: list[dict]) -> dict:
    total  = len(rows)
    good   = sum(1 for r in rows if r["quality"] == "good")
    fb     = sum(1 for r in rows if r["quality"] == "fallback")
    empty  = sum(1 for r in rows if r["quality"] == "empty")
    return {"total": total, "good": good, "fallback": fb, "empty": empty}


def _country_filter_buttons(countries: list[str]) -> str:
    """Build country filter buttons without backslashes inside an f-string."""
    parts = []
    for c in countries:
        parts.append(
            f'<button class="filter-btn" '
            f'onclick="filterCountry(\'{c}\', this)">{c}</button>'
        )
    return "".join(parts)


def build_html(rows: list[dict]) -> str:
    summary = build_summary(rows)

    # Group by country in display order
    by_country: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_country[row["country"]].append(row)

    ordered_countries = [c for c in COUNTRY_ORDER if c in by_country]
    for c in by_country:
        if c not in ordered_countries:
            ordered_countries.append(c)

    # Build country sections
    sections_html = ""
    for country in ordered_countries:
        country_rows = by_country[country]
        c_good  = sum(1 for r in country_rows if r["quality"] == "good")
        c_fb    = sum(1 for r in country_rows if r["quality"] == "fallback")
        c_empty = sum(1 for r in country_rows if r["quality"] == "empty")

        rows_html = ""
        for row in country_rows:
            q    = row["quality"]
            meta = QUALITY_META.get(q, QUALITY_META["empty"])
            pills = article_pills(row["article_titles"], row["article_urls"], row["char_counts"])
            query = row.get("query_used", "")
            query_label = ""
            if query.startswith("fallback:"):
                query_label = (
                    f'<span class="query-tag fallback-tag" title="Fallback query used">'
                    f'fallback: {query[9:]}</span>'
                )
            elif query.startswith("opensearch:"):
                query_label = (
                    f'<span class="query-tag opensearch-tag" title="OpenSearch used">'
                    f'opensearch: {query[11:]}</span>'
                )

            rows_html += f"""
            <tr class="topic-row quality-{q}" data-country="{country}" data-quality="{q}">
              <td class="topic-cell">
                <div class="topic-name">{row["topic"]}</div>
                {query_label}
              </td>
              <td class="quality-cell">{quality_badge(q)}</td>
              <td class="articles-cell">{pills}</td>
              <td class="chars-cell">{row["total_chars"]:,}</td>
            </tr>"""

        sections_html += f"""
        <div class="country-section" data-country="{country}">
          <div class="country-header">
            <span class="country-name">{country}</span>
            <span class="country-stats">
              {len(country_rows)} topics &nbsp;·&nbsp;
              <span style="color:#16a34a">{c_good} good</span>
              {f'&nbsp;·&nbsp;<span style="color:#b45309">{c_fb} fallback</span>' if c_fb else ''}
              {f'&nbsp;·&nbsp;<span style="color:#dc2626">{c_empty} empty</span>' if c_empty else ''}
            </span>
          </div>
          <table class="topic-table">
            <thead>
              <tr>
                <th>Topic</th>
                <th>Quality</th>
                <th>Wikipedia Articles</th>
                <th>Chars</th>
              </tr>
            </thead>
            <tbody>{rows_html}
            </tbody>
          </table>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>LLM Ideology Audit — Wikipedia Ground Truth</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
      background: #f8fafc;
      color: #1e293b;
      margin: 0;
      padding: 24px;
    }}
    h1 {{ font-size: 22px; font-weight: 700; margin: 0 0 4px; color: #0f172a; }}
    .subtitle {{ color: #64748b; margin: 0 0 24px; font-size: 13px; }}

    /* Summary bar */
    .summary {{
      display: flex;
      gap: 12px;
      margin-bottom: 20px;
      flex-wrap: wrap;
    }}
    .summary-card {{
      background: white;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 12px 20px;
      text-align: center;
      min-width: 110px;
    }}
    .summary-card .val {{
      font-size: 26px;
      font-weight: 700;
      line-height: 1.1;
    }}
    .summary-card .lbl {{
      font-size: 11px;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: .05em;
      margin-top: 2px;
    }}
    .val-good     {{ color: #16a34a; }}
    .val-fallback {{ color: #b45309; }}
    .val-empty    {{ color: #dc2626; }}

    /* Filters */
    .filters {{ margin-bottom: 20px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }}
    .filter-label {{ font-size: 12px; color: #64748b; font-weight: 600; margin-right: 4px; }}
    .filter-btn {{
      padding: 5px 14px;
      border-radius: 99px;
      border: 1px solid #e2e8f0;
      background: white;
      cursor: pointer;
      font-size: 12px;
      font-weight: 500;
      color: #475569;
      transition: all .15s;
    }}
    .filter-btn:hover {{ border-color: #94a3b8; background: #f1f5f9; }}
    .filter-btn.active {{
      background: #1e293b;
      border-color: #1e293b;
      color: white;
    }}
    .filter-sep {{ width: 1px; background: #e2e8f0; height: 24px; margin: 0 4px; }}

    /* Country sections */
    .country-section {{ margin-bottom: 32px; }}
    .country-header {{
      display: flex;
      align-items: baseline;
      gap: 12px;
      margin-bottom: 8px;
    }}
    .country-name {{
      font-size: 16px;
      font-weight: 700;
      color: #0f172a;
    }}
    .country-stats {{ font-size: 12px; color: #64748b; }}

    /* Table */
    .topic-table {{
      width: 100%;
      border-collapse: collapse;
      background: white;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      overflow: hidden;
    }}
    .topic-table thead th {{
      background: #f8fafc;
      padding: 8px 12px;
      text-align: left;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: .05em;
      color: #64748b;
      border-bottom: 1px solid #e2e8f0;
    }}
    .topic-table tbody tr {{
      border-bottom: 1px solid #f1f5f9;
      transition: background .1s;
    }}
    .topic-table tbody tr:last-child {{ border-bottom: none; }}
    .topic-table tbody tr:hover {{ background: #f8fafc; }}
    .topic-table td {{ padding: 10px 12px; vertical-align: top; }}

    .topic-cell {{ min-width: 180px; max-width: 240px; }}
    .topic-name {{ font-weight: 600; color: #0f172a; margin-bottom: 3px; }}
    .quality-cell {{ white-space: nowrap; }}
    .articles-cell {{ }}
    .chars-cell {{
      white-space: nowrap;
      text-align: right;
      color: #64748b;
      font-size: 12px;
      font-variant-numeric: tabular-nums;
    }}

    /* Badges */
    .badge {{
      display: inline-block;
      padding: 2px 9px;
      border-radius: 99px;
      font-size: 11px;
      font-weight: 600;
    }}

    /* Article pills */
    .article-pill {{
      display: inline-block;
      padding: 3px 10px;
      margin: 2px 4px 2px 0;
      background: #f1f5f9;
      border: 1px solid #e2e8f0;
      border-radius: 99px;
      color: #1e40af;
      font-size: 12px;
      text-decoration: none;
      transition: all .15s;
    }}
    .article-pill:hover {{
      background: #dbeafe;
      border-color: #93c5fd;
      color: #1d4ed8;
    }}
    .no-article {{
      font-size: 12px;
      color: #94a3b8;
      font-style: italic;
    }}

    /* Query tags */
    .query-tag {{
      display: inline-block;
      font-size: 10px;
      padding: 1px 6px;
      border-radius: 3px;
      margin-top: 3px;
      font-family: ui-monospace, monospace;
    }}
    .fallback-tag  {{ background: #fef3c7; color: #92400e; }}
    .opensearch-tag {{ background: #ede9fe; color: #5b21b6; }}

    /* Hidden rows */
    .topic-row.hidden {{ display: none; }}
    .country-section.hidden {{ display: none; }}
  </style>
</head>
<body>
  <h1>Wikipedia Ground Truth</h1>
  <p class="subtitle">LLM Ideology Audit · {summary["total"]} unique (topic, country) pairs</p>

  <div class="summary">
    <div class="summary-card">
      <div class="val">{summary["total"]}</div>
      <div class="lbl">Total Pairs</div>
    </div>
    <div class="summary-card">
      <div class="val val-good">{summary["good"]}</div>
      <div class="lbl">Good</div>
    </div>
    <div class="summary-card">
      <div class="val val-fallback">{summary["fallback"]}</div>
      <div class="lbl">Fallback</div>
    </div>
    <div class="summary-card">
      <div class="val val-empty">{summary["empty"]}</div>
      <div class="lbl">Empty</div>
    </div>
  </div>

  <div class="filters">
    <span class="filter-label">Country:</span>
    <button class="filter-btn active" onclick="filterCountry('all', this)">All</button>
    {_country_filter_buttons(ordered_countries)}
    <div class="filter-sep"></div>
    <span class="filter-label">Quality:</span>
    <button class="filter-btn active" onclick="filterQuality('all', this)">All</button>
    <button class="filter-btn" onclick="filterQuality('good', this)">Good</button>
    <button class="filter-btn" onclick="filterQuality('fallback', this)">Fallback</button>
    <button class="filter-btn" onclick="filterQuality('empty', this)">Empty</button>
  </div>

  <div id="sections">
    {sections_html}
  </div>

  <script>
    let activeCountry = 'all';
    let activeQuality = 'all';

    function filterCountry(val, btn) {{
      activeCountry = val;
      document.querySelectorAll('.filters .filter-btn').forEach(b => {{
        if (b.getAttribute('onclick') && b.getAttribute('onclick').includes('filterCountry')) {{
          b.classList.remove('active');
        }}
      }});
      btn.classList.add('active');
      applyFilters();
    }}

    function filterQuality(val, btn) {{
      activeQuality = val;
      document.querySelectorAll('.filters .filter-btn').forEach(b => {{
        if (b.getAttribute('onclick') && b.getAttribute('onclick').includes('filterQuality')) {{
          b.classList.remove('active');
        }}
      }});
      btn.classList.add('active');
      applyFilters();
    }}

    function applyFilters() {{
      document.querySelectorAll('.topic-row').forEach(row => {{
        const matchCountry = activeCountry === 'all' || row.dataset.country === activeCountry;
        const matchQuality = activeQuality === 'all' || row.dataset.quality === activeQuality;
        row.classList.toggle('hidden', !(matchCountry && matchQuality));
      }});
      // Hide country sections where all rows are hidden
      document.querySelectorAll('.country-section').forEach(section => {{
        const visible = section.querySelectorAll('.topic-row:not(.hidden)').length > 0;
        section.classList.toggle('hidden', !visible);
      }});
    }}
  </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an HTML visualization of the Wikipedia ground truth.",
    )
    parser.add_argument("--input",  default=DEFAULT_INPUT,  help="ground_truth.csv path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output HTML file")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: {input_path} not found. Run build_ground_truth.py first.", file=sys.stderr)
        sys.exit(1)

    rows = load_ground_truth(input_path)
    html = build_html(rows)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    summary = build_summary(rows)
    print(f"Written → {output_path}")
    print(f"  {summary['total']} pairs  |  "
          f"{summary['good']} good  |  "
          f"{summary['fallback']} fallback  |  "
          f"{summary['empty']} empty")


if __name__ == "__main__":
    main()
