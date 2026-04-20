#!/usr/bin/env python3
"""
Leaderboard Data Builder for LLM Ideology Audit
================================================
Reads one or more judge_scores_*.csv files, joins model metadata, and
outputs a single leaderboard_data.json file consumed by leaderboard.html.

The output contains aggregate statistics broken down by:
  • Model
  • Country of origin (derived from model_metadata.json)
  • Audit category
  • Prompt type (Standardized / Pluralistic / Biased)
  • Relevant country in the prompt (US / China / India / France)
  • Individual topic (for deep-dive views)

It also extracts flagged "interesting" responses for qualitative review.

Usage:
    # Build from all judge_scores_*.csv files in current directory
    python build_leaderboard_data.py

    # Specify input files and output path
    python build_leaderboard_data.py \\
        --input judge_scores_opus.csv judge_scores_sarvam30b.csv \\
        --output leaderboard_data.json \\
        --metadata model_metadata.json

    # Limit interesting responses kept per model
    python build_leaderboard_data.py --max-interesting 50

Requirements:
    pip install (standard library only)
"""

import argparse
import csv
import json
import logging
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Score dimensions
# ---------------------------------------------------------------------------

SCORE_DIMS = [
    ("relevance_accuracy_score",    "Relevance & Accuracy"),
    ("plurality_breadth_score",     "Plurality & Breadth"),
    ("coherence_conciseness_score", "Coherence & Conciseness"),
]

DIM_KEYS = [d[0] for d in SCORE_DIMS]
DIM_LABELS = {k: v for k, v in SCORE_DIMS}


# ---------------------------------------------------------------------------
# Model metadata helpers
# ---------------------------------------------------------------------------

def load_model_metadata(metadata_file: str) -> list[dict]:
    """Load model_metadata.json."""
    with open(metadata_file, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("models", [])


def resolve_model_meta(model_id: str, metadata: list[dict]) -> dict:
    """
    Match a model ID string against model_metadata patterns.
    Returns the best match or a generic fallback.
    """
    model_id_lower = model_id.lower()
    for entry in metadata:
        pattern = entry.get("model_id_pattern", "").lower()
        if pattern and pattern in model_id_lower:
            return entry
    # Fallback — unknown model
    return {
        "model_id_pattern": model_id,
        "display_name": model_id,
        "provider": "Unknown",
        "country_of_origin": "Unknown",
        "flag": "🏳️",
        "color": "#6B7280",
    }


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_scores(files: list[str]) -> list[dict]:
    """Load and merge all judge_scores CSV files."""
    all_rows = []
    for fpath in files:
        try:
            with open(fpath, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = [dict(r) for r in reader]
            logger.info(f"  {fpath}: {len(rows)} rows")
            all_rows.extend(rows)
        except Exception as e:
            logger.error(f"Failed to load {fpath}: {e}")
    logger.info(f"Total rows loaded: {len(all_rows)}")
    return all_rows


def clean_row(row: dict) -> Optional[dict]:
    """
    Validate and coerce a score row.
    Returns None if the row should be excluded (error, missing scores).
    """
    if row.get("judge_error") and row["judge_error"].strip():
        return None
    for key in DIM_KEYS:
        val = row.get(key, "")
        try:
            row[key] = int(float(str(val).strip()))
        except (ValueError, TypeError):
            return None
        if not (1 <= row[key] <= 5):
            return None
    # Coerce overall_score
    try:
        row["overall_score"] = float(row.get("overall_score", 0) or 0)
    except (ValueError, TypeError):
        scores = [row[k] for k in DIM_KEYS]
        row["overall_score"] = round(sum(scores) / len(scores), 3)
    # Coerce interesting_flag
    flag = str(row.get("interesting_flag", "")).lower()
    row["interesting_flag"] = flag in ("true", "1", "yes")
    return row


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def compute_stats(values: list[float]) -> dict:
    """Return mean, std, median, min, max, n for a list of numeric values."""
    if not values:
        return {"mean": None, "std": None, "median": None, "min": None, "max": None, "n": 0}
    n = len(values)
    mean = round(statistics.mean(values), 3)
    std = round(statistics.stdev(values), 3) if n > 1 else 0.0
    median = round(statistics.median(values), 3)
    return {
        "mean": mean,
        "std": std,
        "median": median,
        "min": min(values),
        "max": max(values),
        "n": n,
    }


def score_block(rows: list[dict]) -> dict:
    """Compute per-dimension stats + overall for a list of scored rows."""
    block = {}
    all_scores = []
    for dim_key, dim_label in SCORE_DIMS:
        vals = [r[dim_key] for r in rows]
        block[dim_key] = compute_stats(vals)
        block[dim_key]["label"] = dim_label
        all_scores.extend(vals)
    block["overall"] = compute_stats(all_scores)
    return block


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def build_model_summary(rows: list[dict], meta: dict) -> dict:
    """Build the top-level summary card for one model."""
    return {
        "model_id": rows[0].get("Model", "unknown"),
        "display_name": meta["display_name"],
        "provider": meta["provider"],
        "country_of_origin": meta["country_of_origin"],
        "flag": meta["flag"],
        "color": meta.get("color", "#6B7280"),
        "n_responses": len(rows),
        "scores": score_block(rows),
    }


def group_by(rows: list[dict], key_fn) -> dict[str, list[dict]]:
    """Group rows by a key function."""
    groups: dict[str, list] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    return dict(groups)


def build_breakdown(model_rows_map: dict[str, list[dict]], key_fn) -> dict:
    """
    For each model, compute score_block for each group produced by key_fn.
    Returns {model_display_name: {group_name: score_block}}.
    """
    result = {}
    for model_name, rows in model_rows_map.items():
        grouped = group_by(rows, key_fn)
        result[model_name] = {
            group: {**score_block(group_rows), "n": len(group_rows)}
            for group, group_rows in sorted(grouped.items())
        }
    return result


def build_country_origin_summary(
    model_summaries: list[dict],
    model_rows_map: dict[str, list[dict]],
) -> list[dict]:
    """
    Aggregate across all models with the same country of origin.
    """
    by_country: dict[str, list[dict]] = defaultdict(list)
    country_meta: dict[str, dict] = {}
    for summary in model_summaries:
        country = summary["country_of_origin"]
        by_country[country].extend(
            model_rows_map[summary["display_name"]]
        )
        if country not in country_meta:
            country_meta[country] = {
                "flag": summary["flag"],
                "providers": [],
            }
        country_meta[country]["providers"].append(summary["provider"])

    result = []
    for country, rows in sorted(by_country.items()):
        meta = country_meta[country]
        result.append({
            "country_of_origin": country,
            "flag": meta["flag"],
            "providers": sorted(set(meta["providers"])),
            "n_responses": len(rows),
            "scores": score_block(rows),
        })
    return result


def build_topic_matrix(model_rows_map: dict[str, list[dict]]) -> list[dict]:
    """
    Build a flat list of {topic, country, category, model_scores: {...}} records
    for the topic-level heatmap.
    """
    # Collect all (topic, country, category) combinations
    all_keys: set[tuple] = set()
    for rows in model_rows_map.values():
        for row in rows:
            all_keys.add((
                row.get("Topic", ""),
                row.get("Country", ""),
                row.get("Category", ""),
            ))

    topic_list = []
    for topic, country, category in sorted(all_keys):
        entry = {
            "topic": topic,
            "country": country,
            "category": category,
            "model_scores": {},
        }
        for model_name, rows in model_rows_map.items():
            topic_rows = [
                r for r in rows
                if r.get("Topic") == topic and r.get("Country") == country
            ]
            if topic_rows:
                entry["model_scores"][model_name] = {
                    "overall": round(
                        statistics.mean(r["overall_score"] for r in topic_rows), 3
                    ),
                    **{
                        dim_key: round(
                            statistics.mean(r[dim_key] for r in topic_rows), 3
                        )
                        for dim_key in DIM_KEYS
                    },
                    "n": len(topic_rows),
                }
        topic_list.append(entry)
    return topic_list


def extract_interesting(
    model_rows_map: dict[str, list[dict]],
    max_per_model: int = 50,
) -> list[dict]:
    """Extract flagged interesting responses for qualitative review."""
    interesting = []
    for model_name, rows in model_rows_map.items():
        flagged = [r for r in rows if r.get("interesting_flag")]
        # Sort by overall_score ascending (most surprising low scores first) + descending
        # Interleave extremes for variety
        sorted_low = sorted(flagged, key=lambda r: r["overall_score"])
        sorted_high = sorted(flagged, key=lambda r: r["overall_score"], reverse=True)
        seen = set()
        selected = []
        for r in sorted_low + sorted_high:
            uid = r.get("Prompt UID", "") + r.get("Model", "")
            if uid not in seen:
                seen.add(uid)
                selected.append(r)
            if len(selected) >= max_per_model:
                break

        for r in selected:
            interesting.append({
                "model": model_name,
                "model_id": r.get("Model", ""),
                "topic": r.get("Topic", ""),
                "category": r.get("Category", ""),
                "country": r.get("Country", ""),
                "prompt_type": r.get("Prompt Type", ""),
                "prompt": r.get("Prompt Text", r.get("Prompt", ""))[:500],
                "response": r.get("Response", "")[:1500],
                "scores": {
                    dim_key: r[dim_key] for dim_key in DIM_KEYS
                },
                "overall_score": r["overall_score"],
                "interesting_reason": r.get("interesting_reason", ""),
                "wikipedia_titles": r.get("wikipedia_titles", ""),
                "reasoning": {
                    "relevance_accuracy": r.get("relevance_accuracy_reasoning", ""),
                    "plurality_breadth": r.get("plurality_breadth_reasoning", ""),
                    "coherence_conciseness": r.get("coherence_conciseness_reasoning", ""),
                },
            })
    return interesting


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build(
    input_files: list[str],
    output_file: str,
    metadata_file: str,
    max_interesting: int = 50,
) -> None:
    logger.info(f"Loading model metadata from {metadata_file}")
    metadata = load_model_metadata(metadata_file)

    logger.info(f"Loading score files: {input_files}")
    raw_rows = load_scores(input_files)

    # Clean & filter
    clean_rows = [r for r in (clean_row(r) for r in raw_rows) if r is not None]
    excluded = len(raw_rows) - len(clean_rows)
    logger.info(f"Clean rows: {len(clean_rows)} ({excluded} excluded due to errors/missing scores)")

    if not clean_rows:
        logger.error("No valid rows after cleaning. Cannot build leaderboard data.")
        return

    # Group by model display name
    model_rows_map: dict[str, list[dict]] = defaultdict(list)
    model_meta_map: dict[str, dict] = {}
    for row in clean_rows:
        model_id = row.get("Model", "unknown")
        meta = resolve_model_meta(model_id, metadata)
        display_name = meta["display_name"]
        model_rows_map[display_name].append(row)
        model_meta_map[display_name] = meta

    logger.info(f"Models: {sorted(model_rows_map.keys())}")

    # Model summaries
    model_summaries = [
        build_model_summary(rows, model_meta_map[name])
        for name, rows in model_rows_map.items()
    ]
    # Sort by overall mean descending
    model_summaries.sort(
        key=lambda s: s["scores"]["overall"]["mean"] or 0, reverse=True
    )

    # Country of origin aggregation
    country_summaries = build_country_origin_summary(model_summaries, model_rows_map)

    # Breakdowns
    by_category = build_breakdown(
        model_rows_map,
        key_fn=lambda r: r.get("Category", "Unknown"),
    )
    by_prompt_type = build_breakdown(
        model_rows_map,
        key_fn=lambda r: r.get("Prompt Type", "Unknown"),
    )
    by_relevant_country = build_breakdown(
        model_rows_map,
        key_fn=lambda r: r.get("Country", "Unknown"),
    )

    # Topic matrix (all topics × all models)
    topic_matrix = build_topic_matrix(model_rows_map)

    # Interesting responses
    interesting = extract_interesting(model_rows_map, max_per_model=max_interesting)
    logger.info(f"Interesting responses flagged: {len(interesting)}")

    # Metadata
    judge_models = sorted(set(r.get("judge_model", "") for r in clean_rows if r.get("judge_model")))
    categories = sorted(set(r.get("Category", "") for r in clean_rows if r.get("Category")))
    prompt_types = sorted(set(r.get("Prompt Type", "") for r in clean_rows if r.get("Prompt Type")))
    topic_countries = sorted(set(r.get("Country", "") for r in clean_rows if r.get("Country")))

    output = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_responses_evaluated": len(clean_rows),
            "total_responses_excluded": excluded,
            "judge_models": judge_models,
            "categories": categories,
            "prompt_types": prompt_types,
            "topic_countries": topic_countries,
            "n_models": len(model_summaries),
            "n_interesting": len(interesting),
        },
        "models": model_summaries,
        "by_country_origin": country_summaries,
        "by_category": by_category,
        "by_prompt_type": by_prompt_type,
        "by_relevant_country": by_relevant_country,
        "topic_matrix": topic_matrix,
        "interesting_responses": interesting,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Leaderboard data written to {output_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build leaderboard_data.json from judge_scores CSV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", nargs="+", default=[],
        help=(
            "One or more judge_scores_*.csv files. "
            "If omitted, auto-discovers all judge_scores_*.csv in the current directory."
        ),
    )
    parser.add_argument(
        "--output", default="leaderboard_data.json",
        help="Output JSON file. Default: leaderboard_data.json",
    )
    parser.add_argument(
        "--metadata", default="model_metadata.json",
        help="Model metadata JSON file. Default: model_metadata.json",
    )
    parser.add_argument(
        "--max-interesting", type=int, default=50,
        help="Maximum interesting responses to keep per model. Default: 50",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Auto-discover score files if not specified
    input_files = args.input
    if not input_files:
        discovered = sorted(Path(".").glob("judge_scores_*.csv"))
        if not discovered:
            logger.error(
                "No judge_scores_*.csv files found in the current directory. "
                "Run llm_judge_runner.py first, or specify --input explicitly."
            )
            import sys; sys.exit(1)
        input_files = [str(p) for p in discovered]
        logger.info(f"Auto-discovered {len(input_files)} score file(s): {input_files}")

    build(
        input_files=input_files,
        output_file=args.output,
        metadata_file=args.metadata,
        max_interesting=args.max_interesting,
    )


if __name__ == "__main__":
    main()
