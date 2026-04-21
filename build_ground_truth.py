#!/usr/bin/env python3
"""
build_ground_truth.py — Pre-fetch Wikipedia reference material for the
LLM Ideology Audit prompt bank.

Run this once before running llm_judge_runner.py. It fetches Wikipedia
articles for every unique (topic, country) pair in the prompt bank,
writes results to wiki_cache.json (which the judge runner reads as
immutable ground truth), and exports a human-readable ground_truth.csv
for quality inspection and manual correction.

Once built, the judge runner will use the cache as-is without making
further Wikipedia requests. To refresh the ground truth (e.g. after
adding new prompts), re-run this script.

Usage:
    python build_ground_truth.py
    python build_ground_truth.py --input llm-ideology-audit-prompts.csv
    python build_ground_truth.py --refresh          # re-fetch all, ignore cache
    python build_ground_truth.py --dry-run          # list pairs, no fetching
"""

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_INPUT  = "llm-ideology-audit-prompts.csv"
DEFAULT_CACHE  = "wiki_cache.json"
DEFAULT_OUTPUT = "ground_truth.csv"
DEFAULT_DELAY  = 1.0  # seconds between (topic, country) fetches

# ---------------------------------------------------------------------------
# Prompt bank loading
# ---------------------------------------------------------------------------

def load_prompt_bank(path: Path) -> list[dict]:
    """Load the prompt bank CSV (or XLSX) and return all rows as dicts."""
    if path.suffix.lower() in (".xlsx", ".xls"):
        try:
            import openpyxl
        except ImportError:
            logger.error("Install openpyxl to read Excel files: pip install openpyxl")
            sys.exit(1)
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        rows = list(ws.values)
        if not rows:
            logger.error("Empty worksheet.")
            sys.exit(1)
        headers = [str(h) if h is not None else "" for h in rows[0]]
        return [dict(zip(headers, row)) for row in rows[1:]]
    else:
        with open(path, encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))


def extract_unique_pairs(rows: list[dict]) -> list[tuple[str, str]]:
    """
    Return unique (topic, country) pairs in prompt-bank order.
    Handles both 'Country' and 'Relevant Country' column names.
    """
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str]] = []
    for row in rows:
        topic   = str(row.get("Topic", "") or "").strip()
        country = str(
            row.get("Country") or row.get("Relevant Country") or ""
        ).strip()
        if topic and (topic, country) not in seen:
            seen.add((topic, country))
            pairs.append((topic, country))
    return pairs

# ---------------------------------------------------------------------------
# Ground truth CSV helpers
# ---------------------------------------------------------------------------

def quality_label(result: dict) -> str:
    """
    Classify the quality of a Wikipedia fetch result.

    'good'     — articles found via primary full-text or OpenSearch query
    'fallback' — articles found only after falling back to broader queries
    'empty'    — no articles found at all
    """
    articles = result.get("articles", [])
    if not articles:
        return "empty"
    if result.get("query_used", "").startswith("fallback:"):
        return "fallback"
    return "good"


def build_csv_row(topic: str, country: str, result: dict) -> dict:
    articles = result.get("articles", [])
    titles   = [a["title"] for a in articles]
    urls     = [a["url"]   for a in articles]
    chars    = [len(a.get("extract", "")) for a in articles]
    return {
        "topic":           topic,
        "country":         country,
        "article_count":   len(articles),
        "article_titles":  json.dumps(titles, ensure_ascii=False),
        "article_urls":    json.dumps(urls,   ensure_ascii=False),
        "char_counts":     json.dumps(chars),
        "total_chars":     sum(chars),
        "query_used":      result.get("query_used", ""),
        "quality":         quality_label(result),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-fetch Wikipedia ground truth for the LLM Ideology Audit.\n"
            "Run once before llm_judge_runner.py; the judge runner then uses\n"
            "wiki_cache.json as immutable ground truth."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", default=DEFAULT_INPUT, metavar="PATH",
        help="Prompt bank CSV or XLSX",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT, metavar="PATH",
        help="Ground truth inspection CSV",
    )
    parser.add_argument(
        "--cache", default=DEFAULT_CACHE, metavar="PATH",
        help="wiki_cache.json path (shared with llm_judge_runner.py)",
    )
    parser.add_argument(
        "--delay", type=float, default=DEFAULT_DELAY, metavar="SECS",
        help="Seconds between Wikipedia fetches",
    )
    parser.add_argument(
        "--max-articles", type=int, default=3,
        help="Maximum Wikipedia articles per topic (default: 3)",
    )
    parser.add_argument(
        "--max-chars-per-article", type=int, default=3000,
        help="Max characters extracted per article (default: 3000)",
    )
    parser.add_argument(
        "--max-total-chars", type=int, default=8000,
        help="Max combined characters across all articles (default: 8000)",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Re-fetch all pairs, ignoring existing valid cache entries",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List unique (topic, country) pairs and exit without fetching",
    )
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)
    cache_path  = Path(args.cache)

    if not input_path.exists():
        logger.error(f"Prompt bank not found: {input_path}")
        sys.exit(1)

    # Import WikipediaFetcher from the judge runner so logic stays DRY
    try:
        from llm_judge_runner import WikipediaFetcher
    except ImportError as e:
        logger.error(f"Could not import WikipediaFetcher from llm_judge_runner.py: {e}")
        sys.exit(1)

    # Load prompt bank and extract unique pairs
    rows  = load_prompt_bank(input_path)
    pairs = extract_unique_pairs(rows)
    logger.info(
        f"Prompt bank: {len(rows)} prompts → "
        f"{len(pairs)} unique (topic, country) pairs"
    )

    if args.dry_run:
        print(f"\n{'#':<4} {'TOPIC':<42} COUNTRY")
        print("─" * 65)
        for i, (topic, country) in enumerate(pairs, 1):
            print(f"{i:<4} {topic:<42} {country}")
        print(f"\n{len(pairs)} pairs total. No fetching performed (--dry-run).")
        return

    # Initialise fetcher pointing at the shared cache.
    # strict_cache=False is essential here — this script's entire purpose is
    # to populate the cache, so it must always fetch on a miss.
    fetcher = WikipediaFetcher(
        cache_file=str(cache_path),
        max_articles=args.max_articles,
        max_chars_per_article=args.max_chars_per_article,
        max_total_chars=args.max_total_chars,
        strict_cache=False,
    )

    if args.refresh:
        # Wipe data entries but preserve any existing metadata keys
        logger.info("--refresh: clearing all existing cache entries.")
        fetcher.cache = {
            k: v for k, v in fetcher.cache.items()
            if k.startswith("_")
        }

    # Fetch Wikipedia for each pair
    csv_rows: list[dict] = []
    counts = {"good": 0, "fallback": 0, "empty": 0, "cached": 0}

    for i, (topic, country) in enumerate(pairs, 1):
        cache_key = f"{topic}|||{country}"
        entry = fetcher.cache.get(cache_key, {})

        if not args.refresh and fetcher._is_valid_cache_entry(entry):
            logger.info(f"[{i:>3}/{len(pairs)}] Cached:   {topic} ({country})")
            result = entry
            counts["cached"] += 1
        else:
            logger.info(f"[{i:>3}/{len(pairs)}] Fetching: {topic} ({country})")
            result = fetcher.get(topic, country)
            ql = quality_label(result)
            counts[ql] += 1
            if ql == "empty":
                logger.warning(f"  No articles found for: {topic} ({country})")
            elif ql == "fallback":
                logger.info(f"  Fallback used: {result.get('query_used', '')}")
            # Polite rate-limiting between fetches (not after the last one)
            if i < len(pairs):
                time.sleep(args.delay)

        csv_rows.append(build_csv_row(topic, country, result))

    # Stamp the cache with build metadata so the judge runner can detect it
    fetcher.cache["_ground_truth_meta"] = {
        "built_at":   datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "pair_count": len(pairs),
        "good":       counts["good"] + counts["cached"],
        "fallback":   counts["fallback"],
        "empty":      counts["empty"],
    }
    fetcher._save_cache()
    logger.info(f"Cache written → {cache_path}")

    # Write ground_truth.csv for human inspection
    fieldnames = [
        "topic", "country", "article_count", "article_titles",
        "article_urls", "char_counts", "total_chars", "query_used", "quality",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    logger.info(f"Inspection CSV written → {output_path}")

    # Summary report
    n_empty    = counts["empty"]
    n_fallback = counts["fallback"]
    n_cached   = counts["cached"]
    n_good     = counts["good"]
    print(f"""
Ground truth build complete
───────────────────────────
  Pairs processed  : {len(pairs)}
  Articles found   : {n_good + n_fallback + n_cached}  (good: {n_good + n_cached}, fallback: {n_fallback})
  No article found : {n_empty}
  From cache       : {n_cached}

  Cache file       : {cache_path}
  Inspection CSV   : {output_path}
""")
    if n_empty:
        print(
            f"  ⚠  {n_empty} topic(s) returned no Wikipedia articles. "
            f"Review 'quality=empty' rows in {output_path}."
        )
    if n_fallback:
        print(
            f"  ℹ  {n_fallback} topic(s) required fallback queries. "
            f"Review 'quality=fallback' rows to confirm the articles are appropriate."
        )
    print(
        f"\nRun llm_judge_runner.py — it will use {cache_path} as "
        f"immutable ground truth without making further Wikipedia requests."
    )


if __name__ == "__main__":
    main()
