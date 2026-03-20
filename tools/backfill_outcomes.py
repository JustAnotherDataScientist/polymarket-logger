"""
backfill_outcomes.py
─────────────────────
Rétrograde les outcomes officiels Polymarket dans les parquets live.

Pour chaque slug dans les parquets :
  1. Fetch le marché via Gamma API
  2. Si résolu → écrit l'outcome officiel (1=UP, 0=DOWN) sur toutes les lignes
  3. Réécrit le parquet in-place (atomic: tmp → rename)

Usage:
    python backfill_outcomes.py                    # tous les parquets
    python backfill_outcomes.py --date 2026-03-14  # un jour spécifique
    python backfill_outcomes.py --dry-run          # affiche sans écrire

Notes:
    - Seuls les marchés résolus sont mis à jour
    - Les marchés non résolus sont ignorés (outcome="" intact)
    - Rate limit : 1 req/s vers Gamma API (évite les bans)
"""
from __future__ import annotations

import argparse
import time
import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"
LIVE_DIR   = "historical_data/live"
RATE_LIMIT = 0.3   # seconds between API calls


def fetch_market_outcome(slug: str) -> tuple[str | None, float | None]:
    """
    Returns (outcome, close_price) for a resolved market.
    outcome = "1" (UP won) or "0" (DOWN won) or None (not resolved yet).
    close_price = Polymarket resolution price or None.
    """
    try:
        r = requests.get(
            f"{GAMMA_BASE}/markets/slug/{slug}",
            timeout=10
        )
        if not r.ok:
            return None, None
        data = r.json()
        m = data[0] if isinstance(data, list) else data

        # Check if resolved
        closed   = m.get("closed", False)
        resolved = m.get("resolved", False)
        if not (closed or resolved):
            return None, None

        # Get winner from outcomePrices or winner field
        # outcomePrices: ["1", "0"] or ["0", "1"] where 1 = won
        # outcomes: ["Up", "Down"] or ["Yes", "No"]
        import json
        outcomes = m.get("outcomes", "[]")
        if isinstance(outcomes, str):
            try: outcomes = json.loads(outcomes)
            except: outcomes = []

        outcome_prices = m.get("outcomePrices", "[]")
        if isinstance(outcome_prices, str):
            try: outcome_prices = json.loads(outcome_prices)
            except: outcome_prices = []

        if outcome_prices and outcomes:
            # Find which outcome won (price == "1" or closest to 1)
            prices = [float(p) for p in outcome_prices]
            winner_idx = prices.index(max(prices))
            winner_label = str(outcomes[winner_idx]).lower() if winner_idx < len(outcomes) else ""

            # Map to 1/0: "up" or "yes" = 1, "down" or "no" = 0
            if winner_label in ("up", "yes"):
                outcome = "1"
            elif winner_label in ("down", "no"):
                outcome = "0"
            else:
                log.warning("Unknown winner label '%s' for slug %s", winner_label, slug)
                return None, None

            # Resolution price from Polymarket
            resolution_price = m.get("resolutionPrice") or m.get("marketMakerAmount")
            if resolution_price:
                try: resolution_price = float(resolution_price)
                except: resolution_price = None

            return outcome, resolution_price

        # Fallback: check winner field directly
        winner = m.get("winner", "")
        if winner:
            outcome = "1" if str(winner).lower() in ("up", "yes", "1") else "0"
            return outcome, None

        return None, None

    except Exception as e:
        log.error("Error fetching slug %s: %s", slug, e)
        return None, None


def backfill_file(path: Path, dry_run: bool = False) -> dict:
    """Process one parquet file. Returns stats dict."""
    # Phase 1: load ONLY slug + outcome columns to find what needs updating (~10MB vs 2GB)
    df_meta = pd.read_parquet(path, columns=["slug", "outcome"])
    stats = {"total_slugs": 0, "updated": 0, "already_set": 0, "unresolved": 0, "errors": 0}

    slugs = df_meta["slug"].unique()
    stats["total_slugs"] = len(slugs)
    log.info("%s: %d slugs to check", path.name, len(slugs))

    updated_outcomes = {}   # slug → outcome
    updated_prices   = {}   # slug → close_price

    for slug in slugs:
        # Check if already fully labeled
        slug_rows = df_meta[df_meta["slug"] == slug]
        existing  = slug_rows["outcome"].replace("", None).dropna()

        if len(existing) > 0 and set(existing.unique()) <= {"0", "1"}:
            stats["already_set"] += 1
            continue  # already labeled

        # Fetch from Polymarket
        outcome, close_price = fetch_market_outcome(slug)
        time.sleep(RATE_LIMIT)

        if outcome is None:
            stats["unresolved"] += 1
            log.debug("  %s → not resolved yet", slug)
            continue

        updated_outcomes[slug] = outcome
        if close_price is not None:
            updated_prices[slug] = close_price

        log.info("  %s → outcome=%s  close_price=%s", slug, outcome, close_price)
        stats["updated"] += 1

    del df_meta  # free memory before full load

    if not updated_outcomes:
        log.info("%s: nothing to update", path.name)
        return stats

    if dry_run:
        log.info("[DRY-RUN] Would update %d slugs in %s", len(updated_outcomes), path.name)
        return stats

    # Phase 2: load full file, apply updates, rewrite
    # Use pyarrow directly to avoid pandas peak memory doubling
    table = pq.read_table(path)
    df = table.to_pandas()
    del table  # free arrow memory

    for slug, outcome in updated_outcomes.items():
        mask = df["slug"] == slug
        df.loc[mask, "outcome"] = outcome

    for slug, price in updated_prices.items():
        mask = df["slug"] == slug
        df.loc[mask, "coin_price_at_window_close"] = str(price)

    # Atomic write
    tmp = path.with_suffix(".tmp")
    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=False),
        str(tmp), compression="snappy", row_group_size=2000
    )
    del df  # free before rename
    tmp.replace(path)
    log.info("%s: ✓ updated %d slugs", path.name, len(updated_outcomes))

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date",    default=None, help="YYYY-MM-DD — specific date only")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change, don't write")
    parser.add_argument("--dir",     default=LIVE_DIR, help=f"Parquet directory (default: {LIVE_DIR})")
    args = parser.parse_args()

    live_dir = Path(args.dir)
    if not live_dir.exists():
        log.error("Directory not found: %s", live_dir)
        return

    if args.date:
        files = [live_dir / f"{args.date}.parquet"]
        files = [f for f in files if f.exists()]
    else:
        files = sorted(live_dir.glob("*.parquet"))

    if not files:
        log.error("No parquet files found")
        return

    log.info("Processing %d file(s)%s", len(files), " [DRY-RUN]" if args.dry_run else "")

    total = {"total_slugs": 0, "updated": 0, "already_set": 0, "unresolved": 0, "errors": 0}
    for f in files:
        stats = backfill_file(f, dry_run=args.dry_run)
        for k in total:
            total[k] += stats.get(k, 0)

    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"  Files processed:  {len(files)}")
    print(f"  Total slugs:      {total['total_slugs']}")
    print(f"  Already labeled:  {total['already_set']}")
    print(f"  Newly labeled:    {total['updated']}")
    print(f"  Not resolved yet: {total['unresolved']}")
    if args.dry_run:
        print("\n  [DRY-RUN] — no files were modified")


if __name__ == "__main__":
    main()
