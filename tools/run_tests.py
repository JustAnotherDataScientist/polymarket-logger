"""
run_tests.py — PM Watcher v4 — Test Suite
──────────────────────────────────────────
Tests:
  1. Cohérence logique des données
  2. Backfill outcomes (slugs résolus vs non résolus)

Usage:
    python3 run_tests.py
    python3 run_tests.py --date 2026-03-16
    python3 run_tests.py --verbose
"""
from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS = "✓"
FAIL = "✗"
WARN = "⚠"

class TestResult:
    def __init__(self, name: str):
        self.name    = name
        self.passed  = 0
        self.failed  = 0
        self.warned  = 0
        self.details = []

    def ok(self, msg: str):
        self.passed += 1
        self.details.append((PASS, msg))

    def fail(self, msg: str):
        self.failed += 1
        self.details.append((FAIL, msg))

    def warn(self, msg: str):
        self.warned += 1
        self.details.append((WARN, msg))

    def print(self, verbose: bool = False):
        status = PASS if self.failed == 0 else FAIL
        print(f"\n{status} {self.name}  "
              f"({self.passed} passed, {self.failed} failed, {self.warned} warnings)")
        if verbose or self.failed > 0:
            for icon, msg in self.details:
                print(f"    {icon} {msg}")


def load_parquets(live_dir: str, date: str | None) -> pd.DataFrame:
    p = Path(live_dir)
    if date:
        # Chercher d'abord le fichier final fusionné, puis les chunks du jour
        final = p / f"{date}.parquet"
        chunks = sorted(p.glob(f"{date}_*.parquet"))
        files = ([final] if final.exists() else []) + chunks
    else:
        # Tous les fichiers parquet : finaux (YYYY-MM-DD.parquet)
        # + chunks en cours (YYYY-MM-DD_NNNN.parquet)
        finals = sorted(p.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].parquet"))
        chunks = sorted(p.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_*.parquet"))
        files  = finals + chunks
    if not files:
        print(f"No parquet files in {live_dir}")
        sys.exit(1)
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"WARNING: could not read {f.name}: {e}")
    return pd.concat(dfs, ignore_index=True)


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.replace("", np.nan), errors="coerce")


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Cohérence logique
# ─────────────────────────────────────────────────────────────────────────────

def test_logic(df: pd.DataFrame, verbose: bool) -> TestResult:
    r = TestResult("LOGICAL CONSISTENCY")

    for coin in ["btc", "eth", "sol", "xrp"]:
        for iv in ["5", "15"]:
            sub = df[(df["coin"] == coin) & (df["interval_min"].astype(str) == iv)]
            if len(sub) == 0:
                r.warn(f"{coin}-{iv}m: no rows")
                continue
            k = f"{coin}-{iv}m"

            # 1. bid < ask (always)
            bid = to_num(sub["up_bid"])
            ask = to_num(sub["up_ask"])
            mask = bid.notna() & ask.notna()
            violations = (bid[mask] >= ask[mask]).sum()
            if violations == 0:
                r.ok(f"{k}: bid < ask ✓")
            else:
                r.fail(f"{k}: {violations} rows where bid >= ask")

            # 2. mid = (bid + ask) / 2
            mid = to_num(sub["up_mid"])
            expected_mid = (bid + ask) / 2
            diff = (mid - expected_mid).abs().dropna()
            if (diff > 0.001).sum() == 0:
                r.ok(f"{k}: up_mid = (bid+ask)/2 ✓")
            else:
                r.fail(f"{k}: {(diff > 0.001).sum()} rows where mid ≠ (bid+ask)/2")

            # 3. prices in [0, 1]
            for col in ["up_bid", "up_ask", "up_mid"]:
                v = to_num(sub[col]).dropna()
                out = ((v < 0) | (v > 1)).sum()
                if out == 0:
                    r.ok(f"{k}: {col} ∈ [0,1] ✓")
                else:
                    r.fail(f"{k}: {col} has {out} values outside [0,1]")

            # 4. vol >= 0
            vol = to_num(sub["coin_vol_60s_bps"]).dropna()
            neg = (vol < 0).sum()
            if neg == 0:
                r.ok(f"{k}: coin_vol_60s_bps >= 0 ✓")
            else:
                r.fail(f"{k}: {neg} negative vol values")

            # 5. seconds_remaining in [0, 900]
            sec = to_num(sub["seconds_remaining"]).dropna()
            max_sec = int(iv) * 60
            out = ((sec < 0) | (sec > max_sec + 5)).sum()
            if out == 0:
                r.ok(f"{k}: seconds_remaining ∈ [0, {max_sec}] ✓")
            else:
                r.fail(f"{k}: {out} rows with seconds_remaining out of range")

            # 6. outcome only 0, 1 or empty
            outcomes = sub["outcome"].replace("", np.nan).dropna().unique()
            invalid = set(str(o) for o in outcomes) - {"0", "1"}
            if not invalid:
                r.ok(f"{k}: outcome values valid ✓  ({len(outcomes)} unique: {sorted(outcomes)})")
            else:
                r.fail(f"{k}: invalid outcome values: {invalid}")

            # 7. Binance book: bn_best_bid < bn_best_ask
            bn_bid = to_num(sub["bn_best_bid"])
            bn_ask = to_num(sub["bn_best_ask"])
            mask2  = bn_bid.notna() & bn_ask.notna()
            if mask2.sum() == 0:
                r.warn(f"{k}: no Binance book data")
            else:
                viols = (bn_bid[mask2] >= bn_ask[mask2]).sum()
                if viols == 0:
                    r.ok(f"{k}: bn_bid < bn_ask ✓  ({mask2.sum()} rows)")
                else:
                    r.fail(f"{k}: {viols} rows where bn_bid >= bn_ask")

            # 8. bn_depth_imbalance in [-1, 1]
            imb = to_num(sub["bn_depth_imbalance"]).dropna()
            if len(imb) == 0:
                r.warn(f"{k}: no bn_depth_imbalance data")
            else:
                out2 = ((imb < -1.01) | (imb > 1.01)).sum()
                if out2 == 0:
                    r.ok(f"{k}: bn_depth_imbalance ∈ [-1,1] ✓")
                else:
                    r.fail(f"{k}: {out2} imbalance values outside [-1,1]")

            # DOWN book checks (only when data is present)
            down_bid = to_num(sub["down_bid"]) if "down_bid" in sub.columns else None
            down_ask = to_num(sub["down_ask"]) if "down_ask" in sub.columns else None
            if down_bid is not None and down_ask is not None:
                mask_d = down_bid.notna() & down_ask.notna()
                if mask_d.sum() > 10:
                    # down_bid < down_ask
                    viols = (down_bid[mask_d] >= down_ask[mask_d]).sum()
                    if viols == 0:
                        r.ok(f"{k}: down_bid < down_ask ✓  ({mask_d.sum()} rows)")
                    else:
                        r.fail(f"{k}: {viols} rows where down_bid >= down_ask")
                    # down prices ∈ [0,1]
                    for col in ["down_bid", "down_ask"]:
                        v = to_num(sub[col]).dropna()
                        out = ((v < 0) | (v > 1)).sum()
                        if out == 0:
                            r.ok(f"{k}: {col} ∈ [0,1] ✓")
                        else:
                            r.fail(f"{k}: {col} has {out} values outside [0,1]")
                    # down_mid ≈ 1 - up_mid (when both available)
                    if "down_mid" in sub.columns and "up_mid" in sub.columns:
                        dm = to_num(sub["down_mid"])
                        um = to_num(sub["up_mid"])
                        mask_both = dm.notna() & um.notna()
                        if mask_both.sum() > 10:
                            diff = ((dm[mask_both] + um[mask_both]) - 1.0).abs()
                            # Allow up to 5% spread difference
                            bad = (diff > 0.05).sum()
                            if bad == 0:
                                r.ok(f"{k}: down_mid + up_mid ≈ 1.0 ✓")
                            else:
                                r.warn(f"{k}: {bad} rows where down_mid + up_mid ≠ 1 (spread)")
                else:
                    r.warn(f"{k}: insufficient DOWN book data (market may be one-sided)")

            # 9. edge_bps = (analytical_prob_up - up_mid) * 10000
            prob = to_num(sub["analytical_prob_up"])
            edge = to_num(sub["edge_bps"])
            mask3 = prob.notna() & mid.notna() & edge.notna()
            if mask3.sum() > 10:
                expected_edge = (prob[mask3] - mid[mask3]) * 10000
                diff2 = (edge[mask3] - expected_edge).abs()
                if (diff2 > 1.0).sum() == 0:
                    r.ok(f"{k}: edge_bps = (prob-mid)*10000 ✓")
                else:
                    r.fail(f"{k}: {(diff2 > 1.0).sum()} rows where edge_bps formula wrong")

            # 10. No duplicate timestamps per slug
            dupes = sub.groupby(["slug", "timestamp_utc"]).size()
            n_dupes = (dupes > 1).sum()
            if n_dupes == 0:
                r.ok(f"{k}: no duplicate (slug, timestamp) ✓")
            else:
                r.fail(f"{k}: {n_dupes} duplicate (slug, timestamp) pairs")

            # 11. Colonnes JSON book présentes et parsables
            import json as _json
            for json_col in ["up_book_bids", "up_book_asks"]:
                if json_col not in sub.columns:
                    r.fail(f"{k}: colonne {json_col} manquante ← v6 requis")
                    continue
                sample_vals = sub[json_col].dropna().head(20)
                bad = 0
                for v in sample_vals:
                    try:
                        parsed = _json.loads(v) if isinstance(v, str) else v
                        if not isinstance(parsed, list):
                            bad += 1
                    except Exception:
                        bad += 1
                if bad == 0:
                    non_empty = sum(
                        1 for v in sample_vals
                        if ((_json.loads(v) if isinstance(v, str) else v) or [])
                    )
                    r.ok(f"{k}: {json_col} JSON valide ✓  ({non_empty}/{len(sample_vals)} non-vides)")
                else:
                    r.fail(f"{k}: {json_col} {bad} valeurs non-parsables")

    r.print(verbose)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Backfill outcomes
# ─────────────────────────────────────────────────────────────────────────────

GAMMA_BASE = "https://gamma-api.polymarket.com"

def fetch_resolution(slug: str) -> tuple[str | None, str | None]:
    """Returns (outcome, resolution_source) or (None, reason)."""
    try:
        r = requests.get(f"{GAMMA_BASE}/markets/slug/{slug}", timeout=10)
        if not r.ok:
            return None, f"API {r.status_code}"
        data = r.json()
        m = data[0] if isinstance(data, list) else data

        closed   = m.get("closed", False)
        resolved = m.get("resolved", False)
        if not (closed or resolved):
            return None, "not_resolved"

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
            prices = [float(p) for p in outcome_prices]
            winner_idx   = prices.index(max(prices))
            winner_label = str(outcomes[winner_idx]).lower() if winner_idx < len(outcomes) else ""
            if winner_label in ("up", "yes"):
                return "1", "polymarket"
            elif winner_label in ("down", "no"):
                return "0", "polymarket"
            return None, f"unknown_label:{winner_label}"

        return None, "no_outcome_prices"
    except Exception as e:
        return None, f"error:{e}"


def test_backfill(df: pd.DataFrame, verbose: bool, max_slugs: int = 20) -> TestResult:
    r = TestResult("BACKFILL OUTCOMES")

    slugs = df["slug"].unique()
    r.ok(f"Total slugs in parquet: {len(slugs)}")

    # Check current outcome coverage
    outcomes_filled = df.groupby("slug")["outcome"].apply(
        lambda x: (x.replace("", np.nan).dropna().isin(["0","1"])).any()
    )
    n_labeled   = outcomes_filled.sum()
    n_unlabeled = (~outcomes_filled).sum()
    r.ok(f"Already labeled: {n_labeled}/{len(slugs)} slugs")

    if n_unlabeled > 0:
        r.warn(f"Not yet labeled: {n_unlabeled} slugs (may be unresolved or need backfill)")

    # Check Polymarket resolution for a sample of slugs
    sample = list(df["slug"].unique())[:max_slugs]
    r.ok(f"Checking {len(sample)} slugs against Polymarket API...")

    resolved_poly   = 0
    unresolved_poly = 0
    mismatches      = 0
    errors          = 0

    for slug in sample:
        poly_outcome, reason = fetch_resolution(slug)
        time.sleep(0.2)  # rate limit

        # Get pm_watcher outcome for this slug
        local_outcome = df[df["slug"] == slug]["outcome"].replace("", np.nan).dropna()
        local_val = local_outcome.iloc[0] if len(local_outcome) > 0 else None

        if poly_outcome is None:
            if reason == "not_resolved":
                unresolved_poly += 1
                if verbose:
                    r.ok(f"  {slug[:40]}... → not resolved yet")
            else:
                errors += 1
                r.warn(f"  {slug[:40]}... → API error: {reason}")
        else:
            resolved_poly += 1
            if local_val is None:
                r.warn(f"  {slug[:40]}... → resolved={poly_outcome} but NOT in parquet (needs backfill)")
            elif str(local_val) == str(poly_outcome):
                if verbose:
                    r.ok(f"  {slug[:40]}... → outcome={poly_outcome} ✓ matches parquet")
            else:
                mismatches += 1
                r.fail(f"  {slug[:40]}... → Poly={poly_outcome} but parquet={local_val} ← MISMATCH")

    r.ok(f"Resolved on Polymarket: {resolved_poly}/{len(sample)}")
    r.ok(f"Not yet resolved:       {unresolved_poly}/{len(sample)}")
    if errors:
        r.warn(f"API errors:             {errors}/{len(sample)}")
    if mismatches == 0:
        r.ok(f"Outcome mismatches:     0 ✓")
    else:
        r.fail(f"Outcome mismatches:     {mismatches} ← run backfill_outcomes.py")

    r.print(verbose)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date",    default=None, help="YYYY-MM-DD")
    parser.add_argument("--dir",     default="historical_data/live")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip-backfill", action="store_true",
                        help="Skip Polymarket API calls (faster)")
    parser.add_argument("--cron", action="store_true",
                        help="Mode cron : écrit le résultat dans logs/tests_status.json")
    args = parser.parse_args()

    print("=" * 60)
    print("PM WATCHER v4 — TEST SUITE")
    print("=" * 60)

    df = load_parquets(args.dir, args.date)
    print(f"Loaded: {len(df):,} rows | {len(df.columns)} cols")
    print(f"Period: {df['timestamp_utc'].min()} → {df['timestamp_utc'].max()}")

    results = []

    # Test 1: Logic
    results.append(test_logic(df, args.verbose))

    # Test 2: Backfill
    if not args.skip_backfill:
        results.append(test_backfill(df, args.verbose))
    else:
        print("\n[skipped] BACKFILL OUTCOMES (--skip-backfill)")

    # Summary
    total_passed = sum(r.passed for r in results)
    total_failed = sum(r.failed for r in results)
    total_warned = sum(r.warned for r in results)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Passed:   {total_passed}")
    print(f"  Failed:   {total_failed}")
    print(f"  Warnings: {total_warned}")

    if total_failed == 0:
        print("\n  ✅ All tests passed — data ready for backtest + ML")
    else:
        print(f"\n  ❌ {total_failed} test(s) failed — review above")

    # Mode cron : écrire le résultat dans un fichier JSON pour health_check
    if args.cron:
        import json as _json
        from pathlib import Path as _Path
        status = {
            "timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z",
            "passed": total_passed,
            "failed": total_failed,
            "warnings": total_warned,
            "ok": total_failed == 0,
        }
        status_file = _Path(__file__).parent / "logs" / "tests_status.json"
        status_file.parent.mkdir(exist_ok=True)
        status_file.write_text(_json.dumps(status, indent=2))

    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
