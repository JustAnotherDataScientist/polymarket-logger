#!/usr/bin/env python3
"""
tests/run_all.py — Run the full validation suite.

Usage (from project root):
  python tests/run_all.py historical_data
  python tests/run_all.py historical_data --date 2026-03-20
  python tests/run_all.py historical_data --skip-backtest
"""

import sys, os, subprocess, argparse
from datetime import datetime, timezone

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
TOOLS_DIR = os.path.join(PROJECT_ROOT, 'tools')


def run_script(path, args, description, cwd=None):
    print(f"\n{'#'*70}")
    print(f"# {description}")
    print(f"# python {os.path.relpath(path, PROJECT_ROOT)} {' '.join(args)}")
    print(f"{'#'*70}\n")
    result = subprocess.run(
        [sys.executable, path] + args,
        cwd=cwd or PROJECT_ROOT,
    )
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('base_dir', nargs='?', default='historical_data')
    parser.add_argument('--date', default=None)
    parser.add_argument('--skip-backtest', action='store_true')
    parser.add_argument('--skip-crossval', action='store_true')
    args = parser.parse_args()

    events_dir = os.path.join(args.base_dir, 'events')
    live_dir = os.path.join(args.base_dir, 'live')

    if not os.path.isdir(events_dir) and os.path.isdir(os.path.join(args.base_dir, 'poly_books')):
        events_dir = args.base_dir

    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  Event Store Validation Suite                               ║")
    print(f"║  {now:55s}  ║")
    print(f"║  Events: {events_dir:51s}  ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")

    codes = []

    # 1. Exploration (in tools/)
    codes.append(run_script(
        os.path.join(TOOLS_DIR, 'explore_events.py'),
        [events_dir],
        "STEP 1: Data Exploration",
    ))

    # 2. Event quality (in tests/)
    qa_args = [events_dir]
    if args.date:
        qa_args += ['--date', args.date]
    codes.append(run_script(
        os.path.join(TESTS_DIR, 'test_events.py'),
        qa_args,
        "STEP 2: Event Quality Validation",
    ))

    # 3. Cross-validation (in tests/)
    if not args.skip_crossval and os.path.isdir(live_dir):
        codes.append(run_script(
            os.path.join(TESTS_DIR, 'test_crossval.py'),
            [args.base_dir],
            "STEP 3: Cross-Validation (events vs snapshots)",
        ))
    else:
        print(f"\n[SKIP] Cross-validation" +
              (" (--skip-crossval)" if args.skip_crossval else f" (no {live_dir})"))

    # 4. Backtest validation (in tests/, cwd=project root for imports)
    if not args.skip_backtest:
        bt_args = [events_dir]
        if args.date:
            bt_args += ['--date', args.date]
        codes.append(run_script(
            os.path.join(TESTS_DIR, 'test_backtest.py'),
            bt_args,
            "STEP 4: Backtest Engine Validation",
            cwd=PROJECT_ROOT,
        ))
    else:
        print(f"\n[SKIP] Backtest validation (--skip-backtest)")

    failures = sum(1 for c in codes if c != 0)
    print(f"\n{'═'*70}")
    if failures == 0:
        print(f"  ✓ ALL {len(codes)} STEPS PASSED")
    else:
        print(f"  ✗ {failures}/{len(codes)} STEPS HAD FAILURES")
    print(f"{'═'*70}")
    return failures


if __name__ == '__main__':
    sys.exit(main())
