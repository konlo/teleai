"""run_test_set.py — CLI Test Runner for the 200-question benchmark suite.

Usage examples:
  # Run all 200 tests
  python test_set/run_test_set.py --all

  # Run only Level 1 tests
  python test_set/run_test_set.py --level 1

  # Run only Level 2 tests
  python test_set/run_test_set.py --level 2

  # Run a specific test by ID
  python test_set/run_test_set.py --id L1_042
  python test_set/run_test_set.py --id L2_077

  # Filter by type: schema | synonym | table | chart
  python test_set/run_test_set.py --level 1 --type chart
  python test_set/run_test_set.py --all --type schema

  # Filter by category keyword
  python test_set/run_test_set.py --all --category "시각화"

  # Show summary only (no stdout per test)
  python test_set/run_test_set.py --all --quiet

  # Show prompt + code for a specific test without running it
  python test_set/run_test_set.py --show L2_082
"""

import os
import sys
import json
import time
import argparse
import io
import traceback
from pathlib import Path

# ── Headless Matplotlib setup ─────────────────────────────────────────────────
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

# ── Paths & data loading ──────────────────────────────────────────────────────
TEST_SET_DIR = Path(__file__).parent
DATA_DIR     = TEST_SET_DIR / "data"

df_bank    = pd.read_csv(DATA_DIR / "bank_loan.csv")
df_titanic = pd.read_csv(DATA_DIR / "titanic.csv")

sys.path.insert(0, str(TEST_SET_DIR))

from level1.definitions_part1 import get_level1_part1
from level1.definitions_part2 import get_level1_part2
from level1.definitions_part3 import get_level1_part3
from level1.definitions_part4 import get_level1_part4
from level2.definitions_part1 import get_level2_part1
from level2.definitions_part2 import get_level2_part2
from level2.definitions_part3 import get_level2_part3
from level2.definitions_part4 import get_level2_part4
from level3.definitions_part5 import get_level3_part5

LEVEL1_SPECS = (get_level1_part1() + get_level1_part2() +
                get_level1_part3() + get_level1_part4())
LEVEL2_SPECS = (get_level2_part1() + get_level2_part2() +
                get_level2_part3() + get_level2_part4())
ALL_SPECS    = LEVEL1_SPECS + LEVEL2_SPECS

# ── Helpers ───────────────────────────────────────────────────────────────────

def run_spec(spec: dict) -> dict:
    code = spec.get("python_code", "")
    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    has_figure = False
    error_msg  = None
    try:
        plt.close("all")
        exec_globals = {
            "__builtins__": __builtins__,
            "pd": pd, "np": np, "plt": plt, "matplotlib": matplotlib,
            "df_bank":    df_bank.copy(),
            "df_titanic": df_titanic.copy(),
        }
        exec(code, exec_globals)
        has_figure = len(plt.get_fignums()) > 0
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout
        plt.close("all")

    return {
        **spec,
        "status":     "PASS" if error_msg is None else "FAIL",
        "stdout":     captured.getvalue(),
        "has_figure": has_figure,
        "error":      error_msg,
    }


def print_result(result: dict, quiet: bool = False):
    icon = "✅" if result["status"] == "PASS" else "❌"
    print(f"\n{icon} [{result['id']}] {result['category']} — {result['type'].upper()}")
    print(f"   Prompt : {result['prompt']}")
    if not quiet:
        if result["stdout"]:
            print("   Output :")
            for line in result["stdout"].strip().split("\n"):
                print(f"     {line}")
        if result["has_figure"]:
            print("   Figure : ✓ matplotlib figure generated")
        if result["error"]:
            print(f"   ERROR  : {result['error'].split(chr(10))[0]}")


def show_spec(spec: dict):
    """Print prompt and code for inspection without running."""
    print(f"\n{'='*60}")
    print(f"  ID       : {spec['id']}")
    print(f"  Category : {spec['category']}")
    print(f"  Type     : {spec['type']}")
    print(f"  Difficulty: {spec.get('difficulty', 'N/A')}")
    print(f"  Table    : {spec['target_table']}")
    print(f"  Synonym  : {spec.get('synonym_mapping')}")
    print(f"\n  Prompt:\n    {spec['prompt']}")
    print(f"\n  Python Code:\n{'─'*60}")
    for line in spec["python_code"].split("\n"):
        print(f"  {line}")
    print(f"{'─'*60}")


def print_summary(results: list, level_name: str = ""):
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = len(results) - passed
    charts = sum(1 for r in results if r["has_figure"] and r["status"] == "PASS")
    label  = f"  [{level_name}]" if level_name else "  [ALL]"
    print(f"\n{'='*60}")
    print(f"{label} {passed}/{len(results)} PASS  "
          f"({failed} FAIL, {charts} figures generated)")
    print(f"{'='*60}")
    if failed:
        print("\n  Failed tests:")
        for r in results:
            if r["status"] == "FAIL":
                err = (r["error"] or "").split("\n")[0]
                print(f"    ✗ {r['id']:8s}  {err}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Telly Chatbot Benchmark Test Runner (200 Q&A test set)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all",   action="store_true", help="Run all 200 tests")
    group.add_argument("--level", type=int, choices=[1, 2], help="Run Level 1 or Level 2 only")
    group.add_argument("--id",    type=str, help="Run a single test by ID (e.g. L1_042)")
    group.add_argument("--show",  type=str, help="Show prompt+code for an ID without running")

    parser.add_argument("--type",     type=str, help="Filter by type: schema|synonym|table|chart")
    parser.add_argument("--category", type=str, help="Filter by category keyword (Korean supported)")
    parser.add_argument("--quiet",    action="store_true", help="Suppress per-test stdout output")
    parser.add_argument("--list",     action="store_true", help="List all test IDs and prompts")

    args = parser.parse_args()

    # ── --list mode ───────────────────────────────────────────────────────────
    if args.list:
        for s in ALL_SPECS:
            print(f"  {s['id']:8s}  [{s['type']:6s}]  {s['prompt'][:60]}...")
        return

    # ── --show mode ───────────────────────────────────────────────────────────
    if args.show:
        matches = [s for s in ALL_SPECS if s["id"].upper() == args.show.upper()]
        if not matches:
            print(f"ID '{args.show}' not found. Use --list to see all IDs.")
            sys.exit(1)
        show_spec(matches[0])
        return

    # ── Select specs ──────────────────────────────────────────────────────────
    if args.id:
        specs = [s for s in ALL_SPECS if s["id"].upper() == args.id.upper()]
        if not specs:
            print(f"ID '{args.id}' not found. Use --list to see all IDs.")
            sys.exit(1)
        level_name = args.id
    elif args.level == 1:
        specs = LEVEL1_SPECS
        level_name = "Level 1"
    elif args.level == 2:
        specs = LEVEL2_SPECS
        level_name = "Level 2"
    elif args.all:
        specs = ALL_SPECS
        level_name = "All"
    else:
        parser.print_help()
        return

    # ── Apply filters ─────────────────────────────────────────────────────────
    if args.type:
        specs = [s for s in specs if s.get("type", "").lower() == args.type.lower()]
        level_name += f" [{args.type}]"

    if args.category:
        specs = [s for s in specs if args.category in (s.get("category") or "")]
        level_name += f" ['{args.category}']"

    if not specs:
        print("No tests match the given filters.")
        return

    # ── Run ───────────────────────────────────────────────────────────────────
    print(f"\nRunning {len(specs)} test(s) for: {level_name}")
    print(f"Data: {DATA_DIR}")

    results = []
    for i, spec in enumerate(specs, 1):
        t0 = time.time()
        result = run_spec(spec)
        elapsed = time.time() - t0
        icon = "✅" if result["status"] == "PASS" else "❌"
        print(f"  [{i:>3}/{len(specs)}] {icon} {result['id']:8s}  ({elapsed:.2f}s)  fig={result['has_figure']}")
        if not args.quiet and (result["stdout"] or result["error"]):
            print_result(result, quiet=args.quiet)
        results.append(result)

    print_summary(results, level_name)


if __name__ == "__main__":
    main()
