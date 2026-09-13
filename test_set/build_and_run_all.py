"""build_and_run_all.py
Execute all 200 test specifications (Level 1: 100, Level 2: 100) against
bank_loan.csv and titanic.csv, capture stdout and figure generation status,
and serialize results to benchmark_level1.json and benchmark_level2.json.
"""
import os
import sys
import json
import time
import traceback
import io
from pathlib import Path

# ── Headless Matplotlib setup (must be BEFORE any other mpl import) ──────────
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
TEST_SET_DIR = Path(__file__).parent
DATA_DIR = TEST_SET_DIR / "data"

# ── Load shared dataframes ────────────────────────────────────────────────────
df_bank    = pd.read_csv(DATA_DIR / "bank_loan.csv")
df_titanic = pd.read_csv(DATA_DIR / "titanic.csv")

print(f"Loaded bank_loan:  {df_bank.shape}")
print(f"Loaded titanic:    {df_titanic.shape}")

# ── Load all spec modules ─────────────────────────────────────────────────────
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

level1_specs = (
    get_level1_part1() +
    get_level1_part2() +
    get_level1_part3() +
    get_level1_part4()
)

level2_specs = (
    get_level2_part1() +
    get_level2_part2() +
    get_level2_part3() +
    get_level2_part4()
)

level3_specs = get_level3_part5()

print(f"\nLevel 1 specs loaded: {len(level1_specs)}")
print(f"Level 2 specs loaded: {len(level2_specs)}")
print(f"\nLevel 3 specs loaded: {len(level3_specs)}")

# ── Execution helper ──────────────────────────────────────────────────────────
def run_spec(spec: dict, df_bank: pd.DataFrame, df_titanic: pd.DataFrame) -> dict:
    """Execute a single spec's python_code and capture output + figure flag."""
    code = spec.get("python_code", "")
    
    # Capture stdout
    captured_stdout = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured_stdout
    
    has_figure = False
    error_msg  = None
    
    try:
        # Close any lingering figures before execution
        plt.close("all")
        
        exec_globals = {
            "__builtins__": __builtins__,
            "pd": pd,
            "np": np,
            "plt": plt,
            "matplotlib": matplotlib,
            "df_bank": df_bank.copy(),
            "df_titanic": df_titanic.copy(),
        }
        
        exec(code, exec_globals)
        
        # Check if any figure was created
        has_figure = len(plt.get_fignums()) > 0
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout
        plt.close("all")
    
    stdout_text = captured_stdout.getvalue()
    
    return {
        "id":                   spec.get("id"),
        "category":             spec.get("category"),
        "type":                 spec.get("type"),
        "difficulty":           spec.get("difficulty"),
        "prompt":               spec.get("prompt"),
        "target_table":         spec.get("target_table"),
        "expected_output_type": spec.get("expected_output_type"),
        "status":               "PASS" if error_msg is None else "FAIL",
        "stdout":               stdout_text,
        "has_figure":           has_figure,
        "error":                error_msg,
    }


# ── Run all tests ─────────────────────────────────────────────────────────────
def run_all(specs, level_name):
    results = []
    passed  = 0
    failed  = 0
    
    print(f"\n{'='*60}")
    print(f"  Running {level_name} — {len(specs)} tests")
    print(f"{'='*60}")
    
    for i, spec in enumerate(specs, 1):
        t0 = time.time()
        result = run_spec(spec, df_bank, df_titanic)
        elapsed = time.time() - t0
        
        status_icon = "✓" if result["status"] == "PASS" else "✗"
        print(f"  [{i:>3}/{len(specs)}] {status_icon} {result['id']:8s}  "
              f"({elapsed:.2f}s)  fig={result['has_figure']}  "
              f"{'  FAIL: ' + (result['error'] or '').split(chr(10))[0] if result['status'] == 'FAIL' else ''}")
        
        results.append(result)
        if result["status"] == "PASS":
            passed += 1
        else:
            failed += 1
    
    print(f"\n  {level_name} Summary: {passed} PASS / {failed} FAIL / {len(specs)} TOTAL")
    return results, passed, failed


level1_results, l1_pass, l1_fail = run_all(level1_specs, "Level 1")
level2_results, l2_pass, l2_fail = run_all(level2_specs, "Level 2")

# ── Save JSON benchmarks ───────────────────────────────────────────────────────
out_l1 = TEST_SET_DIR / "benchmark_level1.json"
out_l2 = TEST_SET_DIR / "benchmark_level2.json"

with open(out_l1, "w", encoding="utf-8") as f:
    json.dump(level1_results, f, ensure_ascii=False, indent=2)

with open(out_l2, "w", encoding="utf-8") as f:
    json.dump(level2_results, f, ensure_ascii=False, indent=2)

print(f"\nSaved: {out_l1}")
print(f"Saved: {out_l2}")

# ── Generate execution_report.md ───────────────────────────────────────────────
total_pass = l1_pass + l2_pass
total_fail = l1_fail + l2_fail
total      = len(level1_results) + len(level2_results)

report_lines = [
    "# Test Execution Report",
    "",
    f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
    "",
    "## Summary",
    "",
    f"| Level   | PASS | FAIL | Total |",
    f"|---------|------|------|-------|",
    f"| Level 1 | {l1_pass}  | {l1_fail}  | {len(level1_results)}   |",
    f"| Level 2 | {l2_pass}  | {l2_fail}  | {len(level2_results)}   |",
    f"| **All** | **{total_pass}** | **{total_fail}** | **{total}** |",
    "",
]

# Failed items detail
all_fails = [r for r in level1_results + level2_results if r["status"] == "FAIL"]
if all_fails:
    report_lines += ["## Failed Tests", ""]
    for r in all_fails:
        err_first = (r["error"] or "").split("\n")[0]
        report_lines.append(f"- **{r['id']}** ({r['category']}): `{err_first}`")
    report_lines.append("")
else:
    report_lines += ["## Failed Tests", "", "✅ All tests passed!", ""]

# Chart coverage
l1_charts = sum(1 for r in level1_results if r["has_figure"] and r["status"] == "PASS")
l2_charts = sum(1 for r in level2_results if r["has_figure"] and r["status"] == "PASS")
report_lines += [
    "## Coverage",
    "",
    f"- Level 1 chart tests with figure generated: {l1_charts}",
    f"- Level 2 chart tests with figure generated: {l2_charts}",
    "",
]

report_path = TEST_SET_DIR / "execution_report.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"Saved: {report_path}")
print(f"\n{'='*60}")
print(f"  FINAL: {total_pass}/{total} PASS  ({total_fail} FAIL)")
print(f"{'='*60}")
