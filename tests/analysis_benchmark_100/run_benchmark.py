"""CLI Runner for the 100 Data Analysis Benchmark Test Suite.

Usage:
  .venv/bin/python tests/analysis_benchmark_100/run_benchmark.py --all
  .venv/bin/python tests/analysis_benchmark_100/run_benchmark.py --id TEST_001
  .venv/bin/python tests/analysis_benchmark_100/run_benchmark.py --category "시계열"
  .venv/bin/python tests/analysis_benchmark_100/run_benchmark.py --list
"""
import argparse
import io
import json
import sys
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent

import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', tempfile.gettempdir())
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_data():
    cust_file = BENCHMARK_DIR / "customer_analytics.csv"
    tx_file = BENCHMARK_DIR / "transaction_history.csv"
    if not cust_file.exists() or not tx_file.exists():
        from dataset_generator import generate_datasets
        return generate_datasets(BENCHMARK_DIR)
    return pd.read_csv(cust_file), pd.read_csv(tx_file)

def load_benchmark_cases():
    json_path = BENCHMARK_DIR / "benchmark_cases.json"
    if not json_path.exists():
        raise FileNotFoundError(f"benchmark_cases.json not found in {BENCHMARK_DIR}. Run build_and_run_benchmark.py first.")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_single_case(case, df_cust, df_tx, verbose=True):
    code = case["python_code"]
    plt.close('all')
    
    exec_globals = {
        'pd': pd,
        'np': np,
        'plt': plt,
        'df': df_cust.copy(),
        'df_tx': df_tx.copy()
    }
    
    stdout_buf = io.StringIO()
    orig_stdout = sys.stdout
    error_msg = None
    has_fig = False
    
    try:
        sys.stdout = stdout_buf
        exec(code, exec_globals)
        has_fig = len(plt.get_fignums()) > 0
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
    finally:
        sys.stdout = orig_stdout
        plt.close('all')
        
    output_text = stdout_buf.getvalue().strip()
    status = "PASS" if error_msg is None else "FAIL"
    
    if verbose:
        print(f"\n[{status}] {case['id']}: {case['question']}")
        print(f"  카테고리: {case['category']} | 난이도: {case['difficulty']} | 출력 유형: {case['expected_output_type']}")
        print(f"  시각화(차트) 생성 여부: {'생성됨 (Figure created)' if has_fig else '없음 (Text/Table only)'}")
        if output_text:
            print("  --- 실행 출력 결과 (Stdout) ---")
            lines = output_text.splitlines()
            for line in lines[:8]:
                print(f"    {line}")
            if len(lines) > 8:
                print(f"    ... ({len(lines)-8} lines truncated)")
        if error_msg:
            print(f"  --- 에러 상세 ---")
            print(f"    {error_msg.splitlines()[-1]}")
            
    return {
        "id": case["id"],
        "status": status,
        "has_fig": has_fig,
        "error": error_msg
    }

def main():
    parser = argparse.ArgumentParser(description="100 Data Analysis Benchmark Runner")
    parser.add_argument("--all", action="store_true", help="Run all 100 benchmark test cases")
    parser.add_argument("--id", type=str, help="Run a specific test case by ID (e.g. TEST_001)")
    parser.add_argument("--category", type=str, help="Run all tests in a category (e.g. '시계열')")
    parser.add_argument("--list", action="store_true", help="List all 100 questions")
    args = parser.parse_args()
    
    cases = load_benchmark_cases()
    
    if args.list:
        print(f"{'ID':<10} | {'Category':<22} | {'Diff':<8} | {'Type':<8} | Question")
        print("-" * 90)
        for c in cases:
            print(f"{c['id']:<10} | {c['category']:<20} | {c['difficulty']:<8} | {c['expected_output_type']:<8} | {c['question'][:45]}...")
        return

    df_cust, df_tx = load_data()
    
    target_cases = []
    if args.id:
        target_cases = [c for c in cases if c["id"].upper() == args.id.upper()]
        if not target_cases:
            print(f"Error: Test case with ID '{args.id}' not found.")
            return
    elif args.category:
        target_cases = [c for c in cases if args.category in c["category"]]
        if not target_cases:
            print(f"Error: No test cases found matching category '{args.category}'.")
            return
    else:
        # Default to --all if no specific filter
        target_cases = cases
        
    print(f"=== Running Benchmark Tests ({len(target_cases)} cases) ===")
    pass_cnt = 0
    for case in target_cases:
        res = run_single_case(case, df_cust, df_tx, verbose=True)
        if res["status"] == "PASS":
            pass_cnt += 1
            
    print("\n" + "=" * 50)
    print(f"최종 결과: {pass_cnt}/{len(target_cases)} 통과 (성공률: {(pass_cnt/len(target_cases))*100:.1f}%)")
    print("=" * 50)

if __name__ == "__main__":
    main()
