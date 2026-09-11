"""Benchmark Builder and Automated Runner.
Executes all 100 benchmark test cases against the synthetic datasets,
captures stdout and matplotlib figures, verifies zero execution errors,
and generates benchmark_cases.json and benchmark_execution_report.md.
"""
import io
import json
import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg') # Headless mode for automated testing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BENCHMARK_DIR = Path(__file__).resolve().parent

from test_definitions_part1 import get_all_test_specs as get_p1
from test_definitions_part2 import get_part2_specs as get_p2
from test_definitions_part3 import get_part3_specs as get_p3
from test_definitions_part4 import get_part4_specs as get_p4
from test_definitions_part5 import get_part5_specs as get_p5

def load_all_specs():
    specs = []
    specs.extend(get_p1())
    specs.extend(get_p2())
    specs.extend(get_p3())
    specs.extend(get_p4())
    specs.extend(get_p5())
    return specs

def execute_case(spec, df_cust, df_tx):
    code = spec['python_code']
    plt.close('all')
    
    # Safe execution scope
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
        fignums = plt.get_fignums()
        has_fig = len(fignums) > 0
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
    finally:
        sys.stdout = orig_stdout
        plt.close('all')
        
    captured_text = stdout_buf.getvalue().strip()
    status = "FAIL" if error_msg else "PASS"
    
    return {
        "status": status,
        "stdout": captured_text,
        "has_figure": has_fig,
        "error": error_msg
    }

def main():
    cust_path = BENCHMARK_DIR / "customer_analytics.csv"
    tx_path = BENCHMARK_DIR / "transaction_history.csv"
    
    if not cust_path.exists() or not tx_path.exists():
        from dataset_generator import generate_datasets
        df_cust, df_tx = generate_datasets(BENCHMARK_DIR)
    else:
        df_cust = pd.read_csv(cust_path)
        df_tx = pd.read_csv(tx_path)
        
    specs = load_all_specs()
    print(f"Total Test Cases Loaded: {len(specs)}")
    
    results = []
    pass_count = 0
    fail_count = 0
    
    for i, spec in enumerate(specs, 1):
        res = execute_case(spec, df_cust, df_tx)
        
        # Merge execution output into spec for the final JSON artifact
        case_data = dict(spec)
        case_data["execution_result"] = {
            "status": res["status"],
            "has_figure": res["has_figure"],
            "stdout_preview": res["stdout"][:500] if len(res["stdout"]) > 500 else res["stdout"]
        }
        
        if res["status"] == "PASS":
            pass_count += 1
        else:
            fail_count += 1
            print(f"FAILED {spec['id']}: {res['error']}")
            
        results.append(case_data)
        
    print(f"\nExecution Summary: PASS={pass_count}/100, FAIL={fail_count}/100")
    
    # Save benchmark_cases.json
    out_json = BENCHMARK_DIR / "benchmark_cases.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_json}")
    
    # Generate execution report Markdown
    report_path = BENCHMARK_DIR / "benchmark_execution_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 100 Data Analysis Benchmark Execution Report\n\n")
        f.write(f"- **Total Tests**: {len(results)}\n")
        f.write(f"- **Passed**: {pass_count}\n")
        f.write(f"- **Failed**: {fail_count}\n")
        f.write(f"- **Success Rate**: {(pass_count/len(results))*100:.1f}%\n\n")
        f.write("## Category Breakdown\n\n")
        
        cats = {}
        for r in results:
            c = r["category"]
            cats[c] = cats.get(c, 0) + (1 if r["execution_result"]["status"] == "PASS" else 0)
        for c, cnt in cats.items():
            f.write(f"- **{c}**: {cnt}/10 Passed\n")
            
        f.write("\n## 100 Test Cases Verification Table\n\n")
        f.write("| ID | Category | Difficulty | Output Type | Figure Generated | Status |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in results:
            fig_str = "Yes" if r["execution_result"]["has_figure"] else "No"
            f.write(f"| {r['id']} | {r['category']} | {r['difficulty']} | {r['expected_output_type']} | {fig_str} | {r['execution_result']['status']} |\n")
            
    print(f"Saved: {report_path}")

if __name__ == "__main__":
    main()
