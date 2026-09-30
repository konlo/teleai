#!/usr/bin/env python3
"""Summarize one retained agent run without exporting the raw log."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from core.analysis_agent.support_report import summarize, brief


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log',type=Path,required=True,help='Conversation runtime.jsonl path')
    selector=parser.add_mutually_exclusive_group()
    selector.add_argument('--run-id')
    selector.add_argument('--error-id')
    parser.add_argument('--brief',action='store_true',help='print a short manual-report checklist')
    args=parser.parse_args()
    report=summarize(args.log,run_id=args.run_id,error_id=args.error_id)
    print(brief(report) if args.brief else json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report['found'] else 1


if __name__=='__main__':raise SystemExit(main())
