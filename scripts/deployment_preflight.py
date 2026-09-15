#!/usr/bin/env python3
"""Run the read-only Telly deployment gate without printing secrets."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from core.deployment_preflight import evaluate_deployment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("local-desktop", "private-single-user", "multi-user"),
        default="local-desktop",
    )
    parser.add_argument("--json", action="store_true", help="print machine-readable output")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    report = evaluate_deployment(os.environ, profile=args.profile, project_root=ROOT)
    if args.json:
        print(json.dumps(report.public(), ensure_ascii=False, indent=2))
    else:
        print(f"Telly deployment preflight: {args.profile}")
        for check in report.checks:
            print(f"[{check.status.upper()}] {check.name}: {check.message}")
        print("READY" if report.ready else "NOT READY")
    return 0 if report.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
