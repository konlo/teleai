#!/usr/bin/env python3
"""Apply the configured single-process capacity gate to a benchmark report."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from core.runtime_capacity import CapacityPolicy, evaluate_capacity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    report = json.loads(args.input.read_text())
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.input),
        **evaluate_capacity(report, CapacityPolicy.from_mapping(os.environ)),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
