#!/usr/bin/env python3
"""Preview or atomically switch a code release; never touch persistent data."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import uuid


def switch_release(releases: Path, current: Path, target_name: str, *, apply: bool) -> dict[str, str | bool]:
    releases = releases.resolve(strict=True)
    if not target_name or Path(target_name).name != target_name or target_name in {".", ".."}:
        raise ValueError("target must be one release directory name")
    target = releases / target_name
    if target.is_symlink() or not target.is_dir() or not (target / "requirements-agent.txt").is_file():
        raise ValueError("target must be a real release directory with requirements-agent.txt")
    if current.parent.resolve(strict=True) == releases:
        raise ValueError("current link must be outside the releases directory")
    if current.exists() and not current.is_symlink():
        raise ValueError("current must be a symlink")
    previous = current.resolve(strict=True) if current.is_symlink() else None
    if previous is not None and previous.parent != releases:
        raise ValueError("current points outside the releases directory")
    if apply:
        temporary = current.with_name(f".{current.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.symlink_to(target, target_is_directory=True)
            os.replace(temporary, current)
        finally:
            temporary.unlink(missing_ok=True)
    return {"applied": apply, "previous": str(previous) if previous else "", "target": str(target)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--releases", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--target", required=True, help="name of an existing release directory")
    parser.add_argument("--apply", action="store_true", help="perform the atomic symlink switch")
    args = parser.parse_args()
    try:
        result = switch_release(args.releases, args.current, args.target, apply=args.apply)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"release switch rejected: {exc}\n")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
