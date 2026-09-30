#!/usr/bin/env python3
"""Inspect this checkout/interpreter without reading secrets or contacting services.

Run with the service's Python executable. This is a fresh process: it cannot
prove which code an already running Streamlit process has loaded.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    'main.py', 'pages/Telly.py', 'ui/analysis_page.py', 'ui/legacy_telly.py',
    'core/analysis_agent/remote_completion.py', 'core/analysis_agent/memory.py',
    'core/analysis_agent/runtime.py', 'core/analysis_loop.py', 'core/analysis_runtime.py',
)


def git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(['git', '-C', str(root), *args],
                                capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect(root: Path, expected_revision: str | None = None) -> dict:
    packages = {name: package_version(name) for name in
                ('langchain', 'langchain-core', 'langgraph', 'streamlit', 'sqlglot')}
    try:
        modern = int((packages['langchain'] or '').split('.')[0]) >= 1
        route = 'ui/analysis_page.py' if modern else 'ui/legacy_telly.py'
    except ValueError:
        route = 'unavailable: langchain version missing or invalid'
    included = None
    resolved = None
    if expected_revision:
        resolved = git(root, 'rev-parse', '--verify', '--end-of-options',
                       expected_revision + '^{commit}')
        if resolved and git(root, 'rev-parse', '--verify', 'HEAD'):
            try:
                result = subprocess.run(
                    ['git', '-C', str(root), 'merge-base', '--is-ancestor', resolved, 'HEAD'],
                    capture_output=True, timeout=5)
                included = {0: True, 1: False}.get(result.returncode)
            except (OSError, subprocess.SubprocessError):
                pass
    status = git(root, 'status', '--porcelain', '--untracked-files=no')
    return {
        'scope': 'checkout_and_fresh_interpreter_only',
        'running_server_verified': False,
        'checkout': str(root.resolve()),
        'python': sys.executable,
        'python_version': sys.version.split()[0],
        'branch': git(root, 'branch', '--show-current'),
        'revision': git(root, 'rev-parse', '--verify', 'HEAD'),
        'tracked_changes': bool(status) if status is not None else None,
        'expected_revision': resolved,
        'expected_revision_in_history': included,
        'packages': packages,
        'telly_page_for_this_interpreter': route,
        'files_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                         if (root / name).is_file() else None for name in FILES},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-revision', help='Known fix commit to check in local Git history')
    args = parser.parse_args()
    print(json.dumps(collect(ROOT, args.expected_revision), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
