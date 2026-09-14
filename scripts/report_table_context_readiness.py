#!/usr/bin/env python3
"""Report runtime TableContext freshness and alias coverage without querying data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.analysis_catalog import load_saved_reference_context, resolve_table_context
from utils.analysis_datasets import DatasetStore


def build_report(storage_dir: Path) -> dict:
    tables = []
    for item in load_saved_reference_context(storage_dir):
        resolution = resolve_table_context([item], DatasetStore(), item['table'])
        columns = item.get('columns', [])
        tables.append({
            'table': item['table'],
            'training_status': item.get('training_status'),
            'observed_at': item.get('observed_at'),
            'freshness': item.get('freshness'),
            'schema_fingerprint': item.get('schema_fingerprint'),
            'column_count': len(columns),
            'columns_with_aliases': sum(bool(column.get('aliases')) for column in columns),
            'alias_count': sum(len(column.get('aliases', [])) for column in columns),
            'runtime_resolution': resolution.get('status'),
            'refresh_query': resolution.get('refresh_query'),
        })
    return {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'storage_dir': str(storage_dir),
        'tables': tables,
        'summary': {
            'table_count': len(tables),
            'fresh_tables': sum(table['freshness'] == 'fresh' for table in tables),
            'stale_tables': sum(table['freshness'] == 'stale' for table in tables),
            'tables_needing_refresh': sum(table['runtime_resolution'] == 'needs_refresh' for table in tables),
        },
        'safety': {
            'databricks_calls': 0,
            'raw_rows_read': False,
            'credentials_read': False,
            'refresh_queries_executed': 0,
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--storage-dir', type=Path, default=ROOT / '.telly_table_context')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    report = build_report(args.storage_dir)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end='')
    return 2 if not report['tables'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
