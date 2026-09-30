"""Evaluate latest-record-per-key distribution using synthetic data and a real model."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from PIL import Image
from langchain_core.messages import AIMessage, ToolMessage
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.model_provider import build_analysis_chat_model
from core.analysis_agent.policy import RuntimePolicy
from utils.analysis_datasets import stored_dataset_digest
from utils.analysis_datasets import DatasetStore
from core.analysis_sql import local_query
from core.analysis_runtime_tools import build_analysis_tools
from core.analysis_tool_contract import AnalysisToolContext


def run_sql_contract(fixture, output):
    """Separate model planning failures from SQL tool failures; never count as agent success."""
    frame = pd.DataFrame(fixture['rows'], columns=fixture['columns'])
    key, product, clock = ['"' + c.replace('"', '""') + '"' for c in fixture['columns'][:3]]
    ranked = (f'SELECT {key}, {product}, ROW_NUMBER() OVER '
              f'(PARTITION BY {key} ORDER BY {clock} DESC) AS row_rank FROM data')
    queries = {
        'window_subquery': f'SELECT {product}, COUNT(*) AS device_count FROM ({ranked}) ranked '
                           f'WHERE row_rank = 1 GROUP BY {product}',
        'qualify_rows': f'SELECT {key}, {product}, {clock} FROM data QUALIFY ROW_NUMBER() OVER '
                         f'(PARTITION BY {key} ORDER BY {clock} DESC) = 1',
    }
    store = DatasetStore()
    raw = store.register(frame, source=fixture['source'], coverage='complete',
                         predicate_known=True, snapshot='synthetic-v1')
    before = stored_dataset_digest(store, raw.id)
    context = AnalysisToolContext(store, {}, [], lambda **kwargs: None)
    tools = {t.name: t for t in build_analysis_tools(context)}
    results = []
    for name, query in queries.items():
        data, truncated, _ = local_query(frame, query)
        try:
            result = tools['local_analysis_sql'].run(dataset_id=raw.id, query=query)
        except Exception as exc:
            result = {'error_type': type(exc).__name__, 'message': str(exc)}
        results.append({'case': name, 'query': query, 'duckdb_rows': data.to_dict(orient='records'),
                        'truncated': truncated, 'tool_result': result})
    selected = results[-1]['tool_result']['dataset']['id']
    counted = tools['local_analysis_sql'].run(dataset_id=selected,
        query=f'SELECT {product}, COUNT(*) AS device_count FROM data GROUP BY {product}',
        current_result_only=True)
    data = store.frames[counted['dataset']['id']]
    counts = dict(zip(data[fixture['columns'][1]], data['device_count']))
    assert counts == fixture['expected_counts'], counts
    chart = tools['render_chart_spec'].run(dataset_id=counted['dataset']['id'], kind='bar',
        x=fixture['columns'][1], y='device_count', aggregation='none',
        title='Synthetic fixture: latest record per device', y_label='Unique devices')
    card = context.artifacts[chart['cards'][0]['id']]
    with Image.open(BytesIO(card.image)) as image:
        image.verify()
    (output.parent / 'reference_expected.png').write_bytes(card.image)
    assert stored_dataset_digest(store, raw.id) == before
    report = {'mode': 'manually specified SQL through production tools; NOT autonomous agent success',
              'results': results, 'reference_counts': counts, 'raw_preserved': True,
              'reference_chart': 'reference_expected.png'}
    (output.parent / 'sql_contract.json').write_text(json.dumps(report, ensure_ascii=False,
                                                            indent=2, default=str)+'\n')


def run_case(case, fixture, output, repeat, model_planning=False):
    rename = case.get('rename', {})
    key, product, clock = [rename.get(c, c) for c in fixture['columns'][:3]]
    frame = pd.DataFrame(fixture['rows'], columns=fixture['columns']).rename(columns=rename)
    for col in (clock, 'ingested_at'):
        frame[col] = pd.to_datetime(frame[col])
    reference = frame.sort_values(clock).drop_duplicates(key, keep='last')
    assert dict(zip(reference[key], reference[product])) == fixture['expected_latest']
    assert dict(Counter(reference[product])) == fixture['expected_counts']
    policy = RuntimePolicy(model_timeout_seconds=45)
    model = build_analysis_chat_model(policy, provider='databricks')
    with tempfile.TemporaryDirectory(prefix='teleai-latest-') as directory:
        runtime = GraphAnalysisRuntime(directory, 'synthetic-eval', case['id'], model, policy=policy)
        try:
            raw = runtime.datasets.register(frame, source=fixture['source'], coverage='complete',
                                            predicate_known=True, snapshot='synthetic-v1')
            runtime.select_dataset(raw.id)
            before = stored_dataset_digest(runtime.datasets, raw.id)
            started = time.monotonic()
            from contextlib import nullcontext
            from unittest.mock import patch
            planning = (patch.object(runtime.recovery, '_next_local', return_value=None)
                        if model_planning else nullcontext())
            with planning:
                result = runtime.submit(case['prompt'])
            recovery = runtime.inspect()['recovery']
            calls, observations = [], []
            for message in runtime.events():
                if isinstance(message, AIMessage):
                    calls.extend(message.tool_calls)
                if isinstance(message, ToolMessage):
                    try:
                        value = json.loads(message.content)
                    except (ValueError, TypeError):
                        value = {'text': str(message.content)}
                    observations.append({'tool': message.name, 'result': value})
            charts = []
            for chart_id in runtime.artifacts:
                card = runtime.artifacts[chart_id]
                data = runtime.datasets.frames[card.dataset_id]
                info = runtime.datasets.metadata[card.dataset_id]
                image_path = output.parent / f'{case["id"]}-{repeat}-{len(charts)}.png'
                image_path.write_bytes(card.image)
                with Image.open(BytesIO(card.image)) as image:
                    image.verify()
                counts = None
                if product in data:
                    if key in data and len(data) == len(fixture['expected_latest']) and data[key].is_unique:
                        counts = dict(Counter(data[product]))
                    else:
                        numeric = [c for c in data if c != product and pd.api.types.is_numeric_dtype(data[c])]
                        if len(numeric) == 1 and data[product].is_unique:
                            counts = {str(row[product]): float(row[numeric[0]]) for _, row in data.iterrows()}
                charts.append({'kind': card.kind, 'columns': card.columns, 'image': image_path.name,
                    'accepted_by_completion': chart_id in recovery.get('artifact_ids', []),
                    'dataset_id': card.dataset_id, 'query': info.query, 'counts': counts,
                    'data': data.to_dict(orient='records'),
                    'counts_match': counts == fixture['expected_counts']})
            preserved = stored_dataset_digest(runtime.datasets, raw.id) == before
            answer = result.get('text', '')
            if case.get('clarification'):
                # Two plausible, conflicting time columns: choosing either without asking is unsafe.
                matched = (not charts and result['status'] == 'answered' and
                           any(word in answer for word in ['어떤', '어느', '선택', '알려', '의미하']) and
                           any(word in answer for word in ['시간', '기준', 'event_time', 'ingested_at']))
            else:
                matched = (result['status'] == 'answered'
                           and bool(recovery.get('latest_selection_evidence'))
                           and any(c['counts_match'] and c['accepted_by_completion'] for c in charts))
            datasets = [{'id': identity, 'query': info.query, 'parent_id': info.parent_id,
                         'rows': runtime.datasets.frames[identity].to_dict(orient='records')}
                        for identity, info in runtime.datasets.metadata.items() if identity != raw.id]
            return {'id': case['id'], 'repeat': repeat, 'prompt': case['prompt'],
                'status': 'PASS' if matched and preserved and not runtime.inspect()['requests'] else 'FAIL',
                'agent_status': result['status'], 'answer': answer, 'charts': charts, 'datasets': datasets,
                'raw_preserved': preserved, 'calls': calls, 'observations': observations,
                'state': recovery, 'model_calls': recovery.get('model_calls'),
                'model_planning_required': model_planning,
                'elapsed_seconds': round(time.monotonic()-started, 3)}
        finally:
            runtime.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--id', action='append')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--contracts-only', action='store_true')
    parser.add_argument('--model-planning', action='store_true',
                        help='Diagnostic only: disable deterministic local tool dispatch.')
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    os.environ['LANGSMITH_TRACING'] = 'false'
    os.environ['LANGCHAIN_TRACING_V2'] = 'false'
    fixture = json.loads((ROOT / 'tests/fixtures/latest_per_key.json').read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source = pd.DataFrame(fixture['rows'], columns=fixture['columns'])
    source.to_csv(args.output.parent / 'synthetic_device_events.csv', index=False)
    source.sort_values(fixture['columns'][2]).drop_duplicates(fixture['columns'][0], keep='last').to_csv(
        args.output.parent / 'expected_latest_rows.csv', index=False)
    run_sql_contract(fixture, args.output)
    if args.contracts_only:
        print('Reference SQL pipeline and PNG verified; see sql_contract.json.')
        return
    report = {'mode': 'production graph with Databricks model configured; actual model call count per result; synthetic local data',
              'generated_at': datetime.now(timezone.utc).isoformat(), 'warehouse_sql_executions': 0,
              'model_planning_required': args.model_planning,
              'limitations': ['Not the user table or real schema; source data is a 10-row synthetic fixture.',
                             'No remote warehouse loading or large-data performance measurement.'], 'results': []}
    for repeat in range(1, args.repeats+1):
        for case in fixture['cases']:
            if args.id and case['id'] not in args.id:
                continue
            result = run_case(case, fixture, args.output, repeat, args.model_planning)
            report['results'].append(result)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str)+'\n')
            print(json.dumps({k: result[k] for k in ['id', 'repeat', 'status', 'agent_status',
                                                   'model_calls', 'elapsed_seconds']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
