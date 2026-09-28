"""Repeated real-model evaluation of fixed numeric audit cases; no warehouse SQL."""
import json
import argparse
import os
import time
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from langchain_core.messages import AIMessage, ToolMessage
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.model_provider import build_analysis_chat_model
from core.analysis_agent.policy import RuntimePolicy
from utils.analysis_datasets import stored_dataset_digest


def main():
    cases = [
        ('scalar_chart', '`sensor reading` 평균과 히스토그램을 보여줘.', 'mean=2.5; histogram'),
        ('filtered_scalar_chart', '`sensor reading` >= 3인 행의 `sensor reading` 평균과 히스토그램을 보여줘.', 'mean=3.5; histogram of 3,4'),
        ('grouped_scalar', '`cohort`별로 `sensor reading` 평균을 알려줘.', 'A=1.5; B=3.5'),
        ('custom_chart', '`sensor reading` 평균과 구간 수 5개인 히스토그램을 보여줘.', 'mean=2.5; histogram with 5 bins'),
    ]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    os.environ['LANGSMITH_TRACING'] = 'false'
    os.environ['LANGCHAIN_TRACING_V2'] = 'false'
    rows = []
    for name, request, oracle, repeat in [(*case, i) for i in range(1, args.repeats + 1) for case in cases]:
        with tempfile.TemporaryDirectory(prefix='teleai-audit-') as root:
            model = build_analysis_chat_model(RuntimePolicy(model_timeout_seconds=45), provider='databricks')
            runtime = GraphAnalysisRuntime(root, 'audit', name, model)
            try:
                raw = runtime.datasets.register(pd.DataFrame({
                    'sensor reading': ['1', '2', '3', '4', 'missing'],
                    'cohort': ['A', 'A', 'B', 'B', 'B'],
                }), source='fixture.unseen_audit', coverage='unknown')
                runtime.select_dataset(raw.id)
                before = stored_dataset_digest(runtime.datasets, raw.id)
                prompt = '현재 로딩된 표본에서 ' + request + ' "missing" 문자열만 결측값으로 처리해.'
                started = time.monotonic()
                result = runtime.submit(prompt)
                state = runtime.inspect()['recovery']
                calls, observations, chart_specs = [], [], []
                for message in runtime.events():
                    if isinstance(message, AIMessage):
                        calls.extend(c['name'] for c in message.tool_calls)
                    if isinstance(message, ToolMessage):
                        try:
                            value = json.loads(message.content)
                        except (ValueError, TypeError):
                            value = {}
                        observations.append({'tool': message.name, 'status': value.get('status'),
                                             'error_code': value.get('error_code')})
                        if message.name == 'render_chart_spec':
                            chart_specs.append(value.get('chart_spec'))
                evidence = [runtime.datasets.frames[i].to_dict(orient='records')
                            for i in state.get('evidence_ids', [])]
                expected_mean = 3.5 if name == 'filtered_scalar_chart' else 2.5
                mean_ok = any(len(row) == 1 and next(iter(row.values())) == expected_mean for frame in evidence for row in frame)
                chart_ok = bool(state.get('artifact_ids'))
                if name == 'custom_chart':
                    chart_ok = chart_ok and any(s and s.get('bins') == 5 for s in chart_specs)
                # Grouped output needs two independently specified groups;
                # profile text does not count as a grouped calculation.
                group_ok = any({row.get('cohort'): next((v for k, v in row.items() if k != 'cohort'), None) for row in frame}
                               == {'A': 1.5, 'B': 3.5} for frame in evidence)
                oracle_pass = (result['status'] == 'answered' and
                               (group_ok if name == 'grouped_scalar' else mean_ok and chart_ok))
                rows.append({'case': name, 'repeat': repeat, 'prompt': prompt, 'oracle': oracle,
                    'oracle_pass': oracle_pass,
                    'status': result['status'], 'error_type': result.get('error_type'),
                    'model_calls': state.get('model_calls'), 'elapsed_seconds': round(time.monotonic()-started, 3), 'tools': calls, 'observations': observations,
                    'chart_count': len(state.get('artifact_ids', [])),
                    'chart_specs': chart_specs,
                    'chart_reasons': [runtime.artifacts[i].reason for i in state.get('artifact_ids', [])],
                    'intent': {k: state.get(k) for k in ('profile_kind', 'calculation',
                        'operations', 'scalar_grouping', 'chart_spec_requested', 'required_columns')},
                    'evidence': evidence,
                    'raw_preserved': stored_dataset_digest(runtime.datasets, raw.id) == before,
                    'numeric_branch_bound': bool(state.get('numeric_prepared_dataset')),
                    'answer': result.get('text', '')})
            finally:
                runtime.close()
        records = []
        for row in rows:
            records.append({**row, 'id': row['case'] + ':' + str(row['repeat']),
                'status': 'PASS' if row['oracle_pass'] else 'FAIL', 'agent_status':row['status'],
                'final_output': row['answer'], 'reference_facts': row['oracle'],
                'runtime_metadata': {'recovery_model_calls':row['model_calls']},
                'tool_calls':[{'tool':t} for t in row['tools']]})
        report = {'mode':'live-databricks-model', 'git_baseline':'89effbb',
            'data':'synthetic five-row fixture; production recovery enabled',
            'warehouse_sql_executions':0, 'results':records,
            'limitations':['Fixed four audit cases; not unseen questions or an overall quality estimate.',
                           'Model configured but deterministic paths may make zero model calls.',
                           'No injected errors; failures are preserved without manual retry.']}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({k:records[-1][k] for k in ('id','status','agent_status','model_calls','elapsed_seconds')},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
