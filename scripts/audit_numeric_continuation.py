"""Offline audit of continuation breadth; this is not an LLM quality score.

The scripted model prepares numeric data once, then deliberately refuses further
inference. Each case uses a fresh store and an independently stated result.
"""
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from langchain_core.messages import AIMessage, ToolMessage
from core.analysis_agent.runtime import GraphAnalysisRuntime
from tests.test_analysis_numeric import PrepareOnlyModel
from utils.analysis_datasets import stored_dataset_digest


def main():
    cases = [
        ('scalar_chart', '`sensor reading` 평균과 히스토그램을 보여줘.', 'mean=2.5; histogram'),
        ('filtered_scalar_chart', '`sensor reading` >= 3인 행의 `sensor reading` 평균과 히스토그램을 보여줘.', 'mean=3.5; histogram of 3,4'),
        ('grouped_scalar', '`cohort`별로 `sensor reading` 평균을 알려줘.', 'A=1.5; B=3.5'),
        ('custom_chart', '`sensor reading` 평균과 구간 수 5개인 히스토그램을 보여줘.', 'mean=2.5; histogram with 5 bins'),
    ]
    rows = []
    for name, request, oracle in cases:
        with tempfile.TemporaryDirectory(prefix='teleai-audit-') as root:
            model = PrepareOnlyModel()
            runtime = GraphAnalysisRuntime(root, 'audit', name, model)
            try:
                raw = runtime.datasets.register(pd.DataFrame({
                    'sensor reading': ['1', '2', '3', '4', 'missing'],
                    'cohort': ['A', 'A', 'B', 'B', 'B'],
                }), source='fixture.unseen_audit', coverage='unknown')
                runtime.select_dataset(raw.id)
                model.dataset_id = raw.id
                before = stored_dataset_digest(runtime.datasets, raw.id)
                prompt = '현재 로딩된 표본에서 ' + request + ' "missing" 문자열만 결측값으로 처리해.'
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
                mean_ok = any(row.get('average') == expected_mean for frame in evidence for row in frame)
                chart_ok = bool(state.get('artifact_ids'))
                if name == 'custom_chart':
                    chart_ok = chart_ok and any(s and s.get('bins') == 5 for s in chart_specs)
                # Grouped output needs two independently specified groups;
                # profile text does not count as a grouped calculation.
                group_ok = any({row.get('cohort'): row.get('average') for row in frame}
                               == {'A': 1.5, 'B': 3.5} for frame in evidence)
                oracle_pass = (result['status'] == 'answered' and
                               (group_ok if name == 'grouped_scalar' else mean_ok and chart_ok))
                rows.append({'case': name, 'prompt': prompt, 'oracle': oracle,
                    'oracle_pass': oracle_pass,
                    'status': result['status'], 'error_type': result.get('error_type'),
                    'model_calls': model.calls, 'tools': calls, 'observations': observations,
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
    report = {'mode': 'offline scripted model; continuation coverage, not actual model failure rate',
              'git_baseline': '89effbb', 'warehouse_sql_executions': 0,
              'limitations': ['A real model may recover through additional inference.',
                              'Second inference intentionally raises AssertionError, not a transient outage.'],
              'cases': rows}
    output = ROOT / 'docs/evaluation/2026-09-27_numeric_continuation_audit.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps([{k: r[k] for k in ('case', 'status', 'model_calls', 'chart_count',
                                        'numeric_branch_bound', 'raw_preserved')} for r in rows], indent=2))


if __name__ == '__main__':
    main()
