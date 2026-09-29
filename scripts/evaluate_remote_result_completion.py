"""Real-model approval/result delivery with a synthetic SQL executor (no warehouse SQL)."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
import sqlglot
from sqlglot import exp
from core.analysis_sql import local_query
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.model_provider import build_analysis_chat_model
from core.analysis_agent.policy import RuntimePolicy
from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manual-approval', action='store_true')
    args = parser.parse_args()
    load_dotenv(ROOT / '.env')
    os.environ['LANGSMITH_TRACING'] = 'false'
    os.environ['LANGCHAIN_TRACING_V2'] = 'false'
    fixture = json.loads((ROOT / 'tests/fixtures/remote_result_completion.json').read_text())
    frame = pd.DataFrame(fixture['rows'])
    calls = []
    def factory(datasets):
        def execute(envelope):
            calls.append(envelope['query'])
            tree = sqlglot.parse_one(envelope['query'], read='databricks')
            for table in tree.find_all(exp.Table):
                if '.'.join(filter(None, [table.catalog, table.db, table.name])) != fixture['source']:
                    raise ValueError('Synthetic executor rejects unknown sources')
                table.set('this', exp.to_identifier('data'))
                table.set('db', None)
                table.set('catalog', None)
            data, truncated, _ = local_query(frame, tree.sql(dialect='duckdb'))
            info = datasets.register(data, source=fixture['source'], query=envelope['query'],
                coverage='truncated' if truncated else 'complete', predicate_known=True,
                snapshot=datetime.now(timezone.utc).isoformat())
            return {'status':'ready', 'dataset':asdict(info)}
        return execute
    context = [{'table':fixture['source'], 'observed_at':datetime.now(timezone.utc).isoformat(),
                'columns':[{'name':c,'dtype':'string'} for c in frame]}]
    policy = RuntimePolicy(model_timeout_seconds=45, require_remote_approval=args.manual_approval)
    model = build_analysis_chat_model(policy, provider='databricks')
    with tempfile.TemporaryDirectory(prefix='teleai-remote-result-') as root:
        runtime = GraphAnalysisRuntime(root, 'eval', 'receipt', model, policy=policy,
            connection_identity='synthetic', remote_factory=factory,
            reference_context_loader=lambda:context)
        try:
            prompt = f"{fixture['source']}에서 table_name, table_type을 조회해서 실제 결과 목록을 보여줘."
            proposal = runtime.submit(prompt)
            before = len(calls)
            result = proposal
            if proposal['status'] == 'awaiting_approval' and len(proposal['requests']) == 1:
                result = runtime.respond(proposal['requests'][0]['id'], approved=True)
            text = str(result.get('text', '')).replace('\\_', '_')
            passed = (before == (0 if args.manual_approval else 1) and len(calls) == 1 and result['status'] == 'answered'
                      and all(row['table_name'] in text for row in fixture['rows'])
                      and '도착하면' not in text)
            report = {'mode':'real Databricks model; synthetic local SQL executor; production graph',
                'status':'PASS' if passed else 'FAIL', 'prompt':prompt,
                'model':model.model_name, 'warehouse_sql_executions':0,
                'synthetic_sql_executions':len(calls), 'executions_before_approval':before,
                'queries':calls, 'proposal_status':proposal['status'], 'result':result,
                'messages':[{'type':m.type,'content':m.content,'tool_calls':getattr(m,'tool_calls',[])}
                            for m in runtime.events()],
                'recovery':runtime.inspect()['recovery'],
                'limitations':['Synthetic table listing only; not a live Databricks SQL connection test.']}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str)+'\n')
            print(json.dumps({k:report[k] for k in ['status','model','synthetic_sql_executions',
                                                  'executions_before_approval']},ensure_ascii=False))
        finally:
            runtime.close()


if __name__ == '__main__':
    main()
