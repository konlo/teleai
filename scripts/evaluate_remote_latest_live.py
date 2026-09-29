"""Real read-only warehouse journey plus an independent latest-row SQL oracle."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
from core.analysis_agent.databricks import ConnectionConfig, make_executor
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.model_provider import build_analysis_chat_model
from core.analysis_agent.policy import RuntimePolicy
from core.analysis_catalog import _quoted_table
from utils.analysis_remote_latest import _quote


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'key', 'order', 'value'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bins',type=int,help='Evaluate a numeric histogram instead of category counts')
    parser.add_argument('--stale-schema', action='store_true')
    parser.add_argument('--tie-break', help='Explicit secondary descending order column')
    parser.add_argument('--drop-missing-before-selection', action='store_true')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    load_dotenv(ROOT/'.env')
    config = ConnectionConfig.from_env()
    source = args.source.split('.')
    if len(source) != 3:
        raise ValueError('A full catalog.schema.table is required')
    from databricks import sql
    queries = []
    def direct(query):
        queries.append(query)
        with sql.connect(server_hostname=config.server_hostname, http_path=config.http_path,
                access_token=config.access_token, catalog=config.catalog, schema=config.schema) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                names = [d[0] for d in cursor.description]
                rows=cursor.fetchmany(301)
                if len(rows)>300:raise ValueError('Independent oracle transfer limit exceeded')
                return [dict(zip(names,row)) for row in rows]
    literal = lambda v: "'"+v.replace("'", "''")+"'"
    report = {'source': args.source, 'status': 'NOT_COMPLETED', 'queries': queries}
    stage = 'schema'
    try:
        rows = direct('SELECT column_name, data_type FROM '+_quoted_table(source[0]+'.information_schema.columns')
            +' WHERE table_schema = '+literal(source[1])+' AND table_name = '+literal(source[2])+' ORDER BY ordinal_position')
        if not rows:
            raise ValueError('No observed schema')
        reference = [{'table': args.source, 'observed_at': datetime.now(timezone.utc).isoformat(),
            'columns': [{'name': row['column_name'], 'dtype': row['data_type']} for row in rows]}]
        report['observed_schema'] = reference
        if args.stale_schema:
            reference[0]['observed_at'] = '2020-01-01T00:00:00+00:00'
            report['stale_schema_injected'] = True
        policy = RuntimePolicy(model_timeout_seconds=45)
        model = build_analysis_chat_model(policy, provider='databricks')
        def factory(datasets):
            execute = make_executor(config, datasets)
            def tracked(envelope):
                queries.append(envelope['query'])
                return execute(envelope)
            return tracked
        stage = 'agent'
        with tempfile.TemporaryDirectory(prefix='telly-remote-latest-') as root:
            r = GraphAnalysisRuntime(root, 'live-eval', 'latest', model, policy=policy,
                connection_identity=config.identity(), remote_factory=factory,
                reference_context_loader=lambda: reference)
            try:
                prompt = (args.source+' 테이블에서 '+args.key+'별 unique한 값마다 '+args.order
                    +'가 가장 큰 마지막 행의 '+args.value+' 분포를 그려줘. '
                    +(f'{args.bins}개 구간 histogram으로 그려줘.' if args.bins else args.value+'는 범주야.'))
                if args.tie_break:prompt+=' 동률이면 '+args.tie_break+'이 가장 큰 행을 사용해줘.'
                if args.drop_missing_before_selection:prompt+=' 결측행을 최신행 선택 전에 제외해줘.'
                started = time.monotonic()
                result = r.submit(prompt)
                state = r.inspect()['recovery']
                proof = state.get('latest_selection_evidence') or {}
                report.update(prompt=prompt, agent_status=result['status'], answer=result.get('text'),
                    elapsed_seconds=round(time.monotonic()-started, 3), model_calls=state.get('model_calls'),
                    remote_result_rows=r.datasets.metadata[proof['input_result_id']].rows if proof else None,
                    input_rows=proof.get('input_rows'), selected_keys=proof.get('selected_keys'),
                    excluded_rows=proof.get('excluded_rows'),tie_break=args.tie_break,
                    null_policy='drop_before_selection' if args.drop_missing_before_selection else 'reject')
                if not proof:
                    report['status'] = 'INCOMPLETE'
                    return
                actual = (proof['chart_spec']['counts'] if args.bins else
                    {row[args.value]: row[proof['count_column']] for row in proof['counts']})
                stage = 'oracle'
                # ROW_NUMBER independently checks winners; agent DENSE_RANK proof
                # already establishes absence of maximum ties and NULLs.
                order_columns=[args.order]+([args.tie_break] if args.tie_break else [])
                oracle_filter=''
                if args.drop_missing_before_selection:
                    # Independently express eligibility with conjunctions, rather than
                    # copying the planner's negated disjunction over aliased columns.
                    types={c['name']:c['dtype'].lower() for c in reference[0]['columns']}
                    parts=[]
                    for c in [args.key,*order_columns,args.value]:
                        q=_quote(c);parts.append(q+' IS NOT NULL')
                        if types[c] in ('float','double'):
                            parts.extend(['NOT isnan('+q+')',q+" < CAST('Infinity' AS DOUBLE)",q+" > CAST('-Infinity' AS DOUBLE)"])
                    oracle_filter=' WHERE '+' AND '.join(parts)
                oracle_query = ('SELECT CAST('+_quote(args.value)+' AS STRING) AS category, COUNT(*) AS frequency FROM '
                    +'(SELECT '+_quote(args.value)+', ROW_NUMBER() OVER (PARTITION BY '+_quote(args.key)
                    +' ORDER BY '+', '.join(_quote(c)+' DESC' for c in order_columns)+') AS rn FROM '+_quoted_table(args.source)+oracle_filter
                    +') chosen WHERE rn = 1 GROUP BY '+_quote(args.value)+' ORDER BY category')
                oracle = {row['category']: row['frequency'] for row in direct(oracle_query)}
                if args.bins:
                    import numpy as np
                    values=np.array([float(v) for v in oracle])
                    counts,edges=np.histogram(values,bins=args.bins,weights=np.array(list(oracle.values())))
                    oracle=counts.tolist()
                    report['oracle_edges']=edges.tolist()
                    report['actual_edges']=proof['chart_spec']['edges']
                    if not np.allclose(edges,proof['chart_spec']['edges']):raise ValueError('Histogram edges differ')
                report.update(actual_counts=actual, oracle_counts=oracle,
                    status='PASS' if actual == oracle else 'FAIL')
                for identity in state['artifact_ids']:
                    args.output.with_suffix('.png').write_bytes(r.artifacts[identity].image)
                before = len(queries)
                repeated = r.submit(prompt)
                report.update(reuse_status=repeated['status'], reuse_queries=len(queries)-before,
                    reused='재사용' in repeated.get('text', ''))
                if not report['reused'] or report['reuse_queries']:
                    report['status'] = 'FAIL'
            finally:
                r.close()
    except Exception as error:
        # Provider exceptions may include connection information; record type/status only.
        report.update(status='BLOCKED', stage=stage, error_type=type(error).__name__)
        details = getattr(error, 'context', {})
        if isinstance(details, dict):
            report['http_status'] = details.get('http-code')
    finally:
        report['query_count'] = len(queries)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps({k: report.get(k) for k in ('status', 'stage', 'error_type', 'http_status',
            'query_count', 'agent_status', 'input_rows', 'selected_keys', 'remote_result_rows', 'reuse_queries')}))


if __name__ == '__main__':
    main()
