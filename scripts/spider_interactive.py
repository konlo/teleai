"""Interactive public SQLite evaluation, with actual production graph completion."""
from contextlib import closing
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import tempfile
import time

import pandas as pd

INSTRUCTIONS = '''Solve the supplied analytical question using the observed public SQLite schema
and task documentation. The tool query_databricks connects to a read-only public
SQLite database in this evaluation; it does not contact Databricks. Execute
bounded exploratory SELECTs when needed, inspect their results, and continue
until the requested final calculation is actually complete. An exploratory
sample or schema probe does not answer a requested aggregate. Preserve every
requested condition, unit, grouping, join role and formula. Inspect declared
relationships before joining. List all physical source tables separated by | in
the source argument, excluding CTE aliases. Use SQLite syntax and observed data
encodings. Do not invent results, implicit years or business definitions. If a
required meaning is genuinely unknown, explain what remains unresolved.
'''


def sqlite_executor(path, datasets, calls, *, max_rows=1000, timeout=10, max_bytes=16*1024*1024):
    """Read-only bounded transfer; the normal receipt ledger wraps this executor."""
    from core.analysis_sql import validate_query
    from core.analysis_load_plan import source_plan
    from utils.analysis_provenance import query_coverage, raw_conditions

    def execute(envelope):
        query = envelope['query']
        tree = validate_query(query, dialect='sqlite')
        plan = source_plan(envelope['source'], query, dialect='sqlite')
        started = time.monotonic()
        entry = {'query': query, 'status': 'started'}
        calls.append(entry)
        try:
            with closing(sqlite3.connect(f'file:{path.resolve()}?mode=ro', uri=True)) as db:
                db.execute('PRAGMA query_only=ON')
                db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, min(max_bytes,1024*1024))
                db.set_progress_handler(lambda: int(time.monotonic()-started >= timeout), 1000)
                cursor = db.execute(query)
                columns = [item[0] for item in cursor.description or []]
                if not columns or len(columns)>64 or len(set(columns))!=len(columns):
                    raise ValueError('Invalid or excessive output columns')
                records=[];byte_count=0
                for _ in range(max_rows+1):
                    row=cursor.fetchone()
                    if row is None:break
                    byte_count+=sum(len(v.encode('utf-8')) if isinstance(v,str) else len(v) if isinstance(v,bytes) else 8 for v in row)
                    if byte_count>max_bytes:raise ValueError('Public SQLite result byte limit exceeded')
                    records.append(row)
                frame = pd.DataFrame.from_records(records[:max_rows], columns=columns)
                if frame.memory_usage(deep=True).sum()>max_bytes:
                    raise ValueError('Public SQLite result byte limit exceeded')
                conditions = raw_conditions(tree)
                info = datasets.register(frame, source=' | '.join(plan.actual_tables), query=query,
                    grain=plan.grain, aggregation=tree.sql() if plan.grain=='aggregate' else '',
                    conditions=conditions or (), predicate_known=conditions is not None,
                    coverage=query_coverage(tree,truncated=len(records)>max_rows),
                    snapshot=datetime.now(timezone.utc).isoformat())
                entry.update(status='completed', rows=info.rows, coverage=info.coverage, dataset_id=info.id)
                return {'status':'ready','dataset':asdict(info),'preview':frame.head(10).to_dict('records')}
        except Exception as exc:
            entry.update(status='failed', error_type=type(exc).__name__)
            raise
        finally:
            entry['elapsed_seconds']=round(time.monotonic()-started,3)
    return execute


def evaluate_interactive(spider_root, task, model, predictions, *, intent_mode='llm'):
    from scripts.evaluate_spider2_teleai import database_path, schema_context, task_document, check_sqlite_candidate
    from core.analysis_agent.runtime import GraphAnalysisRuntime
    from core.analysis_agent.policy import RuntimePolicy
    from langchain_core.messages import AIMessage, ToolMessage
    case_id=task['instance_id']
    result={'id':case_id,'db':task['db'],'benchmark':'Spider2-Lite SQLite',
        'mode':'Interactive production graph with bounded read-only public SQLite execution; no gold access',
        'remote_executions':0,'local_sql_executions':[],'local_sql_probes':[]}
    started=time.monotonic()
    try:
        path=database_path(spider_root,task)
        contexts=schema_context(path)
        document=task_document(spider_root,task)
        def preflight(query):
            checked=check_sqlite_candidate(path,query)
            result['local_sql_probes'].append({'query':query,**checked})
            return checked
        with tempfile.TemporaryDirectory(prefix='teleai-interactive-spider-') as root:
            runtime=GraphAnalysisRuntime(root,'evaluation',case_id,model,
                policy=RuntimePolicy(require_remote_approval=False),
                connection_identity='public-spider2-readonly-sqlite',
                remote_factory=lambda store:sqlite_executor(path,store,result['local_sql_executions']),
                reference_context_loader=lambda:contexts,reference_document=document,
                sql_dialect='sqlite',proposal_validator=preflight,
                tool_allowlist={'inspect_table_context','inspect_table_relationships','query_databricks'},
                agent_instructions=INSTRUCTIONS,intent_mode=intent_mode)
            try:
                outcome=runtime.submit(task['question'])
                recovery=runtime.inspect().get('recovery') or {}
                measured=dict(recovery)
                runtime.model_attempts.sync(measured)
                events=runtime.events()
                result.update(agent_status=outcome.get('status'),final_output=outcome.get('text',''),
                    model_calls=measured.get('model_calls'),model_retries=measured.get('model_retries'),recovery_status=recovery.get('status'),
                    scope_error=recovery.get('scope_error'),stop_reason=recovery.get('stop_reason'),error_type=outcome.get('error_type'),
                    request_contract={key:recovery.get(key) for key in (
                        'required_sources','required_columns','operations','scope','operation_pending',
                        'scalar_grouping','profile_kind','whole_row_count','calculation')},
                    error_category=outcome.get('error_category'),
                    diagnostics=[json.loads(line) for line in runtime.diagnostics.path.read_text().splitlines()],
                    sql_drafts=[call['args'] for message in events if isinstance(message,AIMessage)
                        for call in message.tool_calls if call['name']=='query_databricks'],
                    observations=[{'tool':m.name,'content':m.content} for m in events if isinstance(m,ToolMessage)])
                # Only SQL backing the graph's verified calculation evidence is final.
                # Neither the first exploration nor the last arbitrary tool call qualifies.
                ids=recovery.get('evidence_ids') or []
                queries={runtime.datasets.metadata[i].query for i in ids if i in runtime.datasets.metadata}
                queries.discard('')
                if (outcome.get('status')=='answered' and recovery.get('status')=='complete'
                        and len(queries)==1 and all(runtime.datasets.metadata[i].coverage=='complete' for i in ids)):
                    candidate=queries.pop()
                    if not any(c.get('status')=='completed' and c['query']==candidate for c in result['local_sql_executions']):
                        raise ValueError('Final calculation has no completed SQLite execution')
                    predictions.mkdir(parents=True,exist_ok=True)
                    dest=predictions/(case_id+'.sql')
                    dest.write_text('\n'.join(line.rstrip() for line in candidate.splitlines())+'\n')
                    result.update(status='SQL_COMPLETED',prediction=str(dest.resolve()))
                else:
                    result['status']=('BLOCKED_PROVIDER' if outcome.get('error_category') else 'NO_VERIFIED_FINAL_SQL')
            finally:
                runtime.close()
    except Exception as exc:
        result.update(status='FAIL',error_type=type(exc).__name__)
    result['elapsed_seconds']=round(time.monotonic()-started,3)
    return result
