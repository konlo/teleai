"""Real-model join discovery -> approval -> SQLite aggregate -> verified answer.

Only the public synthetic fixture is queried. The model provider may be remote;
the query_databricks tool name is the existing approval gateway, not a warehouse
connection in this harness. No gold result is supplied to the model.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage
from core.analysis_agent.model_provider import build_analysis_chat_model
from core.analysis_agent.policy import RuntimePolicy
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_sql import validate_query
from scripts.evaluate_spider2_teleai import schema_context
from utils.analysis_datasets import stored_dataset_digest


def evaluate(provider):
    fixture = json.loads((ROOT/'tests/fixtures/relationship_journey.json').read_text())
    started=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='teleai-relations-') as root:
        path=Path(root)/'public.sqlite'
        with sqlite3.connect(path) as db:
            db.executescript(fixture['ddl'])
            for table, data in fixture['tables'].items():
                pd.DataFrame(data['rows'],columns=data['columns']).to_sql(table,db,if_exists='append',index=False)
        references=schema_context(path);executions=[]
        def factory(datasets):
            def execute(request):
                validate_query(request['query'],dialect='sqlite')
                with sqlite3.connect(f'file:{path}?mode=ro',uri=True) as db:
                    db.execute('PRAGMA query_only=ON')
                    cursor=db.execute(request['query']);rows=cursor.fetchmany(1001)
                    if len(rows)>1000:raise ValueError('Public fixture result cap exceeded')
                    frame=pd.DataFrame(rows,columns=[column[0] for column in cursor.description])
                executions.append(request['query'])
                info=datasets.register(frame,source=request['source'],query=request['query'],
                    grain='aggregate',coverage='complete',predicate_known=True,
                    snapshot=datetime.now(timezone.utc).isoformat())
                return {'status':'ready','dataset':asdict(info)}
            return execute
        model=build_analysis_chat_model(RuntimePolicy(),provider=provider)
        runtime=GraphAnalysisRuntime(root,'evaluation','relationship-journey',model,
            remote_factory=factory,connection_identity='synthetic-sqlite-only',
            reference_context_loader=lambda:references,sql_dialect='sqlite')
        try:
            source=fixture['protected_source'];data=fixture['tables'][source]
            raw=runtime.datasets.register(pd.DataFrame(data['rows'],columns=data['columns']),
                source=source,coverage='complete',predicate_known=True)
            runtime.select_dataset(raw.id);digest=stored_dataset_digest(runtime.datasets,raw.id)
            staged=runtime.submit(fixture['prompt']);before=len(executions)
            result=staged
            if staged['status']=='awaiting_approval' and len(staged['requests'])==1:
                result=runtime.respond(staged['requests'][0]['id'],approved=True)
            state=runtime.inspect()['recovery'];actual=None
            if state.get('evidence_ids'):
                frame=runtime.datasets.frames[state['evidence_ids'][-1]]
                if frame.shape==(1,1):actual=float(frame.iloc[0,0])
            preserved=stored_dataset_digest(runtime.datasets,raw.id)==digest and runtime.context.selected_dataset_id==raw.id
            passed=(staged['status']=='awaiting_approval' and before==0 and len(executions)==1
                and result['status']=='answered' and actual==fixture['expected'] and preserved
                and state.get('join_relationship_basis')=='fresh_database_catalog')
            return {'status':'PASS' if passed else 'FAIL','provider':provider,
                'mode':'real model with synthetic public SQLite and explicit local-test approval',
                'prompt':fixture['prompt'],'agent_status':result['status'],
                'error_type':result.get('error_type'),
                'stop_reason':state.get('stop_reason'),'expected':fixture['expected'],'actual':actual,
                'model_calls':state.get('model_calls'),'executions_before_approval':before,
                'public_sqlite_executions':len(executions),'databricks_sql_executions':0,
                'queries':executions,'raw_preserved':preserved,'final_output':result.get('text',''),
                'tools':[m.name for m in runtime.events() if isinstance(m,ToolMessage)],
                'tool_observations':[{'tool':m.name,'content':str(m.content)[:2000]}
                    for m in runtime.events() if isinstance(m,ToolMessage)],
                'model_actions':[{'calls':m.tool_calls,'text':str(m.content)[:2000]}
                    for m in runtime.events() if isinstance(m,AIMessage)],
                'scope':state.get('scope'),
                'intent':{key:state.get(key) for key in ('join','calculation','chart','data_load','required_sources','operations')},
                'focus_events':[json.loads(line) for line in (runtime.db.directory/'runtime.jsonl').read_text().splitlines()
                    if 'model_tools_focused' in line],
                'errors':[json.loads(line) for line in (runtime.db.directory/'runtime.jsonl').read_text().splitlines()
                    if '"event": "error"' in line],
                'elapsed_seconds':round(time.monotonic()-started,3)}
        finally:runtime.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider',choices=['databricks','ollama'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();load_dotenv(ROOT/'.env')
    os.environ['LANGSMITH_TRACING']='false';os.environ['LANGCHAIN_TRACING_V2']='false'
    result=evaluate(args.provider)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('status','agent_status','model_calls','elapsed_seconds')}))
    raise SystemExit(0 if result['status']=='PASS' else 1)
