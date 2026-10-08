"""Stage a single exact-query load; execute only with its explicitly approved ID.

The prepare command never executes SQL. Do not pass --approved-request until
this exact SQL has user approval. Repeating an executed request is forbidden.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.policy import RuntimePolicy
from core.analysis_agent.databricks import ConnectionConfig,make_executor
from scripts.evaluate_analysis_statistics import ForbiddenModel
from scripts.check_large_data_storage import peak_rss_bytes
from utils.analysis_datasets import stored_dataset_digest

SOURCE='workspace.default.ncr_ride'
QUERY='SELECT * FROM workspace.default.ncr_ride LIMIT 100000'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--approved-request',help='Existing request ID whose exact SQL the user explicitly approved')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env');os.environ['LANGSMITH_TRACING']='false';os.environ['LANGCHAIN_TRACING_V2']='false'
    config=ConnectionConfig.from_env();policy=RuntimePolicy()
    storage=ROOT/'.telly_runtime/scale-validation-20260927'
    r=GraphAnalysisRuntime(storage,'scale-validator','ncr-100k-once',ForbiddenModel(),
        connection_identity=config.identity(),remote_factory=lambda datasets:make_executor(config,datasets,max_rows=100000),policy=policy,intent_mode='contract_fixture')
    try:
        state=r.inspect();pending=state['requests']
        if a.approved_request:
            request=next((q for q in pending if q['id']==a.approved_request),None)
            if not request or request['query']!=QUERY or request['source']!=SOURCE:
                raise ValueError('No matching pending exact-query request; no query executed')
            started=time.monotonic();outcome=r.respond(request['id'],approved=True)
            state=r.inspect();selected=state.get('selected_dataset') or {}
            receipt=r.ledger.get(request['id'])
            dataset_id=(receipt.get('result') or {}).get('dataset',{}).get('id') or selected.get('id')
            info=r.datasets.metadata.get(dataset_id)
            record={'mode':'actual Databricks exact-query approved load','query':QUERY,
                'request_id':request['id'],'status':outcome['status'],'error_type':outcome.get('error_type'),
                'ingestion_status':'completed' if receipt['status']=='completed' and info else receipt['status'],
                'analysis_status':outcome['status'],
                'elapsed_seconds':round(time.monotonic()-started,3),'peak_rss_bytes':peak_rss_bytes(),
                'ledger_status':r.ledger.get(request['id'])['status'],
                'rows':info.rows if info else None,'columns':len(info.columns) if info else None,
                'coverage':info.coverage if info else None,'dataset_id':dataset_id,
                'raw_digest':stored_dataset_digest(r.datasets,dataset_id) if info else None,
                'storage':str(storage),'limitations':['LIMIT result; not full-table coverage.',
                    'Isolated validation conversation; existing user conversation is untouched.',
                    'Peak RSS is not wire transfer byte count.']}
            if info:
                record['file_bytes']=r.db.dataset_file(info.id).stat().st_size if r.db.dataset_file(info.id) else None
                record['runtime_schema']=r.datasets.inspect(info.id).get('dtypes',{})
        elif pending:
            record={'status':'awaiting_approval','query':QUERY,'request_id':pending[0]['id'],'warehouse_sql_executions':0}
        elif state['dataset_ids'] or state.get('uncertain_executions'):
            raise ValueError('This validation scope already has data or uncertain execution; no automatic repeat')
        else:
            # No payload/schema rows are queried until the durable grant exists.
            outcome=r.propose_query(SOURCE,QUERY,'대규모 적재 검증: 최대 100,000행을 별도 검증 대화에 저장합니다.')
            if outcome['status']!='awaiting_approval':raise RuntimeError('Expected approval checkpoint')
            record={'status':outcome['status'],'query':QUERY,'request_id':outcome['requests'][0]['id'],'warehouse_sql_executions':0}
        record['generated_at']=datetime.now(timezone.utc).isoformat()
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({k:record[k] for k in ('status','request_id','rows','ledger_status') if k in record}))
    finally:r.close()
if __name__=='__main__':main()
