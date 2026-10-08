"""Live model numeric-string EDA on an isolated copy of retained warehouse rows.

No new warehouse SQL. The actual model must discover tools and prepare numeric
values; no derived dataset or historical conversation is supplied to it.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.model_provider import build_analysis_chat_model
from core.analysis_agent.policy import RuntimePolicy
from scripts.evaluate_analysis_statistics import ForbiddenModel
from utils.analysis_datasets import project_dataset,stored_dataset_digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--storage',type=Path,required=True);p.add_argument('--owner',required=True)
    p.add_argument('--conversation',required=True);p.add_argument('--dataset-id',required=True)
    p.add_argument('--column',required=True);p.add_argument('--missing-value',action='append',default=[])
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from dotenv import load_dotenv
    import pandas as pd
    import pyarrow.parquet as pq
    from langchain_core.messages import AIMessage,ToolMessage
    load_dotenv(ROOT/'.env');os.environ['LANGSMITH_TRACING']='false';os.environ['LANGCHAIN_TRACING_V2']='false'
    model=build_analysis_chat_model(RuntimePolicy(),provider='databricks',environ=dict(os.environ))
    source=GraphAnalysisRuntime(a.storage,a.owner,a.conversation,ForbiddenModel(),intent_mode='contract_fixture')
    try:
        info=source.datasets.metadata[a.dataset_id];before=stored_dataset_digest(source.datasets,info.id)
        values=project_dataset(source.datasets,info.id,[a.column])[a.column]
        expected=float(pd.to_numeric(values.mask(values.isin(a.missing_value)),errors='raise').mean())
        with tempfile.TemporaryDirectory(prefix='teleai-live-numeric-') as root:
            r=GraphAnalysisRuntime(root,'evaluation','numeric-repair',model)
            try:
                parquet=pq.ParquetFile(source.db.dataset_file(info.id))
                copied=r.datasets.register_batches((batch.to_pandas() for batch in parquet.iter_batches(batch_size=1024)),
                    columns=info.columns,source=info.source,max_rows=info.rows,query=info.query,
                    coverage=info.coverage,snapshot=info.snapshot,predicate_known=info.predicate_known,conditions=info.conditions)
                r.select_dataset(copied.id)
                copied_digest=stored_dataset_digest(r.datasets,copied.id)
                prompt=(f'현재 로딩된 표본에서 `{a.column}` 평균과 히스토그램을 보여줘. '
                        f'수치 변환 시 {json.dumps(a.missing_value,ensure_ascii=False)} 문자열만 결측값으로 처리하고 원본과 모든 행은 보존해.')
                started=time.monotonic();outcome=r.submit(prompt);state=r.inspect()['recovery']
                evidence=state.get('evidence_ids',[]);cards=state.get('artifact_ids',[])
                actual=float(r.datasets.frames[evidence[-1]].iloc[0,0]) if evidence else None
                png=bool(cards) and all(r.artifacts[c].image.startswith(b'\x89PNG') for c in cards)
                events=r.events();calls=[{'tool':c['name']} for m in events if isinstance(m,AIMessage) for c in m.tool_calls]
                observed=[]
                for m in events:
                    if isinstance(m,ToolMessage):
                        try:v=json.loads(m.content)
                        except (ValueError,TypeError):v={}
                        observed.append({'tool':m.name,'status':v.get('status'),'error_code':v.get('error_code')})
                report={'generated_at':datetime.now(timezone.utc).isoformat(),'mode':'actual Databricks model; isolated retained actual warehouse data',
                    'request':prompt,'rows':info.rows,'status':'PASS' if outcome['status']=='answered' and actual is not None and abs(actual-expected)<1e-8 and png else 'FAIL',
                    'agent_status':outcome['status'],'error_type':outcome.get('error_type'),'stop_reason':state.get('stop_reason'),
                    'actual':actual,'expected':expected,'chart_count':len(cards),'model_calls':state.get('model_calls'),
                    'elapsed_seconds':round(time.monotonic()-started,3),'tools':calls,'observations':observed,
                    'raw_preserved':stored_dataset_digest(r.datasets,copied.id)==copied_digest,
                    'original_raw_preserved':stored_dataset_digest(source.datasets,info.id)==before,
                    'warehouse_sql_executions':0,'answer':outcome.get('text',''),
                    'limitations':['One explicitly specified missing-value policy; ambiguity resolution is not tested.',
                        'Retained warehouse sample, not full population. No SQL or browser action in this run.']}
            finally:r.close()
    finally:source.close()
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','model_calls','chart_count','elapsed_seconds')}))
    return 0 if report['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
