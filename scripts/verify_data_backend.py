"""Bounded live adapter journey; no LLM scoring and no raw rows in the report."""
import argparse
import json
import logging
from pathlib import Path
import sys
import time
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


class VerificationFailure(Exception):
    def __init__(self, phase, checks, runtime=None):
        self.phase=phase
        self.checks=checks
        self.run_id=runtime.diagnostics.run_id if runtime else None
        self.stop_reason=(runtime.inspect().get('recovery') or {}).get('stop_reason') if runtime else None
        self.error_id=runtime.diagnostics.last_error_id if runtime else None
        super().__init__(phase)


def verify(backend, *, cold_start=False):
    import pandas as pd
    from langchain_core.language_models.chat_models import BaseChatModel
    from core.analysis_agent.runtime import GraphAnalysisRuntime
    from core.analysis_agent.policy import RuntimePolicy
    from core.analysis_catalog import _quoted_table
    from utils.analysis_datasets import stored_dataset_digest

    class NoInference(BaseChatModel):
        @property
        def _llm_type(self):return 'live-backend-deterministic-verification'
        def bind_tools(self, tools, **kwargs):return self
        def _generate(self, *args, **kwargs):
            raise AssertionError('This verification requires grounded deterministic tools, not model-generated evidence')

    checks=[];queries=[];config=backend.config
    if backend.name=='databricks':
        if not config.catalog or not config.schema:
            raise ValueError('DATABRICKS_CATALOG/DATABRICKS_SCHEMA required for the bounded live journey')
        source=config.catalog+'.information_schema.tables'
        query=(f'SELECT table_catalog, table_schema, table_name, table_type FROM {_quoted_table(source)} '
            + "WHERE table_schema = '"+config.schema.replace("'","''")+"' ORDER BY table_name LIMIT 100")
    else:
        source='information_schema.tables'
        query=("SELECT table_schema AS table_schema, table_name AS table_name, table_type AS table_type "
            "FROM information_schema.tables WHERE table_schema = '"+config.database.replace("'","''")
            + "' ORDER BY table_name LIMIT 100")
    # Keep failed checkpoints/diagnostics local so a failure can be traced.
    root=ROOT/'.telly_runtime/backend-verification'/uuid4().hex
    def factory(datasets):
        execute=backend.executor_factory(config,datasets,max_rows=1000)
        def recorded(envelope):
            queries.append(envelope['query'])
            return execute(envelope)
        return recorded
    runtime=GraphAnalysisRuntime(root,'backend-verifier','journey',NoInference(),
        connection_identity=config.identity(),remote_factory=factory,sql_dialect=backend.dialect,
        reference_context_loader=(lambda:[]) if cold_start else backend.context_loader,summary_trigger_tokens=100000,
        policy=RuntimePolicy(max_remote_rows=1000),source_namespace=backend.namespace,intent_mode='contract_fixture')
    try:
        def require(result, phase):
            if result.get('status')!='answered':
                raise VerificationFailure(phase,checks,runtime)
            checks.append({'phase':phase,'status':'PASS','elapsed_seconds':result.get('elapsed_seconds')})
        listing=runtime.propose_query(source,query,'설정한 namespace의 실제 테이블 목록을 확인합니다.')
        require(listing,'table_list')
        listed=next(info for info in runtime.datasets.metadata.values() if info.query==query)
        names=runtime.datasets.frames[listed.id]
        if names.empty:raise VerificationFailure('no_accessible_table',checks,runtime)
        name=str(names.iloc[0]['table_name'])
        table=(config.catalog+'.'+config.schema+'.'+name if backend.name=='databricks'
               else config.database+'.'+name)
        if cold_start:
            require(runtime.submit(table+' 컬럼들을 보여줘'),'cold_start_schema_discovery')
            if len(queries)!=2 or not queries[-1].endswith(' LIMIT 0'):
                raise VerificationFailure('unexpected_schema_query',checks,runtime)
        else:
            require(runtime.propose_query(table,'SELECT * FROM '+_quoted_table(table)+' LIMIT 0',
                '현재 컬럼명과 자료형만 확인합니다. 원본 행은 로딩하지 않습니다.'),'zero_row_schema')
        require(runtime.submit(table+' 컬럼들을 보여줘'),'columns_followup')
        schema=runtime.inspect()['recovery'].get('metadata_evidence') or {}
        require(runtime.propose_query(table,'SELECT * FROM '+_quoted_table(table)+' LIMIT 10',
            '접속 및 원본 보존 검증에 필요한 최대 10행만 가져옵니다.'),'bounded_raw_load')
        raw_id=runtime.context.selected_dataset_id
        raw=runtime.datasets.metadata[raw_id]
        before=stored_dataset_digest(runtime.datasets,raw_id)
        frame=runtime.datasets.frames[raw_id]
        numeric=[c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c].dtype)]
        if not numeric:raise VerificationFailure('no_numeric_column_in_first_table',checks,runtime)
        # Prefer a small observed cardinality; never send sampled values to a model.
        column=min(numeric,key=lambda c:frame[c].nunique())
        require(runtime.submit(table+' '+column+' histogram을 보여줘'),'remote_histogram')
        proof=runtime.inspect()['recovery'];card=runtime.artifacts[proof['artifact_ids'][0]]
        if not card.image.startswith(b'\x89PNG\r\n\x1a\n'):raise AssertionError('Missing PNG')
        query_count=len(queries)
        require(runtime.submit(column+' histogram을 다시 보여줘'),'histogram_reuse')
        if len(queries)!=query_count:raise AssertionError('Histogram repeat resubmitted SQL')
        if runtime.context.selected_dataset_id!=raw_id or stored_dataset_digest(runtime.datasets,raw_id)!=before:
            raise AssertionError('Protected raw selection or digest changed')
        checks.append({'phase':'raw_preservation_and_no_requery','status':'PASS'})
        return {'status':'PASS','backend':backend.name,'scope':'live_adapter_and_deterministic_journey_not_llm_quality',
            'cold_start':cold_start,
            'checks':checks,'sql_executions':len(queries),'schema_columns':len(schema.get('columns',[])),
            'raw_rows':raw.rows,'png_bytes':len(card.image)}
    finally:runtime.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=('databricks','mysql'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cold-start',action='store_true',help='Verify discovery without saved table profiles')
    args=parser.parse_args()
    from dotenv import load_dotenv
    from core.analysis_agent.backends import load_data_backend
    load_dotenv(ROOT/'.env')
    # Silence third-party console logs while retaining our scoped diagnostics.
    logging.getLogger().handlers[:]=[logging.NullHandler()]
    started=time.monotonic()
    try:
        result=verify(load_data_backend(ROOT,name=args.backend),cold_start=args.cold_start)
    except Exception as error:
        context=getattr(error,'context',{})
        result={'status':'BLOCKED','backend':args.backend,'error_type':type(error).__name__,
            'http_status':context.get('http-code') if isinstance(context,dict) else None,
            'scope':'live_adapter_and_deterministic_journey_not_llm_quality'}
        if isinstance(error,VerificationFailure):
            result.update(phase=error.phase,checks=error.checks,run_id=error.run_id,
                          stop_reason=error.stop_reason,error_id=error.error_id)
    result['elapsed_seconds']=round(time.monotonic()-started,3)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
