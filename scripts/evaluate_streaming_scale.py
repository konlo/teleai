"""Synthetic million-row transport through production ingestion, then graph EDA.

No network is connected. This measures transport/staging/EDA contracts and
process peak RSS, not actual Databricks network throughput or model reasoning.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.policy import RuntimePolicy
from core.analysis_databricks import execute_approved
from scripts.check_large_data_storage import synthetic_frame,peak_rss_bytes
from scripts.evaluate_analysis_statistics import ForbiddenModel
from utils.analysis_datasets import stored_dataset_digest


class SyntheticCursor:
    def __init__(self, fixture, rows, failure_at=None):
        self.fixture,self.rows,self.failure_at=fixture,rows,failure_at
        self.description=[(name,) for name in fixture['columns']]
        self.position=0;self.calls=0;self.max_batch=0;self.executions=0
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def execute(self,query):self.executions+=1
    def fetchmany(self,size):
        self.calls+=1;self.max_batch=max(self.max_batch,size)
        if self.failure_at is not None and self.position>=self.failure_at:
            raise ConnectionError('synthetic transfer interrupted')
        if self.position>=self.rows:return []
        # Generate only the next bounded chunk, preserving global sequence.
        count=min(size,self.rows-self.position)
        import numpy as np
        indexes=np.arange(self.position,self.position+count,dtype='int64')
        data=[]
        for definition in self.fixture['columns'].values():
            kind=definition['kind']
            if kind=='sequence':data.append(indexes)
            elif kind=='numeric_cycle':data.append((indexes%definition['modulus'])/definition['divisor'])
            elif kind=='category':data.append(np.asarray(definition['values'],dtype=object)[indexes%len(definition['values'])])
            else:raise ValueError(kind)
        self.position+=count
        return list(zip(*data))
    def cursor(self):return self


def evaluate(rows):
    fixture=json.loads((ROOT/'tests/fixtures/large_data_workload.json').read_text())
    # One cycle is an independent, bounded mathematical oracle for repeated data.
    assert rows%1000==0, 'Use a whole number of fixture cycles'
    oracle=synthetic_frame(fixture,1000)['measurement']
    config=SimpleNamespace(server_hostname='synthetic.invalid',http_path='',access_token='',catalog='',schema='')
    request=SimpleNamespace(status='executing',source=fixture['source'],query=f"SELECT * FROM {fixture['source']} LIMIT {rows}")
    started=time.monotonic();baseline=peak_rss_bytes()
    report={'created_at':datetime.now(timezone.utc).isoformat(),'mode':'synthetic cursor, production ingestion and graph',
        'rows_requested':rows,'network_calls':0,'model_calls':0,'baseline_peak_rss_bytes':baseline,
        'limitations':['No actual warehouse/network, provider model or UI is used.',
            'Five-column cyclic fixture is not a universal data-width or performance guarantee.',
            'Peak RSS includes interpreter/imports; cache cap is not an RSS ceiling.',
            'Test runtime row policy is set to the synthetic workload size; product defaults are unchanged.']}
    with tempfile.TemporaryDirectory(prefix='teleai-streaming-scale-') as root:
        policy=RuntimePolicy(max_remote_rows=rows)
        r=GraphAnalysisRuntime(root,'evaluation','streaming',ForbiddenModel(),policy=policy)
        try:
            cursor=SyntheticCursor(fixture,rows)
            before=time.monotonic();result=execute_approved(request,config,r.datasets,max_rows=rows,connect=lambda **_:cursor)
            raw_id=result['dataset']['id'];r.select_dataset(raw_id)
            digest=stored_dataset_digest(r.datasets,raw_id)
            report['ingestion']={'elapsed_seconds':round(time.monotonic()-before,3),
                'peak_rss_bytes':peak_rss_bytes(),'rows':result['dataset']['rows'],
                'coverage':result['dataset']['coverage'],'fetch_calls':cursor.calls,'max_batch':cursor.max_batch,
                'executions':cursor.executions,'file_backed':r.db.dataset_file(raw_id) is not None,
                'file_bytes':r.db.dataset_file(raw_id).stat().st_size,'cache_bytes':r.datasets.frames.bytes}
            cases=[('현재 로딩된 표본에서 measurement 평균을 알려줘',float(oracle.mean()),False),
                ('현재 로딩된 표본에서 measurement >= 90인 행의 measurement 평균을 알려줘',float(oracle[oracle>=90].mean()),False),
                ('현재 로딩된 원본 표본에서 measurement 평균과 measurement 히스토그램을 보여줘. 이전 measurement 조건은 적용하지 말고 보유 데이터만 사용해.',float(oracle.mean()),True)]
            turns=[]
            for prompt,expected,chart in cases:
                outcome=r.submit(prompt);state=r.inspect()['recovery'];ids=state.get('evidence_ids',[])
                actual=float(r.datasets.frames[ids[-1]].iloc[0,0]) if ids else None
                png=bool(state.get('artifact_ids')) and all(r.artifacts[c].image.startswith(b'\x89PNG') for c in state['artifact_ids'])
                ok=outcome['status']=='answered' and actual is not None and abs(actual-expected)<1e-9 and (not chart or png)
                turns.append({'status':'PASS' if ok else 'FAIL','prompt':prompt,'actual':actual,'expected':expected,
                    'elapsed_seconds':outcome.get('elapsed_seconds'),'peak_rss_bytes':peak_rss_bytes(),
                    'model_calls':state.get('model_calls'),'chart_count':len(state.get('artifact_ids',[])),
                    'raw_preserved':stored_dataset_digest(r.datasets,raw_id)==digest,'stop_reason':state.get('stop_reason')})
            report['eda']=turns
            # A transport failure after dozens of batches must not publish a candidate.
            ids_before=set(r.datasets.metadata);selected=r.context.selected_dataset_id
            broken=SyntheticCursor(fixture,rows,failure_at=50_000);failure=None
            try:execute_approved(request,config,r.datasets,max_rows=rows,connect=lambda **_:broken)
            except ConnectionError as exc:failure=type(exc).__name__
            report['interrupted_transfer']={'error_type':failure,'received_rows':broken.position,
                'no_partial_publication':set(r.datasets.metadata)==ids_before,
                'selected_preserved':r.db.selected_dataset_id()==selected,
                'raw_preserved':stored_dataset_digest(r.datasets,raw_id)==digest,
                'staging_clean':not list(r.db.directory.glob('*.staging.parquet'))}
            previous_limit=r.datasets.max_frame_bytes
            limited=SyntheticCursor(fixture,rows);limit_error=None
            try:
                r.datasets.max_frame_bytes=128*1024
                execute_approved(request,config,r.datasets,max_rows=rows,connect=lambda **_:limited)
            except MemoryError as exc:limit_error=type(exc).__name__
            finally:r.datasets.max_frame_bytes=previous_limit
            report['byte_limit']={'limit_bytes':128*1024,'error_type':limit_error,
                'received_rows':limited.position,'no_partial_publication':set(r.datasets.metadata)==ids_before,
                'raw_preserved':stored_dataset_digest(r.datasets,raw_id)==digest,
                'selected_preserved':r.db.selected_dataset_id()==selected,
                'staging_clean':not list(r.db.directory.glob('*.staging.parquet'))}
            r.close();r=GraphAnalysisRuntime(root,'evaluation','streaming',ForbiddenModel(),policy=policy)
            report['reopen']={'raw_preserved':stored_dataset_digest(r.datasets,raw_id)==digest,
                'rows':r.datasets.metadata[raw_id].rows,'selected_preserved':r.context.selected_dataset_id==selected}
            checks=[report['ingestion']['rows']==rows,report['ingestion']['max_batch']<=1024,
                report['ingestion']['file_backed'],all(t['status']=='PASS' and t['raw_preserved'] and t['model_calls']==0 for t in turns),
                failure=='ConnectionError',all(v is True for k,v in report['interrupted_transfer'].items() if k not in {'error_type','received_rows'}),
                report['reopen']['raw_preserved'],report['reopen']['selected_preserved'],
                limit_error=='MemoryError',all(v is True for k,v in report['byte_limit'].items()
                    if k not in {'error_type','limit_bytes','received_rows'})]
            report['status']='PASS' if all(checks) else 'FAIL'
        finally:r.close()
    report['elapsed_seconds']=round(time.monotonic()-started,3);report['peak_rss_bytes']=peak_rss_bytes()
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rows',type=int,default=1_000_000);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=evaluate(a.rows);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'status':result['status'],'peak_rss_bytes':result['peak_rss_bytes']}))
    return 0 if result['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
