"""Resume a completed load and verify EDA using its stored data; no remote SQL.

Schema and numeric columns come from the actual persisted dataset. The Pandas
oracle is independent of the graph's DuckDB/chart tools. Never issue a reload.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from core.analysis_agent.runtime import GraphAnalysisRuntime
from scripts.evaluate_analysis_statistics import ForbiddenModel
from scripts.check_large_data_storage import peak_rss_bytes
from utils.analysis_datasets import stored_dataset_digest,project_dataset


def evaluate(storage,owner,conversation,request_id,missing_values):
    import pandas as pd
    started=time.monotonic();remote=[]
    def factory(_):
        def forbidden(envelope):
            remote.append(1)
            raise AssertionError('Re-execution forbidden; use persisted receipt')
        return forbidden
    def runtime():return GraphAnalysisRuntime(storage,owner,conversation,ForbiddenModel(),
                         connection_identity='stored-verification-no-network',remote_factory=factory)
    r=runtime()
    report={'generated_at':datetime.now(timezone.utc).isoformat(),
            'mode':'actual stored Databricks data, production graph, independent Pandas oracle',
            'limitations':['No actual LLM inference in this EDA test.',
                           'LIMIT result is a sample, not the complete table.',
                           'Only numerical summaries, histogram, filtering/reset and restart are evaluated.']}
    try:
        receipt=r.ledger.get(request_id)
        if receipt['status']!='completed':raise ValueError('Requires a completed durable receipt')
        raw_id=receipt['result']['dataset']['id'];info=r.datasets.metadata[raw_id]
        digest=stored_dataset_digest(r.datasets,raw_id)
        report.update(raw_id=raw_id,rows=info.rows,columns=len(info.columns),query=info.query,
                      coverage=info.coverage,raw_digest=digest)
        if r.agent.get_state(r.config).next:
            recovered=r.resume()
            report['receipt_recovery']={k:recovered.get(k) for k in ('status','elapsed_seconds','error_type')}
            if recovered['status']!='answered':raise AssertionError('Receipt recovery incomplete')
        if r.context.selected_dataset_id!=raw_id:r.select_dataset(raw_id)
        schema=r.datasets.inspect(raw_id).get('dtypes',{})
        report['runtime_schema']=schema
        # Column choice uses observed data types, never a familiar table name.
        column=next((c for c in info.columns if pd.api.types.is_numeric_dtype(schema.get(c,''))),None)
        analysis_id=raw_id
        if column is None:
            for candidate in info.columns:
                if schema.get(candidate)!='object':continue
                series=project_dataset(r.datasets,raw_id,[candidate])[candidate]
                values=pd.to_numeric(series.mask(series.isin(missing_values)),errors='coerce')
                if values.notna().any() and (values.notna() | series.isna() | series.isin(missing_values)).all():
                    column=candidate;break
            if column is None:raise ValueError('No numeric column under the declared missing-value policy')
            from core.analysis_agent.tools import local_tools
            tool=next(t for t in local_tools(r.context) if t.name=='prepare_numeric_dataset')
            prepared=tool.invoke({'dataset_id':raw_id,'columns':[column],'missing_values':missing_values})
            if prepared['status']!='ready':raise ValueError('Numeric preparation was rejected')
            analysis_id=prepared['dataset']['id'];r.select_dataset(analysis_id)
            report['preparation']={'mode':'controller-selected tool, not autonomous LLM planning',
                'column':column,'missing_values':missing_values,'conversion':prepared['conversion'],
                'rows_preserved':prepared['dataset']['rows']==info.rows,'dataset_id':analysis_id}
        values=pd.to_numeric(project_dataset(r.datasets,analysis_id,[column])[column],errors='raise')
        threshold=float(values.median());mean=float(values.mean());filtered=float(values[values>=threshold].mean())
        cases=[(f'현재 로딩된 표본에서 `{column}` 평균을 알려줘',mean,False),
               (f'현재 로딩된 표본에서 `{column}` >= {threshold}인 행의 `{column}` 평균을 알려줘',filtered,False),
               (f'현재 로딩된 원본 표본에서 `{column}` 평균과 `{column}` 히스토그램을 보여줘. 이전 `{column}` 조건은 적용하지 말고 보유 데이터만 사용해.',mean,True)]
        turns=[]
        for prompt,expected,chart in cases:
            outcome=r.submit(prompt);state=r.inspect()['recovery'];ids=state.get('evidence_ids',[])
            actual=float(r.datasets.frames[ids[-1]].iloc[0,0]) if ids and outcome['status']=='answered' else None
            cards=state.get('artifact_ids',[])
            png=bool(cards) and all(r.artifacts[c].image.startswith(b'\x89PNG') for c in cards)
            ok=outcome['status']=='answered' and actual is not None and abs(actual-expected)<1e-8 and (not chart or png)
            turns.append({'status':'PASS' if ok else 'FAIL','prompt':prompt,'actual':actual,'expected':expected,
                          'model_calls':state.get('model_calls'), 'chart_count':len(cards),
                          'raw_preserved':stored_dataset_digest(r.datasets,raw_id)==digest,
                          'elapsed_seconds':outcome.get('elapsed_seconds'),'stop_reason':state.get('stop_reason')})
            if chart and png:
                report['_image']=r.artifacts[cards[-1]].image
            if r.agent.get_state(r.config).next:break
        report['eda']=turns
        r.close();r=runtime()
        report['reopen']={'raw_preserved':stored_dataset_digest(r.datasets,raw_id)==digest,
                          'selected_preserved':r.context.selected_dataset_id==analysis_id,
                          'rows':r.datasets.metadata[raw_id].rows}
        report['remote_reexecutions']=len(remote)
        report['status']='PASS' if (len(turns)==3 and all(t['status']=='PASS' and t['raw_preserved'] and t['model_calls']==0 for t in turns)
             and report['reopen']['raw_preserved'] and report['reopen']['selected_preserved'] and not remote) else 'FAIL'
    finally:r.close()
    report['elapsed_seconds']=round(time.monotonic()-started,3);report['peak_rss_bytes']=peak_rss_bytes()
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--storage',type=Path,required=True);p.add_argument('--owner',required=True)
    p.add_argument('--conversation',required=True);p.add_argument('--request-id',required=True)
    p.add_argument('--missing-value',action='append',default=[],help='Explicit validation policy for numeric conversion; never guessed')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=evaluate(a.storage,a.owner,a.conversation,a.request_id,a.missing_value)
    image=report.pop('_image',None)
    if image:
        # Actual aggregate plot, without source rows or identifiers.
        image_path=a.output.with_suffix('.png');image_path.write_bytes(image)
        report['histogram_file']=str(image_path)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','rows','remote_reexecutions','peak_rss_bytes')}))
    return 0 if report['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
