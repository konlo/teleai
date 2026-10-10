"""Capture this browser fixture from persisted state, without rerunning SQL/LLM."""
from pathlib import Path
import sys,json,sqlite3
import pandas as pd
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from core.analysis_agent.assets import AssetDB,PersistentDatasets,PersistentCharts
from core.analysis_agent.execution_plan import ExecutionPlans
from core.analysis_agent.goal_grounding import bound_to_current
from langgraph.checkpoint.sqlite import SqliteSaver
from utils.analysis_datasets import stored_dataset_digest
from utils.analysis_image_validation import validate_chart_image

ROOT=Path(__file__).resolve().parents[3]
cid,stage=sys.argv[1:3]
db=AssetDB(ROOT/'.telly_runtime/v1/mysql_eval/44bc5ed80f4a4583ada9f905','local-owner',cid)
store=PersistentDatasets(db)
connection=sqlite3.connect(f'file:{db.directory}/graph.sqlite?mode=ro',uri=True,check_same_thread=False)
state=SqliteSaver(connection).get_tuple({'configurable':{'thread_id':'conversation'}}).checkpoint['channel_values']
current=state.get('recovery',{})
records=[json.loads(s) for s in (db.directory/'runtime.jsonl').read_text().splitlines()]
runs=[e for e in records if e.get('event')=='run_completed']
last=runs[-1];events=[e for e in records if e.get('run_id')==last['run_id']]
roots=[i for i in store.metadata.values() if not i.parent_id and i.grain=='raw']
raw=roots[0] if roots else None
frame=store.frames[raw.id] if raw else pd.DataFrame()
report={'conversation':cid,'stage':stage,'run':last,'goal':current.get('goal'),
    'scope':current.get('scope'),
    'model_calls':current.get('model_calls'),'sent_calls':current.get('sent_calls'),
    'decision_support_bound':bound_to_current(current),
    'decision_support':current.get('goal_decision_evidence'),
    'plan':ExecutionPlans(db).inspect(current.get('request_id','')),
    'datasets':[{'id':i.id,'source':i.source,'rows':i.rows,'columns':list(i.columns),
                'parent_id':i.parent_id,'digest':stored_dataset_digest(store,i.id)} for i in store.metadata.values()],
    'selected_dataset':db.selected_dataset_id(),
    'additional_queries_this_run':sum(e.get('event')=='remote_query_started' for e in events),
    'total_queries':sum(e.get('event')=='remote_query_started' for e in records),
    'tool_calls':sum(e.get('event')=='tool_started' for e in events),'diagnostics':events}
oracle={}
if raw:
    oracle={'raw_rows':len(frame),'raw_columns':list(frame.columns),
        'row_scope_matches':bool(frame.age.between(30,40).all()
                                and frame.education.isin(['primary','secondary']).all()),
        'expected_balance_mean':float(frame.balance.mean()),
        'expected_education_frequencies':frame.education.value_counts().to_dict()}
if stage=='01':
    oracle['passed']=bool(last['status']=='answered' and raw and len(frame)==10
        and list(frame.columns)==['age','balance','education'] and oracle['row_scope_matches'])
elif stage=='02':
    values=[]
    for key in current.get('evidence_ids',[]):
        f=store.frames[key]
        if len(f)==1:values.extend(f.select_dtypes(include='number').to_numpy().ravel().tolist())
    oracle['observed_scalar_values']=values
    oracle['passed']=bool(last['status']=='answered' and
        any(np.isclose(v,oracle['expected_balance_mean']) for v in values) and report['additional_queries_this_run']==0)
elif stage=='03':
    charts=PersistentCharts(db);report['charts']=[]
    for key in current.get('artifact_ids',[]):
        card=charts[key];validate_chart_image(card.image)
        report['charts'].append({'id':key,'kind':card.kind,'render_spec':card.render_spec,'scope':card.scope,
                               'columns':list(card.columns),'dataset_id':card.dataset_id,'png_valid':True})
    def frequencies(spec):
        if isinstance(spec,dict):
            labels,counts=spec.get('labels'),spec.get('counts')
            if isinstance(labels,list) and isinstance(counts,list) and len(labels)==len(counts):
                yield dict(zip(labels,counts))
            for value in spec.values():yield from frequencies(value)
        elif isinstance(spec,list):
            for value in spec:yield from frequencies(value)
    oracle['chart_frequencies_match']=any(c['kind']=='bar' and c['columns']==['education']
        and any(counts==oracle['expected_education_frequencies'] for counts in frequencies(c['render_spec']))
        for c in report['charts'])
    oracle['passed']=bool(last['status']=='answered' and oracle['chart_frequencies_match']
        and report['additional_queries_this_run']==0)
if stage!='01':
    first=json.loads(Path(__file__).with_name('01.json').read_text())
    baseline={d['id']:d['digest'] for d in first['datasets']}
    report['originals_preserved']=all(stored_dataset_digest(store,k)==v for k,v in baseline.items())
    oracle['passed']=oracle['passed'] and report['originals_preserved']
report['oracle']=oracle
report['passed']=oracle.get('passed',False) and report['decision_support_bound']
Path(__file__).with_name(stage+'.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str)+'\n')
print(json.dumps({k:report[k] for k in ('stage','passed','model_calls','tool_calls','additional_queries_this_run')},ensure_ascii=False))
connection.close();db.close()
