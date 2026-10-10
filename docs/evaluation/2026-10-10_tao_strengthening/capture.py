"""Read this browser test session only; no model/SQL re-execution."""
from pathlib import Path
import sys,json,sqlite3,zipfile
from io import BytesIO
import pandas as pd,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from core.analysis_agent.assets import AssetDB,PersistentDatasets
from langgraph.checkpoint.sqlite import SqliteSaver
from utils.analysis_datasets import stored_dataset_digest
ROOT=Path(__file__).resolve().parents[3]
CID='2d20b7a7-a716-4330-b60b-a4345e13443e'
db=AssetDB(ROOT/'.telly_runtime/v1/mysql_eval/44bc5ed80f4a4583ada9f905','local-owner',CID)
store=PersistentDatasets(db)
conn=sqlite3.connect(f'file:{db.directory}/graph.sqlite?mode=ro',uri=True,check_same_thread=False)
values=SqliteSaver(conn).get_tuple({'configurable':{'thread_id':'conversation'}}).checkpoint['channel_values']
r=values.get('recovery',{})
report={'conversation':CID,'status':r.get('status'),'goal':r.get('goal'),
        'model_calls':r.get('model_calls'),'tool_calls':r.get('tool_calls'),'elapsed_model_seconds':r.get('model_seconds'),
        'selected_dataset_id':db.selected_dataset_id(),'dataset_count':len(store.metadata),
        'datasets':[{'id':i.id,'source':i.source,'rows':i.rows,'columns':list(i.columns),'coverage':i.coverage,
            'parent_id':i.parent_id,'digest':stored_dataset_digest(store,i.id)} for i in store.metadata.values()],
        'result_spec':r.get('custom_analysis_spec'),'execution_plan':None,
        'new_evidence':{k:r.get(k) for k in ('custom_analysis_evidence','advanced_eda_evidence','export_evidence','table_preview_evidence')}}
raw=next(i for i in store.metadata.values() if not i.parent_id)
frame=store.frames[raw.id]
report['original_preserved']=stored_dataset_digest(store,raw.id)=='06e8443c9fb358517faac21487988d5be8e970e0984b9fffcaf7849dc7e74a7e'
from core.analysis_agent.execution_plan import ExecutionPlans
report['execution_plan']=ExecutionPlans(db).inspect(r.get('request_id',''))
charts=db.metadata('chart')
report['charts']=[]
for key,card in charts.items():
 if card.get('kind')=='correlation_heatmap':
  expected=float(np.corrcoef(frame['age'],frame['balance'])[0,1])
  actual=card['render_spec']['summary']['correlation']['age']['balance']
  report['charts'].append({'id':key,'expected_correlation':expected,'actual_correlation':actual,'oracle_pass':abs(expected-actual)<1e-12,'scope':card['scope']})
report['exports']=[]
for key,meta in db.metadata('export').items():
 _,payload=db.get(key,'export')
 with zipfile.ZipFile(BytesIO(payload)) as archive:
  manifest=json.loads(archive.read('manifest.json'))
  exported=pd.read_csv(BytesIO(archive.read('result.csv')))
  report['exports'].append({'id':key,'manifest':manifest,'csv_rows':len(exported),'original_values_equal':exported.equals(frame),
      'exported_columns_match_receipt':exported.columns.tolist()==meta.get('columns',list(frame.columns)),
      'exported_values_equal_to_source':exported.equals(frame[meta.get('columns',list(frame.columns))])})
if r.get('custom_analysis_evidence'):
 output=store.frames[r['custom_analysis_evidence']['output_dataset_id']]
 ages=sorted(frame['age'].tolist());expected=[(ages[n-1]+ages[n])/2 for n in range(1,len(ages))]
 result=output.select_dtypes(include='number')
 report['python_oracle']={'expected_non_null_rolling_mean':expected,'output_columns':list(output.columns),
  'matches_one_numeric_column':any(np.allclose(result[c].dropna().to_numpy(),expected) for c in result if len(result[c].dropna())==len(expected)),
  'requested_age_rolling_matches':bool('age_rolling' in output and len(output['age_rolling'].dropna())==len(expected)
      and np.allclose(output['age_rolling'].dropna().to_numpy(),expected)),
  'sorted_age_matches':bool('age' in output and output['age'].tolist()==ages),
  'rows':len(output)}
records=[json.loads(s) for s in (db.directory/'runtime.jsonl').read_text().splitlines()]
report['runs']=[{k:e.get(k) for k in ('run_id','status','elapsed_seconds','error_id')} for e in records if e.get('event')=='run_completed']
report['last_run_diagnostics']=[e for e in records if e.get('run_id')==records[-1].get('run_id')]
report['tool_calls']=sum(e.get('event')=='tool_started' for e in report['last_run_diagnostics'])
report['remote_queries_started']=sum(e.get('event')=='remote_query_started' for e in records)
path=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).with_name('browser_evidence.json')
path.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str)+'\n')
print(json.dumps({k:report[k] for k in ('status','dataset_count','original_preserved','remote_queries_started')},ensure_ascii=False))
conn.close();db.close()
