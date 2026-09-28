"""Fixed paraphrase/follow-up acceptance on synthetic data, with independent values.

No Databricks SQL executor is connected. Expected values never enter the model
context. Unsupported or unverified output is a failure, not an implicit pass.
"""
import argparse
from io import BytesIO
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
from PIL import Image
import sqlglot
from sqlglot import exp
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.policy import RuntimePolicy
from core.analysis_agent.model_provider import build_analysis_chat_model
from utils.analysis_datasets import stored_dataset_digest
from core.analysis_agent.intent_scope import scope_matches


def evaluate(spec,fixture,model):
    with tempfile.TemporaryDirectory(prefix='teleai-language-') as directory:
        frame=pd.DataFrame(fixture['rows'],columns=fixture['columns'])
        refs=[{'table':fixture['source'],'observed_at':datetime.now(timezone.utc).isoformat(),
               'columns':[{**c,'dtype':str(frame[c['name']].dtype)} for c in fixture['metadata']]}]
        r=GraphAnalysisRuntime(directory,'evaluation',spec['id'],model,reference_context_loader=lambda:refs)
        try:
            raw=r.datasets.register(frame,source=fixture['source'],coverage='complete',predicate_known=True)
            r.select_dataset(raw.id);digest=stored_dataset_digest(r.datasets,raw.id);turns=[]
            for turn in spec['turns']:
                started=time.monotonic();outcome=r.submit(turn['prompt']);state=r.inspect()['recovery']
                actual=None;scope_ok=False;operation_ok=False
                if state.get('evidence_ids'):
                    info=r.datasets.metadata[state['evidence_ids'][-1]]
                    result=r.datasets.frames[info.id]
                    if result.shape==(1,1):actual=float(result.iloc[0,0])
                    scope_ok=scope_matches(info,{'conditions':turn.get('conditions',[]),'any_conditions':[],
                        'unresolved':[],'columns':[]},dialect='duckdb')
                    tree=sqlglot.parse_one(info.query,read='duckdb')
                    operation_ok=turn['operation'] in {node.sql_name() for node in tree.find_all(exp.AggFunc)}
                charts=state.get('artifact_ids',[])
                chart_verified=not turn.get('chart')
                for key in charts:
                    card=r.artifacts[key]
                    with Image.open(BytesIO(card.image)) as image:
                        image.verify()
                    chart_info=r.datasets.metadata[card.dataset_id]
                    lineage_ok=card.dataset_id==raw.id
                    column=fixture['columns'][0]
                    # A complete value-frequency aggregate is a valid exact
                    # histogram input; compare every bin weight independently.
                    if chart_info.parent_id==raw.id and len(chart_info.columns)==2:
                        weights=[name for name in chart_info.columns if name!=column]
                        counts=r.datasets.frames[chart_info.id]
                        expected=frame[column].value_counts().sort_index()
                        if len(weights)==1 and column in counts:
                            actual_counts=counts.set_index(column)[weights[0]].sort_index()
                            lineage_ok=(actual_counts.index.equals(expected.index)
                                and actual_counts.tolist()==expected.tolist())
                    chart_verified=(card.kind=='histogram' and lineage_ok
                        and tuple(card.columns)==(column,)) or chart_verified
                valid=(outcome['status']=='answered' and actual is not None
                    and abs(actual-turn['expected'])<1e-8 and scope_ok and operation_ok
                    and digest==stored_dataset_digest(r.datasets,raw.id)
                    and chart_verified)
                turns.append({'prompt':turn['prompt'],'status':'PASS' if valid else 'FAIL',
                    'actual':actual,'expected':turn['expected'],'scope_verified':scope_ok,
                    'operation_verified':operation_ok,'chart_verified':chart_verified,
                    'agent_status':outcome['status'],'error_type':outcome.get('error_type'),
                    'output':outcome.get('text',''),'recovery_scope':state.get('scope'),
                    'operations':state.get('operations'),'model_calls':state.get('model_calls'),
                    'model_recovery':r.inspect()['model_recovery'],'raw_preserved':digest==stored_dataset_digest(r.datasets,raw.id),
                    'chart_count':len(charts),'elapsed_seconds':round(time.monotonic()-started,3)})
                if outcome['status']=='incomplete':break
            return {'id':spec['id'],'status':'PASS' if len(turns)==len(spec['turns']) and all(t['status']=='PASS' for t in turns) else 'FAIL','turns':turns}
        finally:r.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--provider',choices=['databricks','ollama'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--id',action='append');args=p.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env');os.environ['LANGSMITH_TRACING']='false';os.environ['LANGCHAIN_TRACING_V2']='false'
    model=build_analysis_chat_model(RuntimePolicy(),provider=args.provider)
    fixture=json.loads((ROOT/'tests/fixtures/natural_language_journeys.json').read_text());results=[]
    for spec in fixture['cases']:
        if args.id and spec['id'] not in args.id:continue
        results.append(evaluate(spec,fixture,model))
        report={'provider':args.provider,'data':'synthetic local only','remote_sql_executions':0,'results':results}
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({'id':spec['id'],'status':results[-1]['status']}),flush=True)
    raise SystemExit(0 if all(r['status']=='PASS' for r in results) else 1)
