"""Million-row staged selection with fixture oracle, source hash and real PNGs."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import resource
import sys
import tempfile
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.policy import RuntimePolicy
from migration.test_persistent_runtime import QuietModel
from utils.analysis_image_validation import validate_chart_image

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    fixture=json.loads((ROOT/'tests/fixtures/latest_filters.json').read_text())
    key,order,value,measure=fixture['columns']
    replicas,cycles=fixture['scale']['replicas'],fixture['scale']['cycles']
    def batches():
        for cycle in range(cycles):
            for start in range(0,replicas,1000):
                rows=[[f'{k}_{i}',v+cycle*2,c,m]
                    for i in range(start,min(start+1000,replicas)) for k,v,c,m in fixture['rows']]
                yield pd.DataFrame(rows,columns=fixture['columns'])
    def open_runtime(root):
        return GraphAnalysisRuntime(root,'eval','staged-latest',QuietModel(),
            policy=RuntimePolicy(frame_cache_bytes=0))
    report={'mode':'synthetic data; deterministic production graph; no live model or remote SQL',
        'input_rows':len(fixture['rows'])*replicas*cycles,'cases':[]}
    with tempfile.TemporaryDirectory() as root:
        r=open_runtime(root)
        try:
            raw=r.datasets.register_batches(batches(),columns=fixture['columns'],source=fixture['source'],
                max_rows=report['input_rows'],coverage='complete',predicate_known=True)
            r.select_dataset(raw.id)
            file=r.datasets.db.dataset_file(raw.id);digest=sha256(file.read_bytes()).hexdigest()
            project=r.datasets.frames.project;get=type(r.datasets.frames).__getitem__
            def no_project(identity,columns):
                if identity==raw.id:raise AssertionError('Full original projection')
                return project(identity,columns)
            def no_get(cache,identity):
                if identity==raw.id:raise AssertionError('Full original materialization')
                return get(cache,identity)
            for stage in ('before','after'):
                start=time.monotonic()
                with patch.object(r.datasets.frames,'project',side_effect=no_project),patch.object(type(r.datasets.frames),'__getitem__',no_get):
                    outcome=r.submit(fixture['prompt']+' '+fixture[stage])
                state=r.inspect()['recovery'];proof=state.get('latest_selection_evidence') or {}
                actual={row[value]:row[proof['count_column']] for row in proof.get('counts',[])}
                expected={k:v*replicas for k,v in fixture['expected_'+stage].items()}
                assert outcome['status']=='answered' and actual==expected,(outcome,actual,expected)
                assert r.context.selected_dataset_id==raw.id and sha256(file.read_bytes()).hexdigest()==digest
                chosen=proof['dataset']['id'];info=r.datasets.metadata[chosen]
                assert info.row_selection['filter_stage']==stage+'_selection'
                assert info.row_selection['conditions']==[fixture['condition']]
                card=r.artifacts[proof['cards'][0]['id']];validate_chart_image(card.image)
                image=args.output.with_name(stage+'.png');image.write_bytes(card.image)
                report['cases'].append({'stage':stage,'status':'PASS','counts':actual,'selected_keys':proof['selected_keys'],
                    'execution_mode':proof['execution_mode'],'seconds':round(time.monotonic()-start,3),
                    'source_preserved':True,'model_calls':state.get('model_calls'),
                    'remote_queries':len(state.get('remote_query_ids',[])),'image':image.name})
            report['frame_cache_bytes']=r.datasets.frames.bytes
        finally:r.close()
        r=open_runtime(root)
        try:
            assert r.context.selected_dataset_id==raw.id
            assert r.datasets.metadata[chosen].row_selection==info.row_selection
            assert sha256(r.datasets.db.dataset_file(raw.id).read_bytes()).hexdigest()==digest
            # Persisted chart is readable, not just dataset metadata.
            validate_chart_image(r.artifacts[card.id].image)
            report['restart_preserved']=True
        finally:r.close()
    report['peak_process_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    report['status']='PASS'
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
