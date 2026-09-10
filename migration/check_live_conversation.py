"""Actual local model four-turn evaluation with runtime reopen between turns."""
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import urlparse
from dotenv import load_dotenv
import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.messages import ToolMessage
from migration.graph_runtime import GraphAnalysisRuntime


def main():
    load_dotenv()
    endpoint=os.getenv('OLLAMA_BASE_URL','http://localhost:11434')
    if urlparse(endpoint).hostname not in {'localhost','127.0.0.1','::1'}:raise ValueError('localhost only')
    model=ChatOllama(model=os.getenv('OLLAMA_MODEL','gemma4:e4b'),base_url=endpoint,
        reasoning=True,temperature=0,num_ctx=16384,num_predict=4096,client_kwargs={'timeout':60})
    fixture=json.loads(Path('tests/fixtures/analysis_acceptance.json').read_text())
    results=[]
    path=Path('docs/v1_conversation_evidence.json')
    with tempfile.TemporaryDirectory() as root:
        r=GraphAnalysisRuntime(root,'test-owner','conversation',model)
        r.datasets.register(pd.DataFrame(fixture['rows']),source=fixture['source'],coverage='complete',predicate_known=True)
        r.close()
        for turn in fixture['turns']:
            r=GraphAnalysisRuntime(root,'test-owner','conversation',model)
            before=len(r.events())
            result=r.submit(turn['prompt'])
            observations=[json.loads(m.content) for m in r.events()[before:]
                          if isinstance(m,ToolMessage) and m.name=='local_analysis_sql']
            scalar=None
            for obs in observations:
                preview=obs.get('preview',[])
                if len(preview)==1 and len(preview[0])==1:scalar=float(next(iter(preview[0].values())))
            record={'prompt':turn['prompt'],'expected':turn['expected'],'actual':scalar,
                    'status':'PASS' if result['status']=='answered' and scalar==turn['expected'] else 'FAIL',
                    'result':result,'observations':observations}
            results.append(record)
            path.write_text(json.dumps({'model':model.model,'database_calls':0,'turns':results},ensure_ascii=False,indent=2)+'\n')
            print(json.dumps({k:v for k,v in record.items() if k not in {'observations','result'}},ensure_ascii=False),flush=True)
            r.close()
            if result['status']!='answered':break
    return int(len(results)!=4 or any(r['status']!='PASS' for r in results))

if __name__=='__main__':raise SystemExit(main())
