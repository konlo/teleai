"""Local provider compatibility smoke test; synthetic data and no remote DB tool."""
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv
import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from core.analysis_tool_contract import AnalysisToolContext
from utils.analysis_datasets import DatasetStore
from migration.v1_prototype import build_local_agent


def main():
    load_dotenv()
    endpoint=os.getenv('OLLAMA_BASE_URL','http://localhost:11434')
    if urlparse(endpoint).hostname not in {'localhost','127.0.0.1','::1'}:
        raise ValueError('Only localhost Ollama is permitted')
    fixture=json.loads(Path('tests/fixtures/analysis_acceptance.json').read_text())
    store=DatasetStore()
    store.register(pd.DataFrame(fixture['rows']),source=fixture['source'],coverage='complete',predicate_known=True)
    def forbidden(**kwargs):raise RuntimeError('Remote execution unavailable')
    context=AnalysisToolContext(store,{},[],forbidden)
    model=ChatOllama(model=os.getenv('OLLAMA_MODEL','gemma4:e4b'),base_url=endpoint,
        reasoning=True,temperature=0,num_ctx=16384,num_predict=4096,client_kwargs={'timeout':60})
    report={'provider':'ChatOllama','model':model.model,'database_calls':0,'status':'FAIL'}
    started=time.monotonic()
    import tempfile
    try:
        with tempfile.TemporaryDirectory() as tmp:
            with SqliteSaver.from_conn_string(str(Path(tmp)/'checkpoint.sqlite')) as saver:
                agent=build_local_agent(model,context,saver)
                output=agent.invoke({'messages':[{'role':'user','content':fixture['turns'][0]['prompt']}]},
                    {'configurable':{'thread_id':'live-local'},'recursion_limit':12})
                observations=[json.loads(m.content) for m in output['messages']
                              if isinstance(m,ToolMessage) and m.name=='local_analysis_sql']
                values=[float(next(iter(o['preview'][0].values()))) for o in observations
                        if len(o.get('preview',[]))==1 and len(o['preview'][0])==1]
                report.update(status='PASS' if values and values[-1]==40 else 'FAIL',
                    observations=observations,answer=output['messages'][-1].content,
                    tool_messages=sum(isinstance(m,ToolMessage) for m in output['messages']))
    except Exception as exc:
        report['error_type']=type(exc).__name__
    report['elapsed_seconds']=round(time.monotonic()-started,2)
    Path('docs/v1_live_compatibility.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return int(report['status']!='PASS')

if __name__=='__main__':raise SystemExit(main())
