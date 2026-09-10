"""Real model summary + referential follow-up, using only a local synthetic fixture."""
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import urlparse
from dotenv import load_dotenv
import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
from core.analysis_agent.runtime import GraphAnalysisRuntime


def main():
    load_dotenv()
    endpoint=os.getenv('OLLAMA_BASE_URL','http://localhost:11434')
    if urlparse(endpoint).hostname not in {'localhost','127.0.0.1','::1'}:raise ValueError('localhost only')
    model=ChatOllama(model=os.getenv('OLLAMA_MODEL','gemma4:e4b'),base_url=endpoint,
        reasoning=True,temperature=0,num_ctx=16384,num_predict=4096,client_kwargs={'timeout':60})
    fixture=json.loads(Path('tests/fixtures/analysis_acceptance.json').read_text())
    with tempfile.TemporaryDirectory() as root:
        r=GraphAnalysisRuntime(root,'test-owner','memory',model,summary_trigger_tokens=1000,summary_keep_messages=2)
        info=r.datasets.register(pd.DataFrame(fixture['rows']),source=fixture['source'],coverage='complete',predicate_known=True)
        history=[HumanMessage(content=f"분석 대상은 {info.id}이고 period는 2026-08, segment는 A야. 다음 요청에도 이 조건을 유지해줘."),
                 AIMessage(content='확정한 기간과 그룹을 유지합니다.')]
        for _ in range(10):history.extend([HumanMessage(content='앞서 정한 조건은 변경하지 않습니다. '*20),AIMessage(content='같은 조건을 유지합니다.')])
        r.agent.update_state(r.config,{'messages':history},as_node='model');r.events()
        before=len(r.events())
        result=r.submit('그 조건으로 value 중앙값을 계산해줘.')
        observations=[json.loads(m.content) for m in r.events()[before:] if isinstance(m,ToolMessage) and m.name=='local_analysis_sql']
        values=[float(next(iter(o['preview'][0].values()))) for o in observations if len(o.get('preview',[]))==1 and len(o['preview'][0])==1]
        state=r.agent.get_state(r.config).values['messages']
        summarized=any(m.additional_kwargs.get('lc_source')=='summarization' for m in state)
        report={'status':'PASS' if result['status']=='answered' and values and values[-1]==20 and summarized else 'FAIL',
            'model':model.model,'result':result,'observations':observations,'summarized':summarized,
            'visible_messages':len(r.events()),'model_messages':len(state),'database_calls':0}
        Path('docs/production_memory_evidence.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(report,ensure_ascii=False,indent=2));r.close()
    return int(report['status']!='PASS')
if __name__=='__main__':raise SystemExit(main())
