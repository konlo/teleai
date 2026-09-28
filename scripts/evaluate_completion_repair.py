"""Inject one invalid dataset ID, then measure real-model repair with local rescue disabled.

Uses a synthetic dataset and never connects a Databricks SQL executor.
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from pydantic import Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from core.analysis_agent.model_provider import build_analysis_chat_model
from core.analysis_agent.policy import RuntimePolicy
from core.analysis_agent.runtime import GraphAnalysisRuntime
from scripts.evaluate_analysis_agent import fixture_reference_context
from utils.analysis_datasets import stored_dataset_digest

class InjectedModel(BaseChatModel):
    delegate:Any
    tracker:dict=Field(default_factory=dict)
    @property
    def _llm_type(self):return 'single-fault-then-live-model'
    def bind_tools(self,tools,**kwargs):
        return self.model_copy(update={'delegate':self.delegate.bind_tools(tools,**kwargs)})
    def _generate(self,messages,stop=None,run_manager=None,**kwargs):
        if not self.tracker.get('injected'):
            self.tracker['injected']=True
            msg=AIMessage(content='',tool_calls=[{'name':'aggregate_dataset','args':{'dataset_id':'invalid-dataset','aggregation':'mean','value_column':'reading'},'id':'injected-first-fault'}])
        else:
            self.tracker['live_calls']=self.tracker.get('live_calls',0)+1
            msg=self.delegate.invoke(messages)
        return ChatResult(generations=[ChatGeneration(message=msg)])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider', choices=('ollama', 'databricks'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    os.environ['LANGSMITH_TRACING'] = 'false'
    os.environ['LANGCHAIN_TRACING_V2'] = 'false'
    policy=RuntimePolicy(model_timeout_seconds=45)
    model=InjectedModel(delegate=build_analysis_chat_model(policy,provider=args.provider))
    with tempfile.TemporaryDirectory() as root:
        r=GraphAnalysisRuntime(root,'evaluation','repair',model,policy=policy)
        frame=pd.DataFrame({'reading':[2.,4.,10.,20.], 'segment':['a','a','b','b']})
        raw=r.datasets.register(frame,source='fixture.observations',coverage='complete',predicate_known=True,snapshot='fixture:v1')
        r.context.reference_context[:]=[fixture_reference_context(raw.source,frame)]
        r.select_dataset(raw.id)
        digest=stored_dataset_digest(r.datasets,raw.id)
        t=time.monotonic()
        with patch.object(r.recovery,'_next_local',return_value=None),patch.object(r.recovery,'_budget_local_rescue',return_value=None):
            result=r.submit('reading 평균을 알려줘')
        rec=r.inspect()['recovery']
        report={'type':'one injected invalid ID followed by actual model; deterministic local rescue disabled','provider':args.provider,'status':result['status'],'text':result.get('text'),'elapsed_seconds':round(time.monotonic()-t,3),'model_calls':rec['model_calls'],'tracker':model.tracker,'raw_unchanged':digest==stored_dataset_digest(r.datasets,raw.id),'stop_reason':rec.get('stop_reason'),'expected_mean':9.,'evidence_ids':rec['evidence_ids']}
        report['passed'] = (result['status'] == 'answered' and report['raw_unchanged']
            and bool(rec['evidence_ids']) and float(r.datasets.frames[rec['evidence_ids'][-1]].iloc[0,0]) == 9.0
            and model.tracker.get('live_calls', 0) > 0)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(json.dumps(report,ensure_ascii=False),flush=True)
        r.close()

    return 0 if report["passed"] else 1

if __name__ == '__main__':
    raise SystemExit(main())
