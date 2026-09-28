"""Live model chart decision followed by an injected provider outage; local data only."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import httpx
from openai import APITimeoutError
import pandas as pd
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from scripts.evaluate_autonomous_paths import FaultThenLiveModel
from scripts.evaluate_analysis_agent import fixture_reference_context
from core.analysis_agent.model_provider import build_analysis_chat_model
from core.analysis_agent.policy import RuntimePolicy
from core.analysis_agent.runtime import GraphAnalysisRuntime
from utils.analysis_datasets import stored_dataset_digest


class OutageAfterChart(FaultThenLiveModel):
    def _generate(self,messages,stop=None,run_manager=None,**kwargs):
        chart_done=False
        for message in messages:
            if isinstance(message,ToolMessage) and message.name in {'render_chart_spec','prepare_histogram','render_histogram'}:
                try:result=json.loads(message.content)
                except (TypeError,ValueError):continue
                chart_done=chart_done or bool(result.get('cards'))
        if chart_done:
            self.tracker['injected_timeouts']=self.tracker.get('injected_timeouts',0)+1
            raise APITimeoutError(request=httpx.Request('POST','https://example.invalid/injected-outage'))
        self.tracker['live_calls']=self.tracker.get('live_calls',0)+1
        return ChatResult(generations=[ChatGeneration(message=self.delegate.invoke(messages))])


def evaluate(delegate):
    with tempfile.TemporaryDirectory(prefix='teleai-local-continuation-') as root:
        model=OutageAfterChart(delegate=delegate)
        r=GraphAnalysisRuntime(root,'evaluation','local-continuation',model)
        frame=pd.DataFrame({'reading':[2.,4.,10.,20.]})
        raw=r.datasets.register(frame,source='unfamiliar.observations',coverage='complete',predicate_known=True)
        r.context.reference_context[:]=[fixture_reference_context(raw.source,frame)]
        r.select_dataset(raw.id)
        digest=stored_dataset_digest(r.datasets,raw.id)
        progress=[];r.on_progress=progress.append
        original=r.recovery._next_local
        def plan(current,calls):
            return original(current,calls) if current.get('local_continuation') else None
        prompt='먼저 reading 히스토그램을 만들고, 이어서 reading 평균도 계산해줘.'
        try:
            with patch.object(r.recovery,'_next_local',side_effect=plan),patch.object(
                    r.recovery,'_cached_chart_call',return_value=None):
                outcome=r.submit(prompt)
            state=r.inspect()['recovery']
            ids=state.get('evidence_ids',[])
            actual=float(r.datasets.frames[ids[-1]].iloc[0,0]) if ids else None
            charts=state.get('artifact_ids',[])
            valid_png=bool(charts) and all(r.artifacts[c].image.startswith(b'\x89PNG') for c in charts)
            log=[json.loads(line) for line in r.diagnostics.path.read_text().splitlines()]
            recovery_events=[e for e in log if e['event']=='automatic_local_continuation']
            unchanged=digest==stored_dataset_digest(r.datasets,raw.id)
            passed=(outcome['status']=='answered' and actual==9. and valid_png and len(charts)==1
                    and unchanged and len(recovery_events)==1 and not r.inspect()['requests']
                    and model.tracker.get('injected_timeouts')==3)
            return {'status':'PASS' if passed else 'FAIL','prompt':prompt,'outcome':outcome,
                'expected_mean':9.,'actual_mean':actual,'chart_count':len(charts),'valid_png':valid_png,
                'raw_unchanged':unchanged,'tracker':model.tracker,'model_calls':state['model_calls'],
                'model_recovery':r.inspect()['model_recovery'],'recovery_events':recovery_events,
                'executed_tools':[e['tool'] for e in log if e['event']=='tool_started'],
                'tool_calls':[c for m in r.events() if isinstance(m,AIMessage) for c in m.tool_calls],
                'recovery_state':{k:state.get(k) for k in ('scope','scope_error','required_columns',
                    'operations','kind','current_result_only','failed','histogram_plan')},
                'progress':progress,'approval_requests':len(r.inspect()['requests'])}
        finally:r.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env');os.environ['LANGSMITH_TRACING']='false';os.environ['LANGCHAIN_TRACING_V2']='false'
    result=evaluate(build_analysis_chat_model(RuntimePolicy(model_timeout_seconds=45),provider='databricks'))
    report={'generated_at':datetime.now(timezone.utc).isoformat(),'provider':'databricks',
        'mode':'live model initial decisions + injected APITimeoutError + production local continuation',
        'limitations':['Synthetic four-row local fixture; no warehouse SQL executor connected.',
            'Outage is injected, not an observed Databricks outage. This is not an official benchmark score.',
            'Deterministic planner is disabled before outage to exercise model decisions; restored only for continuation.'],
        'result':result}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'tracker':result['tracker']},ensure_ascii=False))
    return 0 if result['status']=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())
