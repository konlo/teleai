"""Run only in the isolated v1 environment. All model responses here are scripted."""
import json
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

import pandas as pd
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from core.analysis_tool_contract import AnalysisToolContext
from utils.analysis_datasets import DatasetStore
from migration.v1_prototype import build_local_agent


class ScriptModel(BaseChatModel):
    tool_name: str
    arguments: dict

    @property
    def _llm_type(self): return 'deterministic-contract-model'

    def bind_tools(self, tools, **kwargs): return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if isinstance(messages[-1], ToolMessage):
            message = AIMessage(content='도구 결과 확인')
        else:
            message = AIMessage(content='',tool_calls=[dict(name=self.tool_name,
                args=self.arguments,id=str(uuid4()))])
        return ChatResult(generations=[ChatGeneration(message=message)])


class V1Contracts(unittest.TestCase):
    def test_current_tools_and_sqlite_reopen(self):
        fixture=json.loads(Path('tests/fixtures/analysis_acceptance.json').read_text())
        store=DatasetStore()
        info=store.register(pd.DataFrame(fixture['rows']),source=fixture['source'],coverage='complete',predicate_known=True)
        def forbidden(**kwargs): raise AssertionError('remote unavailable')
        context=AnalysisToolContext(store,{},[],forbidden)
        model=ScriptModel(tool_name='local_analysis_sql',arguments={
            'dataset_id':info.id,'query':'SELECT COUNT(*) AS n FROM data'})
        config={'configurable':{'thread_id':'one'}}
        with tempfile.TemporaryDirectory() as tmp:
            db=str(Path(tmp)/'state.sqlite')
            with SqliteSaver.from_conn_string(db) as saver:
                agent=build_local_agent(model,context,saver)
                result=agent.invoke({'messages':[{'role':'user','content':'행 수 계산'}]},config)
                observation=next(m for m in result['messages'] if isinstance(m,ToolMessage))
                self.assertEqual(json.loads(observation.content)['preview'][0]['n'],len(fixture['rows']))
                count=len(result['messages'])
            with SqliteSaver.from_conn_string(db) as saver:
                agent=build_local_agent(model,context,saver)
                self.assertEqual(len(agent.get_state(config).values['messages']),count)
                self.assertFalse(agent.get_state({'configurable':{'thread_id':'other'}}).values)

    def test_interrupt_reopen_reject_and_approve(self):
        for decision in ['reject','approve']:
            calls=[]
            def remote_stub(query: str) -> str:
                """A local counter only; no database connection."""
                calls.append(query)
                return 'fixture result'
            model=ScriptModel(tool_name='remote_stub',arguments={'query':'SELECT 1'})
            config={'configurable':{'thread_id':decision}}
            def build(saver):
                return create_agent(model,tools=[remote_stub],checkpointer=saver,middleware=[
                    HumanInTheLoopMiddleware(interrupt_on={'remote_stub':{'allowed_decisions':['approve','reject']}})])
            with tempfile.TemporaryDirectory() as tmp:
                db=str(Path(tmp)/'state.sqlite')
                with SqliteSaver.from_conn_string(db) as saver:
                    result=build(saver).invoke({'messages':[{'role':'user','content':'fixture'}]},config)
                    self.assertTrue(result['__interrupt__'])
                    self.assertEqual(calls,[])
                with SqliteSaver.from_conn_string(db) as saver:
                    agent=build(saver)
                    before=agent.get_state(config)
                    self.assertTrue(before.tasks[0].interrupts)
                    # Read-only inspection must leave the pending checkpoint unchanged.
                    self.assertEqual(agent.get_state(config).config,before.config)
                    result=agent.invoke(Command(resume={'decisions':[{'type':decision}]}),config)
                    self.assertEqual(calls,['SELECT 1'] if decision=='approve' else [])
                    self.assertTrue(any(isinstance(m,ToolMessage) for m in result['messages']))

if __name__=='__main__':unittest.main()
