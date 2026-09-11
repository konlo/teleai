import json
from pathlib import Path
import tempfile
import unittest
from pydantic import Field
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.outputs import ChatResult,ChatGeneration
from migration.test_persistent_runtime import QuietModel
from core.analysis_agent.runtime import GraphAnalysisRuntime


class MemoryModel(QuietModel):
    summary: str
    seen: list = Field(default_factory=list)
    def _generate(self,messages,**kwargs):
        if str(messages[0].content).startswith('이 분석 대화를 이어가기'):
            text=self.summary
        else:
            self.seen.append([m.content for m in messages])
            text='조건을 유지했습니다.'
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


class ProductionMemoryTests(unittest.TestCase):
    def test_compaction_preserves_visible_history_and_context_after_reopen(self):
        fixture=json.loads(Path('tests/fixtures/analysis_acceptance.json').read_text())
        facts=json.dumps(fixture['rows'][0],ensure_ascii=False)
        model=MemoryModel(summary=facts)
        with tempfile.TemporaryDirectory() as root:
            runtime=GraphAnalysisRuntime(root,'owner','thread',model,summary_trigger_tokens=100,summary_keep_messages=2)
            initial=[HumanMessage(content='확정 조건: '+facts),AIMessage(content='조건을 유지합니다.')]
            for _ in range(8):initial.extend([HumanMessage(content='기존 조건 유지. '*80),AIMessage(content='확인했습니다.')])
            runtime.agent.update_state(runtime.config,{'messages':initial},as_node='model')
            runtime.resume()  # Complete after-model middleware for seeded history.
            runtime.events()
            before=len(runtime.events())
            self.assertEqual(runtime.submit('이전 조건으로 이어서 설명해줘')['status'],'answered')
            self.assertEqual(len(runtime.events()),before+2)
            context=runtime.agent.get_state(runtime.config).values['messages']
            self.assertLess(len(context),len(runtime.events()))
            self.assertTrue(any(m.additional_kwargs.get('lc_source')=='summarization' for m in context))
            self.assertTrue(any(facts in str(text) for text in model.seen[-1]))
            self.assertIn(facts,runtime.events()[0].content)
            runtime.close()
            runtime=GraphAnalysisRuntime(root,'owner','thread',model)
            self.assertEqual(len(runtime.events()),before+2)
            runtime.close()
