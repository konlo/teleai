import json
from pathlib import Path
import tempfile
import unittest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from migration.test_persistent_runtime import QuietModel
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.diagnostics import Diagnostics


class RecoveryModel(QuietModel):
    def _generate(self, messages, **kwargs):
        if isinstance(messages[-1], ToolMessage):
            observation=json.loads(messages[-1].content)
            if observation.get('error_code')=='dataset_not_loaded':
                message=AIMessage(content='',tool_calls=[{'name':'inspect_table_context','args':{'table':'fixture.table'},'id':'schema'}])
            else:
                message=AIMessage(content='저장된 컬럼 정보를 확인했습니다.')
        else:
            message=AIMessage(content='',tool_calls=[{'name':'inspect_dataset','args':{'dataset_id':'fixture.table'},'id':'bad-id'}])
        return ChatResult(generations=[ChatGeneration(message=message)])


class DiagnosticsTests(unittest.TestCase):
    def test_wrong_table_id_is_observation_then_recovers_without_database(self):
        with tempfile.TemporaryDirectory() as root:
            runtime=GraphAnalysisRuntime(root,'owner','conversation',RecoveryModel())
            runtime.context.reference_context.append({'table':'fixture.table','columns':[]})
            self.assertEqual(runtime.submit('저장된 테이블 구조 확인')['status'],'answered')
            self.assertEqual(runtime.inspect()['state'],'idle')
            records=[json.loads(line) for line in runtime.diagnostics.path.read_text().splitlines()]
            self.assertTrue(any(r['event']=='tool_rejected' for r in records))
            self.assertTrue(any(r.get('tool')=='inspect_table_context' for r in records))
            self.assertEqual(records[-1]['event'],'run_completed')
            runtime.close()

    def test_error_has_location_and_id_but_no_exception_payload(self):
        with tempfile.TemporaryDirectory() as root:
            log=Diagnostics(root)
            try:raise ValueError('secret-token-and-private-row')
            except ValueError as exc:error_id=log.failure(exc,run_id='test',stage='tool')
            text=log.path.read_text()
            self.assertNotIn('secret-token',text)
            record=json.loads(text)
            self.assertEqual(record['error_id'],error_id)
            self.assertEqual(record['error_type'],'ValueError')
            self.assertTrue(record['frames'])


class ContextBudgetTests(unittest.TestCase):
    def test_large_legacy_discovery_is_compacted_without_losing_transcript(self):
        from langchain_core.messages import HumanMessage
        with tempfile.TemporaryDirectory() as root:
            runtime=GraphAnalysisRuntime(root,'owner','context',QuietModel())
            fixture=json.loads(Path('tests/fixtures/analysis_acceptance.json').read_text())
            profile={'table':'fixture.table','columns':[{'name':'field','top_values':fixture['rows']*300}]}
            runtime.context.reference_context.append(profile)
            original=json.dumps({'datasets':[], 'available_tables':[profile]})
            messages=[HumanMessage(content='테이블 목록'),AIMessage(content='',tool_calls=[{'name':'list_analysis_context','args':{},'id':'catalog'}]),
                      ToolMessage(content=original,name='list_analysis_context',tool_call_id='catalog',id='observation'),AIMessage(content='목록 확인')]
            runtime.agent.update_state(runtime.config,{'messages':messages},as_node='model')
            runtime.resume()  # Complete after-model middleware for seeded history.
            runtime.events()
            self.assertEqual(runtime.submit('이어서 확인')['status'],'answered')
            archived=next(m for m in runtime.events() if m.id=='observation')
            self.assertEqual(archived.content,original)
            model_message=next(m for m in runtime.agent.get_state(runtime.config).values['messages'] if m.id=='observation')
            self.assertLess(len(model_message.content),1000)
            self.assertEqual(runtime.inspect()['state'],'idle')
            runtime.close()


class ToolOutcomeTests(unittest.TestCase):
    def test_final_claim_after_failed_tool_is_replaced(self):
        from core.analysis_agent.memory import ToolOutcomeMiddleware
        from langchain_core.messages import HumanMessage
        guard=ToolOutcomeMiddleware()
        messages=[HumanMessage(content='chart'),ToolMessage(content=json.dumps({'status':'error','error_code':'dataset_not_loaded'}),tool_call_id='failed'),AIMessage(content='완료했습니다.',id='final')]
        result=guard.after_model({'messages':messages},None)
        self.assertEqual(result['messages'][0].additional_kwargs['analysis_status'],'needs_data')
        self.assertNotIn('완료했습니다.',result['messages'][0].content)
        messages.insert(-1,ToolMessage(content='{"status":"ready"}',tool_call_id='repaired'))
        self.assertIsNone(guard.after_model({'messages':messages},None))
