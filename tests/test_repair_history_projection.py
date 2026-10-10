"""Failed-code compaction preserves the latest repair and remote uncertainty."""
import json
import unittest
from types import SimpleNamespace
from langchain_core.messages import AIMessage,HumanMessage,ToolMessage,SystemMessage
from core.analysis_agent.repair_history import compact_failed_calls
from core.analysis_agent.model_context import ModelContextBudgetMiddleware,payload_bytes
from tests.test_model_context_budget import Request

def pair(key,name='execute_analysis_python',size=4000):
    return [AIMessage(content='',tool_calls=[{'id':key,'name':name,'args':{'code':'x'*size}}]),
        ToolMessage(content=json.dumps({'status':'error','error_code':'python_contract_violation'}),tool_call_id=key),
        SystemMessage(content='도구 실패를 관찰했습니다. '+('detail '*100))]

class RepairHistoryTests(unittest.TestCase):
    def test_budget_repair_preserves_current_request_latest_code_and_original_history(self):
        messages=[HumanMessage(content='Only the retained result; preserve source and filters')]+pair('old')+pair('latest')
        request=Request(messages,[],{'recovery':{}},SystemMessage(content='Current scope: observation >= 10'))
        original=[m.model_dump() for m in messages]
        result=ModelContextBudgetMiddleware(SimpleNamespace(num_ctx=8000,num_predict=1000)).wrap_model_call(request,lambda r:r)
        self.assertLess(payload_bytes(result.system_message,result.messages,result.tools)+512+64*(len(result.messages)+1),7000)
        self.assertIs(result.messages[0],messages[0])
        latest=next(m for m in result.messages if isinstance(m,AIMessage))
        self.assertEqual(latest.tool_calls[0]['id'],'latest')
        self.assertEqual(latest.tool_calls[0]['args'],messages[4].tool_calls[0]['args'])
        self.assertEqual([m.model_dump() for m in messages],original)
    def test_remote_unknown_and_successful_tool_pairs_are_retained(self):
        remote=pair('remote','query_databricks',20)
        success=[AIMessage(content='',tool_calls=[{'id':'done','name':'inspect_dataset','args':{}}]),
            ToolMessage(content='{"status":"ready"}',tool_call_id='done')]
        messages=remote+success+pair('old')+pair('latest')
        result=compact_failed_calls(messages)
        self.assertTrue(all(m in result for m in remote+success))
        self.assertNotIn(messages[5],result)
    def test_non_object_tool_observations_are_preserved(self):
        messages=[ToolMessage(content='null',tool_call_id='malformed')]+pair('only')
        self.assertIs(compact_failed_calls(messages),messages)

if __name__=='__main__':unittest.main()
