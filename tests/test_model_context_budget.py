from dataclasses import dataclass, replace
from types import SimpleNamespace
import json
import unittest
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage,SystemMessage
from core.analysis_agent.model_context import (ModelContextBudgetMiddleware,
    ModelContextBudgetExceeded,ProgressiveToolsMiddleware,payload_bytes,prompt_catalog)
from core.analysis_agent.tools import local_tools
from core.analysis_tool_contract import AnalysisToolContext
from utils.analysis_datasets import DatasetStore


@dataclass
class Request:
    messages:list
    tools:list
    state:dict
    system_message:object
    def override(self,**kwargs):return replace(self,**kwargs)


class ContextBudgetTests(unittest.TestCase):
    def test_builtin_emergency_projection_preserves_current_request_and_receipt(self):
        from core.analysis_agent.model_context import COMPACT_INSTRUCTIONS,MINIMAL_INSTRUCTIONS
        system=SystemMessage(content=COMPACT_INSTRUCTIONS+'\n'+'x'*7000,
            additional_kwargs={'telly_builtin_instructions':COMPACT_INSTRUCTIONS})
        messages=[HumanMessage(content='reading의 같은 조건을 유지해줘'),
            AIMessage(content='',tool_calls=[{'id':'proof','name':'inspect_dataset','args':{'dataset_id':'exact'}}]),
            ToolMessage(tool_call_id='proof',name='inspect_dataset',content='{"status":"ready","dataset_id":"exact","coverage":"complete"}')]
        request=Request(messages,self.tools(),{},system)
        projected=ModelContextBudgetMiddleware(SimpleNamespace(num_ctx=16384,num_predict=4096)).wrap_model_call(request,lambda r:r)
        self.assertEqual(projected.messages,messages)
        self.assertIn(MINIMAL_INSTRUCTIONS,projected.system_message.content)
        self.assertNotIn(COMPACT_INSTRUCTIONS,projected.system_message.content)
        self.assertIn('x'*7000,projected.system_message.content)
        self.assertIn('synchronous read-only SQL',projected.system_message.content)
        self.assertIn('Preserve originals/selection',projected.system_message.content)

    def test_many_relevant_snapshots_cannot_displace_current_scope_and_tool_receipt(self):
        from core.analysis_agent.input_views import CATALOG_KEY
        block=json.dumps({'datasets':[{'id':str(i),'source':'lab.sensor','columns':['reading'],
            'query':'SELECT reading FROM lab.sensor WHERE reading >= 3 '+(' '*100)} for i in range(100)],
            'available_tables':[{'table':'lab.sensor'}]})
        policy='Protected custom instruction '+('p'*8200)
        system=SystemMessage(content=policy+'\n'+block,additional_kwargs={CATALOG_KEY:block})
        current={'required_sources':['lab.sensor'],'scope':{'sources':['lab.sensor'],
            'conditions':[{'column':'reading','op':'ge','value':3}]}}
        messages=[HumanMessage(content='같은 조건의 reading 분포를 그려줘'),
            AIMessage(content='',tool_calls=[{'id':'receipt','name':'inspect_dataset','args':{'dataset_id':'exact'}}]),
            ToolMessage(tool_call_id='receipt',name='inspect_dataset',content='{"status":"ready","dataset_id":"exact","coverage":"complete"}')]
        request=Request(messages,self.tools(),{'recovery':current},system)
        projected=ModelContextBudgetMiddleware(SimpleNamespace(num_ctx=16384,num_predict=4096)).wrap_model_call(request,lambda r:r)
        self.assertEqual(projected.messages,messages)
        self.assertIn(policy,projected.system_message.content)
        self.assertIn('list_analysis_context',projected.system_message.content)
        self.assertNotIn('SELECT reading FROM lab.sensor',projected.system_message.content)
        self.assertEqual(request.state['recovery']['scope']['conditions'][0]['value'],3)
        budget=payload_bytes(projected.system_message,projected.messages,projected.tools)+512+128*len(projected.tools)+64*(len(messages)+1)
        self.assertLessEqual(budget,12288)

    def test_budget_report_exposes_sizes_only_not_prompt_or_schema_content(self):
        from core.analysis_agent.support_report import public_input_budget
        report=public_input_budget({'payload_bytes':10549,'template_headroom':1088,
            'input_budget_units':12288,'system_prompt':'PRIVATE prompt',
            'components':{'system_bytes':5797,'message_content_bytes':'PRIVATE row',
                'tool_names':['inspect_table_context','PRIVATE credentials !'],'prompt':'secret'}})
        self.assertEqual(report['payload_bytes'],10549)
        self.assertEqual(report['components']['system_bytes'],5797)
        self.assertIsNone(report['components']['message_content_bytes'])
        self.assertEqual(report['components']['tool_names'],['inspect_table_context'])
        self.assertNotIn('PRIVATE',json.dumps(report))
        self.assertNotIn('secret',json.dumps(report))

    def test_last_resort_tool_discovery_removes_finished_thinking_without_losing_pairs(self):
        tools=self.tools()
        messages=[HumanMessage(content='관측된 스키마를 설명해줘'),
            AIMessage(content='',additional_kwargs={'reasoning_content':'internal reasoning '*200},
                tool_calls=[{'id':'schema','name':'inspect_table_context','args':{'table':'lab.observed'}}]),
            ToolMessage(name='inspect_table_context',tool_call_id='schema',content='{"status":"ready"}')]
        request=Request(messages,tools,{},SystemMessage(content='protected custom instruction '+('a'*9000)))
        compact=ModelContextBudgetMiddleware(SimpleNamespace(num_ctx=16384,num_predict=4096)).wrap_model_call(request,lambda r:r)
        self.assertEqual({t.name for t in compact.tools},{'search_analysis_tools','read_analysis_skill'})
        self.assertNotIn('reasoning_content',compact.messages[1].additional_kwargs)
        self.assertIn('reasoning_content',messages[1].additional_kwargs)
        self.assertEqual(compact.messages[1].tool_calls,messages[1].tool_calls)
        self.assertEqual(compact.messages[2],messages[2])
        self.assertIn('a'*9000,compact.system_message.content)

    def test_wide_schema_after_inspection_compacts_catalog_and_preserves_all_column_types(self):
        from core.analysis_agent.input_views import CATALOG_KEY
        columns=[{'name':f'field_{i}','dtype':'object'} for i in range(107)]
        columns[3]['comment']='Recorded date as a string; format must be verified'
        result={'status':'ready','authority':'approved_select_star_result','scope':'snapshot only',
            'table_context':{'table':'lab.wide','columns':columns,'freshness':'current_loaded_schema','schema_fingerprint':'abc'}}
        tool=ToolMessage(name='inspect_table_context',tool_call_id='schema',content=json.dumps(result))
        current={'request_text':'시간 관련 column을 확인해줘','required_sources':['lab.wide'],
            'scope':{'sources':['lab.wide'],'conditions':[{'column':'cohort','op':'eq','value':'B'}]},
            'confirmed_analysis':{'required_sources':['lab.old'],'status':'complete'}}
        catalog={'datasets':[{'id':str(i),'source':f'lab.old_{i}'} for i in range(150)]+[{'id':'actual','source':'lab.wide'}],
            'available_tables':[{'table':f'lab.old_{i}'} for i in range(100)]}
        block=json.dumps(catalog)
        system=SystemMessage(content='CUSTOM policy never change filters\nENV:'+block+'\nEND policy',
            additional_kwargs={CATALOG_KEY:block})
        messages=[HumanMessage(content='old history'*500),HumanMessage(content='시간 관련 column을 확인해줘'),
            AIMessage(content='',tool_calls=[{'id':'schema','name':'inspect_table_context','args':{'table':'lab.wide'}}]),tool]
        request=Request(messages,self.tools(),{'recovery':current},system)
        compact=ModelContextBudgetMiddleware(SimpleNamespace(num_ctx=16384,num_predict=4096)).wrap_model_call(request,lambda r:r)
        schema=json.loads(compact.messages[-1].content)
        observed=schema['table_context']
        self.assertEqual(observed['columns_by_dtype']['object'],[c['name'] for c in columns])
        self.assertEqual(observed['column_details'][0]['comment'],columns[3]['comment'])
        self.assertEqual(schema['authority'],result['authority'])
        self.assertEqual(compact.messages[-1].tool_call_id,'schema')
        self.assertEqual(json.loads(tool.content),result)
        self.assertIn('CUSTOM policy never change filters',compact.system_message.content)
        self.assertIn('END policy',compact.system_message.content)
        self.assertNotIn('lab.old_99',compact.system_message.content)
        self.assertIn('actual',compact.system_message.content)
        self.assertIn('cohort',compact.messages[0].content)
        budget=payload_bytes(compact.system_message,compact.messages,compact.tools)+512+128*len(compact.tools)+64*(len(compact.messages)+1)
        self.assertLessEqual(budget,12288)

    def test_current_wide_tool_preview_is_projected_but_receipt_and_pairs_are_preserved(self):
        data={'status':'ready','dataset':{'id':'wide','source':'lab.wide','coverage':'sampled',
             'rows':10,'columns':[f'c{i}' for i in range(107)],'conditions':[{'column':'c3','op':'ge','value':30}]},
             'preview':[{f'c{i}':'v'*20 for i in range(107)} for _ in range(10)]}
        messages=[HumanMessage(content='c3 분석해줘'),AIMessage(content='',tool_calls=[
            {'id':'query','name':'query_databricks','args':{'source':'lab.wide','query':'SELECT * FROM lab.wide LIMIT 10'}}]),
            ToolMessage(name='query_databricks',tool_call_id='query',content=json.dumps(data))]
        r=Request(messages,[],{'recovery':{'required_columns':['c3']}},SystemMessage(content='policy'))
        compact=ModelContextBudgetMiddleware(SimpleNamespace(num_ctx=16384,num_predict=4096)).wrap_model_call(r,lambda r:r)
        result=json.loads(compact.messages[-1].content)
        self.assertEqual(result['dataset'],data['dataset'])
        self.assertEqual(compact.messages[-1].tool_call_id,'query')
        self.assertEqual(compact.messages[:2],messages[:2])
        self.assertEqual(len(result['preview']),5)
        self.assertIn('c3',result['preview'][0])
        self.assertLessEqual(len(result['preview'][0]),8)
        self.assertEqual(json.loads(messages[-1].content),data)

    def test_wide_catalog_is_on_demand_and_confirmed_preview_does_not_duplicate_schema(self):
        catalog={'datasets':[{'id':'target','source':'lab.target','columns':['reading']+[f'c{i}' for i in range(1000)],
            'coverage':'sampled','grain':'raw'}, {'id':'old','source':'lab.old','columns':['irrelevant']*1000}],
            'skills':[],'available_tables':[{'table':'lab.target','schema_fingerprint':'x'*10000,'freshness':'fresh'}]}
        current={'required_sources':['lab.target'],'required_columns':['reading'],
                 'confirmed_analysis':{'status':'complete','required_sources':['lab.target'],
                    'scope':{'conditions':[{'column':'cohort','op':'eq','value':'B'}]},
                    'table_preview_evidence':{'dataset_id':'target','source':'lab.target','rows':10,
                        'columns':catalog['datasets'][0]['columns']}}}
        view=prompt_catalog(catalog,current)
        self.assertEqual(view['datasets'][0]['columns'],['reading'])
        self.assertNotIn('columns',view['datasets'][1])
        self.assertEqual(view['datasets'][0]['column_count'],1001)
        self.assertEqual(view['datasets'][0]['more_columns_tool'],'inspect_dataset')
        self.assertEqual(len(catalog['datasets'][0]['columns']),1001)
        request=Request([HumanMessage(content='옛 대화'+'a'*20000),HumanMessage(content='reading 분포를 보여줘')],
            self.tools(),{'recovery':current},SystemMessage(content=json.dumps(view)))
        compact=ModelContextBudgetMiddleware(SimpleNamespace(num_ctx=16384,num_predict=4096)).wrap_model_call(request,lambda r:r)
        self.assertIn('cohort',compact.messages[0].content)
        self.assertIn('"rows": 10',compact.messages[0].content)
        self.assertNotIn('c999',compact.messages[0].content)
        self.assertLess(payload_bytes(compact.system_message,compact.messages,compact.tools),12288)

    def tools(self):
        return local_tools(AnalysisToolContext(DatasetStore(),{},[],lambda **kwargs:None))

    def test_large_offered_schema_counts_before_inference_and_never_retries(self):
        tools=[self.tools()[0].model_copy(update={'description':'a'*16000})];called=[]
        request=Request([HumanMessage(content='계산해주세요')],tools,{},SystemMessage(content='policy'))
        self.assertGreater(payload_bytes(request.system_message,request.messages,tools),16000)
        with self.assertRaises(ModelContextBudgetExceeded):
            ModelContextBudgetMiddleware(SimpleNamespace(num_ctx=16384,num_predict=4096)).wrap_model_call(
                request,lambda r:called.append(r))
        self.assertEqual(called,[])

    def test_oversized_menu_falls_back_to_discovery_with_output_reserved(self):
        tools=self.tools();request=Request([HumanMessage(content='고급 분석해줘')],tools,{},SystemMessage(content='policy'))
        compact=ModelContextBudgetMiddleware(SimpleNamespace(num_ctx=16384,num_predict=4096)).wrap_model_call(request,lambda r:r)
        self.assertIn('search_analysis_tools',{t.name for t in compact.tools})
        self.assertLess(len(compact.tools),len(tools))
        self.assertLess(payload_bytes(compact.system_message,compact.messages,compact.tools),12288)
        self.assertEqual(len(request.tools),len(tools))

    def test_old_turn_compaction_preserves_current_pairs_scope_system_and_transcript(self):
        current=[HumanMessage(content='같은 조건에서 x와 y 그려줘'),
                 AIMessage(content='',tool_calls=[{'id':'call','name':'inspect_dataset','args':{'dataset_id':'real'}}]),
                 ToolMessage(name='inspect_dataset',tool_call_id='call',content='{"status":"ready"}')]
        messages=[HumanMessage(content='옛 대화'+('가'*10000)),AIMessage(content='이전 결과')]+current
        state={'recovery':{'confirmed_analysis':{'required_sources':['lab.changed'],
            'required_columns':['x','y'],'scope':{'conditions':[{'column':'cohort','op':'eq','value':'B'}]}}}}
        system=SystemMessage(content='CUSTOM: never erase source conditions')
        r=Request(messages,[],state,system)
        observed=[]
        ModelContextBudgetMiddleware(SimpleNamespace(num_ctx=16384,num_predict=4096)).wrap_model_call(r,observed.append)
        compact=observed[0]
        self.assertEqual(compact.messages[1:],current)
        self.assertIn('lab.changed',compact.messages[0].content)
        self.assertIn('cohort',compact.messages[0].content)
        self.assertEqual(compact.system_message,system)
        self.assertEqual(r.messages,messages);self.assertEqual(len(r.messages),5)

    def test_current_request_cannot_be_silently_truncated(self):
        request=Request([HumanMessage(content='요청'+('a'*16000))],[],{},SystemMessage(content='policy'))
        with self.assertRaises(ModelContextBudgetExceeded):
            ModelContextBudgetMiddleware(SimpleNamespace(num_ctx=16384,num_predict=4096)).wrap_model_call(request,lambda r:r)
        self.assertEqual(len(request.messages[0].content),16002)

    def test_tool_discovery_exposes_full_contract_without_changing_executor(self):
        tools=self.tools();names={t.name for t in tools}
        request=Request([HumanMessage(content='피벗 분석해줘')],tools,{'recovery':{}},SystemMessage(content='policy'))
        middleware=ProgressiveToolsMiddleware(tools)
        first=middleware.wrap_model_call(request,lambda r:r)
        self.assertLess(len(first.tools),len(tools));self.assertNotIn('pivot_dataset',{t.name for t in first.tools})
        discovery=ToolMessage(name='search_analysis_tools',tool_call_id='discovery',
            content=json.dumps({'matches':[{'name':'pivot_dataset'},{'name':'invented_tool'}]}))
        next_request=replace(request,messages=request.messages+[discovery])
        second=middleware.wrap_model_call(next_request,lambda r:r)
        self.assertIn('pivot_dataset',{t.name for t in second.tools})
        self.assertNotIn('invented_tool',{t.name for t in second.tools})
        self.assertEqual({t.name for t in tools},names)
        next_request=replace(next_request,messages=next_request.messages+[HumanMessage(content='다른 요청')])
        self.assertNotIn('pivot_dataset',{t.name for t in middleware.wrap_model_call(next_request,lambda r:r).tools})


if __name__=='__main__':unittest.main()
