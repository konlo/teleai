"""Scope selection is not a column predicate; real filters still require proof."""
import json
from copy import deepcopy
from pathlib import Path
import unittest
from jsonschema import Draft202012Validator
from core.analysis_agent.population_requirements import validate, model_schema, canonical_delta
from core.analysis_agent.population_audit import audit
from core.analysis_agent.population_review_diagnostics import summary
from tests.test_population_audit_local_repair import delta
import tests.test_population_audit_local_repair as repair
from tests.test_llm_goal import goal

F=json.loads((Path(__file__).parent/'fixtures/unfiltered_population.json').read_text())


def cleared():
    value=delta();value.update(change='clear',evidence_quote=F['request'])
    value['restriction_review']={'clauses':[{'quote':F['request'],'kind':'source_scope',
        'scope_basis':'unfiltered_source','columns':[],'status':'resolved'}],'question':''}
    return value


class UnfilteredScopeTests(unittest.TestCase):
    def schema(self):return [{'table':F['source'],'columns':[{'name':n} for n in (F['column'],F['other_column'])]}]

    def test_source_scope_contract_has_distinct_grammar_and_clears_prior_filters(self):
        raw=cleared();Draft202012Validator(model_schema()).validate(raw['restriction_review'])
        i=repair.PopulationAuditLocalRepairTests().interpreter([raw])
        previous=[{'column':F['other_column'],'op':'ge','value':5}]
        current={'request_id':'new','request_text':F['request'],'confirmed_analysis':{
            'status':'complete','required_sources':[F['source']],
            'scope':{'conditions':previous,'any_conditions':[]}}}
        plan=goal('chart',{'kind':'histogram'},sources=[F['source']],columns=[F['column']])
        result=audit(i,current,{'tables':self.schema()},plan,{'population_basis':'unfiltered_source'})
        self.assertEqual(result['conditions'],[])
        self.assertEqual(result['any_conditions'],[])
        self.assertEqual(current['confirmed_analysis']['scope']['conditions'],previous)
        self.assertEqual(i.population_model.invoke.call_count,1)

    def test_resolved_axis_without_filter_is_still_rejected(self):
        raw=delta();raw.update(change='modify',evidence_quote=F['request'])
        review={'clauses':[{'quote':F['request'],'kind':'predicate','columns':[F['column']],
            'status':'resolved'}],'question':''}
        with self.assertRaisesRegex(ValueError,'no matching population predicate'):
            validate(review,F['request'],raw,raw,self.schema())

    def test_scope_cannot_disguise_a_filter_or_preserve_old_population(self):
        for edit in ('conditions','change','columns','status'):
            with self.subTest(edit=edit):
                raw=cleared();review=raw['restriction_review']
                if edit=='conditions':raw[edit]=[{'column':F['other_column'],'op':'ge','value':5}]
                elif edit=='change':raw[edit]='keep'
                elif edit=='columns':review['clauses'][0]['columns']=[F['column']]
                else:review['clauses'][0]['status']='unresolved'
                with self.assertRaises(ValueError):validate(review,F['request'],raw,raw,self.schema())

    def test_all_matching_rows_keeps_real_predicates(self):
        raw=delta();raw.update(change='modify',evidence_quote=F['filtered_request'],
            conditions=[{'column':F['other_column'],'op':'ge','value':5}])
        review={'clauses':[{'quote':F['filtered_request'],'kind':'predicate',
            'columns':[F['other_column']],'status':'resolved'}],'question':''}
        validate(review,F['filtered_request'],raw,raw,self.schema())
        raw['conditions']=[]
        with self.assertRaises(ValueError):validate(review,F['filtered_request'],raw,raw,self.schema())

    def test_diagnostics_retains_shapes_but_no_quotes_values_or_unobserved_names(self):
        raw=cleared();raw['conditions']=[{'column':F['column'],'op':'eq','value':'PRIVATE_VALUE'}]
        raw['any_conditions']=[{'column':'PRIVATE_COLUMN','op':'BAD_OPERATOR','value':'PRIVATE_VALUE'}]
        view=summary(raw,raw['restriction_review'],F['request'],self.schema())
        encoded=json.dumps(view)
        for private in ('PRIVATE_VALUE','PRIVATE_COLUMN','BAD_OPERATOR',F['request']):self.assertNotIn(private,encoded)
        self.assertEqual(view['restriction_shapes'][0]['kind'],'source_scope')
        self.assertTrue(view['restriction_shapes'][0]['quote_in_request'])

    def test_local_repair_corrects_axis_confusion_without_new_goal(self):
        bad=delta();bad.update(change='modify',evidence_quote=F['request'])
        bad['restriction_review']={'clauses':[{'quote':F['request'],'kind':'predicate',
            'columns':[F['column']],'status':'resolved'}],'question':''}
        i=repair.PopulationAuditLocalRepairTests().interpreter([bad,cleared()])
        current={'request_id':'new','request_text':F['request']}
        plan=goal('chart',{'kind':'histogram'},sources=[F['source']],columns=[F['column']])
        original=deepcopy(plan)
        result=audit(i,current,{'tables':self.schema()},plan)
        self.assertEqual(result['conditions'],[]);self.assertEqual(plan,original)
        self.assertEqual(i.population_model.invoke.call_count,2)

    def test_malformed_response_cannot_break_diagnostic_collection(self):
        raw={'change':[],'conditions':[{'column':{},'op':[]}], 'any_conditions':'bad'}
        review={'clauses':[{'kind':{},'status':[],'columns':[{}],'quote':{}}]}
        view=summary(raw,review,F['request'],self.schema())
        self.assertEqual(view['change'],'<invalid>')
        self.assertEqual(view['restriction_shapes'][0]['kind'],'<invalid>')

    def test_explicit_scope_is_compiled_without_changing_model_meaning(self):
        raw=cleared();raw['change']='modify'
        corrected=canonical_delta(raw,raw['restriction_review'],F['request'],self.schema())
        self.assertEqual(corrected['change'],'clear');self.assertEqual(raw['change'],'modify')
        validate(corrected['restriction_review'],F['request'],corrected,corrected,self.schema())
        for edit in ('quote','conditions','mixed'):
            bad=deepcopy(raw)
            if edit=='quote':bad['restriction_review']['clauses'][0]['quote']='unrelated private request'
            elif edit=='conditions':bad['conditions']=[{'column':F['other_column'],'op':'ge','value':5}]
            else:bad['restriction_review']['clauses'].append({'quote':F['request'],
                'kind':'predicate','columns':[F['other_column']],'status':'unresolved'})
            self.assertEqual(canonical_delta(bad,bad['restriction_review'],F['request'],self.schema()),bad)

    def test_extraneous_reference_quote_does_not_require_another_semantic_call(self):
        from core.analysis_agent.population_basis import read
        from tests.test_population_requirement_guard import reader
        i=reader([{'basis':'source_population','quote':F['request'],
            'changes_filters':True,'filter_change_quote':F['request']}])
        current={'request_id':'new','request_text':F['request']}
        value=read(i,current,{'verified_previous':{'chart_available':True}})
        self.assertEqual(value['population_basis'],'source_population')
        self.assertTrue(value['changes_filters']);self.assertEqual(value['filter_change_quote'],F['request'])
        self.assertEqual(i.selection_model.invoke.call_count,1)

    def test_support_report_shows_contract_shape_without_model_output(self):
        from core.analysis_agent.support_report import public_population_review
        event=summary(cleared(),cleared()['restriction_review'],F['request'],self.schema())
        event.update(attempt=1,quote='PRIVATE_QUOTE',content='PRIVATE_MODEL_OUTPUT')
        view=public_population_review(event)
        self.assertEqual(view['restrictions'][0]['kind'],'source_scope')
        self.assertNotIn('PRIVATE',json.dumps(view))

    def test_schema_handle_cannot_pin_a_source_histogram_to_zero_rows(self):
        import pandas as pd
        from unittest.mock import Mock
        from utils.analysis_datasets import DatasetStore
        from core.analysis_tool_contract import AnalysisToolContext
        from core.analysis_runtime_tools import build_analysis_tools
        store=DatasetStore();remote=Mock()
        probe=store.register(pd.DataFrame({F['column']:pd.Series([],dtype=float)}),
            source=F['source'],query='SELECT * FROM '+F['source']+' LIMIT 0',
            coverage='complete',predicate_known=True)
        ctx=AnalysisToolContext(store,{},[],remote)
        tool=next(t.run for t in build_analysis_tools(ctx) if t.name=='prepare_histogram')
        result=tool(F['source'],F['column'],dataset_id=probe.id)
        self.assertEqual(result['status'],'planned')
        self.assertIn('schema_only_dataset_is_not_population',result['applied_corrections'])
        self.assertIn('COUNT(*)',result['histogram_plan']['query'])
        self.assertEqual(tool(F['source'],F['column'],dataset_id=probe.id,
            current_result_only=True)['status'],'needs_data')
        from core.analysis_agent.dataset_preflight import unknown_dataset_error
        alias={'name':'prepare_histogram','args':{'source':F['source'],
            'column':F['column'],'dataset_id':F['source']}}
        self.assertIsNone(unknown_dataset_error(alias,ctx))
        result=tool(**alias['args'])
        self.assertEqual(result['status'],'planned')
        self.assertIn('source_name_is_not_dataset_id',result['applied_corrections'])
        self.assertNotIn(probe.id,result['histogram_plan']['query'])
        for invalid in ({**alias['args'],'dataset_id':'invented'},
                        {**alias['args'],'current_result_only':True}):
            self.assertEqual(unknown_dataset_error({'name':alias['name'],'args':invalid},ctx)['error_code'],'unknown_dataset_id')
        remote.assert_not_called();self.assertEqual(store.metadata[probe.id].rows,0)

    def test_histogram_menu_remains_expandable_after_discovery(self):
        from types import SimpleNamespace
        from langchain_core.messages import HumanMessage, ToolMessage
        from core.analysis_agent.model_context import ProgressiveToolsMiddleware
        names=['prepare_histogram','prepare_source_scatter','query_databricks',
               'search_analysis_tools','get_analysis_tool_contract','execute_analysis_python']
        tools=[SimpleNamespace(name=n) for n in names+[f'optional_{i}' for i in range(6)]]
        def request(messages):
            r=SimpleNamespace(tools=tools,messages=messages,state={'recovery':{'chart':True,'kind':'histogram'}})
            r.override=lambda **changes:SimpleNamespace(**{**vars(r),**changes})
            return r
        m=ProgressiveToolsMiddleware(tools)
        first=m.wrap_model_call(request([HumanMessage(content='distribution')]),lambda r:r)
        offered={t.name for t in first.tools}
        self.assertIn('search_analysis_tools',offered);self.assertIn('prepare_histogram',offered)
        self.assertNotIn('prepare_source_scatter',offered)
        observed=ToolMessage(name='get_analysis_tool_contract',tool_call_id='discover',
            content=json.dumps({'status':'ready','contract':{'name':'execute_analysis_python'}}))
        expanded=m.wrap_model_call(request([HumanMessage(content='distribution'),observed]),lambda r:r)
        self.assertIn('execute_analysis_python',{t.name for t in expanded.tools})
