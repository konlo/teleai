"""A typed language decision owns permission to alter a confirmed population."""
from contextlib import ExitStack
from types import SimpleNamespace
import json,tempfile,unittest
from unittest.mock import patch
import pandas as pd
from core.analysis_agent.population_basis import validate
from core.analysis_agent.population_audit import audit
from core.analysis_agent.runtime import GraphAnalysisRuntime
from tests.test_llm_goal import GoalModel,goal
from tests.test_population_null_authority import BadNullAudit
from utils.analysis_datasets import stored_dataset_digest

class PopulationChangeAuthorityTests(unittest.TestCase):
    def test_explicit_delta_needs_quote_and_legacy_has_no_new_authority(self):
        legacy=validate({'basis':'source_population','quote':''},'their mean')
        self.assertNotIn('changes_filters',legacy)
        for value in [
            {'basis':'source_population','quote':'','changes_filters':True,'filter_change_quote':''},
            {'basis':'source_population','quote':'','changes_filters':True,'filter_change_quote':'old restriction'},
            {'basis':'source_population','quote':'','changes_filters':False,'filter_change_quote':'mean'}]:
            with self.subTest(value=value),self.assertRaises(ValueError):validate(value,'their mean')
        changed=validate({'basis':'source_population','quote':'','changes_filters':True,'filter_change_quote':'reading < 20'},'reading < 20의 평균')
        self.assertTrue(changed['changes_filters'])
        # Unchanged old source must never replace an explicitly switched source.
        interpreter=SimpleNamespace(population_model=None,diagnostics=SimpleNamespace(emit=lambda *a,**k:None))
        current={'request_id':'x','request_text':'new source mean','confirmed_analysis':{
            'status':'complete','required_sources':['lab.old'],'scope':{'conditions':[{'column':'reading','op':'ge','value':10}]}}}
        new=goal('calculation',{'operations':['AVG']},sources=['lab.new'],columns=['reading'])
        with patch('core.analysis_agent.population_audit.model_for',side_effect=RuntimeError('normal new-source audit')):
            with self.assertRaisesRegex(RuntimeError,'normal new-source audit'):
                audit(interpreter,current,{},new,{'population_basis':'source_population','changes_filters':False})

    def test_restart_keeps_range_despite_hallucinated_duplicate_bound(self):
        with tempfile.TemporaryDirectory() as root:
            conditions=[{'column':'reading','op':'ge','value':10}]
            count=goal('row_count',conditions=conditions)
            count['source_reference']='selected_dataset'
            mean=goal('calculation',{'operations':['AVG']},columns=['reading'],conditions=[*conditions,{'column':'reading','op':'between','value':[10,10]}])
            mean['source_reference']='previous_analysis'
            model=GoalModel(goals=[count,count,mean,mean],calls=[
                {'name':'local_analysis_sql','args':{'dataset_id':'$fixture','query':'SELECT COUNT(*) FROM data','requested_conditions':conditions}},
                {'name':'local_analysis_sql','args':{'dataset_id':'$fixture','query':'SELECT AVG(reading) FROM data','requested_conditions':conditions}}])
            r=GraphAnalysisRuntime(root,'owner','change-audit',model)
            raw=r.datasets.register(pd.DataFrame({'reading':[2.,4.,10.,20.]}),source='lab.observations',coverage='complete',predicate_known=True)
            r.select_dataset(raw.id);model.evaluation_dataset_id=raw.id
            before=stored_dataset_digest(r.datasets,raw.id)
            try:
                for prompt,expected in [('reading >= 10인 행의 건수를 알려줘',2.),('그중 reading 평균을 알려줘',15.)]:
                    if expected==15.:r.recovery.goal_interpreter.population_model=BadNullAudit()
                    with ExitStack() as stack:
                        for method in ['_next_local','_budget_local_rescue','_cached_chart_call']:
                            stack.enter_context(patch.object(r.recovery,method,return_value=None))
                        stack.enter_context(patch('core.analysis_agent.task_selection.select',return_value={
                            'mode':'execute','capabilities':['row_count' if expected==2. else 'calculation'],
                            'source_reference':'selected_dataset' if expected==2. else 'previous_analysis',
                            'source_mentions':[],'chart_kind':'','metadata_kind':'','chart_edit_fields':[],
                            'output_columns':[] if expected==2. else ['reading'],'group_column':'',
                            'scalar_operations':[] if expected==2. else ['AVG'],
                            'population_basis':'source_population','result_reference_quote':'','current_result_only':False,
                            'changes_filters':expected==2.,'filter_change_quote':prompt if expected==2. else ''}))
                        result=r.submit(prompt)
                    self.assertEqual(result['status'],'answered',result)
                    state=r.inspect()['recovery'];self.assertEqual(state['scope']['conditions'],conditions)
                    self.assertEqual(float(r.datasets.frames[state['evidence_ids'][-1]].iloc[0,0]),expected)
                    self.assertEqual(stored_dataset_digest(r.datasets,raw.id),before)
                    if expected==2.:r.close();r=GraphAnalysisRuntime(root,'owner','change-audit',model)
                events=[json.loads(line) for line in r.diagnostics.path.read_text().splitlines()]
                self.assertTrue(any(e['event']=='goal_population_contract_applied' and e.get('basis')=='unchanged_filters' for e in events))
            finally:r.close()
