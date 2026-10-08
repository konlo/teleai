"""First-pass journey defects: fallible audits, unfinished subjects, partial negation."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import tempfile
import unittest
from types import SimpleNamespace

import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage
from tests.test_agent_failure_batch import SubjectModel
from tests.test_llm_goal import goal, GoalModel
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.source_references import source_mentions


def catalog():
    return [{'table':source,'observed_at':datetime.now(timezone.utc).isoformat(),
             'columns':[{'name':column,'dtype':'int'}]}
            for source,column in [('lab.observations','reading'),('lab.other','other_value')]]


class JourneyRepairTests(unittest.TestCase):
    def test_different_repair_errors_can_progress_to_a_valid_combined_metadata_goal(self):
        fixed=goal('metadata',{'kind':'dtypes'})
        mode=deepcopy(fixed);mode['mode']='explain'
        duplicate=deepcopy(fixed);duplicate['tasks'].insert(0,{'capability':'metadata','options':{'kind':'columns'}})
        malformed=deepcopy(fixed);malformed['tasks'][0]['options']['limit']=10
        model=GoalModel(goals=[mode,malformed,fixed,fixed])
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','progressive-validation',model,sql_dialect='mysql',
                reference_context_loader=catalog)
            try:
                result=r.submit('observations의 이름과 DB타입 모두, 차트는 만들지 말아줘')
                self.assertEqual(result['status'],'answered',result)
                self.assertEqual(model.goal_calls,4)
                self.assertEqual(r.inspect()['recovery']['metadata_kind'],'dtypes')
                self.assertEqual(len(r.artifacts),0)
            finally:r.close()

    def test_redundant_names_and_types_tasks_preserve_both_outputs(self):
        from core.analysis_agent.goal_normalization import normalize
        plan=goal('metadata',{'kind':'dtypes'})
        plan['tasks'].insert(0,{'capability':'metadata','options':{'kind':'columns'}})
        model=GoalModel(goals=[plan])
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','redundant-metadata',model,sql_dialect='mysql',
                reference_context_loader=catalog)
            try:
                result=r.submit('observations 이름과 타입 모두, 그림은 없이')
                self.assertEqual(result['status'],'answered',result)
                self.assertEqual(model.goal_calls,2)
                self.assertEqual(r.inspect()['recovery']['metadata_evidence']['table'],'lab.observations')
                self.assertEqual(len(r.artifacts),0)
                unsafe=deepcopy(plan);unsafe['tasks'][0]['options']['kind']='numeric_columns'
                self.assertEqual(normalize(unsafe),unsafe)
                malformed=deepcopy(plan);malformed['tasks'][0]['unexpected']='must remain invalid'
                self.assertEqual(normalize(malformed),malformed)
            finally:r.close()

    def test_independent_population_read_replaces_anchored_equality(self):
        from core.analysis_agent.population_audit import audit
        plan=goal('chart',{'kind':'histogram'},columns=['reading'],
                  conditions=[{'column':'reading','op':'between','value':[12,19]},
                              {'column':'segment','op':'eq','value':'X'}])
        plan['source_reference']='previous_analysis'
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','population-audit',GoalModel(goals=[plan]),
                sql_dialect='mysql',reference_context_loader=catalog)
            try:
                class PopulationModel:
                    def invoke(self,messages):
                        payload=json.loads(messages[-1].content)
                        self.payload=payload
                        return AIMessage(content=json.dumps({'conditions':[
                            {'column':'reading','op':'between','value':[12,19]},
                            {'column':'segment','op':'in','value':['X','Y']}],'any_conditions':[]}))
                reader=PopulationModel();i=r.recovery.goal_interpreter;i.population_model=reader
                i.model_recovery=None
                current={'request_id':'new','request_text':'Keep range; include Y too',
                         'confirmed_analysis':{'required_sources':['lab.observations'],
                                               'scope':{'conditions':plan['conditions']}}}
                result=audit(i,current,{'tables':catalog()},plan)
                self.assertEqual(result['conditions'][1]['value'],['X','Y'])
                self.assertNotIn('proposed_goal',reader.payload)
                self.assertEqual(reader.payload['previous_verified_population']['conditions'],plan['conditions'])
            finally:r.close()

    def test_preview_output_limit_is_not_an_inherited_population(self):
        from core.analysis_agent.population_audit import applies
        plan=goal('row_preview',{'limit':10},sources=['lab.other'])
        current={'confirmed_analysis':{'required_sources':['lab.observations'],
                                      'scope':{'conditions':[{'column':'reading','op':'gt','value':12}]}}}
        self.assertFalse(applies(plan,current))
        plan['conditions']=[{'column':'other_value','op':'gt','value':12}]
        self.assertTrue(applies(plan,current))

    def test_population_audit_cannot_turn_row_count_into_an_unmentioned_id_filter(self):
        from core.analysis_agent.population_audit import audit
        plan=goal('row_preview',{'limit':10},sources=['lab.other'])
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','ungrounded-population',GoalModel(goals=[plan]),
                sql_dialect='mysql',reference_context_loader=catalog)
            try:
                reader=SimpleNamespace(invoke=lambda messages:AIMessage(content=json.dumps({
                    'conditions':[{'column':'other_value','op':'eq','value':'10'}],'any_conditions':[]})))
                i=r.recovery.goal_interpreter;i.population_model=reader;i.model_recovery=None
                current={'request_id':'new','request_text':'other row 10개만 표로 보여줘'}
                with self.assertRaisesRegex(ValueError,'ungrounded filter column'):
                    audit(i,current,{'tables':catalog()},plan)
            finally:r.close()

    def test_llm_value_list_finishes_after_distinct_receipt_without_planner_call(self):
        from dataclasses import asdict
        reference=[{'table':'lab.observations','observed_at':datetime.now(timezone.utc).isoformat(),
                    'columns':[{'name':'label','dtype':'longtext'}]}]
        model=GoalModel(goals=[goal('value_list',columns=['label'])])
        queries=[]
        def factory(store):
            def execute(envelope):
                queries.append(envelope['query'])
                info=store.register(pd.DataFrame({'label':['A','B']}),source='lab.observations',
                    query=envelope['query'],coverage='unknown',predicate_known=False)
                return {'status':'ready','dataset':asdict(info)}
            return execute
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','values-receipt',model,sql_dialect='mysql',
                connection_identity='fixture',remote_factory=factory,reference_context_loader=lambda:reference)
            try:
                result=r.submit('observations label에 들어있는 값 종류 모두 알려줘')
                self.assertEqual(result['status'],'answered',result)
                self.assertEqual(len(queries),1)
                self.assertEqual(r.inspect()['recovery']['model_calls'],2)
                self.assertEqual(r.inspect()['recovery']['value_list_evidence']['values'],['A','B'])
            finally:r.close()

    def test_inventory_uses_namespace_even_with_a_spurious_source_reference(self):
        plan=goal('table_list',{'catalog':'lab'},sources=[])
        plan['source_reference']='selected_dataset'
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','inventory-reference',GoalModel(goals=[plan]),
                sql_dialect='mysql',reference_context_loader=catalog)
            try:
                from core.analysis_agent.goal_contract import compile_goal, pending_state
                state=pending_state(HumanMessage(content='사용 가능한 테이블',id='new'),{},r.context)
                compiled=compile_goal(state,plan,r.context)
                self.assertEqual(compiled['required_sources'],['information_schema.tables'])
                self.assertEqual(compiled['catalog_discovery_source'],'information_schema.tables')
            finally:r.close()

    def test_literal_identity_is_not_a_substring_or_ambiguous_alias(self):
        context=SimpleNamespace(reference_context=catalog(),datasets=SimpleNamespace(metadata={}))
        self.assertEqual(source_mentions('other에는 어떤 항목이 있지?',context)[0]['source'],'lab.other')
        self.assertEqual(source_mentions('otherworld 컬럼',context),[])
        context.reference_context.append({'table':'another.other'})
        self.assertEqual(source_mentions('other 컬럼',context),[])
        self.assertEqual(source_mentions('lab.other 컬럼',context)[0]['source'],'lab.other')

    def test_wrong_audit_cannot_override_literal_named_table(self):
        first=goal('metadata',{'kind':'columns'})
        switched=goal('metadata',{'kind':'columns'},sources=['lab.other']);switched['source_reference']='explicit'
        model=SubjectModel(goals=[first,first,switched,switched])
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','audit-conflict',model,sql_dialect='mysql',reference_context_loader=catalog)
            try:
                r.recovery.goal_interpreter.reference_model=model
                self.assertEqual(r.submit('observations 컬럼')['status'],'answered')
                self.assertEqual(r.submit('other에는 어떤 컬럼이 있지?')['status'],'answered')
                self.assertEqual(r.inspect()['recovery']['metadata_evidence']['table'],'lab.other')
                self.assertEqual(model.reference_calls,1)
            finally:r.close()

    def test_failed_named_subject_can_be_inspected_without_claiming_a_result(self):
        first=goal('metadata',{'kind':'columns'})
        invalid={'not_a_goal':True}
        following=goal('metadata',{'kind':'dtypes'},sources=[]);following['source_reference']='previous_analysis'
        model=GoalModel(goals=[first,first,invalid,invalid,following,following])
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','unfinished-subject',model,sql_dialect='mysql',reference_context_loader=catalog)
            try:
                self.assertEqual(r.submit('observations 컬럼')['status'],'answered')
                self.assertEqual(r.submit('other 컬럼을 알려줘')['status'],'blocked')
                self.assertEqual(r.submit('그 테이블 타입만 보여줘')['status'],'answered')
                state=r.inspect()['recovery']
                self.assertEqual(state['metadata_evidence']['table'],'lab.other')
                self.assertEqual(state['requested_subject']['status'],'user_named_not_execution_evidence')
                self.assertEqual(len(r.datasets.metadata),0)
            finally:r.close()

    def test_prior_human_text_survives_assistant_interleaving(self):
        model=GoalModel(goals=[goal('metadata',{'kind':'columns'})])
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','history',model,sql_dialect='mysql',reference_context_loader=catalog)
            try:
                messages=[HumanMessage(content='other where reading is between 12 and 19',id='old'),
                          AIMessage(content='not completed'),HumanMessage(content='same population',id='new')]
                current={'request_id':'new','request_text':'same population'}
                payload=r.recovery.goal_interpreter.payload(current,messages)
                self.assertEqual(payload['conversation_text_not_evidence'][0]['text'],messages[0].content)
                self.assertEqual(payload['requested_previous_subject']['sources'],['lab.other'])
                self.assertFalse(payload['verified_previous'].get('status')=='complete')
            finally:r.close()

    def test_partial_negation_mode_conflict_is_repaired_not_task_dropped(self):
        fixed=goal('metadata',{'kind':'dtypes'})
        wrong=deepcopy(fixed);wrong['mode']='explain'
        model=GoalModel(goals=[wrong,fixed,fixed])
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','negation',model,sql_dialect='mysql',reference_context_loader=catalog)
            try:
                result=r.submit('observations 컬럼 이름과 타입 모두 알려줘. 차트는 만들지 마')
                self.assertEqual(result['status'],'answered',result)
                self.assertEqual(r.inspect()['recovery']['metadata_kind'],'dtypes')
                self.assertFalse(r.inspect()['recovery']['chart'])
                self.assertEqual(len(r.artifacts),0)
            finally:r.close()


if __name__=='__main__':unittest.main()
