"""Reproduced boundary failures; scripted models test contracts, not language scores."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from tests.test_actual_agent_evaluation import EvaluationModel
from tests.test_llm_goal import goal, GoalModel
from core.analysis_agent.approvals import ApprovalLedger, QueryRejected
from core.analysis_agent.dtypes import family
from core.analysis_agent.goal_contract import response_schema, validate_goal
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.sql_preflight import dialect_error
from utils.analysis_datasets import stored_dataset_digest


class RepairModel(GoalModel):
    proposals: int = 0

    def _generate(self, messages, **kwargs):
        if any('goal_schema_v1' in str(m.content) for m in messages):
            return super()._generate(messages, **kwargs)
        self.proposals+=1
        alias='initial_total' if self.proposals==1 else 'total'
        call={'name':'query_databricks','id':'repair-'+str(self.proposals),
              'args':{'source':'lab.observations',
                      'query':f'SELECT SUM(reading) AS {alias} FROM lab.observations',
                      'reason':'Compute the requested sum without changing its scope.'}}
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content='',tool_calls=[call]))])


class SubjectModel(GoalModel):
    reference_calls:int=0

    def _generate(self,messages,**kwargs):
        if any('subject_reference_v1' in str(m.content) for m in messages):
            self.reference_calls+=1
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content='{"reference":"previous_analysis"}'))])
        return super()._generate(messages,**kwargs)


class FailureBatchTests(unittest.TestCase):
    def test_independent_reference_audit_repairs_wrong_original_before_metadata_execution(self):
        schemas=[{'table':source,'observed_at':datetime.now(timezone.utc).isoformat(),
                  'columns':[{'name':column,'dtype':'int'}]}
                 for source,column in [('lab.observations','reading'),('lab.other','other_value')]]
        first=goal('metadata',{'kind':'columns'});first['sources']=['lab.other']
        wrong=goal('metadata',{'kind':'columns'});wrong['source_reference']='selected_dataset'
        fixed=goal('metadata',{'kind':'columns'});fixed['source_reference']='previous_analysis';fixed['sources']=[]
        model=SubjectModel(goals=[first,first,wrong,fixed,fixed])
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','subject-audit',model,sql_dialect='mysql',reference_context_loader=lambda:schemas)
            try:
                r.recovery.goal_interpreter.reference_model=model
                raw=r.datasets.register(pd.DataFrame({'reading':[1,2]}),source='lab.observations',coverage='complete',predicate_known=True)
                r.select_dataset(raw.id)
                self.assertEqual(r.submit('other 컬럼을 보여줘')['status'],'answered')
                self.assertEqual(r.submit('컬럼 다시 보여줘')['status'],'answered')
                self.assertEqual(r.inspect()['recovery']['metadata_evidence']['table'],'lab.other')
                self.assertEqual(model.reference_calls,1)
                self.assertEqual(r.context.selected_dataset_id,raw.id)
            finally:r.close()

    def test_followup_reference_cannot_revert_to_an_unrelated_selected_original(self):
        schemas=[{'table':source,'observed_at':datetime.now(timezone.utc).isoformat(),
                  'columns':[{'name':column,'dtype':'int'}]}
                 for source,column in [('lab.observations','reading'),('lab.other','other_value')]]
        first=goal('metadata',{'kind':'columns'});first['sources']=['lab.other']
        wrong=goal('metadata',{'kind':'columns'});wrong['source_reference']='previous_analysis'
        fixed=goal('metadata',{'kind':'columns'});fixed['source_reference']='previous_analysis';fixed['sources']=[]
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','subject-reference',GoalModel(goals=[first,first,wrong,wrong,fixed,fixed]),
                sql_dialect='mysql',reference_context_loader=lambda:schemas)
            try:
                raw=r.datasets.register(pd.DataFrame({'reading':[1,2]}),source='lab.observations',coverage='complete',predicate_known=True)
                r.select_dataset(raw.id);digest=stored_dataset_digest(r.datasets,raw.id)
                self.assertEqual(r.submit('other 컬럼을 보여줘')['status'],'answered')
                self.assertEqual(r.submit('컬럼 다시 보여줘')['status'],'blocked')
                self.assertIn('verified subject',r.inspect()['recovery']['goal_contract_error'])
                self.assertEqual(r.submit('같은 테이블 컬럼을 보여줘')['status'],'answered')
                self.assertEqual(r.inspect()['recovery']['metadata_evidence']['table'],'lab.other')
                self.assertEqual(r.context.selected_dataset_id,raw.id)
                self.assertEqual(stored_dataset_digest(r.datasets,raw.id),digest)
            finally:r.close()

    def test_grouped_histogram_requires_an_observed_numeric_measure_before_sql(self):
        from core.analysis_agent.goal_contract import compile_goal, pending_state
        from langchain_core.messages import HumanMessage
        schema=[{'table':'lab.observations','observed_at':datetime.now(timezone.utc).isoformat(),
                 'columns':[{'name':'reading','dtype':'longtext'},{'name':'cohort','dtype':'int'}]}]
        plan=goal('chart',{'kind':'histogram','axes':{'x':'reading'},'category':'cohort'},columns=['reading','cohort'])
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','bad-group',GoalModel(goals=[plan]),sql_dialect='mysql',reference_context_loader=lambda:schema)
            try:
                state=pending_state(HumanMessage(content='분포를 그려줘',id='new'),{},r.context)
                with self.assertRaisesRegex(ValueError,'requires a numeric x measure'):
                    compile_goal(state,plan,r.context)
                self.assertEqual(len(r.datasets.metadata),0)
            finally:r.close()
        for kind in ('bar','scatter'):
            with self.subTest(kind=kind),self.assertRaisesRegex(ValueError,'category groups a numeric histogram only'):
                validate_goal(goal('chart',{'kind':kind,'category':'cohort'},columns=['reading','cohort']))

    def test_filter_column_is_not_an_implicit_numeric_bar_axis(self):
        data=pd.DataFrame({'reading':range(20),'cohort':['A','B']*10})
        plan=goal('chart',{'kind':'bar','axes':{'x':'cohort'}},columns=['cohort','reading'],
            conditions=[{'column':'reading','op':'between','value':[3,8]}])
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','plot-roles',GoalModel(goals=[plan]),sql_dialect='mysql')
            try:
                raw=r.datasets.register(data,source='lab.observations',coverage='complete',predicate_known=True)
                r.select_dataset(raw.id)
                result=r.submit('범위 안의 cohort 빈도를 그려줘')
                self.assertEqual(result['status'],'answered',result)
                state=r.inspect()['recovery'];chart=r.artifacts[state['artifact_ids'][-1]]
                self.assertEqual(state['required_columns'],['cohort'])
                self.assertEqual(chart.render_spec['total_count'],6)
                self.assertEqual(dict(zip(chart.render_spec['labels'],chart.render_spec['counts'])),{'A':3,'B':3})
            finally:r.close()
    def test_axis_adjustment_cannot_create_a_new_population(self):
        first=goal('chart',{'kind':'histogram'},columns=['reading'])
        adjustment=goal('chart_adjust',{'y_max':10},columns=['reading'])
        adjustment['sources']=[]
        adjustment['columns']=[]
        invalid=goal('chart',{'kind':'bar','category':'reading'},columns=['reading'])
        schema=[{'table':'lab.observations','observed_at':datetime.now(timezone.utc).isoformat(),
                 'columns':[{'name':'reading','dtype':'longtext'}]}]
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','chart-adjust',GoalModel(goals=[first,first,invalid,invalid,adjustment,adjustment,first,first]),
                sql_dialect='mysql',reference_context_loader=lambda:schema)
            try:
                raw=r.datasets.register(pd.DataFrame({'reading':['A']*12+['B']*3}),
                    source='lab.observations',coverage='complete',predicate_known=True)
                r.select_dataset(raw.id)
                self.assertEqual(r.submit('reading 분포를 그려줘')['status'],'answered')
                before=r.artifacts[r.inspect()['recovery']['artifact_ids'][-1]]
                self.assertEqual(r.submit('미지원 표현을 시험한다')['status'],'blocked')
                self.assertEqual(r.inspect()['recovery']['confirmed_analysis']['artifact_ids'],[before.id])
                result=r.submit('y축을 10으로 바꿔줘')
                self.assertEqual(result['status'],'answered',result)
                after=r.artifacts[r.inspect()['recovery']['artifact_ids'][-1]]
                self.assertEqual(after.dataset_id,before.dataset_id)
                self.assertEqual(after.render_spec['counts'],before.render_spec['counts'])
                self.assertEqual(after.render_spec['y_limits'],[0.0,10.0])
                self.assertNotEqual(after.image,before.image)
                self.assertEqual(r.context.selected_dataset_id,raw.id)
                self.assertEqual(r.submit('다시 reading 전체 분포를 그려줘')['status'],'answered')
                restored=r.artifacts[r.inspect()['recovery']['artifact_ids'][-1]]
                self.assertEqual(restored.dataset_id,before.dataset_id)
                self.assertEqual(restored.render_spec['counts'],before.render_spec['counts'])
                self.assertFalse(restored.render_spec.get('y_limits'))
                self.assertEqual(after.render_spec['y_limits'],[0.0,10.0])
            finally:r.close()

    def test_distribution_uses_complete_frequency_across_multiple_raw_previews(self):
        source='lab.observations'
        sql='SELECT `reading`, COUNT(*) AS `__frequency` FROM `lab`.`observations` WHERE `reading` IS NOT NULL GROUP BY `reading`'
        schema=[{'table':source,'observed_at':datetime.now(timezone.utc).isoformat(),
                 'columns':[{'name':'reading','dtype':'longtext'}]}]
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','counts',GoalModel(goals=[goal('chart',{'kind':'histogram'},columns=['reading'])]),
                sql_dialect='mysql',reference_context_loader=lambda:schema)
            try:
                previews=[r.datasets.register(pd.DataFrame({'reading':['A','B']}),source=source,
                    coverage='sampled',predicate_known=False,query=f'SELECT * FROM lab.observations LIMIT {limit}')
                    for limit in (2,10)]
                exact=r.datasets.register(pd.DataFrame({'reading':['A','B'],'__frequency':[100000,400000]}),
                    source=source,coverage='complete',predicate_known=True,grain='aggregate',query=sql,aggregation=sql)
                r.select_dataset(previews[0].id)
                before={p.id:stored_dataset_digest(r.datasets,p.id) for p in previews}
                result=r.submit('reading 컬럼의 전체 분포를 이미지로 보여줘')
                self.assertEqual(result['status'],'answered',result)
                state=r.inspect()['recovery'];chart=r.artifacts[state['artifact_ids'][-1]]
                self.assertEqual(chart.dataset_id,exact.id)
                self.assertEqual(chart.kind,'bar')
                self.assertEqual(chart.render_spec['total_count'],500000)
                self.assertEqual(dict(zip(chart.render_spec['labels'],chart.render_spec['counts'])),{'A':100000,'B':400000})
                self.assertTrue(chart.image.startswith(b'\x89PNG'))
                for ident,digest in before.items():self.assertEqual(stored_dataset_digest(r.datasets,ident),digest)
            finally:r.close()

    def test_weighted_histogram_respects_requested_bins_and_population(self):
        from utils.analysis_charts import histogram_from_counts
        from utils.analysis_datasets import DatasetStore
        store=DatasetStore()
        info=store.register(pd.DataFrame({'reading':[0,1,2,3],'__frequency':[10,20,30,40]}),
            source='lab.sensor',coverage='complete',grain='aggregate',predicate_known=True,
            query='SELECT reading, COUNT(*) AS __frequency FROM lab.sensor GROUP BY reading',
            aggregation='SELECT reading, COUNT(*) AS __frequency FROM lab.sensor GROUP BY reading')
        card=histogram_from_counts(store,info.id,'reading','__frequency',bins=2)
        self.assertEqual(card.render_spec['bin_counts'],[30.0,70.0])
        self.assertEqual(card.render_spec['total_count'],100)
        self.assertEqual(card.render_spec['bins'],2)
        self.assertTrue(card.image.startswith(b'\x89PNG'))

    def test_type_names_cannot_collide_with_numeric_substrings(self):
        for dtype in ('longtext','mediumtext','tinytext','text','varchar(100)','string','object','boolean'):
            with self.subTest(dtype=dtype):self.assertEqual(family(dtype),'categorical')
        for dtype in ('Int64','uint32','float64','bigint unsigned','decimal(20,3)','double precision'):
            with self.subTest(dtype=dtype):self.assertEqual(family(dtype),'numeric')
        for dtype in ('POINT','interval','unknown'):
            self.assertNotEqual(family(dtype),'numeric')

    def test_provider_grammar_and_validator_share_capability_option_keys(self):
        from core.analysis_agent.goal_schema import OPTIONS
        branches=response_schema()['properties']['tasks']['items']['anyOf']
        self.assertEqual(len(branches),len(OPTIONS))
        for branch in branches:
            props=branch['properties'];name=props['capability']['const']
            options=props['options'].get('anyOf',[props['options']])
            for option in options:
                self.assertTrue(set(option['properties']).issubset(OPTIONS[name]))
                if name=='chart' and option['properties']['kind'].get('const')=='bar':
                    self.assertNotIn('category',option['properties'])
                    self.assertNotIn('bins',option['properties'])
                self.assertFalse(option['additionalProperties'])
        with self.assertRaisesRegex(ValueError,r'chart.options.*title.*allowed keys'):
            validate_goal(goal('chart',{'kind':'histogram','title':'unexpected'},columns=['reading']))

    def test_range_is_atomic_and_tautological_or_cannot_execute(self):
        bad=goal('calculation',{'operations':['SUM']},columns=['reading'])
        bad['any_conditions']=[{'column':'reading','op':'ge','value':3},
                               {'column':'reading','op':'le','value':8}]
        with self.assertRaisesRegex(ValueError,'not a range'):validate_goal(bad)
        good=goal('calculation',{'operations':['SUM']},columns=['reading'],conditions=[
            {'column':'reading','op':'between','value':[3,8]}])
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','atomic',GoalModel(goals=[good]),sql_dialect='mysql')
            try:
                raw=r.datasets.register(pd.DataFrame({'reading':range(20)}),source='lab.observations',coverage='complete',predicate_known=True)
                r.select_dataset(raw.id)
                result=r.submit('reading 3~8의 합계')
                self.assertEqual(result['status'],'answered',result)
                self.assertEqual(r.datasets.frames[r.inspect()['recovery']['evidence_ids'][0]].iloc[0,0],33)
            finally:r.close()

    def test_semantic_review_can_replace_wrong_obligation_before_execution(self):
        data=pd.DataFrame({'reading':['A','B','A']})
        model=GoalModel(goals=[goal('metadata',{'kind':'categorical_columns'},columns=['reading']),
                              goal('value_list',columns=['reading'])])
        schema=[{'table':'lab.observations','observed_at':datetime.now(timezone.utc).isoformat(),
                 'columns':[{'name':'reading','dtype':'longtext'}]}]
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','semantic',model,sql_dialect='mysql',reference_context_loader=lambda:schema)
            try:
                raw=r.datasets.register(data,source='lab.observations',coverage='complete',predicate_known=True)
                r.select_dataset(raw.id)
                result=r.submit('이 컬럼에 어떤 값들이 있는지 보여줘')
                self.assertEqual(result['status'],'answered',result)
                current=r.inspect()['recovery']
                self.assertTrue(current['value_list_requested'])
                self.assertIsNone(current['metadata_kind'])
                self.assertEqual(model.goal_calls,2)
                self.assertIn('A',result['text']);self.assertIn('B',result['text'])
            finally:r.close()

    def test_semantic_review_repairs_missing_conjunct_and_range_before_any_chart(self):
        data=pd.DataFrame({'reading':range(20),'cohort':['A','B']*10})
        wrong=goal('chart',{'kind':'histogram'},columns=['reading'],conditions=[
            {'column':'reading','op':'in','value':[3,8]}])
        correct=goal('chart',{'kind':'histogram'},columns=['reading'],conditions=[
            {'column':'reading','op':'ge','value':3},{'column':'reading','op':'le','value':8},
            {'column':'cohort','op':'eq','value':'B'}])
        schema=[{'table':'lab.observations','observed_at':datetime.now(timezone.utc).isoformat(),
                 'columns':[{'name':c,'dtype':str(data[c].dtype)} for c in data]}]
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','conjuncts',GoalModel(goals=[wrong,correct]),
                sql_dialect='mysql',reference_context_loader=lambda:schema)
            try:
                raw=r.datasets.register(data,source='lab.observations',coverage='complete',predicate_known=True)
                r.select_dataset(raw.id)
                result=r.submit('cohort가 B이고 reading이 3~8인 범위를 시각화해줘')
                self.assertEqual(result['status'],'answered',result)
                state=r.inspect()['recovery'];chart=r.artifacts[state['artifact_ids'][-1]]
                self.assertEqual(chart.render_spec['total_count'],3)
                self.assertEqual(len(r.artifacts),1)
                self.assertEqual(state['goal']['conditions'],correct['conditions'])
            finally:r.close()

    def test_mysql_dialect_rejection_is_not_imposed_on_databricks(self):
        sql='SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY reading) FROM lab.observations'
        self.assertEqual(dialect_error(sql,'mysql')['error_code'],'unsupported_sql_dialect')
        self.assertIsNone(dialect_error(sql,'databricks'))
        self.assertIsNone(dialect_error('SELECT AVG(reading) FROM lab.observations','mysql'))

    def test_repeated_server_rejections_exhaust_repair_without_third_submission(self):
        executed=[]
        def execute(envelope):
            executed.append(envelope['query']);raise QueryRejected(1064,'42000')
        schema=[{'table':'lab.observations','observed_at':datetime.now(timezone.utc).isoformat(),
                 'columns':[{'name':'reading','dtype':'bigint'}]}]
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'owner','bounded-repair',
                RepairModel(goals=[goal('calculation',{'operations':['SUM']},columns=['reading'])]),
                sql_dialect='mysql',reference_context_loader=lambda:schema,
                remote_factory=lambda _:execute,connection_identity='fixture')
            try:
                result=r.submit('observations의 reading 합계를 보여줘')
                self.assertNotEqual(result['status'],'answered')
                self.assertEqual(len(executed),2)
                self.assertEqual(r.inspect()['recovery']['stop_reason'],'sql_repair_budget')
                self.assertEqual(r.ledger.uncertain(),[])
            finally:r.close()

    def test_server_rejection_repairs_new_sql_and_keeps_original_and_receipt_boundaries(self):
        executed=[]
        def factory(store):
            def execute(envelope):
                executed.append(envelope['query'])
                if len(executed)==1:raise QueryRejected(1064,'42000')
                info=store.register(pd.DataFrame({'total':[190]}),source='lab.observations',
                    coverage='complete',predicate_known=True,grain='aggregate',
                    aggregation=envelope['query'],query=envelope['query'])
                return {'status':'ready','dataset':asdict(info)}
            return execute
        schema=[{'table':'lab.observations','observed_at':datetime.now(timezone.utc).isoformat(),
                 'columns':[{'name':'reading','dtype':'bigint'}]}]
        with tempfile.TemporaryDirectory() as root:
            model=RepairModel(goals=[goal('calculation',{'operations':['SUM']},columns=['reading'])])
            r=GraphAnalysisRuntime(root,'owner','repair',model,sql_dialect='mysql',
                reference_context_loader=lambda:schema,remote_factory=factory,connection_identity='fixture')
            try:
                raw=r.datasets.register(pd.DataFrame({'reading':[5,7]}),source='lab.other',coverage='complete',predicate_known=True)
                r.select_dataset(raw.id);digest=stored_dataset_digest(r.datasets,raw.id)
                result=r.submit('observations의 reading 합계만 계산해줘')
                self.assertEqual(result['status'],'answered',result)
                self.assertEqual(len(executed),2);self.assertNotEqual(*executed)
                state=r.inspect()['recovery']
                self.assertEqual(len(state['sql_rejected_call_ids']),1)
                self.assertEqual(r.ledger.get(state['sql_rejected_call_ids'][0])['status'],'failed')
                self.assertEqual(r.ledger.uncertain(),[])
                self.assertIn('190',result['text'])
                self.assertEqual(r.context.selected_dataset_id,raw.id)
                self.assertEqual(stored_dataset_digest(r.datasets,raw.id),digest)
            finally:r.close()

    def test_unknown_ledger_cannot_clear_from_unrelated_fingerprint_or_disconnect(self):
        with tempfile.TemporaryDirectory() as root:
            ledger=ApprovalLedger(Path(root)/'ledger.sqlite',dialect='mysql')
            en=ledger.envelope('lab.events','SELECT COUNT(*) FROM lab.events','count','fixture')
            ledger.propose('call',en);ledger.authorize_automatic('call',en)
            def disconnect(_):raise ConnectionError('lost response')
            with self.assertRaises(ConnectionError):ledger.execute('call',en,disconnect)
            self.assertFalse(ledger.confirm_sql_rejection('call',{**en,'query':'SELECT * FROM lab.events'},1064))
            self.assertFalse(ledger.confirm_sql_rejection('call',en,2013))
            self.assertTrue(ledger.uncertain())
            self.assertTrue(ledger.confirm_sql_rejection('call',en,1064))
            self.assertEqual(ledger.uncertain(),[])
