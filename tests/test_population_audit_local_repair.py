"""A scope reviewer repairs its own invented filters without regenerating a goal."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from core.analysis_agent.population_audit import audit
from core.analysis_agent.runtime import GraphAnalysisRuntime
from tests.test_llm_goal import GoalModel, goal
from utils.analysis_datasets import stored_dataset_digest
from utils.analysis_image_validation import validate_chart_image

FIXTURE=json.loads((Path(__file__).parent/'fixtures/population_histogram_followup.json').read_text())


def delta(column=None, quote=None):
    return {'change':'modify' if column else 'keep','evidence_quote':quote or '',
            'conditions':[{'column':column,'op':'le','value':10}] if column else [],
            'any_conditions':[],'removed_columns':[],'included_values':[]}


class PopulationAuditLocalRepairTests(unittest.TestCase):
    def interpreter(self,replies):
        reader=Mock(spec=['invoke'])
        reader.invoke.side_effect=[AIMessage(content=v if isinstance(v,str) else json.dumps(v)) for v in replies]
        return SimpleNamespace(population_model=reader,model_recovery=None,diagnostics=Mock(),
            budget=SimpleNamespace(wrap_model_call=lambda req,fn:fn(req)))

    def context(self):
        return {'request_id':'histogram','request_text':FIXTURE['histogram_prompt'],
                'confirmed_analysis':{'status':'complete','required_sources':[FIXTURE['source']],
                    'scope':{'conditions':[],'any_conditions':[]},
                    'table_preview_evidence':{'rows':10,'source':FIXTURE['source']}}}

    def plan(self):
        return goal('chart',{'kind':'histogram'},sources=[FIXTURE['source']],columns=[FIXTURE['column']])

    def test_invented_filter_gets_local_repair_before_scope_is_accepted(self):
        bad=delta(FIXTURE['unexpected_filter_column'],FIXTURE['histogram_prompt'])
        interpreter=self.interpreter([bad,delta()])
        original=self.plan()
        result=audit(interpreter,self.context(),{},original)
        self.assertEqual(result['conditions'],[])
        self.assertEqual(interpreter.population_model.invoke.call_count,2)
        repair=interpreter.population_model.invoke.call_args.args[0][-1].content
        self.assertIn('ungrounded filter column',repair)
        self.assertEqual(original,self.plan())

    def test_invalid_json_is_repaired_by_the_same_role(self):
        for malformed in ['not JSON','[]','{"change":"modify"}']:
            with self.subTest(response=malformed):
                interpreter=self.interpreter([malformed,delta()])
                self.assertEqual(audit(interpreter,self.context(),{},self.plan())['conditions'],[])
                self.assertEqual(interpreter.population_model.invoke.call_count,2)

    def test_failed_audit_retains_only_an_unchanged_confirmed_population(self):
        conditions=[{'column':FIXTURE['column'],'op':'ge','value':2}]
        current=self.context();current['confirmed_analysis']['scope']['conditions']=conditions
        plan=self.plan();plan['conditions']=conditions
        bad=delta(FIXTURE['unexpected_filter_column'],current['request_text'])
        interpreter=self.interpreter([bad,bad])
        result=audit(interpreter,current,{},plan)
        self.assertEqual(result['conditions'],conditions)
        self.assertEqual(interpreter.population_model.invoke.call_count,2)

    def test_changed_population_is_not_silently_replaced_with_old_scope(self):
        plan=self.plan();plan['conditions']=[{'column':FIXTURE['column'],'op':'ge','value':2}]
        bad=delta(FIXTURE['unexpected_filter_column'],FIXTURE['histogram_prompt'])
        interpreter=self.interpreter([bad,bad])
        with self.assertRaisesRegex(ValueError,'ungrounded filter column'):
            audit(interpreter,self.context(),{},plan)
        self.assertEqual(interpreter.population_model.invoke.call_count,2)

    def test_new_source_cannot_fall_back_to_an_unrelated_population(self):
        bad=delta(FIXTURE['unexpected_filter_column'],FIXTURE['histogram_prompt'])
        interpreter=self.interpreter([bad,bad])
        current=self.context();current['confirmed_analysis']['required_sources']=['lab.other']
        with self.assertRaisesRegex(ValueError,'ungrounded filter column'):
            audit(interpreter,current,{},self.plan())
        self.assertEqual(interpreter.population_model.invoke.call_count,2)

    def test_four_turn_fixture_creates_histogram_of_source_not_ten_row_preview(self):
        source,column=FIXTURE['source'],FIXTURE['column']
        commands={'table리스트확인':goal('table_list',{'catalog':'','schema':''},sources=[]),
                  '테이블의 컬럼 확인':goal('metadata',{'kind':'columns'},sources=[source]),
                  'row 10개 확인':goal('row_preview',{'limit':10},sources=[source]),
                  FIXTURE['histogram_prompt']:self.plan()}
        class JourneyModel(GoalModel):
            def _generate(self,messages,**kwargs):
                if any('goal_schema_v1' in str(m.content) for m in messages):
                    self.goal_calls+=1
                    payload=json.loads(next(m.content for m in messages if isinstance(m,HumanMessage)))
                    return ChatResult(generations=[ChatGeneration(message=AIMessage(
                        content=json.dumps(commands[payload['request']])))])
                return super()._generate(messages,**kwargs)
        queries=[]
        def factory(datasets):
            def execute(envelope):
                queries.append(envelope['query'])
                self.assertIn('information_schema',envelope['query'])
                frame=pd.DataFrame({'TABLE_SCHEMA':['lab'],'TABLE_NAME':['events'],'TABLE_TYPE':['BASE TABLE']})
                info=datasets.register(frame,source=envelope['source'],query=envelope['query'],
                    coverage='unknown',predicate_known=False)
                return {'status':'ready','dataset':asdict(info)}
            return execute
        reference=[{'table':source,'observed_at':datetime.now(timezone.utc).isoformat(),
                    'columns':[{'name':column,'dtype':'bigint'}]}]
        with tempfile.TemporaryDirectory() as root:
            model=JourneyModel()
            runtime=GraphAnalysisRuntime(root,'owner','four-turn',model,sql_dialect='mysql',
                connection_identity='fixture',remote_factory=factory,source_namespace='lab',
                reference_context_loader=lambda:reference)
            try:
                raw=runtime.datasets.register(pd.DataFrame({column:FIXTURE['values']}),source=source,
                                               coverage='complete',predicate_known=True)
                runtime.select_dataset(raw.id)
                before=stored_dataset_digest(runtime.datasets,raw.id)
                for command in commands:
                    if command==FIXTURE['histogram_prompt']:
                        calls_before_histogram=model.goal_calls
                        interpreter=self.interpreter([
                            delta(FIXTURE['unexpected_filter_column'],command),delta()])
                        runtime.recovery.goal_interpreter.population_model=interpreter.population_model
                    result=runtime.submit(command)
                    self.assertEqual(result['status'],'answered',(command,result,
                        runtime.diagnostics.path.read_text() if result['status']!='answered' else ''))
                    if command=='row 10개 확인':
                        self.assertEqual(runtime.inspect()['recovery']['table_preview_evidence']['rows'],10)
                state=runtime.inspect()['recovery']
                self.assertEqual(state['required_columns'],[column])
                self.assertEqual(state['scope']['conditions'],[])
                card=runtime.artifacts[state['artifact_ids'][-1]]
                self.assertEqual(card.kind,'histogram')
                frame=runtime.datasets.frames[card.dataset_id]
                expected=pd.Series(FIXTURE['values']).value_counts().sort_index()
                actual=frame.set_index(column)['__frequency'].sort_index()
                self.assertEqual(actual.to_dict(),expected.to_dict())
                self.assertEqual(int(actual.sum()),len(FIXTURE['values']))
                validate_chart_image(card.image)
                self.assertEqual(stored_dataset_digest(runtime.datasets,raw.id),before)
                self.assertEqual(runtime.context.selected_dataset_id,raw.id)
                self.assertEqual(len(queries),1)
                self.assertEqual(interpreter.population_model.invoke.call_count,2)
                self.assertEqual(model.goal_calls-calls_before_histogram,2)
            finally:runtime.close()
