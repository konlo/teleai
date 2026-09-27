"""Numeric preparation preserves all rows, source values and exact null policy."""
import tempfile
import unittest
import pandas as pd
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.tools import local_tools
from scripts.evaluate_analysis_statistics import ForbiddenModel
from utils.analysis_datasets import stored_dataset_digest


from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration,ChatResult
from uuid import uuid4

class PrepareOnlyModel(BaseChatModel):
    dataset_id: str = ''
    calls: int = 0
    @property
    def _llm_type(self):return 'prepare-once-then-local-continuation'
    def bind_tools(self,tools,**kwargs):return self
    def _generate(self,messages,**kwargs):
        self.calls+=1
        if self.calls>1:raise AssertionError('Prepared data must complete locally')
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content='',tool_calls=[{
            'name':'prepare_numeric_dataset','id':str(uuid4()),
            'args':{'dataset_id':self.dataset_id,'columns':['sensor reading'],'missing_values':['missing']}}]))])

class NumericPreparationTests(unittest.TestCase):
    def test_large_projection_null_policy_lineage_reuse_and_reopen(self):
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'tests','numeric',ForbiddenModel())
            try:
                frame=pd.DataFrame({'measure':['1.25','missing','3.75']*30000,'keep':['text']*90000})
                raw=r.datasets.register(frame,source='fixture.changed_schema',coverage='complete',snapshot='snapshot')
                digest=stored_dataset_digest(r.datasets,raw.id)
                tool=next(t for t in local_tools(r.context) if t.name=='prepare_numeric_dataset')
                denied=tool.invoke({'dataset_id':raw.id,'columns':['measure']})
                self.assertEqual(denied['error_code'],'numeric_conversion_unresolved')
                self.assertEqual(len(r.datasets.metadata),1)
                args={'dataset_id':raw.id,'columns':['measure'],'missing_values':['missing']}
                result=tool.invoke(args);child=result['dataset']['id']
                self.assertEqual(result['status'],'ready')
                self.assertEqual(result['dataset']['rows'],90000)
                self.assertEqual(result['dataset']['parent_id'],raw.id)
                self.assertEqual(result['dataset']['root_id'],raw.id)
                self.assertEqual(result['conversion'][0]['declared_missing'],30000)
                self.assertAlmostEqual(r.datasets.frames[child]['measure'].mean(),2.5)
                self.assertEqual(stored_dataset_digest(r.datasets,raw.id),digest)
                self.assertEqual(tool.invoke(args)['dataset']['id'],child)
            finally:r.close()
            r=GraphAnalysisRuntime(root,'tests','numeric',ForbiddenModel())
            try:
                self.assertEqual(r.datasets.metadata[child].rows,90000)
                self.assertEqual(stored_dataset_digest(r.datasets,raw.id),digest)
            finally:r.close()

    def test_quoted_measure_used_for_filter_and_average_then_reset_chart(self):
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'tests','quoted',ForbiddenModel())
            try:
                raw=r.datasets.register(pd.DataFrame({'sensor reading':['1','2','3','4','missing']}),
                    source='fixture.new_source',coverage='unknown')
                tool=next(t for t in local_tools(r.context) if t.name=='prepare_numeric_dataset')
                prepared=tool.invoke({'dataset_id':raw.id,'columns':['sensor reading'],'missing_values':['missing']})
                r.select_dataset(prepared['dataset']['id'])
                for prompt,expected in [
                    ('현재 로딩된 표본에서 `sensor reading` >= 3인 행의 `sensor reading` 평균을 알려줘',3.5),
                    ('현재 로딩된 원본 표본에서 `sensor reading` 평균과 히스토그램을 보여줘. 이전 `sensor reading` 조건은 적용하지 말고 보유 데이터만 사용해.',2.5)]:
                    outcome=r.submit(prompt)
                    self.assertEqual(outcome['status'],'answered',outcome)
                    state=r.inspect()['recovery']
                    self.assertAlmostEqual(float(r.datasets.frames[state['evidence_ids'][-1]].iloc[0,0]),expected)
                self.assertEqual(len(state['artifact_ids']),1)
                self.assertEqual(r.datasets.frames[raw.id].iloc[-1,0],'missing')
            finally:r.close()

    def test_model_preparation_continues_both_goals_with_one_model_call(self):
        with tempfile.TemporaryDirectory() as root:
            model=PrepareOnlyModel();r=GraphAnalysisRuntime(root,'tests','auto',model)
            try:
                raw=r.datasets.register(pd.DataFrame({'sensor reading':['1','2','3','missing']}),
                    source='fixture.unseen',coverage='unknown')
                r.select_dataset(raw.id);model.dataset_id=raw.id
                outcome=r.submit('현재 로딩된 표본에서 `sensor reading` 평균과 히스토그램을 보여줘. "missing" 문자열만 결측값으로 처리해.')
                self.assertEqual(outcome['status'],'answered',outcome)
                state=r.inspect()['recovery']
                self.assertEqual(model.calls,1)
                self.assertIn('명시한 문자열 결측 1개',outcome['text'])
                self.assertEqual(len(state['artifact_ids']),1)
                self.assertEqual(float(r.datasets.frames[state['evidence_ids'][-1]].iloc[0,0]),2.)
                self.assertEqual(r.context.selected_dataset_id,raw.id)
                self.assertEqual(r.datasets.frames[raw.id].iloc[-1,0],'missing')
                self.assertFalse(r.recovery._proposed_scope_valid({'name':'prepare_numeric_dataset',
                    'args':{'missing_values':['invented']}},state))
                self.assertFalse(r.recovery._proposed_scope_valid({'name':'prepare_numeric_dataset',
                    'args':{'missing_values':[3]}},state))
                for text in ['"missing" 문자열은 결측 처리하지 말고 그대로 유지해.',
                             '"missing" 문자열의 건수를 알려줘.',
                             'Do not treat "missing" as null.']:
                    self.assertFalse(r.recovery._numeric_missing_policy_valid(
                        {'missing_values':['missing']},{'request_text':text}))
            finally:r.close()

    def test_nonfinite_precision_boolean_and_unknown_tokens_not_dropped(self):
        with tempfile.TemporaryDirectory() as root:
            r=GraphAnalysisRuntime(root,'tests','numeric',ForbiddenModel())
            try:
                tool=next(t for t in local_tools(r.context) if t.name=='prepare_numeric_dataset')
                for values,code in [(['1','inf'],'numeric_conversion_unresolved'),
                                     (['9007199254740993'],'numeric_precision_risk'),
                                     (['9007199254740993',None],'numeric_precision_risk'),
                                     (['1','unexpected'],'numeric_conversion_unresolved'),
                                     ([True,False],'invalid_tool_input')]:
                    raw=r.datasets.register(pd.DataFrame({'different field':values}),source='other.source')
                    before=set(r.datasets.metadata)
                    result=tool.invoke({'dataset_id':raw.id,'columns':['different field'],'missing_values':['NA']})
                    self.assertEqual(result['error_code'],code)
                    self.assertEqual(set(r.datasets.metadata),before)
            finally:r.close()

if __name__=='__main__':unittest.main()
