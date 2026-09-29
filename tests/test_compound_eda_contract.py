"""Independent EDA goals survive preprocessing and chart customization."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.eda_contract import histogram_bins
from core.analysis_runtime_tools import build_analysis_tools
from utils.analysis_datasets import stored_dataset_digest

FIXTURE=json.loads((Path(__file__).parent/'fixtures/compound_numeric.json').read_text())


class PrepareOnce(BaseChatModel):
    arguments: dict = {}
    calls: int = 0
    @property
    def _llm_type(self):return 'prepare-then-verify-continuation'
    def bind_tools(self, tools, **kwargs):return self
    def _generate(self,messages,**kwargs):
        self.calls+=1
        if self.calls>1:raise AssertionError('Expected bounded local continuation after preparation')
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content='',tool_calls=[
            {'name':'prepare_numeric_dataset','args':self.arguments,'id':str(uuid4())}]))])


class CompoundEdaTests(unittest.TestCase):
    def test_all_goals_and_actual_histogram_bins(self):
        for case, rename in [(case, rename) for case in FIXTURE['cases'] for rename in ({}, FIXTURE['renamed_columns'])]:
            fixture={**FIXTURE,'measure':rename.get(FIXTURE['measure'],FIXTURE['measure']),
                     'group':rename.get(FIXTURE['group'],FIXTURE['group'])}
            request=case['request']
            for old,new in rename.items():request=request.replace(old,new)
            with self.subTest(case=case['id'],rename=bool(rename)), tempfile.TemporaryDirectory() as root:
                model=PrepareOnce();r=GraphAnalysisRuntime(root,'test','compound',model)
                try:
                    raw=r.datasets.register(pd.DataFrame(FIXTURE['rows']).rename(columns=rename),source=FIXTURE['source'],coverage='unknown')
                    r.select_dataset(raw.id);before=stored_dataset_digest(r.datasets,raw.id)
                    model.arguments={'dataset_id':raw.id,'columns':[fixture['measure']], 'missing_values':[FIXTURE['missing']]}
                    if case.get('groups'):model.arguments['preserve_columns']=[fixture['group']]
                    elif rename:model.arguments['preserve_columns']=[fixture['group']]
                    prompt='현재 로딩된 표본에서 '+request+' "'+FIXTURE['missing']+'" 문자열만 결측값으로 처리해.'
                    histograms=[];original=Axes.hist
                    def capture(ax,values,*args,**kwargs):
                        result=original(ax,values,*args,**kwargs)
                        histograms.append((list(values),result[0],result[1]))
                        return result
                    with patch.object(Axes,'hist',capture):result=r.submit(prompt)
                    self.assertEqual(result['status'],'answered',result)
                    state=r.inspect()['recovery']
                    self.assertIsNone(state.get('profile_kind'))
                    evidence=r.datasets.frames[state['evidence_ids'][-1]]
                    if case.get('groups'):
                        self.assertEqual(dict(zip(evidence[fixture['group']],evidence['average'])),case['groups'])
                        tools={t.name:t.run for t in build_analysis_tools(r.context)}
                        prepared=state['numeric_prepared_dataset']
                        wrong=tools['local_analysis_sql'](dataset_id=prepared,
                            query='SELECT AVG("'+fixture['measure']+'") AS average FROM data',current_result_only=True)
                        self.assertFalse(r.recovery._valid_calculation(wrong['dataset']['id'],
                            {'dataset_id':prepared,'current_result_only':True},state))
                    else:
                        self.assertEqual(float(evidence.iloc[0,0]),case['mean'])
                        self.assertTrue(state['artifact_ids'])
                        values,counts,edges=histograms[-1]
                        self.assertEqual(sorted(v for v in values if not pd.isna(v)),case['values'])
                        expected,boundaries=np.histogram(case['values'],bins=case.get('bins',20))
                        np.testing.assert_array_equal(counts,expected)
                        np.testing.assert_allclose(edges,boundaries)
                        card=r.artifacts[state['artifact_ids'][-1]]
                        self.assertEqual(card.render_spec['bins'],case.get('bins',20))
                        if case.get('id')=='filtered_scalar_chart':
                            tools={t.name:t.run for t in build_analysis_tools(r.context)}
                            prepared=state['numeric_prepared_dataset']
                            wrong=tools['render_chart_spec'](dataset_id=prepared,kind='histogram',x=fixture['measure'])
                            self.assertFalse(r.recovery._valid_card(r.artifacts[wrong['cards'][0]['id']],state,prepared))
                    self.assertEqual(stored_dataset_digest(r.datasets,raw.id),before)
                    self.assertEqual(model.calls,1)
                finally:r.close()

    def test_bin_wording_and_persisted_card_reject_wrong_completion(self):
        for text in ['구간 수 5개','구간 개수는 5','5개 구간','5 bins','bins=5','bins: 5']:
            self.assertEqual(histogram_bins(text),5,text)
        with tempfile.TemporaryDirectory() as root:
            model=PrepareOnce();r=GraphAnalysisRuntime(root,'test','bins',model)
            raw=r.datasets.register(pd.DataFrame({FIXTURE['measure']:[1,2,3,4]}),source=FIXTURE['source'],coverage='complete',predicate_known=True)
            r.select_dataset(raw.id)
            result=r.submit(f'현재 데이터의 `{FIXTURE["measure"]}` 히스토그램을 5개 구간으로 보여줘')
            self.assertEqual(result['status'],'answered',result)
            state=r.inspect()['recovery'];chart_id=state['artifact_ids'][-1]
            tools={t.name:t.run for t in build_analysis_tools(r.context)}
            wrong=tools['render_chart_spec'](dataset_id=raw.id,kind='histogram',x=FIXTURE['measure'],bins=20)
            wrong_card=r.artifacts[wrong['cards'][0]['id']]
            self.assertFalse(r.recovery._valid_card(wrong_card,state,raw.id))
            r.close();r=GraphAnalysisRuntime(root,'test','bins',PrepareOnce())
            try:self.assertEqual(r.artifacts[chart_id].render_spec['bins'],5)
            finally:r.close()


if __name__=='__main__':unittest.main()
