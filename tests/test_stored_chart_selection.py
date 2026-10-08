"""A UI selection displays the exact saved asset, never a new analysis."""
from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage
from streamlit.testing.v1 import AppTest

from core.analysis_agent.policy import RuntimePolicy
from core.analysis_agent.runtime import GraphAnalysisRuntime
from tests.test_mysql_metadata_contract import NoInference
from utils.analysis_charts import histogram_from_counts
from utils.analysis_datasets import stored_dataset_digest


class StoredChartSelectionTests(unittest.TestCase):
    def add_chart(self, runtime):
        raw=runtime.datasets.register(pd.DataFrame({'reading':[10,15,20,25],
            'cohort':['A','A','B','B']}),source='fixture.observations',
            coverage='complete',predicate_known=True)
        query="SELECT reading, COUNT(*) AS frequency FROM fixture.observations " \
              "WHERE reading BETWEEN 10 AND 20 AND cohort IN ('A','B') GROUP BY reading"
        info=runtime.datasets.register(pd.DataFrame({'reading':[10,15,20],'frequency':[2,4,3]}),
            source=raw.source,grain='aggregate',coverage='complete',predicate_known=True,
            query=query,aggregation=query)
        card=histogram_from_counts(runtime.datasets,info.id,'reading','frequency')
        # Presentation labels must never become another SQL/analysis request.
        card=replace(card,title='reading 10~20 평균 histogram')
        runtime.artifacts[card.id]=card
        runtime.select_dataset(raw.id)
        return raw,info,card

    def test_filtered_count_chart_selection_repeat_and_restart_are_model_free(self):
        for manual in (False,True):
            with self.subTest(manual=manual),tempfile.TemporaryDirectory() as root:
                policy=RuntimePolicy(require_remote_approval=manual)
                def factory(_):
                    return lambda envelope:self.fail('selection must not submit SQL')
                runtime=GraphAnalysisRuntime(root,'owner','selection',NoInference(),
                    connection_identity='fixture',remote_factory=factory,policy=policy,summary_trigger_tokens=1,intent_mode='contract_fixture')
                try:
                    raw,info,card=self.add_chart(runtime)
                    digest=stored_dataset_digest(runtime.datasets,raw.id)
                    for _ in range(2):
                        self.assertEqual(runtime.select_chart(card.id),card.id)
                        state=runtime.inspect()
                        self.assertEqual(state['state'],'idle',state)
                        self.assertEqual(state['selected_dataset']['id'],info.id)
                        self.assertEqual(state['recovery']['model_calls'],0)
                        self.assertEqual(state['recovery']['artifact_ids'],[card.id])
                        self.assertTrue(state['recovery']['current_result_only'])
                        self.assertEqual(runtime.events()[-1].additional_kwargs['analysis_artifact_ids'],[card.id])
                        self.assertIn('저장된 이미지를 표시했습니다',runtime.events()[-1].content)
                        self.assertEqual(stored_dataset_digest(runtime.datasets,raw.id),digest)
                    runtime.close()
                    runtime=GraphAnalysisRuntime(root,'owner','selection',NoInference(),
                        connection_identity='fixture',remote_factory=factory,policy=policy,summary_trigger_tokens=1,intent_mode='contract_fixture')
                    self.assertEqual(runtime.inspect()['state'],'idle')
                    result=runtime.submit('같은 차트를 다시 보여줘')
                    self.assertEqual(result['status'],'answered',result)
                    self.assertEqual(runtime.inspect()['recovery']['model_calls'],0)
                    self.assertEqual(runtime.artifacts[card.id].image,card.image)
                    self.assertEqual(stored_dataset_digest(runtime.datasets,raw.id),digest)
                finally:runtime.close()

    def test_legacy_failed_selection_resumes_from_asset_without_replaying_model(self):
        with tempfile.TemporaryDirectory() as root:
            runtime=GraphAnalysisRuntime(root,'owner','legacy',NoInference(),summary_trigger_tokens=1,intent_mode='contract_fixture')
            try:
                raw,info,card=self.add_chart(runtime)
                human=HumanMessage(id='legacy-selection',content=f'선택한 차트: {card.title}, '
                    f'dataset_id={info.id}, card_id={card.id}',additional_kwargs={'selected_card':card.id})
                runtime.agent.update_state(runtime.config,{'messages':[human,AIMessage(content='선택한 차트를 표시합니다.')],
                    'recovery':{'request_id':human.id,'status':'working','chart':True,'artifact_ids':[],
                                'model_calls':2,'model_seconds':180.}},as_node='model')
                runtime.transcript.record(runtime.agent.get_state(runtime.config).values['messages'])
                runtime.close()
                runtime=GraphAnalysisRuntime(root,'owner','legacy',NoInference(),summary_trigger_tokens=1,intent_mode='contract_fixture')
                self.assertEqual(runtime.inspect()['state'],'incomplete')
                result=runtime.resume()
                self.assertEqual(result['status'],'answered')
                self.assertEqual(runtime.inspect()['state'],'idle')
                self.assertEqual(runtime.inspect()['recovery']['model_calls'],2)
                self.assertEqual(runtime.inspect()['selected_dataset']['id'],info.id)
                self.assertEqual(runtime.events()[-1].additional_kwargs['analysis_artifact_ids'],[card.id])
            finally:runtime.close()

    def test_missing_or_corrupt_chart_does_not_change_selection_or_claim_success(self):
        with tempfile.TemporaryDirectory() as root:
            runtime=GraphAnalysisRuntime(root,'owner','invalid',NoInference(),intent_mode='contract_fixture')
            try:
                raw,info,card=self.add_chart(runtime)
                with self.assertRaises(KeyError):runtime.select_chart('missing')
                with runtime.db.conn:
                    runtime.db.conn.execute('UPDATE assets SET payload=? WHERE id=?',(b'broken-png',card.id))
                with self.assertRaises(ValueError):runtime.select_chart(card.id)
                self.assertEqual(runtime.context.selected_dataset_id,raw.id)
                self.assertEqual(runtime.events(),[])
                self.assertEqual(runtime.inspect()['state'],'idle')
            finally:runtime.close()

    def test_actual_page_selection_shows_image_without_analysis_or_resume_button(self):
        with tempfile.TemporaryDirectory() as root,patch.dict(os.environ,{'TELLY_V1_STORAGE':root}),patch(
                'core.analysis_agent.model_provider.build_analysis_chat_model',return_value=NoInference()):
            app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'ui/analysis_page.py'),default_timeout=30).run()
            runtime=app.session_state['v1_runtime']
            runtime.recovery.intent_mode = 'contract_fixture'  # UI execution/rendering fixture, not an intent score.
            try:
                raw,info,card=self.add_chart(runtime)
                runtime.select_chart(card.id)
                app.run()
                self.assertEqual(len(app.image),1,'Selection should not duplicate its chat image below the conversation')
                button=next(b for b in app.button if b.label=='이 차트 선택')
                button.click().run()
                self.assertFalse(app.exception)
                self.assertEqual(runtime.inspect()['state'],'idle')
                self.assertEqual(runtime.inspect()['recovery']['model_calls'],0)
                self.assertEqual(len(app.image),2,'Two selection turns display one image each')
                self.assertFalse(any(b.label=='미완료 분석 재개' for b in app.button))
                self.assertFalse(app.session_state['v1_notice'])
                self.assertIn('저장된 이미지를 표시했습니다','\n'.join(m.value for m in app.markdown))
                # Root/bottom chat inputs activate the frontend's page-following
                # scroll hook. The composer must remain in the normal main tree.
                self.assertEqual(len(app.main.get('chat_input')),1)
                self.assertEqual(app.main.get('chat_input')[0].key,'telly_chat_input')
                app.chat_input[0].set_value('같은 차트를 다시 보여줘').run()
                self.assertFalse(app.exception)
                self.assertEqual(runtime.inspect()['state'],'idle')
                self.assertEqual(runtime.inspect()['recovery']['model_calls'],0)
                self.assertEqual(len(app.main.get('chat_input')),1)
            finally:runtime.close()
