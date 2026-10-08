"""A previous chart selection cannot masquerade as a new failed request."""
import os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from langchain_core.messages import HumanMessage
from streamlit.testing.v1 import AppTest
from core.analysis_agent.runtime import GraphAnalysisRuntime
from tests.test_mysql_metadata_contract import NoInference
from tests.test_chart_display_journey import CASE

class StaleChartFooterTests(unittest.TestCase):
    def test_selected_histogram_remains_in_history_not_below_unfinished_scatter(self):
        with tempfile.TemporaryDirectory() as root,patch.dict(os.environ,{'TELLY_V1_STORAGE':root}),patch(
                'core.analysis_agent.model_provider.build_analysis_chat_model',return_value=NoInference()):
            app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'ui/analysis_page.py'),default_timeout=30).run()
            runtime=app.session_state['v1_runtime']
            runtime.recovery.intent_mode = 'contract_fixture'  # UI execution/rendering fixture, not an intent score.
            try:
                from tests.test_stored_chart_selection import StoredChartSelectionTests
                _,info,card=StoredChartSelectionTests().add_chart(runtime)
                runtime.select_chart(card.id);app.run()
                self.assertEqual(len(app.get('image')),1)
                human=HumanMessage(id='unfinished-source-scatter',content=CASE['scatter_prompt'])
                runtime.agent.update_state(runtime.config,{'messages':[human],
                    'recovery':{'request_id':human.id,'status':'working','chart':True,'kind':'scatter',
                                'artifact_ids':[],'model_calls':0,'model_seconds':0.}},
                    as_node='ObservedSummarizationMiddleware.before_model')
                runtime.transcript.record(runtime.agent.get_state(runtime.config).values['messages'])
                app.run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.get('image')),1,'Keep the historical selection, omit its stale footer clone')
                self.assertFalse(any(h.value==card.title for h in app.subheader))
                self.assertEqual(runtime.context.selected_dataset_id,info.id)
                self.assertTrue(any('이번 시각화 요청은 아직 완료되지 않았습니다' in w.value for w in app.warning))
                self.assertTrue(any(b.label=='미완료 분석 재개' for b in app.button))
                app.run()
                self.assertEqual(len(app.get('image')),1)
                self.assertEqual(runtime.inspect()['state'],'incomplete')
            finally:runtime.close()

    def test_latest_turn_selection_does_not_cross_new_request_boundary(self):
        from ui.analysis_chart_selection import latest_selected_chart
        from langchain_core.messages import AIMessage
        old=HumanMessage(content='선택한 차트: saved, card_id=legacy')
        current=HumanMessage(content='new request')
        self.assertEqual(latest_selected_chart([old,AIMessage(content='saved')]),'legacy')
        self.assertIsNone(latest_selected_chart([old,current,AIMessage(content='not completed')]))
        self.assertEqual(latest_selected_chart([old,current,HumanMessage(content='chosen',
            additional_kwargs={'selected_card':'structured'})]),'structured')
