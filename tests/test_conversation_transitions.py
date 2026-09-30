"""Verify persisted conversation transitions against external journey oracles."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from migration.test_persistent_runtime import QuietModel
from core.analysis_agent.runtime import GraphAnalysisRuntime
from scripts.evaluate_conversation_context import run_case

FIXTURE=json.loads((Path(__file__).parent/'fixtures/conversation_context_journeys.json').read_text())


class ExplanationModel(QuietModel):
    def _generate(self,messages,**kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(
            content='히스토그램은 수치 구간의 빈도, 막대그래프는 범주별 값을 비교합니다.'))])


class ConversationTransitionTests(unittest.TestCase):
    def test_original_and_renamed_failure_journeys(self):
        for renamed in (False,True):
            fixture=deepcopy(FIXTURE)
            if renamed:
                mapping={'score':'measurement','segment':'category_key','latency_ms':'duration_value',
                    'synthetic.phase_a':'renamed.events_alpha','synthetic.phase_b':'renamed.events_beta'}
                encoded=json.dumps(fixture,ensure_ascii=False)
                for old,new in mapping.items():encoded=encoded.replace(old,new)
                fixture=json.loads(encoded)
            for spec in fixture['cases']:
                if spec['id'] not in {'selective_filter','topic_pause_return','language_source_switch'}:continue
                with self.subTest(case=spec['id'],renamed=renamed):
                    result=run_case(spec,fixture,ExplanationModel(),1)
                    self.assertEqual(result['status'],'PASS',result)

    def test_explanation_restart_and_explicit_ui_selection(self):
        with tempfile.TemporaryDirectory() as root:
            runtime=GraphAnalysisRuntime(root,'test','context',ExplanationModel())
            identities={}
            for source in FIXTURE['datasets']:
                data=runtime.datasets.register(pd.DataFrame(source['rows'],columns=source['columns']),
                    source=source['source'],coverage='complete',predicate_known=True)
                identities[source['source']]=data.id
            first,second=[d['source'] for d in FIXTURE['datasets']]
            runtime.select_dataset(identities[first])
            journey=next(c for c in FIXTURE['cases'] if c['id']=='topic_pause_return')
            self.assertEqual(runtime.submit(journey['turns'][0]['prompt'])['status'],'answered')
            self.assertEqual(runtime.submit(journey['turns'][1]['prompt'])['status'],'answered')
            state=runtime.inspect()['recovery']
            self.assertTrue(state['explanation_only'])
            self.assertFalse(runtime.recovery._proposed_scope_valid({'name':'local_analysis_sql'},state))
            self.assertTrue(state['confirmed_analysis']['scope']['conditions'])
            runtime.close()
            runtime=GraphAnalysisRuntime(root,'test','context',ExplanationModel())
            try:
                result=runtime.submit(journey['turns'][2]['prompt'])
                self.assertEqual(result['status'],'answered',result)
                state=runtime.inspect()['recovery']
                self.assertEqual(float(runtime.datasets.frames[state['evidence_ids'][-1]].iloc[0,0]),20)
                runtime.select_dataset(identities[second])
                prompt=next(c for c in FIXTURE['cases'] if c['id']=='session_isolation')['turns'][1]['prompt']
                result=runtime.submit(prompt)
                self.assertEqual(result['status'],'answered',result)
                state=runtime.inspect()['recovery'];info=runtime.datasets.metadata[state['evidence_ids'][-1]]
                self.assertEqual(info.source,second)
                self.assertEqual(float(runtime.datasets.frames[info.id].iloc[0,0]),200)
            finally:runtime.close()


if __name__=='__main__':unittest.main()
