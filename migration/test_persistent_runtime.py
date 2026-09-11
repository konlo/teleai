import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from migration.test_v1_contracts import ScriptModel
from migration.graph_runtime import GraphAnalysisRuntime
from migration.persistent_assets import AssetDB, PersistentDatasets, PersistentCharts
from utils.analysis_charts import recommend_charts


class QuietModel(ScriptModel):
    tool_name: str='unused'
    arguments: dict={}
    def _generate(self,messages,**kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content='이전 결과를 확인했습니다.'))])


class PersistentRuntimeTests(unittest.TestCase):
    def fixture(self):
        return json.loads(Path('tests/fixtures/analysis_acceptance.json').read_text())

    def test_process_exit_restores_data_image_and_conversation(self):
        with tempfile.TemporaryDirectory() as root:
            runtime=GraphAnalysisRuntime(root,'owner','thread',QuietModel(),cache_bytes=0)
            fixture=self.fixture()
            info=runtime.datasets.register(pd.DataFrame(fixture['rows']),source=fixture['source'],
                coverage='complete',predicate_known=True)
            card=recommend_charts(runtime.datasets,info.id)[0]
            runtime.artifacts[card.id]=card
            runtime.submit('이전 데이터 기억해줘')
            runtime.select_chart(card.id)
            runtime.close()
            # Fresh interpreter, no inherited DataFrame, session, callbacks or cached PNG.
            code='''
import json,sys
from migration.graph_runtime import GraphAnalysisRuntime
from migration.test_persistent_runtime import QuietModel
r=GraphAnalysisRuntime(sys.argv[1],'owner','thread',QuietModel(),cache_bytes=0)
assert len(r.events())==4
assert len(r.datasets.frames[sys.argv[2]])==7
assert r.artifacts[sys.argv[3]].image.startswith(b'\\x89PNG')
assert r.submit('앞선 결과에 이어서 설명해줘')['status']=='answered'
print(json.dumps(r.inspect()))
r.close()
'''
            result=subprocess.run([sys.executable,'-c',code,root,info.id,card.id],capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout)['message_count'],6)
            other=GraphAnalysisRuntime(root,'different-owner','thread',QuietModel())
            self.assertEqual(other.inspect()['dataset_ids'],[])
            self.assertEqual(other.events(),[])
            with self.assertRaises(KeyError):other.artifacts[card.id]
            other.close()

    def test_real_tools_persist_derived_dataset_and_dynamic_catalog(self):
        with tempfile.TemporaryDirectory() as root:
            runtime=GraphAnalysisRuntime(root,'owner','thread',QuietModel())
            fixture=self.fixture()
            info=runtime.datasets.register(pd.DataFrame(fixture['rows']),source=fixture['source'],coverage='complete',predicate_known=True)
            runtime.close()
            model=ScriptModel(tool_name='local_analysis_sql',arguments={'dataset_id':info.id,'query':'SELECT COUNT(*) AS n FROM data'})
            runtime=GraphAnalysisRuntime(root,'owner','thread',model)
            self.assertEqual(runtime.submit('개수 계산')['status'],'answered')
            self.assertEqual(len(runtime.datasets.metadata),2)
            child=next(v for v in runtime.datasets.metadata.values() if v.parent_id)
            self.assertEqual(runtime.datasets.frames[child.id].iloc[0,0],len(fixture['rows']))
            runtime.close()

    def test_cache_budget_and_copy_isolation(self):
        with tempfile.TemporaryDirectory() as root:
            db=AssetDB(root,'owner','thread');store=PersistentDatasets(db,budget=1)
            fixture=self.fixture()
            info=store.register(pd.DataFrame(fixture['rows']),source=fixture['source'])
            frame=store.frames[info.id];frame.iloc[0,0]='mutated'
            self.assertNotEqual(store.frames[info.id].iloc[0,0],'mutated')
            self.assertEqual(store.frames.bytes,0)
            with self.assertRaises(KeyError):store.frames['missing']
            db.close()

    def test_concurrent_controller_call_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            runtime=GraphAnalysisRuntime(root,'owner','thread',QuietModel())
            with runtime._exclusive():
                with self.assertRaises(RuntimeError):runtime.submit('중복 실행')
            self.assertEqual(runtime.events(),[])
            runtime.close()

    def test_budget_failure_retains_unfinished_work(self):
        with tempfile.TemporaryDirectory() as root:
            runtime=GraphAnalysisRuntime(root,'owner','thread',QuietModel(),max_context_chars=1)
            self.assertEqual(runtime.submit('분석')['status'],'incomplete')
            self.assertEqual(runtime.inspect()['state'],'incomplete')
            with self.assertRaises(ValueError):runtime.submit('다른 요청')
            runtime.close()
            runtime=GraphAnalysisRuntime(root,'owner','thread',QuietModel())
            self.assertEqual(runtime.resume()['status'],'answered')
            self.assertEqual(len(runtime.events()),2)
            runtime.close()

if __name__=='__main__':unittest.main()
