"""Images must be decodable, plotted as bars, attached, and actually sent to Streamlit."""
from dataclasses import replace
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from PIL import Image
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, RemoveMessage
from core.analysis_agent.runtime import GraphAnalysisRuntime
from core.analysis_agent.memory import latest_user_request
from migration.test_persistent_runtime import QuietModel
from ui.analysis_chart_delivery import chart_references, chart_delivery_plan
from utils.analysis_charts import render_chart_spec
from utils.analysis_image_validation import validate_chart_image

FIXTURE=json.loads((Path(__file__).parent/'fixtures/analysis_acceptance.json').read_text())
COLUMN=next(k for k,v in FIXTURE['rows'][0].items() if isinstance(v,(int,float)))


class ChartDeliveryTests(unittest.TestCase):
    def prepare(self,root,model=None):
        r=GraphAnalysisRuntime(root,'test','chart',model or QuietModel(),intent_mode='contract_fixture')
        self.addCleanup(r.close)
        info=r.datasets.register(pd.DataFrame(FIXTURE['rows']),source=FIXTURE['source'],
            coverage='complete',predicate_known=True)
        r.select_dataset(info.id)
        return r,info

    def test_corrupt_header_blank_and_transparent_images_are_rejected(self):
        cases=[b'bar chart ...',b'\x89PNG\r\n\x1a\n',b'\x89PNG\r\n\x1a\nnot-an-image']
        for color in ['white',(0,0,0,0)]:
            buffer=BytesIO();Image.new('RGBA',(660,385),color).save(buffer,format='PNG');cases.append(buffer.getvalue())
        for payload in cases:
            with self.subTest(size=len(payload)),self.assertRaises(ValueError):validate_chart_image(payload)

    def test_histogram_contains_rectangular_bins_with_correct_counts(self):
        from matplotlib.axes import Axes
        original=Axes.hist;observed=[]
        def capture(ax,*args,**kwargs):
            output=original(ax,*args,**kwargs)
            observed.append((kwargs,output))
            return output
        with tempfile.TemporaryDirectory() as root:
            r,info=self.prepare(root)
            with patch.object(Axes,'hist',capture):
                card,summary,spec=render_chart_spec(r.datasets,info.id,kind='histogram',x=COLUMN,bins=4)
            kwargs,(counts,edges,bars)=observed[0]
            expected,_=np.histogram(pd.DataFrame(FIXTURE['rows'])[COLUMN],bins=4)
            np.testing.assert_array_equal(counts,expected)
            self.assertEqual(kwargs['histtype'],'bar')
            self.assertEqual(len(bars.patches),4)
            self.assertEqual(sum(rect.get_height() for rect in bars.patches),len(FIXTURE['rows']))
            self.assertGreater(validate_chart_image(card.image)['width'],100)
            self.assertGreater(len(np.unique(np.asarray(Image.open(BytesIO(card.image)).convert('RGB')).reshape(-1,3),axis=0)),10)

    def test_plain_chart_title_cannot_complete_without_attached_image(self):
        class PlaceholderModel(QuietModel):
            def _generate(self,*args,**kwargs):
                from langchain_core.outputs import ChatGeneration,ChatResult
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content='bar chart ...'))])
        with tempfile.TemporaryDirectory() as root:
            r,info=self.prepare(root,PlaceholderModel())
            # A literal model placeholder cannot satisfy the chart contract.
            with patch.object(r.recovery,'_next_local',return_value=None):
                blocked=r.submit(f'{COLUMN} histogram을 보여줘')
            self.assertNotEqual(blocked['status'],'answered',blocked)
            self.assertFalse(chart_references(r.events()[-1]))
            # Normal local recovery must produce the missing image.
            result=r.submit(f'{COLUMN} histogram을 보여줘')
            self.assertEqual(result['status'],'answered',result)
            final=r.events()[-1]
            refs=chart_references(final)
            self.assertTrue(refs)
            self.assertNotEqual(final.content,'bar chart ...')
            for key in refs:validate_chart_image(r.artifacts[key].image)
            old=r.artifacts[refs[0]]
            with self.assertRaises(ValueError):r.artifacts['bad']=replace(old,id='bad',image=b'\x89PNG\r\n\x1a\n')
            self.assertNotIn('bad',r.artifacts)
            with r.db.conn:r.db.conn.execute("UPDATE assets SET payload=? WHERE id=?",(b'\x89PNG\r\n\x1a\n',refs[0]))
            state=r.inspect()['recovery']
            finished=r.recovery._finish(state)
            self.assertNotEqual(finished['messages'][-1].additional_kwargs['analysis_status'],'answered')
            self.assertFalse(chart_references(finished['messages'][-1]))

    def test_final_attachment_displays_even_without_tool_messages_and_survives_rerun(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as root:
            r,info=self.prepare(root)
            result=r.submit(f'{COLUMN} histogram을 보여줘')
            self.assertEqual(result['status'],'answered',result)
            state=r.agent.get_state(r.config)
            final=state.values['messages'][-1]
            human=latest_user_request(state.values['messages'])
            # Simulate persisted history containing only user-facing messages.
            remove=[RemoveMessage(id=m.id) for m in state.values['messages'] if m.id not in {human.id,final.id}]
            r.agent.update_state(r.config,{'messages':remove},as_node='RecoveryMiddleware.after_model')
            with r.db.conn:r.db.conn.execute('DELETE FROM transcript WHERE id NOT IN (?,?)',(human.id,final.id))
            with patch.dict(os.environ,{'TELLY_V1_STORAGE':root}), \
                 patch('core.analysis_agent.model_provider.build_analysis_chat_model',return_value=QuietModel()), \
                 patch('core.analysis_agent.runtime.GraphAnalysisRuntime',return_value=r):
                app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'ui/analysis_page.py'),default_timeout=20).run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.get('image')),1)
                self.assertEqual(len(app.get('image')[0].proto.imgs),1)
                self.assertTrue(app.get('image')[0].proto.imgs[0].url)
                app.run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.get('image')),1)
                # Existing corrupt assets must report the display failure, not silently omit an image.
                key=chart_references(final)[0]
                with r.db.conn:r.db.conn.execute('UPDATE assets SET payload=? WHERE id=?',(b'bad',key))
                app.run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.get('image')),0)
                self.assertTrue(any('이미지를 표시하지 못했습니다' in e.value for e in app.error))

    def test_unaccepted_intermediate_charts_are_not_displayed(self):
        tool=ToolMessage(id='tool',tool_call_id='render',content=json.dumps({'cards':[{'id':'correct'},{'id':'wrong'}]}))
        final=AIMessage(id='final',content='차트',additional_kwargs={'analysis_artifact_ids':['correct']})
        plan=chart_delivery_plan([HumanMessage(id='user',content='histogram'),tool,final])
        self.assertEqual(plan['tool'],['correct'])
        self.assertEqual(plan['final'],['correct'])
        final.additional_kwargs['analysis_artifact_ids']=[]
        self.assertEqual(chart_delivery_plan([tool,final])['tool'],[])
        self.assertEqual(chart_delivery_plan([tool])['tool'],['correct','wrong'])  # legacy fallback

    def test_generic_viz_request_is_not_a_text_only_success(self):
        with tempfile.TemporaryDirectory() as root:
            r,info=self.prepare(root)
            result=r.submit(f'{COLUMN} viz 보여줘')
            if result['status']=='answered':
                self.assertTrue(chart_references(r.events()[-1]),result)
            else:
                self.assertIn(result['status'],{'blocked','exhausted','needs_data'})


if __name__=='__main__':unittest.main()
