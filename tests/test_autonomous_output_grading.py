"""Compound output grading retains the scalar preceding the chart dataset."""
from io import BytesIO
from types import SimpleNamespace
import unittest
import pandas as pd
from PIL import Image, ImageDraw
from scripts.evaluate_autonomous_paths import output_proof

class AutonomousOutputGradingTests(unittest.TestCase):
    def test_independent_scalar_and_histogram_obligations(self):
        image=Image.new('RGB',(200,200),'white');ImageDraw.Draw(image).rectangle((20,20,100,180),fill='blue')
        buffer=BytesIO();image.save(buffer,format='PNG')
        card=SimpleNamespace(kind='histogram',image=buffer.getvalue(),
            render_spec={'bin_edges':[0,5,15,25],'bin_counts':[2,1,1]})
        runtime=SimpleNamespace(datasets=SimpleNamespace(frames={
            'scalar':pd.DataFrame({'mean':[9.]}),'chart-data':pd.DataFrame({'reading':[2,4,10,20]})}),
            artifacts={'card':card})
        state={'evidence_ids':['scalar','chart-data'],'artifact_ids':['card']}
        actual,values,valid=output_proof(runtime,state,9.,True)
        self.assertEqual(actual,9.);self.assertEqual(values,[9.]);self.assertTrue(valid)
        actual,_,_=output_proof(runtime,state,100.,True)
        self.assertNotEqual(actual,100.)
        card.render_spec['bin_counts']=[1,1,1]
        self.assertFalse(output_proof(runtime,state,9.,True)[2])
        self.assertFalse(output_proof(runtime,{**state,'artifact_ids':[]},9.,True)[2])
