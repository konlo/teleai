"""Code generation profile never changes original models or non-Python turns."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from langchain_ollama import ChatOllama
from core.analysis_agent.python_planning import BoundedPythonPlanningMiddleware

def request(model,current):
    req=SimpleNamespace(model=model,state={'recovery':current})
    req.override=lambda **changes:SimpleNamespace(**{**vars(req),**changes})
    return req

class PythonPlanningProfileTests(unittest.TestCase):
    def test_python_ollama_profile_is_bounded_without_mutating_configuration(self):
        model=ChatOllama(model='fixture',reasoning=True,num_predict=4096,num_ctx=16384)
        diag=Mock();req=request(model,{'custom_analysis_spec':{'description':'user task'},'scope':{'conditions':[]}})
        result=BoundedPythonPlanningMiddleware(diag).wrap_model_call(req,lambda r:r)
        self.assertFalse(result.model.reasoning);self.assertEqual(result.model.num_predict,1536)
        self.assertTrue(model.reasoning);self.assertEqual(model.num_predict,4096)
        self.assertIs(result.state,req.state);diag.emit.assert_called_once()
    def test_other_turns_providers_and_smaller_output_limits_are_preserved(self):
        model=ChatOllama(model='fixture',num_predict=512)
        for candidate,current in ((model,{}),(SimpleNamespace(),{'custom_analysis_spec':{'description':'task'}})):
            req=request(candidate,current)
            self.assertIs(BoundedPythonPlanningMiddleware().wrap_model_call(req,lambda r:r),req)
        req=request(model,{'custom_analysis_spec':{'description':'task'}})
        self.assertEqual(BoundedPythonPlanningMiddleware().wrap_model_call(req,lambda r:r).model.num_predict,512)

if __name__=='__main__':unittest.main()
