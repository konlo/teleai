"""Bound the code-emission phase while retaining the agent's goal and tool loop."""
from langchain.agents.middleware import AgentMiddleware
from langchain_ollama import ChatOllama

class BoundedPythonPlanningMiddleware(AgentMiddleware):
    def __init__(self,diagnostics=None):self.diagnostics=diagnostics
    def wrap_model_call(self,request,handler):
        current=request.state.get('recovery') or {}
        model=getattr(request,'model',None)
        if not current.get('custom_analysis_spec') or not isinstance(model,ChatOllama):return handler(request)
        configured=getattr(model,'num_predict',None)
        limit=min(configured,1536) if isinstance(configured,int) and configured>0 else 1536
        revised=model.model_copy(update={'reasoning':False,'num_predict':limit})
        if self.diagnostics:self.diagnostics.emit('model_execution_profile',
            phase='bounded_python_code',reasoning=False,output_tokens=limit,
            provider='ollama',scope_changed=False)
        return handler(request.override(model=revised))
