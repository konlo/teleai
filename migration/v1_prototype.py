"""Isolated compatibility prototype. Not imported by the production page.

Remote execution is deliberately unavailable until the durable ledger exists.
"""
from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from core.analysis_runtime_tools import build_analysis_tools


def local_tools(context):
    return [StructuredTool(name=t.name, description=t.description,
                           args_schema=t.parameters, func=t.run)
            for t in build_analysis_tools(context)
            if t.name != 'propose_databricks_query']


def build_local_agent(model, context, saver):
    from core.analysis_instructions import ANALYSIS_INSTRUCTIONS
    import json
    catalog = next(t for t in build_analysis_tools(context) if t.name=='list_analysis_context').run()
    # Prototype initial context only. Dynamic context/budget is a later runtime feature.
    prompt = ANALYSIS_INSTRUCTIONS + '\n현재 분석 환경:\n' + json.dumps(catalog,ensure_ascii=False,default=str)
    return create_agent(model=model, tools=local_tools(context),
                        system_prompt=prompt, checkpointer=saver)
