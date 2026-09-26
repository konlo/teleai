"""Small tool menus may only be used for unambiguous local scalar requests."""
import unittest
from types import SimpleNamespace
from datetime import datetime, timezone

from langchain.agents.middleware import ModelRequest
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama

from core.analysis_agent.tool_focus import (FocusedScalarToolsMiddleware, SCALAR_TOOLS,
    FocusedRemoteJoinToolsMiddleware, REMOTE_JOIN_TOOLS)
from utils.analysis_datasets import DatasetInfo


class FocusedScalarToolsTests(unittest.TestCase):
    def setUp(self):
        info = DatasetInfo("root-1", "fixture.events", ("measure", "segment"), 10,
                           coverage="complete", predicate_known=True, grain="raw")
        self.context = SimpleNamespace(datasets=SimpleNamespace(metadata={info.id: info}))
        self.middleware = FocusedScalarToolsMiddleware(self.context)
        self.tools = [Tool(name=name, func=lambda _: "", description=name)
                      for name in sorted(SCALAR_TOOLS | {"join_datasets", "query_databricks"})]
        self.model = ChatOllama(model="unused", base_url="http://127.0.0.1:11434")
        self.current = {
            "calculation": True, "operations": ["AVG"], "required_columns": ["measure"],
            "scope": {"conditions": [], "any_conditions": [], "measure_conditions": [],
                      "unresolved": []},
        }

    def visible(self, current):
        request = ModelRequest(model=self.model, messages=[], tools=self.tools,
                               state={"recovery": current})
        return self.middleware.wrap_model_call(
            request, lambda narrowed: {tool.name for tool in narrowed.tools})

    def test_unique_unfiltered_scalar_focuses_only_safe_local_tools(self):
        self.assertEqual(self.visible(self.current), SCALAR_TOOLS)

    def test_ambiguous_scope_and_other_intents_keep_full_tool_menu(self):
        all_tools = {tool.name for tool in self.tools}
        variants = [
            {**self.current, "chart": True},
            {**self.current, "operations": ["AVG", "MAX"]},
            {**self.current, "required_columns": ["missing"]},
            {**self.current, "fresh_source_required": True},
            {**self.current, "required_sources": ["other.events"]},
            {**self.current, "scope": {"conditions": [{"column": "segment", "op": "eq", "value": "a"}]}},
        ]
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertEqual(self.visible(variant), all_tools)
        self.context.datasets.metadata["root-2"] = DatasetInfo(
            "root-2", "fixture.second", ("measure",), 10,
            coverage="complete", predicate_known=True, grain="raw")
        self.assertEqual(self.visible(self.current), all_tools)


class FocusedRemoteJoinTests(unittest.TestCase):
    def setUp(self):
        self.context = SimpleNamespace(datasets=SimpleNamespace(metadata={}),
            reference_context=[{'table':s,'observed_at':datetime.now(timezone.utc).isoformat()}
                               for s in ['fixture.left','fixture.right']])
        self.middleware = FocusedRemoteJoinToolsMiddleware(self.context)
        self.tools = [Tool(name=name,func=lambda _: '',description=name)
                      for name in sorted(REMOTE_JOIN_TOOLS | {'join_datasets','inspect_dataset','aggregate_dataset'})]
        self.current = {'join':True,'requested_join':True,'calculation':True,'required_sources':['fixture.left','fixture.right'],
                        'scope':{'conditions':[], 'unresolved':[]}}

    def visible(self, current, tools=None):
        request = ModelRequest(model=ChatOllama(model='unused'),messages=[],
            tools=self.tools if tools is None else tools,state={'recovery':current})
        return self.middleware.wrap_model_call(request,lambda narrowed:{t.name for t in narrowed.tools})

    def test_missing_sources_focus_approved_remote_path(self):
        self.assertEqual(self.visible(self.current),REMOTE_JOIN_TOOLS)

    def test_local_sources_unresolved_or_other_requests_keep_alternatives(self):
        full={t.name for t in self.tools}
        for variant in [dict(requested_join=False),dict(chart=True),dict(data_load=True),
                        dict(scope={'unresolved':['ambiguous role']}),dict(evidence_ids=['result']),
                        dict(required_sources=['fixture.left','unknown'])]:
            with self.subTest(variant=variant):
                self.assertEqual(self.visible({**self.current,**variant}),full)
        for i,s in enumerate(self.current['required_sources']):
            self.context.datasets.metadata[str(i)]=DatasetInfo(str(i),s,('x',),1,
                grain='raw',coverage='complete',predicate_known=True)
        self.assertEqual(self.visible(self.current),full)

    def test_stale_schema_and_absent_remote_tool_do_not_focus(self):
        self.context.reference_context[0]['observed_at']='2000-01-01T00:00:00Z'
        self.assertEqual(self.visible(self.current),{t.name for t in self.tools})
        tools=[t for t in self.tools if t.name!='query_databricks']
        self.assertEqual(self.visible(self.current,tools),{t.name for t in tools})


if __name__ == "__main__":
    unittest.main()
