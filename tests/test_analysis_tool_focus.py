"""Small tool menus may only be used for unambiguous local scalar requests."""
import unittest
from types import SimpleNamespace

from langchain.agents.middleware import ModelRequest
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama

from core.analysis_agent.tool_focus import FocusedScalarToolsMiddleware, SCALAR_TOOLS
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


if __name__ == "__main__":
    unittest.main()
