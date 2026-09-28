import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from langchain_core.messages import AIMessage, ToolMessage

from core.analysis_agent.runtime import GraphAnalysisRuntime
from migration.test_persistent_runtime import QuietModel


class MultiAggregateRecoveryTests(unittest.TestCase):
    def test_average_and_oldest_are_computed_locally_without_model_or_remote(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = GraphAnalysisRuntime(root, "owner", "multi-aggregate", QuietModel())
            frame = pd.DataFrame({"age": [20, 35, 71], "segment": ["a", "b", "c"]})
            runtime.datasets.register(
                frame, source="arbitrary.dynamic_table", coverage="complete", predicate_known=True
            )
            runtime.context.reference_context[:] = [{
                "table": "arbitrary.dynamic_table",
                "training_status": "fixture_profile",
                "columns": [
                    {"name": "age", "dtype": "int64", "aliases": ["나이"], "top_values": []},
                    {"name": "segment", "dtype": "object", "aliases": [], "top_values": []},
                ],
            }]
            outcome = runtime.submit("고객들의 평균 나이와 최고령 고객의 나이를 알려줘.")
            self.assertEqual(outcome["status"], "answered", outcome)
            calls = [call for message in runtime.events() if isinstance(message, AIMessage)
                     for call in message.tool_calls]
            self.assertEqual([call["name"] for call in calls], ["local_analysis_sql"])
            observation = next(
                json.loads(message.content) for message in runtime.events()
                if isinstance(message, ToolMessage) and message.name == "local_analysis_sql"
            )
            result = runtime.datasets.frames[observation["dataset"]["id"]]
            self.assertEqual(result.to_dict(orient="records"), [{"average": 42.0, "maximum": 71}])
            completed = runtime.inspect()["recovery"]
            self.assertEqual(completed["operations"], ["AVG", "MAX"])
            self.assertEqual(completed["model_calls"], 0)
            runtime.close()


if __name__ == "__main__":
    unittest.main()
