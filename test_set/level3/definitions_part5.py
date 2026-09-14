"""Level 3 registry backed by production agent recovery contracts.

Level 1 and 2 execute reference pandas/matplotlib solutions. Level 3 has a
different purpose: each entry names one deterministic fault-injection test
against the production ``GraphAnalysisRuntime``. Keeping executable test IDs
here prevents placeholder snippets from being reported as agent capability.
"""
from scripts.evaluate_agentic_recovery import CASES


def get_level3_part5() -> list[dict]:
    return [
        {
            "id": case_id,
            "category": "Agentic recovery contract",
            "type": "agentic",
            "difficulty": "Advanced",
            "prompt": capability,
            "target_table": None,
            "synonym_mapping": None,
            "expected_output_type": "contract_result",
            "test": test_id,
        }
        for case_id, capability, test_id in CASES
    ]
