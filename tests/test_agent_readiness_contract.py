"""Acceptance-definition checks: fail closed on gaps or unsafe contract shapes."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator
from scripts.build_agent_readiness_manifest import build
from scripts.evaluate_data_preservation import histogram_matches, run_case, turn_event_count

ROOT = Path(__file__).resolve().parents[1]


class ReadinessContractTests(unittest.TestCase):
    def setUp(self):
        folder = ROOT / 'tests/fixtures/agent_contracts'
        self.schema = json.loads((folder / 'execution.schema.json').read_text())
        self.validator = Draft202012Validator(self.schema)
        self.examples = json.loads((folder / 'examples.json').read_text())

    def test_design_examples_are_valid_and_schema_is_well_formed(self):
        Draft202012Validator.check_schema(self.schema)
        for example in self.examples.values():
            self.validator.validate(example)

    def test_incomplete_candidate_cannot_be_a_ready_protected_root(self):
        for field, value in [('status', 'staging'), ('protected', False), ('digest', None)]:
            candidate = deepcopy(self.examples['protected_root'])
            candidate[field] = value
            self.assertTrue(list(self.validator.iter_errors(candidate)), field)
        aggregate = deepcopy(self.examples['protected_root'])
        aggregate['scope']['grain'] = 'aggregate'
        self.assertTrue(list(self.validator.iter_errors(aggregate)))

    def test_remote_plan_cannot_skip_approval_or_delete_existing_data(self):
        for field, value in [('approval_state', 'not_required'), ('preserve_existing', False),
                             ('sql_fingerprint', None)]:
            plan = deepcopy(self.examples['remote_load'])
            plan[field] = value
            self.assertTrue(list(self.validator.iter_errors(plan)), field)

    def test_unknown_submission_cannot_be_automatically_retryable(self):
        observation = deepcopy(self.examples['unknown_submission'])
        observation['retryable'] = True
        self.assertTrue(list(self.validator.iter_errors(observation)))

    def test_preservation_pass_needs_positive_evidence_and_no_unnecessary_approval(self):
        result = {'case_id': 'fixture-case', 'mode': 'storage_tool', 'status': 'PASS',
                  'requirements': ['D08'], 'root_preserved': True, 'oracle_pass': True,
                  'remote_executions': 0, 'unnecessary_approval_requests': 0,
                  'model_attempts': 0, 'evidence_paths': ['fixture-evidence.json'], 'limitations': []}
        self.validator.validate(result)
        for field, value in [('root_preserved', False), ('oracle_pass', None),
                             ('unnecessary_approval_requests', 1), ('evidence_paths', [])]:
            invalid = dict(result, **{field: value})
            self.assertTrue(list(self.validator.iter_errors(invalid)), field)

    def test_manifest_covers_every_definition_without_inventing_passes(self):
        manifest = build()
        saved = json.loads((ROOT/'tests/fixtures/agent_readiness_manifest.json').read_text())
        self.assertEqual(manifest, saved, 'Regenerate the manifest after changing requirements/definitions')
        requirements = {item['id'] for item in manifest['requirements']}
        journeys = {item['id'] for item in manifest['journeys']}
        self.assertEqual(len(requirements), 34)
        self.assertEqual(len(journeys), 24)
        self.assertEqual(len({case['id'] for case in manifest['reference_cases']}), 200)
        for item in manifest['requirements']:
            self.assertTrue(set(item['journeys']).issubset(journeys))
        for case in manifest['reference_cases']:
            self.assertTrue(set(case['requirements']).issubset(requirements))
            self.assertEqual(case['execution_status'], 'NOT_RUN')
        for journey in manifest['journeys']:
            for name in journey['partial_test_entrypoints']:
                suite = unittest.defaultTestLoader.loadTestsFromName(name)
                self.assertEqual(suite.countTestCases(), 1)
                self.assertNotIn('_FailedTest', str(suite))

    def test_histogram_oracle_rejects_wrong_population_and_clipped_counts(self):
        self.assertTrue(histogram_matches({1.0: 2, 9.0: 1}, {9.0: 1, 1.0: 2}, 3))
        self.assertFalse(histogram_matches({1.0: 2, 9.0: 1}, {1.0: 3}, 3))
        self.assertFalse(histogram_matches({1.0: 2, 9.0: 1}, {1.0: 2, 9.0: 1}, 2))

    def test_tool_attempt_count_is_scoped_to_newest_run(self):
        events = [{'event': 'run_started', 'run_id': 'first'},
                  {'event': 'tool_started', 'run_id': 'first'},
                  {'event': 'run_started', 'run_id': 'second'},
                  {'event': 'tool_started', 'run_id': 'second'},
                  {'event': 'tool_started', 'run_id': 'second'}]
        self.assertEqual(turn_event_count(events, 'tool_started'), 2)

    def test_runtime_evidence_survives_temporary_store_cleanup(self):
        import tempfile
        from scripts.evaluate_analysis_statistics import ForbiddenModel
        spec = json.loads((ROOT/'tests/fixtures/data_preservation_v1.json').read_text())
        case = {'id': 'harness-smoke', 'journeys': ['J21'],
                'turns': [spec['local_model_cases'][0]['turns'][0]]}
        with tempfile.TemporaryDirectory() as directory:
            record = run_case(spec, case, ForbiddenModel(), artifact_dir=Path(directory))
            self.assertEqual(record['status'], 'PASS', record)
            metadata = record['turns'][0]['runtime_metadata']
            self.assertTrue(Path(metadata['path']).is_file())
            self.assertTrue(metadata['diagnostics'])
            self.assertTrue(metadata['charts'])
            self.assertGreater(record['turns'][0]['runtime_tool_calls'], 0)
            self.assertTrue(all(Path(chart['path']).is_file() for chart in metadata['charts']))


if __name__ == '__main__':
    unittest.main()
