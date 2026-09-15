import json
from pathlib import Path
import tempfile
import unittest

from core.deployment_preflight import evaluate_deployment


def valid_environment(storage: str) -> dict[str, str]:
    return {
        "OLLAMA_MODEL": "test-model",
        "OLLAMA_BASE_URL": "https://model.internal.example",
        "DATABRICKS_HOST": "workspace.example",
        "DATABRICKS_HTTP_PATH": "/sql/warehouses/example",
        "DATABRICKS_TOKEN": "must-never-appear",
        "DATABRICKS_CATALOG": "catalog",
        "DATABRICKS_SCHEMA": "schema",
        "TELLY_V1_STORAGE": storage,
    }


class DeploymentPreflightTests(unittest.TestCase):
    def test_valid_local_configuration_is_ready_and_secret_safe(self):
        with tempfile.TemporaryDirectory() as storage:
            report = evaluate_deployment(
                valid_environment(storage),
                profile="local-desktop",
                project_root=Path("/workspace/teleai"),
            )
        self.assertTrue(report.ready)
        self.assertNotIn("must-never-appear", json.dumps(report.public()))

    def test_access_token_alias_is_accepted(self):
        with tempfile.TemporaryDirectory() as storage:
            env = valid_environment(storage)
            env["DATABRICKS_ACCESS_TOKEN"] = env.pop("DATABRICKS_TOKEN")
            report = evaluate_deployment(
                env, profile="local-desktop", project_root=Path("/workspace/teleai")
            )
        self.assertTrue(report.ready)

    def test_missing_databricks_token_fails_without_naming_values(self):
        with tempfile.TemporaryDirectory() as storage:
            env = valid_environment(storage)
            env.pop("DATABRICKS_TOKEN")
            report = evaluate_deployment(
                env, profile="local-desktop", project_root=Path("/workspace/teleai")
            )
        self.assertFalse(report.ready)
        self.assertEqual(
            next(c.status for c in report.checks if c.name == "databricks_configuration"),
            "fail",
        )

    def test_private_profile_requires_external_access_control(self):
        with tempfile.TemporaryDirectory() as storage:
            env = valid_environment(storage)
            report = evaluate_deployment(
                env,
                profile="private-single-user",
                project_root=Path("/workspace/teleai"),
            )
            env["TELLY_EXTERNAL_ACCESS_CONTROL"] = "confirmed"
            confirmed = evaluate_deployment(
                env,
                profile="private-single-user",
                project_root=Path("/workspace/teleai"),
            )
        self.assertFalse(report.ready)
        self.assertTrue(confirmed.ready)

    def test_multi_user_profile_is_blocked_by_identity_contract(self):
        with tempfile.TemporaryDirectory() as storage:
            report = evaluate_deployment(
                valid_environment(storage),
                profile="multi-user",
                project_root=Path("/workspace/teleai"),
            )
        self.assertFalse(report.ready)
        self.assertIn("owner binding", json.dumps(report.public(), ensure_ascii=False))

    def test_relative_storage_and_invalid_policy_fail(self):
        env = valid_environment("relative/storage")
        env["TELLY_MAX_REMOTE_ROWS"] = "0"
        report = evaluate_deployment(
            env, profile="local-desktop", project_root=Path("/workspace/teleai")
        )
        failed = {c.name for c in report.checks if c.status == "fail"}
        self.assertEqual(failed, {"persistent_storage", "runtime_policy"})


if __name__ == "__main__":
    unittest.main()
