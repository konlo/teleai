from contextlib import redirect_stdout
from io import StringIO
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.private_host_release import switch_release
from scripts.private_host_smoke import check_storage, listener_hosts, loopback_only, main as smoke_main


class PrivateHostDeploymentTests(unittest.TestCase):
    def test_listener_must_exist_and_all_addresses_must_be_loopback(self):
        sample = "LISTEN 0 128 127.0.0.1:8502 0.0.0.0:*\nLISTEN 0 128 [::1]:8502 [::]:*"
        self.assertTrue(loopback_only(listener_hosts(sample, 8502)))
        self.assertFalse(loopback_only(listener_hosts(sample, 9999)))
        self.assertFalse(loopback_only(listener_hosts(sample + "\nLISTEN 0 128 0.0.0.0:8502 0.0.0.0:*", 8502)))
        self.assertFalse(loopback_only(["localhost"]))

    def test_storage_must_be_private_and_outside_checkout(self):
        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            checkout = root / "code"
            checkout.mkdir()
            data = root / "data"
            data.mkdir(mode=0o700)
            self.assertTrue(check_storage(data, checkout))
            data.chmod(0o755)
            self.assertFalse(check_storage(data, checkout))
            data.chmod(0o700)
            linked = root / "linked"
            linked.symlink_to(data)
            self.assertFalse(check_storage(linked, checkout))
            internal = checkout / "data"
            internal.mkdir(mode=0o700)
            self.assertFalse(check_storage(internal, checkout))

    def test_smoke_requires_health_and_private_listener(self):
        class HealthResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self, _limit):
                return b"ok"

        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            checkout = root / "checkout"
            checkout.mkdir()
            storage = root / "storage"
            storage.mkdir(mode=0o700)
            arguments = ["smoke", "--storage", str(storage), "--checkout", str(checkout)]
            with patch("sys.argv", arguments), patch("scripts.private_host_smoke.urlopen", return_value=HealthResponse()):
                with patch("scripts.private_host_smoke.subprocess.run") as command:
                    command.return_value.stdout = "LISTEN 0 128 127.0.0.1:8502 0.0.0.0:*"
                    output = StringIO()
                    with redirect_stdout(output):
                        self.assertEqual(smoke_main(), 0)
                    self.assertTrue(json.loads(output.getvalue())["ready"])
                    command.return_value.stdout += "\nLISTEN 0 128 0.0.0.0:8502 0.0.0.0:*"
                    output = StringIO()
                    with redirect_stdout(output):
                        self.assertEqual(smoke_main(), 1)
                    self.assertFalse(json.loads(output.getvalue())["checks"]["loopback_listener"])

    def test_rollback_switch_preserves_storage_and_rejects_unsafe_target(self):
        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            releases = root / "releases"
            releases.mkdir()
            for name in ("old", "new"):
                release = releases / name
                release.mkdir()
                (release / "requirements-agent.txt").write_text("streamlit==1\n")
            current = root / "current"
            current.symlink_to(releases / "new", target_is_directory=True)
            storage = root / "storage"
            storage.mkdir()
            marker = storage / "conversation.bin"
            marker.write_bytes(b"preserve-original")
            preview = switch_release(releases, current, "old", apply=False)
            self.assertFalse(preview["applied"])
            self.assertEqual(current.resolve().name, "new")
            result = switch_release(releases, current, "old", apply=True)
            self.assertTrue(result["applied"])
            self.assertEqual(current.resolve().name, "old")
            self.assertEqual(marker.read_bytes(), b"preserve-original")
            with self.assertRaises(ValueError):
                switch_release(releases, current, "../storage", apply=True)
            with self.assertRaises(ValueError):
                switch_release(releases, current, "missing", apply=True)


if __name__ == "__main__":
    unittest.main()
