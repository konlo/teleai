import unittest

from core.runtime_capacity import CapacityPolicy, evaluate_capacity


def benchmark(*, local_p95=0.15, concurrent_p95=0.56, successes=20, failures=0, rss=400_000_000):
    return {
        "safety": {
            "databricks_calls": 0,
            "model_calls": 0,
            "source_asset_mutated": False,
        },
        "current_local_dataframe_benchmark": {
            "p95_seconds": local_p95,
            "process_peak_rss_bytes": rss - 10_000_000,
        },
        "current_concurrent_local_benchmark": {
            "requests": 20,
            "workers": 4,
            "successes": successes,
            "failures": failures,
            "p95_seconds": concurrent_p95,
            "process_peak_rss_bytes": rss,
        },
    }


class RuntimeCapacityTests(unittest.TestCase):
    def test_release_sample_passes_without_remote_or_model_calls(self):
        result = evaluate_capacity(benchmark(), CapacityPolicy())
        self.assertTrue(result["ready"])
        self.assertEqual(
            result["safety"],
            {"databricks_calls": 0, "model_calls": 0, "source_asset_mutated": False},
        )
        self.assertTrue(all(check["status"] == "pass" for check in result["checks"]))

    def test_remote_or_model_workload_cannot_be_reported_as_safe(self):
        report = benchmark()
        report["safety"]["model_calls"] = 1
        result = evaluate_capacity(report, CapacityPolicy())
        self.assertFalse(result["ready"])
        self.assertEqual(result["checks"][0]["name"], "benchmark_safety")
        self.assertEqual(result["checks"][0]["status"], "fail")

    def test_latency_failure_blocks_release(self):
        result = evaluate_capacity(benchmark(concurrent_p95=1.01), CapacityPolicy())
        self.assertFalse(result["ready"])
        self.assertEqual(
            next(check["status"] for check in result["checks"] if check["name"] == "concurrent_p95"),
            "fail",
        )

    def test_incomplete_concurrent_sample_blocks_release(self):
        result = evaluate_capacity(benchmark(successes=19, failures=1), CapacityPolicy())
        self.assertFalse(result["ready"])

    def test_rss_warning_is_visible_but_critical_blocks(self):
        policy = CapacityPolicy()
        warning = evaluate_capacity(benchmark(rss=policy.rss_warning_bytes), policy)
        critical = evaluate_capacity(benchmark(rss=policy.rss_critical_bytes), policy)
        self.assertTrue(warning["ready"])
        self.assertEqual(warning["checks"][-1]["status"], "warn")
        self.assertFalse(critical["ready"])
        self.assertEqual(critical["checks"][-1]["status"], "fail")

    def test_policy_rejects_invalid_threshold_order(self):
        with self.assertRaises(ValueError):
            CapacityPolicy(rss_warning_bytes=900, rss_critical_bytes=800, memory_limit_bytes=1000)


if __name__ == "__main__":
    unittest.main()
