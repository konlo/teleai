"""Deployment-neutral capacity policy and benchmark acceptance checks."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class CapacityPolicy:
    """Initial single-process limits, overridable by the deployment environment."""

    deterministic_p95_seconds: float = 1.0
    min_concurrent_requests: int = 20
    min_workers: int = 4
    memory_limit_bytes: int = 1024 * 1024 * 1024
    rss_warning_bytes: int = 768 * 1024 * 1024
    rss_critical_bytes: int = 896 * 1024 * 1024

    def __post_init__(self) -> None:
        numeric = asdict(self)
        if any(value <= 0 for value in numeric.values()):
            raise ValueError("capacity policy values must be positive")
        if not self.rss_warning_bytes < self.rss_critical_bytes < self.memory_limit_bytes:
            raise ValueError("RSS thresholds must satisfy warning < critical < memory limit")

    @classmethod
    def from_mapping(cls, env: Mapping[str, str]) -> "CapacityPolicy":
        return cls(
            deterministic_p95_seconds=float(
                env.get("TELLY_DETERMINISTIC_P95_SECONDS", cls.deterministic_p95_seconds)
            ),
            min_concurrent_requests=int(
                env.get("TELLY_CAPACITY_MIN_REQUESTS", cls.min_concurrent_requests)
            ),
            min_workers=int(env.get("TELLY_CAPACITY_MIN_WORKERS", cls.min_workers)),
            memory_limit_bytes=int(env.get("TELLY_MEMORY_LIMIT_BYTES", cls.memory_limit_bytes)),
            rss_warning_bytes=int(env.get("TELLY_RSS_WARNING_BYTES", cls.rss_warning_bytes)),
            rss_critical_bytes=int(env.get("TELLY_RSS_CRITICAL_BYTES", cls.rss_critical_bytes)),
        )

    def public(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class CapacityCheck:
    name: str
    status: str
    observed: int | float | None
    limit: int | float | None
    message: str

    def public(self) -> dict[str, object]:
        return asdict(self)


def evaluate_capacity(report: Mapping[str, object], policy: CapacityPolicy) -> dict[str, object]:
    """Evaluate a controlled benchmark report without running any workload."""
    local = report.get("current_local_dataframe_benchmark")
    concurrent = report.get("current_concurrent_local_benchmark")
    if not isinstance(local, Mapping) or not isinstance(concurrent, Mapping):
        raise ValueError("benchmark report is missing local or concurrent results")

    checks: list[CapacityCheck] = []
    safety = report.get("safety")
    observed_safety = {
        "databricks_calls": safety.get("databricks_calls") if isinstance(safety, Mapping) else None,
        "model_calls": safety.get("model_calls") if isinstance(safety, Mapping) else None,
        "source_asset_mutated": safety.get("source_asset_mutated") if isinstance(safety, Mapping) else None,
    }
    safe_workload = observed_safety == {
        "databricks_calls": 0,
        "model_calls": 0,
        "source_asset_mutated": False,
    }
    checks.append(
        CapacityCheck(
            "benchmark_safety",
            "pass" if safe_workload else "fail",
            int(safe_workload),
            1,
            "모델·Databricks 호출 없이 원본을 변경하지 않은 측정입니다."
            if safe_workload
            else "측정 안전 증거가 없거나 모델·Databricks 호출 또는 원본 변경이 포함됐습니다.",
        )
    )
    requests = int(concurrent.get("requests", 0))
    workers = int(concurrent.get("workers", 0))
    successes = int(concurrent.get("successes", 0))
    failures = int(concurrent.get("failures", 0))
    enough_load = requests >= policy.min_concurrent_requests and workers >= policy.min_workers
    checks.append(
        CapacityCheck(
            "minimum_load_sample",
            "pass" if enough_load else "fail",
            requests,
            policy.min_concurrent_requests,
            f"동시 표본 {requests}건·worker {workers}개를 실행했습니다.",
        )
    )
    complete = requests > 0 and successes == requests and failures == 0
    checks.append(
        CapacityCheck(
            "concurrent_success",
            "pass" if complete else "fail",
            successes,
            requests,
            f"동시 요청 성공 {successes}/{requests}, 실패 {failures}건입니다.",
        )
    )

    for name, benchmark in (("local_p95", local), ("concurrent_p95", concurrent)):
        observed = benchmark.get("p95_seconds")
        valid = isinstance(observed, (int, float)) and observed <= policy.deterministic_p95_seconds
        checks.append(
            CapacityCheck(
                name,
                "pass" if valid else "fail",
                observed if isinstance(observed, (int, float)) else None,
                policy.deterministic_p95_seconds,
                "결정적 로컬 분석 p95가 제한 안에 있습니다."
                if valid
                else "결정적 로컬 분석 p95가 없거나 제한을 초과했습니다.",
            )
        )

    rss_values = [
        value
        for value in (
            local.get("process_peak_rss_bytes"),
            concurrent.get("process_peak_rss_bytes"),
        )
        if isinstance(value, int)
    ]
    peak_rss = max(rss_values, default=None)
    if peak_rss is None or peak_rss >= policy.rss_critical_bytes:
        rss_status = "fail"
        rss_message = "peak RSS가 없거나 위험 기준 이상입니다."
    elif peak_rss >= policy.rss_warning_bytes:
        rss_status = "warn"
        rss_message = "peak RSS가 경고 기준 이상입니다. 용량 확장 또는 재시작 정책을 검토해야 합니다."
    else:
        rss_status = "pass"
        rss_message = "peak RSS가 경고 기준보다 낮습니다."
    checks.append(
        CapacityCheck(
            "peak_rss",
            rss_status,
            peak_rss,
            policy.rss_warning_bytes,
            rss_message,
        )
    )

    return {
        "ready": all(check.status != "fail" for check in checks),
        "policy": policy.public(),
        "checks": [check.public() for check in checks],
        "safety": observed_safety,
    }
