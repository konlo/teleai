"""Secret-safe, read-only deployment checks for the Telly analysis agent."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse

from core.analysis_agent.policy import RuntimePolicy


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    status: str
    message: str

    def public(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class PreflightReport:
    profile: str
    checks: tuple[PreflightCheck, ...]

    @property
    def ready(self) -> bool:
        return all(check.status != "fail" for check in self.checks)

    def public(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "ready": self.ready,
            "checks": [check.public() for check in self.checks],
        }


def _configured(env: Mapping[str, str], *names: str) -> bool:
    return any(bool(env.get(name, "").strip()) for name in names)


def _check_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _storage_check(
    env: Mapping[str, str], profile: str, project_root: Path
) -> PreflightCheck:
    value = env.get("TELLY_V1_STORAGE", "").strip()
    if not value:
        return PreflightCheck(
            "persistent_storage",
            "warn" if profile == "local-desktop" else "fail",
            "TELLY_V1_STORAGE가 없어 프로젝트 내부 기본 저장소를 사용합니다. 배포에는 영속 볼륨의 절대 경로가 필요합니다.",
        )
    path = Path(value).expanduser()
    if not path.is_absolute():
        return PreflightCheck(
            "persistent_storage", "fail", "TELLY_V1_STORAGE는 절대 경로여야 합니다."
        )
    try:
        path.resolve().relative_to(project_root.resolve())
    except ValueError:
        pass
    else:
        status = "warn" if profile == "local-desktop" else "fail"
        return PreflightCheck(
            "persistent_storage",
            status,
            "저장소가 코드 checkout 안에 있습니다. 배포 시 코드 교체와 분리된 영속 볼륨을 사용해야 합니다.",
        )
    probe = path if path.exists() else path.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if not probe.is_dir():
        return PreflightCheck(
            "persistent_storage", "fail", "저장소 상위 경로를 찾을 수 없습니다."
        )
    import os

    if not os.access(probe, os.W_OK):
        return PreflightCheck(
            "persistent_storage", "fail", "저장소 또는 상위 경로에 쓰기 권한이 없습니다."
        )
    return PreflightCheck(
        "persistent_storage", "pass", "코드 checkout과 분리된 쓰기 가능한 절대 경로가 설정되었습니다."
    )


def evaluate_deployment(
    env: Mapping[str, str],
    *,
    profile: str,
    project_root: Path,
) -> PreflightReport:
    """Evaluate configuration without connecting to Ollama or Databricks."""
    if profile not in {"local-desktop", "private-single-user", "multi-user"}:
        raise ValueError(f"unsupported profile: {profile}")

    checks: list[PreflightCheck] = []
    model = env.get("OLLAMA_MODEL", "").strip()
    endpoint = env.get("OLLAMA_BASE_URL", "").strip()
    if not model or not endpoint:
        checks.append(
            PreflightCheck(
                "ollama_configuration",
                "fail",
                "OLLAMA_MODEL과 OLLAMA_BASE_URL을 배포 환경에 명시해야 합니다.",
            )
        )
    elif not _check_url(endpoint):
        checks.append(
            PreflightCheck(
                "ollama_configuration", "fail", "OLLAMA_BASE_URL은 유효한 http(s) URL이어야 합니다."
            )
        )
    else:
        host = (urlparse(endpoint).hostname or "").lower()
        remote_profile = profile != "local-desktop"
        if remote_profile and host in {"localhost", "127.0.0.1", "::1"}:
            checks.append(
                PreflightCheck(
                    "ollama_configuration",
                    "warn",
                    "Ollama가 loopback 주소입니다. 앱과 모델 서버가 같은 배포 호스트에서 실행되는지 확인해야 합니다.",
                )
            )
        else:
            checks.append(
                PreflightCheck(
                    "ollama_configuration", "pass", "Ollama 모델과 endpoint가 명시되었습니다."
                )
            )

    required_connection = (
        _configured(env, "DATABRICKS_HOST"),
        _configured(env, "DATABRICKS_HTTP_PATH"),
        _configured(env, "DATABRICKS_TOKEN", "DATABRICKS_ACCESS_TOKEN"),
        _configured(env, "DATABRICKS_CATALOG"),
        _configured(env, "DATABRICKS_SCHEMA"),
    )
    if all(required_connection):
        checks.append(
            PreflightCheck(
                "databricks_configuration",
                "pass",
                "Databricks 연결 변수 이름이 모두 설정되었습니다. 네트워크 연결이나 SQL은 실행하지 않았습니다.",
            )
        )
    else:
        checks.append(
            PreflightCheck(
                "databricks_configuration",
                "fail",
                "Databricks host, HTTP path, token, catalog, schema 설정이 완전하지 않습니다.",
            )
        )

    checks.append(_storage_check(env, profile, project_root))

    try:
        # RuntimePolicy reads os.environ in production. Recreate its validation
        # against the supplied mapping without mutating process state.
        policy_values = {
            "max_remote_rows": int(env.get("TELLY_MAX_REMOTE_ROWS", RuntimePolicy.max_remote_rows)),
            "max_dataset_columns": int(env.get("TELLY_MAX_DATASET_COLUMNS", RuntimePolicy.max_dataset_columns)),
            "max_dataset_bytes": int(env.get("TELLY_MAX_DATASET_BYTES", RuntimePolicy.max_dataset_bytes)),
            "frame_cache_bytes": int(env.get("TELLY_FRAME_CACHE_BYTES", RuntimePolicy.frame_cache_bytes)),
            "scope_disk_quota_bytes": int(env.get("TELLY_SCOPE_DISK_QUOTA_BYTES", RuntimePolicy.scope_disk_quota_bytes)),
            "retention_days": int(env.get("TELLY_RETENTION_DAYS", RuntimePolicy.retention_days)),
            "model_timeout_seconds": float(env.get("TELLY_MODEL_TIMEOUT_SECONDS", RuntimePolicy.model_timeout_seconds)),
            "turn_slo_seconds": float(env.get("TELLY_TURN_SLO_SECONDS", RuntimePolicy.turn_slo_seconds)),
        }
        if any(value <= 0 for value in policy_values.values()):
            raise ValueError("non-positive policy")
        RuntimePolicy(**policy_values)
    except (TypeError, ValueError):
        checks.append(
            PreflightCheck(
                "runtime_policy", "fail", "TELLY 운영 한도는 모두 0보다 큰 숫자여야 합니다."
            )
        )
    else:
        checks.append(
            PreflightCheck("runtime_policy", "pass", "테이블 중립 운영 한도가 유효합니다.")
        )

    if profile == "local-desktop":
        checks.append(
            PreflightCheck(
                "identity_scope",
                "pass",
                "loopback의 단일 사용자 범위는 현재 local-owner 저장 구조와 일치합니다.",
            )
        )
    elif profile == "private-single-user":
        confirmed = env.get("TELLY_EXTERNAL_ACCESS_CONTROL", "").strip().lower()
        if confirmed in {"1", "true", "yes", "confirmed"}:
            checks.append(
                PreflightCheck(
                    "identity_scope",
                    "warn",
                    "외부 접근제어가 확인됐습니다. 이 revision은 한 명의 사용자만 사용해야 합니다.",
                )
            )
        else:
            checks.append(
                PreflightCheck(
                    "identity_scope",
                    "fail",
                    "현재 앱은 local-owner를 사용합니다. 한 명만 접근하도록 외부 접근제어를 확인해야 합니다.",
                )
            )
    else:
        checks.append(
            PreflightCheck(
                "identity_scope",
                "fail",
                "현재 revision에는 사용자 인증과 사용자별 owner binding이 없어 다중 사용자 배포를 지원하지 않습니다.",
            )
        )

    return PreflightReport(profile=profile, checks=tuple(checks))
