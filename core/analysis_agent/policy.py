"""Operational limits for the persistent analysis agent.

Values are table-neutral and may be overridden per deployment through TELLY_*.
Retention discovery is deliberately non-destructive; a separate maintenance
command must explicitly apply removal of expired conversation scopes.
"""
from dataclasses import asdict, dataclass
import os


def _positive_int(name, default):
    value = int(os.getenv(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_float(name, default):
    value = float(os.getenv(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class RuntimePolicy:
    max_remote_rows: int = 100_000
    max_dataset_columns: int = 256
    max_dataset_bytes: int = 512 * 1024 * 1024
    max_join_rows: int = 100_000
    max_join_expansion_ratio: float = 5.0
    frame_cache_bytes: int = 64 * 1024 * 1024
    scope_disk_quota_bytes: int = 2 * 1024 * 1024 * 1024
    retention_days: int = 30
    model_timeout_seconds: float = 60.0
    turn_slo_seconds: float = 180.0

    @classmethod
    def from_env(cls):
        return cls(
            max_remote_rows=_positive_int("TELLY_MAX_REMOTE_ROWS", cls.max_remote_rows),
            max_dataset_columns=_positive_int("TELLY_MAX_DATASET_COLUMNS", cls.max_dataset_columns),
            max_dataset_bytes=_positive_int("TELLY_MAX_DATASET_BYTES", cls.max_dataset_bytes),
            max_join_rows=_positive_int("TELLY_MAX_JOIN_ROWS", cls.max_join_rows),
            max_join_expansion_ratio=_positive_float(
                "TELLY_MAX_JOIN_EXPANSION_RATIO", cls.max_join_expansion_ratio),
            frame_cache_bytes=_positive_int("TELLY_FRAME_CACHE_BYTES", cls.frame_cache_bytes),
            scope_disk_quota_bytes=_positive_int("TELLY_SCOPE_DISK_QUOTA_BYTES", cls.scope_disk_quota_bytes),
            retention_days=_positive_int("TELLY_RETENTION_DAYS", cls.retention_days),
            model_timeout_seconds=_positive_float("TELLY_MODEL_TIMEOUT_SECONDS", cls.model_timeout_seconds),
            turn_slo_seconds=_positive_float("TELLY_TURN_SLO_SECONDS", cls.turn_slo_seconds),
        )

    def public(self):
        """Safe diagnostics/UI representation without connection information."""
        return asdict(self)
