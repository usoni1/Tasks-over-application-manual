from __future__ import annotations

import logging
import math
from time import monotonic


COMMON_NOISY_LOGGER_NAMES = [
    "databricks",
    "databricks.sdk",
    "databricks.sdk.core",
    "databricks_cli",
    "urllib3",
    "azure",
    "azure.core",
    "azure.core.pipeline",
    "azure.core.pipeline.policies.http_logging_policy",
    "azure.search",
    "azure.search.documents",
    "httpx",
    "httpcore",
    "openai",
    "openai._base_client",
    "openai._client",
    "py4j",
    "py4j.clientserver",
    "py4j.java_gateway",
    "mlflow.models.evaluation.utils.trace",
]


def suppress_noisy_loggers(*extra_logger_names: str) -> None:
    for logger_name in [*COMMON_NOISY_LOGGER_NAMES, *extra_logger_names]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class EvaluationProgressLogger:
    def __init__(self, *, logger: logging.Logger, label: str, total: int, log_every: int | None = None) -> None:
        self.logger = logger
        self.label = label
        self.total = max(int(total), 0)
        self.started_at = monotonic()
        self.completed = 0
        self.ok_count = 0
        self.error_count = 0
        self.log_every = log_every if log_every is not None else self._default_log_every(self.total)

    @staticmethod
    def _default_log_every(total: int) -> int:
        if total <= 0:
            return 1
        return max(1, min(25, total // 20 or 1))

    def log_start(self) -> None:
        self.logger.info("%s progress started: total_cases=%s update_every=%s", self.label, self.total, self.log_every)

    def update(self, *, status: str) -> None:
        self.completed += 1
        if status == "ok":
            self.ok_count += 1
        else:
            self.error_count += 1

        should_log = (
            self.completed == 1
            or self.completed == self.total
            or (self.log_every > 0 and self.completed % self.log_every == 0)
        )
        if should_log:
            self.log_snapshot()

    def log_snapshot(self) -> None:
        elapsed_seconds = max(monotonic() - self.started_at, 0.0)
        rate = (self.completed / elapsed_seconds) if elapsed_seconds > 0 and self.completed > 0 else 0.0
        remaining = max(self.total - self.completed, 0)
        eta_seconds = (remaining / rate) if rate > 0 else None
        percentage = (100.0 * self.completed / self.total) if self.total > 0 else 100.0
        self.logger.info(
            "%s progress: %s/%s (%.1f%%) ok=%s error=%s elapsed=%s eta=%s rate=%.2f case/s",
            self.label,
            self.completed,
            self.total,
            percentage,
            self.ok_count,
            self.error_count,
            format_duration(elapsed_seconds),
            format_duration(eta_seconds),
            rate,
        )
