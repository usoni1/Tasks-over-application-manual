from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from baselines.legal_rag.runtime import resolve_model_name as resolve_legal_model_name
from baselines.legal_rag.runtime import resolve_spark_session as resolve_legal_spark_session


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MLFLOW_EXPERIMENT = "review-pipeline-v1-single-case"


def first_non_empty(*values: str | None) -> str | None:
    for value in values:
        if value is None:
            continue
        stripped = str(value).strip()
        if stripped:
            return stripped
    return None


def load_runtime_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def resolve_execution_env() -> str:
    load_runtime_env()
    execution_env = first_non_empty(os.environ.get("EXECUTION_ENV"), "local") or "local"
    normalized = execution_env.strip().lower()
    if normalized not in {"local", "databricks"}:
        raise RuntimeError("EXECUTION_ENV must be either 'local' or 'databricks'.")
    return normalized


def resolve_model_name() -> str:
    load_runtime_env()
    return resolve_legal_model_name()


def resolve_spark_session(app_name: str | None = None) -> Any:
    load_runtime_env()
    return resolve_legal_spark_session(app_name=app_name or "review-pipeline-v1-single-case")


def resolve_mlflow_experiment() -> str:
    load_runtime_env()
    return (
        first_non_empty(
            os.environ.get("REVIEW_PIPELINE_V1_MLFLOW_EXPERIMENT"),
            os.environ.get("MLFLOW_EXPERIMENT_REVIEW_PIPELINE_V1"),
            DEFAULT_MLFLOW_EXPERIMENT,
        )
        or DEFAULT_MLFLOW_EXPERIMENT
    )


__all__ = [
    "DEFAULT_MLFLOW_EXPERIMENT",
    "PROJECT_ROOT",
    "first_non_empty",
    "load_runtime_env",
    "resolve_execution_env",
    "resolve_mlflow_experiment",
    "resolve_model_name",
    "resolve_spark_session",
]