from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

from .config import PROJECT_ROOT, first_non_empty, resolve_execution_env
from standalone_data import DEFAULT_LOCAL_ICD_STRICT_TABLE, resolve_spark_session as resolve_local_spark_session


DEFAULT_STRICT_TABLE = DEFAULT_LOCAL_ICD_STRICT_TABLE
DEFAULT_MODEL_NAME = "openai:gpt-5"
DEFAULT_MLFLOW_EXPERIMENT = "icd-react-single-case"


def load_runtime_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def resolve_spark_session(app_name: str | None = None) -> Any:
    load_runtime_env()
    return resolve_local_spark_session(execution_env=resolve_execution_env(), app_name=app_name)


def resolve_strict_table() -> str:
    load_runtime_env()
    return (
        first_non_empty(
            os.environ.get("ICD_REACT_STRICT_TABLE"),
            os.environ.get("ICD_STRICT_TABLE"),
            os.environ.get("STRICT_TABLE"),
            DEFAULT_STRICT_TABLE,
        )
        or DEFAULT_STRICT_TABLE
    )


def resolve_model_name() -> str:
    load_runtime_env()
    return (
        first_non_empty(
            os.environ.get("ICD_REACT_MODEL_NAME"),
            os.environ.get("ICD_MODEL_NAME"),
            os.environ.get("OPENAI_CHAT_MODEL"),
            os.environ.get("MODEL_NAME"),
            DEFAULT_MODEL_NAME,
        )
        or DEFAULT_MODEL_NAME
    )


def resolve_mlflow_experiment() -> str:
    load_runtime_env()
    return (
        first_non_empty(
            os.environ.get("ICD_REACT_MLFLOW_EXPERIMENT"),
            os.environ.get("MLFLOW_EXPERIMENT_ICD_REACT"),
            DEFAULT_MLFLOW_EXPERIMENT,
        )
        or DEFAULT_MLFLOW_EXPERIMENT
    )