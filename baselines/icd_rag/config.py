from __future__ import annotations

import os
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def resolve_project_root() -> Path:
    script_path = globals().get("__file__")
    if script_path:
        return Path(script_path).resolve().parents[2]

    cwd = Path.cwd().resolve()
    return cwd


PROJECT_ROOT = resolve_project_root()
DEFAULT_LOCAL_MANUALS_ROOT = Path("data/reference-manuals/ICD-2019-manual")


@dataclass(frozen=True)
class SearchServiceConfig:
    name: str
    endpoint: str
    api_key: str


@dataclass(frozen=True)
class ICDRAGConfig:
    execution_env: str
    manuals_root: Path
    search_service: SearchServiceConfig
    index_name: str
    embedding_deployment: str
    openai_api_type: str
    openai_api_base: str
    openai_api_key: str
    openai_api_version: str
    include_guidelines: bool
    include_tabular: bool
    include_index: bool
    max_chunk_chars: int
    embedding_batch_size: int
    upload_batch_size: int
    chars_per_token_estimate: float
    embedding_cost_per_1m_tokens: float | None


def str_to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def first_non_empty(*values: str | None) -> str | None:
    for value in values:
        if value and value.strip():
            return value.strip()
    return None


def require_env(name: str, value: str | None) -> str:
    if value and value.strip():
        return value.strip()
    raise RuntimeError(f"Missing required environment variable: {name}")


def override_or_env(overrides: Namespace | None, name: str, env_name: str, default: str | None = None) -> str | None:
    if overrides is not None:
        override_value = getattr(overrides, name, None)
        if override_value is not None:
            return str(override_value)
    return first_non_empty(os.environ.get(env_name), default)


def override_or_bool(overrides: Namespace | None, name: str, env_name: str, default: bool) -> bool:
    if overrides is not None:
        override_value = getattr(overrides, name, None)
        if override_value is not None:
            return bool(override_value)
    return str_to_bool(os.environ.get(env_name), default=default)


def resolve_search_service(service_name: str) -> SearchServiceConfig:
    suffix = "" if service_name == "service_1" else f"_{service_name.split('_')[-1]}"
    endpoint = first_non_empty(
        os.environ.get(f"AZURE_SEARCH_ENDPOINT{suffix}"),
        os.environ.get(f"AZURE_AI_SEARCH_ENDPOINT{suffix}"),
    )
    api_key = first_non_empty(
        os.environ.get(f"AZURE_SEARCH_API_KEY{suffix}"),
        os.environ.get(f"AZURE_AI_SEARCH_KEY{suffix}"),
    )
    return SearchServiceConfig(
        name=service_name,
        endpoint=require_env(f"AZURE_SEARCH_ENDPOINT{suffix}", endpoint),
        api_key=require_env(f"AZURE_SEARCH_API_KEY{suffix}", api_key),
    )


def resolve_manuals_root(execution_env: str) -> Path:
    root_value = first_non_empty(
        os.environ.get(f"ICD_MANUALS_ROOT_{execution_env.upper()}"),
        os.environ.get("ICD_MANUALS_ROOT"),
        str(DEFAULT_LOCAL_MANUALS_ROOT) if execution_env == "local" else None,
    )
    root = Path(require_env("ICD_MANUALS_ROOT", root_value)).expanduser()
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()
    return root


def load_config(overrides: Namespace | None = None) -> ICDRAGConfig:
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    execution_env = override_or_env(overrides, "execution_env", "EXECUTION_ENV", "local")
    if execution_env is None:
        execution_env = "local"
    execution_env = execution_env.strip().lower()
    if execution_env not in {"local", "databricks"}:
        raise RuntimeError("EXECUTION_ENV must be either 'local' or 'databricks'")

    search_service_name = override_or_env(overrides, "search_service", "ICD_AZURE_SEARCH_SERVICE", "service_1")
    index_name = override_or_env(overrides, "index_name", "ICD_AZURE_SEARCH_INDEX_NAME", "icd-rag-baseline")
    cost_value = override_or_env(overrides, "embedding_cost_per_1m_tokens", "ICD_EMBEDDING_COST_PER_1M_TOKENS")
    return ICDRAGConfig(
        execution_env=execution_env,
        manuals_root=resolve_manuals_root(execution_env),
        search_service=resolve_search_service(search_service_name or "service_1"),
        index_name=(index_name or "icd-rag-baseline").strip(),
        embedding_deployment=require_env(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
            first_non_empty(
                os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
                os.environ.get("AZURE_OPENAI_EMBEDDING_MODEL"),
            ),
        ),
        openai_api_type=os.environ.get("OPENAI_API_TYPE", "azure").strip(),
        openai_api_base=require_env(
            "OPENAI_API_BASE",
            first_non_empty(os.environ.get("OPENAI_API_BASE"), os.environ.get("AZURE_OPENAI_ENDPOINT")),
        ),
        openai_api_key=require_env(
            "OPENAI_API_KEY",
            first_non_empty(os.environ.get("OPENAI_API_KEY"), os.environ.get("AZURE_OPENAI_API_KEY")),
        ),
        openai_api_version=os.environ.get("OPENAI_API_VERSION", "2025-04-01-preview").strip(),
        include_guidelines=override_or_bool(overrides, "include_guidelines", "ICD_INCLUDE_GUIDELINES", default=True),
        include_tabular=override_or_bool(overrides, "include_tabular", "ICD_INCLUDE_TABULAR", default=True),
        include_index=override_or_bool(overrides, "include_index", "ICD_INCLUDE_INDEX", default=True),
        max_chunk_chars=int(override_or_env(overrides, "max_chunk_chars", "ICD_MAX_CHUNK_CHARS", "12000") or "12000"),
        embedding_batch_size=int(override_or_env(overrides, "embedding_batch_size", "ICD_EMBEDDING_BATCH_SIZE", "32") or "32"),
        upload_batch_size=int(override_or_env(overrides, "upload_batch_size", "ICD_AZURE_UPLOAD_BATCH_SIZE", "100") or "100"),
        chars_per_token_estimate=float(override_or_env(overrides, "chars_per_token_estimate", "ICD_CHARS_PER_TOKEN_ESTIMATE", "4.0") or "4.0"),
        embedding_cost_per_1m_tokens=float(cost_value) if cost_value is not None else None,
    )