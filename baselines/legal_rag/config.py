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
DEFAULT_DOCINTEL_ENDPOINT = "https://ltc-uat-exp.cognitiveservices.azure.com/"
DEFAULT_DATABRICKS_LEGAL_MANUALS_ROOT = Path(
    "/Volumes/usdo_aa_catalog/research_tam_datasets/federal_sentencing/legal_manuals"
)
DEFAULT_LOCAL_LEGAL_MANUALS_ROOT = Path("data/reference-manuals/legal")
DEFAULT_LOCAL_USSG_DOCINTEL_TEXT_ROOT = Path("data/reference-manuals/legal/_docintel_text/ussg")


@dataclass(frozen=True)
class SearchServiceConfig:
    name: str
    endpoint: str
    api_key: str


@dataclass(frozen=True)
class LegalRAGConfig:
    execution_env: str
    manuals_root: Path
    ussg_root: Path
    title18_root: Path
    ussg_docintel_text_root: Path | None
    search_service: SearchServiceConfig
    index_name: str
    embedding_deployment: str
    openai_api_type: str
    openai_api_base: str
    openai_api_key: str
    openai_api_version: str
    include_ussg: bool
    include_title18: bool
    use_docintel_for_ussg: bool
    docintel_endpoint: str | None
    docintel_key: str | None
    docintel_model: str
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


def resolve_base_manuals_root(execution_env: str) -> Path:
    root_value = first_non_empty(
        os.environ.get(f"LEGAL_MANUALS_ROOT_{execution_env.upper()}"),
        os.environ.get("LEGAL_MANUALS_ROOT"),
        str(DEFAULT_DATABRICKS_LEGAL_MANUALS_ROOT)
        if execution_env == "databricks"
        else str(DEFAULT_LOCAL_LEGAL_MANUALS_ROOT),
    )
    root = Path(require_env("LEGAL_MANUALS_ROOT", root_value)).expanduser()
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()
    return root


def resolve_manuals_subdir(base_root: Path, explicit_env_value: str | None, default_relative: str) -> Path:
    if explicit_env_value:
        path = Path(explicit_env_value).expanduser()
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        return path
    return (base_root / default_relative).resolve()


def resolve_optional_path(explicit_env_value: str | None) -> Path | None:
    if not explicit_env_value:
        return None
    path = Path(explicit_env_value).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def load_config(overrides: Namespace | None = None) -> LegalRAGConfig:
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    execution_env = override_or_env(overrides, "execution_env", "EXECUTION_ENV", "local")
    if execution_env is None:
        execution_env = "local"
    execution_env = execution_env.strip().lower()
    if execution_env not in {"local", "databricks"}:
        raise RuntimeError("EXECUTION_ENV must be either 'local' or 'databricks'")

    explicit_manuals_root = first_non_empty(
        os.environ.get(f"LEGAL_MANUALS_ROOT_{execution_env.upper()}"),
        os.environ.get("LEGAL_MANUALS_ROOT"),
    )
    manuals_root = resolve_base_manuals_root(execution_env)
    ussg_root = resolve_manuals_subdir(
        manuals_root,
        first_non_empty(
            os.environ.get(f"LEGAL_USSG_ROOT_{execution_env.upper()}"),
            os.environ.get("LEGAL_USSG_ROOT"),
        ),
        "ussg",
    )
    explicit_title18_root = first_non_empty(
        os.environ.get(f"LEGAL_USC_TITLE18_ROOT_{execution_env.upper()}"),
        os.environ.get("LEGAL_USC_TITLE18_ROOT"),
        os.environ.get(f"LEGAL_TITLE18_ROOT_{execution_env.upper()}"),
        os.environ.get("LEGAL_TITLE18_ROOT"),
    )
    title18_root = resolve_manuals_subdir(
        manuals_root,
        first_non_empty(
            explicit_title18_root,
            "data/reference-manuals/legal/usc_title18" if execution_env == "local" and explicit_manuals_root is None else None,
        ),
        "usc_title18",
    )
    explicit_ussg_docintel_text_root = first_non_empty(
        override_or_env(overrides, "ussg_docintel_text_root", "LEGAL_USSG_DOCINTEL_TEXT_ROOT"),
        os.environ.get(f"LEGAL_USSG_DOCINTEL_TEXT_ROOT_{execution_env.upper()}"),
    )
    ussg_docintel_text_root = resolve_optional_path(
        first_non_empty(
            explicit_ussg_docintel_text_root,
            str(DEFAULT_LOCAL_USSG_DOCINTEL_TEXT_ROOT) if execution_env == "local" and explicit_manuals_root is None else None,
            str(manuals_root / "_docintel_text" / "ussg") if execution_env == "databricks" else None,
            str(manuals_root / "_docintel_text" / "ussg") if execution_env == "local" and explicit_manuals_root is not None else None,
        )
    )

    search_service_name = override_or_env(overrides, "search_service", "LEGAL_AZURE_SEARCH_SERVICE", "service_1")
    index_name = override_or_env(overrides, "index_name", "LEGAL_AZURE_SEARCH_INDEX_NAME", "legal-rag-baseline")
    cost_value = override_or_env(overrides, "embedding_cost_per_1m_tokens", "LEGAL_EMBEDDING_COST_PER_1M_TOKENS")
    docintel_endpoint = first_non_empty(
        override_or_env(overrides, "docintel_endpoint", "LEGAL_DOCINTEL_ENDPOINT"),
        os.environ.get("AZURE_DOCINTEL_ENDPOINT"),
        os.environ.get("AZURE_DOC_ENDPOINT"),
        DEFAULT_DOCINTEL_ENDPOINT if os.environ.get("AZURE_DOCINTEL_KEY") else None,
    )
    docintel_key = first_non_empty(
        override_or_env(overrides, "docintel_key", "LEGAL_DOCINTEL_KEY"),
        os.environ.get("AZURE_DOCINTEL_KEY"),
        os.environ.get("AZURE_DOC_KEY"),
    )

    return LegalRAGConfig(
        execution_env=execution_env,
        manuals_root=manuals_root,
        ussg_root=ussg_root,
        title18_root=title18_root,
        ussg_docintel_text_root=ussg_docintel_text_root,
        search_service=resolve_search_service(search_service_name or "service_1"),
        index_name=(index_name or "legal-rag-baseline").strip(),
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
        include_ussg=override_or_bool(overrides, "include_ussg", "LEGAL_INCLUDE_USSG", default=True),
        include_title18=override_or_bool(overrides, "include_title18", "LEGAL_INCLUDE_TITLE18", default=True),
        use_docintel_for_ussg=override_or_bool(
            overrides,
            "use_docintel_for_ussg",
            "LEGAL_USE_DOCINTEL_FOR_USSG",
            default=True,
        ),
        docintel_endpoint=docintel_endpoint,
        docintel_key=docintel_key,
        docintel_model=(override_or_env(overrides, "docintel_model", "LEGAL_DOCINTEL_MODEL", "prebuilt-layout") or "prebuilt-layout").strip(),
        max_chunk_chars=int(override_or_env(overrides, "max_chunk_chars", "LEGAL_MAX_CHUNK_CHARS", "12000") or "12000"),
        embedding_batch_size=int(override_or_env(overrides, "embedding_batch_size", "LEGAL_EMBEDDING_BATCH_SIZE", "32") or "32"),
        upload_batch_size=int(override_or_env(overrides, "upload_batch_size", "LEGAL_AZURE_UPLOAD_BATCH_SIZE", "100") or "100"),
        chars_per_token_estimate=float(override_or_env(overrides, "chars_per_token_estimate", "LEGAL_CHARS_PER_TOKEN_ESTIMATE", "4.0") or "4.0"),
        embedding_cost_per_1m_tokens=float(cost_value) if cost_value is not None else None,
    )

