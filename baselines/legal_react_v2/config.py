from __future__ import annotations

import os
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from baselines.legal_rag.config import (
    DEFAULT_LOCAL_USSG_DOCINTEL_TEXT_ROOT,
    PROJECT_ROOT,
    first_non_empty,
    resolve_base_manuals_root,
    resolve_manuals_subdir,
    resolve_optional_path,
)
from baselines.legal_rag.title18_paths import has_title18_manual_years


def override_or_env(overrides: Namespace | None, name: str, env_name: str, default: str | None = None) -> str | None:
    if overrides is not None:
        override_value = getattr(overrides, name, None)
        if override_value is not None:
            return str(override_value)
    return first_non_empty(os.environ.get(env_name), default)


def require_path(path: Path, description: str) -> Path:
    if not path.exists():
        raise RuntimeError(f"Missing required {description}: {path}")
    return path


def prefer_existing_path(primary: Path, fallback: Path | None = None) -> Path:
    if primary.exists():
        return primary
    if fallback is not None and fallback.exists():
        return fallback
    return primary


def prefer_existing_optional_path(primary: Path | None, fallback: Path | None = None) -> Path | None:
    if primary is not None and primary.exists():
        return primary
    if fallback is not None and fallback.exists():
        return fallback
    return None


def has_title18_html_years(root: Path) -> bool:
    return has_title18_manual_years(root)


def resolve_execution_env(overrides: Namespace | None = None) -> str:
    execution_env = override_or_env(overrides, "execution_env", "EXECUTION_ENV", "local") or "local"
    execution_env = execution_env.strip().lower()
    if execution_env not in {"local", "databricks"}:
        raise RuntimeError("EXECUTION_ENV must be either 'local' or 'databricks'")
    return execution_env


@dataclass(frozen=True)
class LegalReactV2Config:
    execution_env: str
    manuals_root: Path
    title18_root: Path
    ussg_docintel_text_root: Path | None


def load_config(overrides: Namespace | None = None) -> LegalReactV2Config:
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    execution_env = resolve_execution_env(overrides)
    local_repo_manuals_root = (PROJECT_ROOT / "data" / "reference-manuals" / "legal").resolve()
    local_repo_title18_root = (PROJECT_ROOT / "data" / "reference-manuals" / "legal" / "usc_title18").resolve()
    local_repo_docintel_root = (PROJECT_ROOT / DEFAULT_LOCAL_USSG_DOCINTEL_TEXT_ROOT).resolve()

    manuals_root = resolve_base_manuals_root(execution_env)
    if execution_env == "local":
        manuals_root = prefer_existing_path(manuals_root, local_repo_manuals_root)
    manuals_root = require_path(manuals_root, "legal manuals root")

    explicit_title18_root = first_non_empty(
        os.environ.get(f"LEGAL_USC_TITLE18_ROOT_{execution_env.upper()}"),
        os.environ.get("LEGAL_USC_TITLE18_ROOT"),
        os.environ.get(f"LEGAL_TITLE18_ROOT_{execution_env.upper()}"),
        os.environ.get("LEGAL_TITLE18_ROOT"),
    )
    title18_root = require_path(
        (
            local_repo_title18_root
            if execution_env == "local"
            and has_title18_html_years(local_repo_title18_root)
            and not has_title18_html_years(
                resolve_manuals_subdir(
                    manuals_root,
                    first_non_empty(explicit_title18_root, "data/reference-manuals/legal/usc_title18"),
                    "usc_title18",
                )
            )
            else prefer_existing_path(
                resolve_manuals_subdir(
                    manuals_root,
                    first_non_empty(explicit_title18_root, "data/reference-manuals/legal/usc_title18"),
                    "usc_title18",
                ),
                local_repo_title18_root if execution_env == "local" else None,
            )
        ),
        "Title 18 manuals root",
    )

    explicit_ussg_docintel_text_root = first_non_empty(
        override_or_env(overrides, "ussg_docintel_text_root", "LEGAL_USSG_DOCINTEL_TEXT_ROOT"),
        os.environ.get(f"LEGAL_USSG_DOCINTEL_TEXT_ROOT_{execution_env.upper()}"),
    )
    ussg_docintel_text_root = resolve_optional_path(
        first_non_empty(
            explicit_ussg_docintel_text_root,
            str(DEFAULT_LOCAL_USSG_DOCINTEL_TEXT_ROOT) if execution_env == "local" else None,
            str(manuals_root / "_docintel_text" / "ussg") if execution_env == "databricks" else None,
        )
    )
    if execution_env == "local":
        ussg_docintel_text_root = prefer_existing_optional_path(ussg_docintel_text_root, local_repo_docintel_root)

    return LegalReactV2Config(
        execution_env=execution_env,
        manuals_root=manuals_root,
        title18_root=title18_root,
        ussg_docintel_text_root=ussg_docintel_text_root,
    )