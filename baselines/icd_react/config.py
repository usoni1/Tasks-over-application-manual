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
DEFAULT_DATABRICKS_MANUALS_ROOT = Path("/Volumes/usdo_aa_catalog/research_tam_datasets/mimic/ICD_manual")


@dataclass(frozen=True)
class ICDReactConfig:
    execution_env: str
    manuals_root: Path
    guidelines_pdf_path: Path
    index_xml_path: Path
    tabular_xml_path: Path


def first_non_empty(*values: str | None) -> str | None:
    for value in values:
        if value and value.strip():
            return value.strip()
    return None


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


def resolve_execution_env(overrides: Namespace | None = None) -> str:
    execution_env = override_or_env(overrides, "execution_env", "EXECUTION_ENV", "local") or "local"
    execution_env = execution_env.strip().lower()
    if execution_env not in {"local", "databricks"}:
        raise RuntimeError("EXECUTION_ENV must be either 'local' or 'databricks'")
    return execution_env


def resolve_manuals_root(execution_env: str) -> Path:
    root_value = first_non_empty(
        os.environ.get(f"ICD_MANUALS_ROOT_{execution_env.upper()}"),
        os.environ.get("ICD_MANUALS_ROOT"),
        str(DEFAULT_DATABRICKS_MANUALS_ROOT) if execution_env == "databricks" else str(DEFAULT_LOCAL_MANUALS_ROOT),
    )
    root = Path(root_value or "").expanduser()
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()
    return require_path(root, "ICD manuals root")


def load_config(overrides: Namespace | None = None) -> ICDReactConfig:
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    execution_env = resolve_execution_env(overrides)
    manuals_root = resolve_manuals_root(execution_env)
    guidelines_pdf_path = require_path(manuals_root / "2019-icd10-coding-guidelines-.pdf", "ICD guidelines PDF")
    index_xml_path = require_path(manuals_root / "icd10cm_tabular_2019" / "icd10cm_index_2019.xml", "ICD index XML")
    tabular_xml_path = require_path(manuals_root / "icd10cm_tabular_2019" / "icd10cm_tabular_2019.xml", "ICD tabular XML")
    return ICDReactConfig(
        execution_env=execution_env,
        manuals_root=manuals_root,
        guidelines_pdf_path=guidelines_pdf_path,
        index_xml_path=index_xml_path,
        tabular_xml_path=tabular_xml_path,
    )