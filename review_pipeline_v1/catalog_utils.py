"""Shared helpers for catalog-backed federal sentencing PDFs and Doc Intelligence artifacts.

This module centralizes path and traversal logic for both raw case PDFs and the
Doc Intelligence JSON exports stored under docket-organized catalog folders.
Scripts, agents, and notebooks can reuse the same resolution and iteration
behavior instead of duplicating volume-specific path handling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_CATALOG_CASE_PDF_ROOT = Path("/Volumes/usdo_aa_catalog/research_tam_datasets/federal_sentencing/cases/pdfs")
DEFAULT_CATALOG_DOCINTEL_TEXT_ROOT = Path("/Volumes/usdo_aa_catalog/research_tam_datasets/federal_sentencing/cases/docintel_text")
DEFAULT_LOCAL_CASE_PDF_ROOT = Path("data/download_smoke/pdfs")
DEFAULT_LOCAL_DOCINTEL_TEXT_ROOT = Path("review_pipeline_v1/artifacts/docintel_text")


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path_value: str | None, default_path: Path, project_root: Path | None = None) -> Path:
    if not path_value:
        return default_path

    normalized_value = str(path_value).strip()
    normalized_slashes = normalized_value.replace("\\", "/")
    if normalized_slashes.startswith("dbfs:/Volumes/"):
        return Path(normalized_slashes[len("dbfs:") :])
    if normalized_slashes.startswith("/Volumes/"):
        return Path(normalized_slashes)

    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate

    base_root = project_root or resolve_project_root()
    return (base_root / candidate).resolve()


def resolve_case_pdf_root(
    execution_env: str = "databricks",
    input_root: str | None = None,
    project_root: Path | None = None,
) -> Path:
    default_root = DEFAULT_CATALOG_CASE_PDF_ROOT if execution_env == "databricks" else resolve_path(
        None,
        (project_root or resolve_project_root()) / DEFAULT_LOCAL_CASE_PDF_ROOT,
        project_root=project_root,
    )
    return resolve_path(input_root, default_root, project_root=project_root)


def resolve_docintel_output_root(
    execution_env: str = "databricks",
    output_root: str | None = None,
    project_root: Path | None = None,
) -> Path:
    default_root = DEFAULT_CATALOG_DOCINTEL_TEXT_ROOT if execution_env == "databricks" else resolve_path(
        None,
        (project_root or resolve_project_root()) / DEFAULT_LOCAL_DOCINTEL_TEXT_ROOT,
        project_root=project_root,
    )
    return resolve_path(output_root, default_root, project_root=project_root)


def to_volume_uri(path: Path) -> str | None:
    path_str = str(path).replace("\\", "/")
    if path_str.startswith("dbfs:/Volumes/"):
        return path_str[len("dbfs:") :]
    if path_str.startswith("/Volumes/"):
        return path_str

    drive_prefix = __import__("re").match(r"^[A-Za-z]:(/Volumes/.*)$", path_str)
    if drive_prefix:
        return drive_prefix.group(1)
    return None


def path_from_spark_path(path_str: str) -> Path:
    normalized = path_str.replace("\\", "/")
    if normalized.startswith("dbfs:/Volumes/"):
        normalized = normalized[len("dbfs:") :]
    return Path(normalized)


def uses_spark_volume_io(path: Path, execution_env: str) -> bool:
    return execution_env == "local" and to_volume_uri(path) is not None


def resolve_catalog_spark_session(app_name: str | None = None) -> Any:
    from baselines.legal_rag.runtime import resolve_spark_session

    return resolve_spark_session(app_name=app_name or "review-pipeline-v1-catalog-utils")


def parse_docket_filter(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    return {str(value).strip() for value in values if str(value).strip()}


def infer_docket_id_from_rooted_path(path: Path, input_root: Path) -> str:
    path_uri = to_volume_uri(path)
    root_uri = to_volume_uri(input_root)
    if path_uri is not None and root_uri is not None:
        normalized_root = root_uri.rstrip("/")
        prefix = f"{normalized_root}/"
        if path_uri == normalized_root:
            relative_path = Path()
        elif path_uri.startswith(prefix):
            relative_path = Path(path_uri[len(prefix) :])
        else:
            raise ValueError(f"{str(path)!r} is not in the subpath of {str(input_root)!r}")
    else:
        relative_path = path.relative_to(input_root)

    if not relative_path.parts:
        raise ValueError(f"Could not infer docket id for {path}")

    docket_id = relative_path.parts[0]
    if not docket_id.isdigit():
        raise ValueError(
            f"Expected the first directory under {input_root} to be a numeric docket id, got {docket_id!r} for {path}"
        )
    return docket_id


def infer_docket_id_from_pdf_path(pdf_path: Path, input_root: Path) -> str:
    return infer_docket_id_from_rooted_path(pdf_path, input_root)


def infer_docket_id_from_docintel_path(docintel_path: Path, input_root: Path) -> str:
    return infer_docket_id_from_rooted_path(docintel_path, input_root)


def iter_catalog_case_pdf_paths(
    input_root: Path,
    docket_filter: set[str] | None = None,
    limit: int | None = None,
    sort_paths: bool = False,
):
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    path_iterator = input_root.rglob("*.pdf")
    if sort_paths:
        path_iterator = sorted(path_iterator)

    matched = 0
    for pdf_path in path_iterator:
        if not pdf_path.is_file():
            continue

        docket_id = infer_docket_id_from_pdf_path(pdf_path, input_root)
        if docket_filter is not None and docket_id not in docket_filter:
            continue

        yield pdf_path
        matched += 1
        if limit is not None and matched >= limit:
            return


def count_catalog_case_pdfs(
    input_root: Path,
    docket_filter: set[str] | None = None,
    limit: int | None = None,
    sort_paths: bool = False,
) -> int:
    count = 0
    for _ in iter_catalog_case_pdf_paths(
        input_root=input_root,
        docket_filter=docket_filter,
        limit=limit,
        sort_paths=sort_paths,
    ):
        count += 1
    return count


def preview_catalog_case_pdf_paths(
    input_root: Path,
    docket_filter: set[str] | None = None,
    preview_limit: int = 20,
    sort_paths: bool = False,
) -> list[dict[str, str | int]]:
    preview_rows: list[dict[str, str | int]] = []
    for index, pdf_path in enumerate(
        iter_catalog_case_pdf_paths(
            input_root=input_root,
            docket_filter=docket_filter,
            limit=preview_limit,
            sort_paths=sort_paths,
        ),
        start=1,
    ):
        preview_rows.append(
            {
                "preview_index": index,
                "docket_id": infer_docket_id_from_pdf_path(pdf_path, input_root),
                "pdf_path": str(pdf_path),
                "source_file_name": pdf_path.name,
            }
        )
    return preview_rows


def docintel_output_path_for_pdf(pdf_path: Path, input_root: Path, output_root: Path) -> Path:
    relative_path = pdf_path.relative_to(input_root)
    return (output_root / relative_path).with_suffix(".docintel.json")


def iter_catalog_docintel_paths(
    input_root: Path,
    docket_filter: set[str] | None = None,
    limit: int | None = None,
    sort_paths: bool = False,
    execution_env: str = "local",
    spark: Any | None = None,
):
    if not uses_spark_volume_io(input_root, execution_env) and not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    if uses_spark_volume_io(input_root, execution_env):
        spark_session = spark or resolve_catalog_spark_session(app_name="review-pipeline-v1-docintel-list")
        volume_uri = to_volume_uri(input_root)
        if volume_uri is None:
            raise FileNotFoundError(f"Could not resolve a volume URI for {input_root}")

        root_uri = volume_uri.rstrip("/")
        if docket_filter:
            glob_patterns = []
            for docket_id in sorted(docket_filter):
                glob_patterns.extend(
                    [
                        f"{root_uri}/{docket_id}/*.docintel.json",
                        f"{root_uri}/{docket_id}/*/*.docintel.json",
                    ]
                )
        else:
            glob_patterns = [
                f"{root_uri}/*/*.docintel.json",
                f"{root_uri}/*/*/*.docintel.json",
            ]

        seen_paths: set[str] = set()
        matched = 0
        for glob_pattern in glob_patterns:
            try:
                rows = spark_session.read.format("binaryFile").load(glob_pattern).select("path").collect()
            except Exception as exc:
                if _is_path_not_found_error(exc):
                    continue
                raise
            path_strings = [str(row["path"]) for row in rows]
            if sort_paths:
                path_strings = sorted(path_strings)

            for path_str in path_strings:
                normalized_path = str(path_from_spark_path(path_str))
                if normalized_path in seen_paths:
                    continue
                seen_paths.add(normalized_path)

                docintel_path = path_from_spark_path(path_str)
                yield docintel_path
                matched += 1
                if limit is not None and matched >= limit:
                    return
        return

    path_iterator = input_root.rglob("*.docintel.json")
    if sort_paths:
        path_iterator = sorted(path_iterator)

    matched = 0
    for docintel_path in path_iterator:
        if not docintel_path.is_file():
            continue

        docket_id = infer_docket_id_from_docintel_path(docintel_path, input_root)
        if docket_filter is not None and docket_id not in docket_filter:
            continue

        yield docintel_path
        matched += 1
        if limit is not None and matched >= limit:
            return


def count_catalog_docintel_exports(
    input_root: Path,
    docket_filter: set[str] | None = None,
    limit: int | None = None,
    sort_paths: bool = False,
    execution_env: str = "local",
    spark: Any | None = None,
) -> int:
    count = 0
    for _ in iter_catalog_docintel_paths(
        input_root=input_root,
        docket_filter=docket_filter,
        limit=limit,
        sort_paths=sort_paths,
        execution_env=execution_env,
        spark=spark,
    ):
        count += 1
    return count


def preview_catalog_docintel_paths(
    input_root: Path,
    docket_filter: set[str] | None = None,
    preview_limit: int = 20,
    sort_paths: bool = False,
    execution_env: str = "local",
    spark: Any | None = None,
) -> list[dict[str, str | int]]:
    preview_rows: list[dict[str, str | int]] = []
    for index, docintel_path in enumerate(
        iter_catalog_docintel_paths(
            input_root=input_root,
            docket_filter=docket_filter,
            limit=preview_limit,
            sort_paths=sort_paths,
            execution_env=execution_env,
            spark=spark,
        ),
        start=1,
    ):
        preview_rows.append(
            {
                "preview_index": index,
                "docket_id": infer_docket_id_from_docintel_path(docintel_path, input_root),
                "docintel_path": str(docintel_path),
                "source_file_name": docintel_path.name,
            }
        )
    return preview_rows


def load_docintel_export(
    docintel_path: Path,
    execution_env: str = "local",
    spark: Any | None = None,
) -> dict[str, object]:
    if uses_spark_volume_io(docintel_path, execution_env):
        spark_session = spark or resolve_catalog_spark_session(app_name="review-pipeline-v1-docintel-read")
        volume_uri = to_volume_uri(docintel_path)
        if volume_uri is None:
            raise FileNotFoundError(f"Could not resolve a volume URI for {docintel_path}")

        rows = spark_session.read.format("binaryFile").load(volume_uri).select("content").limit(1).collect()
        if not rows:
            raise FileNotFoundError(f"Doc Intelligence export not found through Spark volume access: {docintel_path}")
        return json.loads(bytes(rows[0]["content"]).decode("utf-8"))

    return json.loads(docintel_path.read_text(encoding="utf-8"))


def _is_path_not_found_error(error: Exception) -> bool:
    message = str(error or "")
    normalized_message = message.upper()
    return "PATH_NOT_FOUND" in normalized_message or "PATH DOES NOT EXIST" in normalized_message