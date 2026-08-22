from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .agent import DocketReviewInput
from .catalog_utils import iter_catalog_docintel_paths, load_docintel_export, parse_docket_filter, resolve_docintel_output_root


DEFAULT_MAX_DOCUMENTS: int | None = None
DEFAULT_MAX_CHARS_PER_DOCUMENT: int | None = None
DEFAULT_MAX_CASE_SUMMARY_CHARS: int | None = None


def read_optional_text_file(path_value: str | None) -> str | None:
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    return path.read_text(encoding="utf-8")


def list_docket_docintel_paths(
    docket_id: str | int,
    *,
    execution_env: str = "local",
    docintel_root: str | None = None,
    spark: Any | None = None,
    limit: int | None = None,
    sort_paths: bool = True,
) -> list[Path]:
    normalized_docket_id = str(docket_id).strip()
    if not normalized_docket_id:
        raise ValueError("docket_id must be non-empty.")

    input_root = resolve_docintel_output_root(execution_env=execution_env, output_root=docintel_root)
    docket_filter = parse_docket_filter([normalized_docket_id])
    return list(
        iter_catalog_docintel_paths(
            input_root=input_root,
            docket_filter=docket_filter,
            limit=limit,
            sort_paths=sort_paths,
            execution_env=execution_env,
            spark=spark,
        )
    )


def load_docket_docintel_documents(
    docket_id: str | int,
    *,
    execution_env: str = "local",
    docintel_root: str | None = None,
    spark: Any | None = None,
    max_documents: int | None = DEFAULT_MAX_DOCUMENTS,
    max_chars_per_document: int | None = DEFAULT_MAX_CHARS_PER_DOCUMENT,
    sort_paths: bool = True,
) -> list[dict[str, Any]]:
    if max_documents is not None and max_documents < 1:
        raise ValueError("max_documents must be at least 1.")
    if max_chars_per_document is not None and max_chars_per_document < 1:
        raise ValueError("max_chars_per_document must be at least 1.")

    docintel_paths = list_docket_docintel_paths(
        docket_id,
        execution_env=execution_env,
        docintel_root=docintel_root,
        spark=spark,
        limit=max_documents,
        sort_paths=sort_paths,
    )
    if not docintel_paths:
        raise FileNotFoundError(f"No Doc Intelligence exports found for docket_id={docket_id}.")

    documents: list[dict[str, Any]] = []
    for docintel_path in docintel_paths:
        payload = load_docintel_export(docintel_path, execution_env=execution_env, spark=spark)
        full_text = str(payload.get("full_text") or "").strip()
        document_text = full_text if max_chars_per_document is None else full_text[:max_chars_per_document]
        documents.append(
            {
                "document_id": build_document_id(docintel_path),
                "source_file_name": str(payload.get("source_file_name") or docintel_path.name),
                "source_pdf_path": str(payload.get("source_pdf_path") or ""),
                "docintel_path": str(docintel_path),
                "document_role": "unknown",
                "why_selected": "Loaded from the docket's catalog Doc Intelligence exports.",
                "page_count": payload.get("page_count"),
                "content_length": payload.get("content_length"),
                "document_text": document_text,
            }
        )
    return documents


def build_document_id(docintel_path: Path) -> str:
    file_name = docintel_path.name
    if file_name.endswith(".docintel.json"):
        return file_name[: -len(".docintel.json")]
    return docintel_path.stem


def build_case_summary_from_documents(
    documents: Sequence[Mapping[str, Any]],
    *,
    max_case_summary_chars: int | None = DEFAULT_MAX_CASE_SUMMARY_CHARS,
) -> str:
    if max_case_summary_chars is not None and max_case_summary_chars < 1:
        raise ValueError("max_case_summary_chars must be at least 1.")

    parts: list[str] = []
    used_chars = 0
    for index, document in enumerate(documents, start=1):
        document_text = str(document.get("document_text") or document.get("text_excerpt") or "").strip()
        block = (
            f"Document {index}: {document.get('source_file_name') or document.get('document_id') or 'unknown'}\n"
            f"Source PDF: {document.get('source_pdf_path') or 'unknown'}\n"
            f"Full text:\n{document_text}"
        ).strip()
        if max_case_summary_chars is not None:
            remaining_chars = max_case_summary_chars - used_chars
            if remaining_chars <= 0:
                break
            if len(block) > remaining_chars:
                block = block[:remaining_chars].rstrip()
        parts.append(block)
        used_chars += len(block) + 2
        if max_case_summary_chars is not None and used_chars >= max_case_summary_chars:
            break
    return "\n\n".join(parts).strip()


def build_review_input_from_docintel(
    docket_id: str | int,
    *,
    execution_env: str = "local",
    docintel_root: str | None = None,
    spark: Any | None = None,
    guideline_year: int | None = None,
    max_documents: int | None = DEFAULT_MAX_DOCUMENTS,
    max_chars_per_document: int | None = DEFAULT_MAX_CHARS_PER_DOCUMENT,
    max_case_summary_chars: int | None = DEFAULT_MAX_CASE_SUMMARY_CHARS,
    reviewer_context: Mapping[str, Any] | None = None,
    sort_paths: bool = True,
) -> DocketReviewInput:
    normalized_docket_id = str(docket_id).strip()
    documents = load_docket_docintel_documents(
        normalized_docket_id,
        execution_env=execution_env,
        docintel_root=docintel_root,
        spark=spark,
        max_documents=max_documents,
        max_chars_per_document=max_chars_per_document,
        sort_paths=sort_paths,
    )
    case_summary = build_case_summary_from_documents(
        documents,
        max_case_summary_chars=max_case_summary_chars,
    )
    merged_reviewer_context: dict[str, Any] = {
        "bundle_source": "catalog_docintel_exports",
        "included_document_count": len(documents),
        "document_role_policy": "unknown_by_default_no_filename_role_inference",
        "calculation_workflow": "stage_1_sentencing_info_calculation_stage_2_global_fact_substantiation",
    }
    if reviewer_context:
        merged_reviewer_context.update(dict(reviewer_context))

    guideline_context = {"guideline_year": guideline_year} if guideline_year is not None else None
    return DocketReviewInput(
        docket_id=normalized_docket_id,
        case_summary=case_summary,
        selected_documents=tuple(documents),
        guideline_context=guideline_context,
        reviewer_context=merged_reviewer_context,
    )


def review_input_to_dict(review_input: DocketReviewInput) -> dict[str, Any]:
    return asdict(review_input)


__all__ = [
    "DEFAULT_MAX_CASE_SUMMARY_CHARS",
    "DEFAULT_MAX_CHARS_PER_DOCUMENT",
    "DEFAULT_MAX_DOCUMENTS",
    "build_case_summary_from_documents",
    "build_document_id",
    "build_review_input_from_docintel",
    "list_docket_docintel_paths",
    "load_docket_docintel_documents",
    "read_optional_text_file",
    "review_input_to_dict",
]