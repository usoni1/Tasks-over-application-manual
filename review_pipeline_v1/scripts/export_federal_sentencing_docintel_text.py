"""
Export federal sentencing case PDFs to text with Azure Document Intelligence.

What this script does:
- Assumes the input root contains one directory per docket, where the directory
  name is the docket id.
- Walks every PDF under each docket directory recursively.
- Sends each PDF to Azure Document Intelligence.
- Writes one JSON output file per source PDF with full extracted text and
  per-page text.
- Writes manifest files summarizing every attempted export, keyed by docket id
  and source PDF path.
- Does not try to infer whether a document is a government memo, defense memo,
  or any other memo type. This script is only responsible for text export.

Typical Databricks usage:
python review_pipeline_v1/scripts/export_federal_sentencing_docintel_text.py \
  --execution-env databricks
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path


def resolve_project_root() -> Path:
    script_path = globals().get("__file__")
    if script_path:
        return Path(script_path).resolve().parents[2]

    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "review_pipeline_v1").exists() and (candidate / "requirements.txt").exists():
            return candidate
    return cwd


PROJECT_ROOT = resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from review_pipeline_v1.catalog_utils import (
    DEFAULT_CATALOG_CASE_PDF_ROOT,
    DEFAULT_CATALOG_DOCINTEL_TEXT_ROOT,
    DEFAULT_LOCAL_CASE_PDF_ROOT,
    count_catalog_case_pdfs,
    docintel_output_path_for_pdf,
    infer_docket_id_from_pdf_path,
    iter_catalog_case_pdf_paths,
    parse_docket_filter,
)


DEFAULT_INPUT_ROOT = DEFAULT_CATALOG_CASE_PDF_ROOT
DEFAULT_OUTPUT_ROOT = DEFAULT_CATALOG_DOCINTEL_TEXT_ROOT
DEFAULT_DOCINTEL_ENDPOINT = "https://ltc-uat-exp.cognitiveservices.azure.com/"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract docket-organized case PDFs with Azure Document Intelligence and write per-PDF JSON plus manifest files."
    )
    parser.add_argument("--execution-env", choices=["local", "databricks"], default="databricks")
    parser.add_argument("--input-root", type=str, default=None, help="Root directory containing one docket-id folder per case.")
    parser.add_argument("--output-root", type=str, default=None, help="Root directory where Doc Intelligence JSON exports will be written.")
    parser.add_argument("--manifest-path", type=str, default=None, help="Optional JSONL manifest path. Defaults to <output-root>/manifest.jsonl.")
    parser.add_argument("--manifest-csv-path", type=str, default=None, help="Optional CSV manifest path. Defaults to <output-root>/manifest.csv.")
    parser.add_argument("--docintel-endpoint", type=str, default=None)
    parser.add_argument("--docintel-key", type=str, default=None)
    parser.add_argument("--docintel-model", type=str, default=None)
    parser.add_argument("--docket-ids", nargs="+", default=None, help="Optional docket-id filter list.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of PDFs to process after filtering.")
    parser.add_argument("--progress-every", type=int, default=1, help="Print a progress update after every N processed PDFs. Use 0 to disable periodic updates.")
    parser.add_argument("--precount-pdfs-for-eta", action="store_true", help="Count matching PDFs before export so ETA can be exact when --limit is not set.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing JSON exports.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args(argv)


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        stream=sys.stdout,
        level=getattr(logging, log_level),
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def resolve_path(path_value: str | None, default_path: Path) -> Path:
    if not path_value:
        return default_path

    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def resolve_input_root(args: argparse.Namespace) -> Path:
    if args.execution_env == "databricks":
        return resolve_path(args.input_root, DEFAULT_INPUT_ROOT)
    return resolve_path(args.input_root, PROJECT_ROOT / DEFAULT_LOCAL_CASE_PDF_ROOT)


def resolve_output_root(args: argparse.Namespace) -> Path:
    if args.execution_env == "databricks":
        return resolve_path(args.output_root, DEFAULT_OUTPUT_ROOT)
    return resolve_path(args.output_root, PROJECT_ROOT / "review_pipeline_v1" / "artifacts" / "docintel_text")


def resolve_manifest_paths(args: argparse.Namespace, output_root: Path) -> tuple[Path, Path]:
    manifest_path = resolve_path(args.manifest_path, output_root / "manifest.jsonl")
    manifest_csv_path = resolve_path(args.manifest_csv_path, output_root / "manifest.csv")
    return manifest_path, manifest_csv_path


def first_non_empty(*values: str | None) -> str | None:
    for value in values:
        if value and value.strip():
            return value.strip()
    return None


def resolve_docintel_settings(args: argparse.Namespace) -> tuple[str, str, str]:
    endpoint = first_non_empty(
        args.docintel_endpoint,
        os.environ.get("LEGAL_DOCINTEL_ENDPOINT"),
        os.environ.get("AZURE_DOCINTEL_ENDPOINT"),
        os.environ.get("AZURE_DOC_ENDPOINT"),
        DEFAULT_DOCINTEL_ENDPOINT if os.environ.get("AZURE_DOCINTEL_KEY") else None,
    )
    key = first_non_empty(
        args.docintel_key,
        os.environ.get("LEGAL_DOCINTEL_KEY"),
        os.environ.get("AZURE_DOCINTEL_KEY"),
        os.environ.get("AZURE_DOC_KEY"),
    )
    model_name = first_non_empty(
        args.docintel_model,
        os.environ.get("LEGAL_DOCINTEL_MODEL"),
        "prebuilt-layout",
    )

    if not endpoint:
        raise RuntimeError("No Azure Document Intelligence endpoint is configured.")
    if not key:
        raise RuntimeError("No Azure Document Intelligence key is configured.")

    return endpoint, key, model_name


def normalize_relative_path(path: Path) -> str:
    return path.as_posix()


def format_duration(total_seconds: float | None) -> str:
    if total_seconds is None:
        return "unknown"
    if total_seconds < 0:
        total_seconds = 0
    total_seconds = int(total_seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def log_progress(
    *,
    processed_count: int,
    exported_count: int,
    skipped_count: int,
    failed_count: int,
    started_at: float,
    last_item_seconds: float | None,
    total_count: int | None,
) -> None:
    elapsed_seconds = max(time.time() - started_at, 0.0)
    docs_per_minute = (processed_count / elapsed_seconds * 60.0) if elapsed_seconds > 0 else 0.0
    eta_seconds = None
    progress_label = f"processed={processed_count} exported={exported_count} skipped={skipped_count} failed={failed_count}"
    if total_count is not None and processed_count > 0:
        remaining = max(total_count - processed_count, 0)
        eta_seconds = (elapsed_seconds / processed_count) * remaining
        progress_label = f"{processed_count}/{total_count} | {progress_label}"

    last_doc_label = format_duration(last_item_seconds) if last_item_seconds is not None else "unknown"
    logging.info(
        "[progress] %s | elapsed=%s | last_doc=%s | rate=%.2f docs/min | eta=%s",
        progress_label,
        format_duration(elapsed_seconds),
        last_doc_label,
        docs_per_minute,
        format_duration(eta_seconds),
    )

def analyze_pdf(pdf_path: Path, client, model_name: str, docket_id: str) -> dict[str, object]:
    with pdf_path.open("rb") as handle:
        poller = client.begin_analyze_document(model_name, document=handle)
        result = poller.result()

    pages: list[dict[str, object]] = []
    for page in result.pages or []:
        lines: list[str] = []
        for line in page.lines or []:
            text = str(getattr(line, "content", "") or "").strip()
            if text:
                lines.append(text)
        pages.append(
            {
                "page_number": page.page_number,
                "text": "\n".join(lines).strip(),
            }
        )

    full_text = str(getattr(result, "content", "") or "").strip()
    return {
        "docket_id": docket_id,
        "source_pdf_path": str(pdf_path),
        "source_file_name": pdf_path.name,
        "docintel_model": model_name,
        "page_count": len(pages),
        "content_length": len(full_text),
        "full_text": full_text,
        "pages": pages,
    }


def read_existing_export_metadata(output_path: Path) -> tuple[int | None, int | None]:
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception:
        return None, None

    page_count = payload.get("page_count")
    content_length = payload.get("content_length")
    return page_count if isinstance(page_count, int) else None, content_length if isinstance(content_length, int) else None


def build_manifest_row(
    *,
    docket_id: str,
    pdf_path: Path,
    input_root: Path,
    output_path: Path,
    status: str,
    page_count: int | None,
    content_length: int | None,
    error_type: str | None = None,
    error_message: str | None = None,
) -> dict[str, object]:
    relative_pdf_path = pdf_path.relative_to(input_root)
    return {
        "docket_id": docket_id,
        "source_pdf_path": str(pdf_path),
        "relative_pdf_path": normalize_relative_path(relative_pdf_path),
        "source_file_name": pdf_path.name,
        "output_json_path": str(output_path),
        "status": status,
        "page_count": page_count,
        "content_length": content_length,
        "error_type": error_type,
        "error_message": error_message,
    }


def write_manifest_jsonl(manifest_path: Path, rows: list[dict[str, object]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True))
            handle.write("\n")


def write_manifest_csv(manifest_csv_path: Path, rows: list[dict[str, object]]) -> None:
    manifest_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "docket_id",
        "source_pdf_path",
        "relative_pdf_path",
        "source_file_name",
        "output_json_path",
        "status",
        "page_count",
        "content_length",
        "error_type",
        "error_message",
    ]
    with manifest_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    from azure.ai.formrecognizer import DocumentAnalysisClient
    from azure.core.credentials import AzureKeyCredential

    input_root = resolve_input_root(args)
    output_root = resolve_output_root(args)
    manifest_path, manifest_csv_path = resolve_manifest_paths(args, output_root)
    endpoint, key, model_name = resolve_docintel_settings(args)
    docket_filter = parse_docket_filter(args.docket_ids)
    total_count = args.limit if args.limit is not None else None
    if args.precount_pdfs_for_eta and total_count is None:
        logging.info("Pre-counting matching PDFs to provide an exact ETA.")
        total_count = count_catalog_case_pdfs(input_root=input_root, docket_filter=docket_filter, limit=args.limit)

    logging.info("Input root: %s", input_root)
    logging.info("Output root: %s", output_root)
    logging.info("Manifest path: %s", manifest_path)
    logging.info("Manifest CSV path: %s", manifest_csv_path)
    if total_count is None:
        logging.info("PDF count for progress: unknown until the run finishes or --precount-pdfs-for-eta is enabled.")
    else:
        logging.info("PDF count for progress: %s", total_count)

    client = DocumentAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key),
    )

    manifest_rows: list[dict[str, object]] = []
    failed = 0
    exported = 0
    skipped = 0
    processed = 0
    started_at = time.time()
    had_any_match = False

    for pdf_path in iter_catalog_case_pdf_paths(input_root=input_root, docket_filter=docket_filter, limit=args.limit):
        had_any_match = True
        item_started_at = time.time()
        processed += 1
        docket_id = infer_docket_id_from_pdf_path(pdf_path, input_root)
        output_path = docintel_output_path_for_pdf(pdf_path, input_root, output_root)
        if output_path.exists() and not args.overwrite:
            page_count, content_length = read_existing_export_metadata(output_path)
            logging.info("Skipping existing export for docket %s: %s", docket_id, output_path)
            skipped += 1
            manifest_rows.append(
                build_manifest_row(
                    docket_id=docket_id,
                    pdf_path=pdf_path,
                    input_root=input_root,
                    output_path=output_path,
                    status="skipped_existing",
                    page_count=page_count,
                    content_length=content_length,
                )
            )
            if args.progress_every > 0 and processed % args.progress_every == 0:
                log_progress(
                    processed_count=processed,
                    exported_count=exported,
                    skipped_count=skipped,
                    failed_count=failed,
                    started_at=started_at,
                    last_item_seconds=time.time() - item_started_at,
                    total_count=total_count,
                )
            continue

        logging.info("Analyzing docket %s PDF %s", docket_id, pdf_path)
        try:
            payload = analyze_pdf(pdf_path=pdf_path, client=client, model_name=model_name, docket_id=docket_id)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
            exported += 1
            manifest_rows.append(
                build_manifest_row(
                    docket_id=docket_id,
                    pdf_path=pdf_path,
                    input_root=input_root,
                    output_path=output_path,
                    status="exported",
                    page_count=payload["page_count"],
                    content_length=payload["content_length"],
                )
            )
        except Exception as exc:
            failed += 1
            logging.error("Failed to analyze docket %s PDF %s with %s: %s", docket_id, pdf_path, type(exc).__name__, exc)
            manifest_rows.append(
                build_manifest_row(
                    docket_id=docket_id,
                    pdf_path=pdf_path,
                    input_root=input_root,
                    output_path=output_path,
                    status="failed",
                    page_count=None,
                    content_length=None,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )

        if args.progress_every > 0 and processed % args.progress_every == 0:
            log_progress(
                processed_count=processed,
                exported_count=exported,
                skipped_count=skipped,
                failed_count=failed,
                started_at=started_at,
                last_item_seconds=time.time() - item_started_at,
                total_count=total_count,
            )

    if not had_any_match:
        raise RuntimeError("No PDFs matched the requested input root and filters.")

    if args.progress_every <= 0 or processed % args.progress_every != 0:
        log_progress(
            processed_count=processed,
            exported_count=exported,
            skipped_count=skipped,
            failed_count=failed,
            started_at=started_at,
            last_item_seconds=None,
            total_count=total_count,
        )

    write_manifest_jsonl(manifest_path, manifest_rows)
    write_manifest_csv(manifest_csv_path, manifest_rows)

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "manifest_path": str(manifest_path),
        "manifest_csv_path": str(manifest_csv_path),
        "docintel_endpoint": endpoint,
        "docintel_model": model_name,
        "pdf_count": processed,
        "exported_count": exported,
        "skipped_existing_count": skipped,
        "failed_count": failed,
        "total_count_for_progress": total_count,
        "elapsed": format_duration(time.time() - started_at),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())