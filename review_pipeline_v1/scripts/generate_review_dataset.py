"""Generate the review dataset for docket verification and write it to catalog-backed Delta tables.

This script runs the current review pipeline over one or more docket bundles,
persists the full artifact JSON into a Delta artifact table, and makes those
artifacts available for the Streamlit verification app.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import monotonic
from typing import Any, Mapping

import mlflow
import mlflow.langchain
from tqdm.auto import tqdm


NOISY_LOGGER_NAMES = [
    "databricks",
    "databricks.sdk",
    "databricks.sdk.core",
    "databricks_cli",
    "urllib3",
]
DEFAULT_ESTIMATED_TOTAL_DOCKETS = 1500
SPARK_SESSION_CLOSED_MARKERS = (
    "INVALID_HANDLE.SESSION_CLOSED",
    "SESSION_CLOSED",
    "Spark Connect Session expired",
    "Session was closed",
)


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


from review_pipeline_v1.agent import (
    DEFAULT_MAX_BATCH_SIZE,
    build_review_agent,
    extract_review_artifact,
    merge_case_facts_into_artifact,
    run_case_facts_extraction,
    run_case_facts_extraction_batch,
    run_docket_review,
    run_docket_review_batch,
    serialize_agent_messages,
)
from review_pipeline_v1.catalog_utils import (
    infer_docket_id_from_docintel_path,
    iter_catalog_docintel_paths,
    parse_docket_filter,
    resolve_docintel_output_root,
)
from review_pipeline_v1.review_store import (
    DEFAULT_FINAL_DATASET_TABLE,
    DEFAULT_REVIEW_ARTIFACT_TABLE,
    DEFAULT_REVIEW_VERSION,
    build_artifact_record,
    build_final_dataset_record,
    ensure_review_tables,
    resolve_review_store_spark,
    upsert_artifact_records,
    upsert_final_dataset_records,
)
from review_pipeline_v1.runtime import resolve_execution_env, resolve_mlflow_experiment, resolve_model_name
from review_pipeline_v1.search_tools import build_search_tools
from review_pipeline_v1.single_case import build_review_input_from_docintel, review_input_to_dict


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def log_docket_trace(
    *,
    experiment_id: str | None,
    docket_id: str,
    guideline_year: int,
    model_name: str,
    max_agent_steps: int,
    batch_size: int,
    max_retries: int,
    review_input_payload: Mapping[str, Any],
    primary_artifact: Mapping[str, Any] | None,
    case_facts_payload: Mapping[str, Any] | None,
    artifact: Mapping[str, Any] | None,
    docket_error: str | None,
    serialized_messages: list[dict[str, Any]],
    case_facts_messages: list[dict[str, Any]],
) -> None:
    root_span = mlflow.start_span_no_context(
        name=f"docket_{docket_id}",
        experiment_id=experiment_id,
        inputs=to_jsonable(
            {
                "docket_id": str(docket_id),
                "guideline_year": guideline_year,
                "model_name": model_name,
                "max_agent_steps": max_agent_steps,
                "batch_size": batch_size,
                "max_retries": max_retries,
                "selected_documents": review_input_payload.get("selected_documents", []),
                "case_summary": review_input_payload.get("case_summary") or "",
            }
        ),
        attributes={"pipeline": "review_pipeline_v1_generate_review_dataset"},
    )
    try:
        primary_span = mlflow.start_span_no_context(
            name="primary_review",
            parent_span=root_span,
            inputs=to_jsonable(review_input_payload),
        )
        try:
            primary_span.set_outputs(
                to_jsonable(
                    {
                        "status": "ok" if primary_artifact is not None else "error",
                        "error": docket_error if primary_artifact is None else "",
                        "artifact": primary_artifact,
                        "message_count": len(serialized_messages),
                    }
                )
            )
        finally:
            primary_span.end()

        case_facts_span = mlflow.start_span_no_context(
            name="case_facts_extraction",
            parent_span=root_span,
            inputs=to_jsonable(
                {
                    "docket_id": str(docket_id),
                    "primary_artifact": primary_artifact,
                }
            ),
        )
        try:
            case_facts_span.set_outputs(
                to_jsonable(
                    {
                        "status": "ok" if case_facts_payload is not None else "error",
                        "error": docket_error if case_facts_payload is None else "",
                        "case_facts_payload": case_facts_payload,
                        "message_count": len(case_facts_messages),
                    }
                )
            )
        finally:
            case_facts_span.end()

        root_span.set_outputs(
            to_jsonable(
                {
                    "status": "ok" if artifact is not None else "error",
                    "error": docket_error or "",
                    "final_total_offense_level": None if artifact is None else artifact.get("final_total_offense_level"),
                    "case_facts_count": 0 if artifact is None else len(artifact.get("case_facts") or []),
                    "artifact": artifact,
                }
            )
        )
    finally:
        root_span.end()


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        stream=sys.stdout,
        level=getattr(logging, log_level),
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for logger_name in NOISY_LOGGER_NAMES:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def configure_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            continue


def is_spark_session_closed_error(exc: Exception) -> bool:
    error_text = f"{type(exc).__name__}: {exc}"
    return any(marker in error_text for marker in SPARK_SESSION_CLOSED_MARKERS)


def run_with_spark_session_retry(
    spark: Any,
    operation,
    *,
    refresh_spark,
    logger: logging.Logger,
    context: str,
    max_attempts: int = 2,
) -> tuple[Any, Any]:
    current_spark = spark
    for attempt in range(1, max_attempts + 1):
        try:
            return current_spark, operation(current_spark)
        except Exception as exc:
            if not is_spark_session_closed_error(exc) or attempt >= max_attempts:
                raise
            logger.warning(
                "Spark session expired during %s; reconnecting and retrying attempt %s/%s.",
                context,
                attempt + 1,
                max_attempts,
            )
            current_spark = refresh_spark()
    raise RuntimeError(f"Spark retry loop exited unexpectedly during {context}.")


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def build_eta_payload(*, started_at_monotonic: float, processed_count: int, estimated_total_count: int) -> dict[str, str | int | None]:
    elapsed_seconds = max(0.0, monotonic() - started_at_monotonic)
    if processed_count <= 0 or estimated_total_count <= processed_count:
        return {
            "processed_count": processed_count,
            "estimated_total_count": estimated_total_count,
            "elapsed": format_duration(elapsed_seconds),
            "eta_remaining": format_duration(0 if estimated_total_count <= processed_count else None),
            "estimated_completion_utc": None if estimated_total_count > processed_count else datetime.now(UTC).isoformat(),
        }

    seconds_per_docket = elapsed_seconds / processed_count
    remaining_count = max(0, estimated_total_count - processed_count)
    remaining_seconds = seconds_per_docket * remaining_count
    return {
        "processed_count": processed_count,
        "estimated_total_count": estimated_total_count,
        "elapsed": format_duration(elapsed_seconds),
        "eta_remaining": format_duration(remaining_seconds),
        "estimated_completion_utc": (datetime.now(UTC) + timedelta(seconds=remaining_seconds)).isoformat(),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate review artifacts and write them to the catalog-backed verification dataset.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    parser.add_argument("--execution-env", choices=["local", "databricks"], default=None)
    parser.add_argument("--docintel-root", type=str, default="/Volumes/usdo_aa_catalog/research_tam_datasets/federal_sentencing/cases/docintel_text")
    parser.add_argument("--docket-ids", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--refresh-case-facts-existing", action="store_true")
    parser.add_argument("--review-version", type=int, default=DEFAULT_REVIEW_VERSION)
    parser.add_argument("--guideline-year", type=int, default=2024)
    parser.add_argument("--artifact-table", type=str, default=DEFAULT_REVIEW_ARTIFACT_TABLE)
    parser.add_argument("--final-dataset-table", type=str, default=DEFAULT_FINAL_DATASET_TABLE)
    parser.add_argument("--mlflow-experiment", type=str, default=None)
    parser.add_argument("--mlflow-run-name", type=str, default=None)
    parser.add_argument("--disable-mlflow", action="store_true")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--max-agent-steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_MAX_BATCH_SIZE)
    parser.add_argument("--max-retries", type=int, default=2)
    return parser.parse_args(argv)


def load_existing_artifacts_by_docket_id(
    spark,
    *,
    artifact_table: str,
    review_version: int,
) -> dict[str, dict[str, Any]]:
    rows = spark.sql(
        f"SELECT docket_id, artifact_json FROM {artifact_table} "
        f"WHERE COALESCE(review_version, {DEFAULT_REVIEW_VERSION}) = {int(review_version)}"
    ).collect()
    artifacts_by_docket_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        docket_id = str(row["docket_id"] or "").strip()
        artifact_json = row["artifact_json"]
        if not docket_id or artifact_json is None:
            continue
        artifacts_by_docket_id[docket_id] = json.loads(str(artifact_json))
    return artifacts_by_docket_id


def iter_docket_ids(args: argparse.Namespace, spark):
    docintel_root = resolve_docintel_output_root(
        execution_env=args.execution_env or resolve_execution_env(),
        output_root=args.docintel_root,
    )
    docket_filter = parse_docket_filter(args.docket_ids)
    seen: set[str] = set()
    yielded = 0
    for docintel_path in iter_catalog_docintel_paths(
        input_root=docintel_root,
        docket_filter=docket_filter,
        limit=None,
        sort_paths=True,
        execution_env=args.execution_env,
        spark=spark,
    ):
        docket_id = infer_docket_id_from_docintel_path(docintel_path, docintel_root)
        if docket_id in seen:
            continue
        seen.add(docket_id)
        yield docket_id
        yielded += 1
        if args.limit is not None and yielded >= args.limit:
            return


def run_generation_for_docket(agent, review_input, *, max_agent_steps: int) -> dict[str, object]:
    primary_result = run_docket_review(
        agent,
        review_input,
        config={"recursion_limit": max_agent_steps},
    )
    primary_artifact = extract_review_artifact(primary_result)
    case_facts_result = run_case_facts_extraction(
        agent,
        review_input,
        primary_artifact,
        config={"recursion_limit": max_agent_steps},
    )
    case_facts_payload = extract_review_artifact(case_facts_result)
    artifact = merge_case_facts_into_artifact(primary_artifact, case_facts_payload)
    serialized_messages = serialize_agent_messages(primary_result.get("messages", []))
    case_facts_messages = serialize_agent_messages(case_facts_result.get("messages", []))
    return {
        "primary_artifact": primary_artifact,
        "case_facts_payload": case_facts_payload,
        "artifact": artifact,
        "serialized_messages": serialized_messages,
        "case_facts_messages": case_facts_messages,
    }


def chunked(values: list[Any], chunk_size: int) -> list[list[Any]]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    return [values[index : index + chunk_size] for index in range(0, len(values), chunk_size)]


def run_primary_stage_with_retries(
    agent: Any,
    review_inputs: list[Any],
    *,
    batch_size: int,
    max_agent_steps: int,
    max_retries: int,
    logger: logging.Logger,
) -> tuple[list[Any | None], list[dict[str, Any] | None], dict[int, str]]:
    total_attempts = max_retries + 1
    raw_results: list[Any | None] = [None] * len(review_inputs)
    parsed_artifacts: list[dict[str, Any] | None] = [None] * len(review_inputs)
    errors_by_index: dict[int, str] = {}
    pending_indices = list(range(len(review_inputs)))

    for attempt in range(1, total_attempts + 1):
        if not pending_indices:
            break
        current_inputs = [review_inputs[index] for index in pending_indices]
        batch_results = run_docket_review_batch(
            agent,
            current_inputs,
            batch_size=batch_size,
            config={"recursion_limit": max_agent_steps},
            show_progress=False,
        )
        next_pending: list[int] = []
        for original_index, batch_result in zip(pending_indices, batch_results):
            if isinstance(batch_result, Exception):
                errors_by_index[original_index] = f"{type(batch_result).__name__}: {batch_result}"
                next_pending.append(original_index)
                continue
            try:
                raw_results[original_index] = batch_result
                parsed_artifacts[original_index] = extract_review_artifact(batch_result)
                errors_by_index.pop(original_index, None)
            except Exception as exc:
                errors_by_index[original_index] = f"{type(exc).__name__}: {exc}"
                next_pending.append(original_index)
        if next_pending and attempt < total_attempts:
            logger.warning(
                "Primary stage retrying %s docket(s) after attempt %s/%s.",
                len(next_pending),
                attempt,
                total_attempts,
            )
        pending_indices = next_pending

    return raw_results, parsed_artifacts, errors_by_index


def run_case_facts_stage_with_retries(
    agent: Any,
    review_inputs: list[Any],
    primary_artifacts: list[dict[str, Any]],
    *,
    batch_size: int,
    max_agent_steps: int,
    max_retries: int,
    logger: logging.Logger,
) -> tuple[list[Any | None], list[dict[str, Any] | None], dict[int, str]]:
    total_attempts = max_retries + 1
    raw_results: list[Any | None] = [None] * len(review_inputs)
    parsed_payloads: list[dict[str, Any] | None] = [None] * len(review_inputs)
    errors_by_index: dict[int, str] = {}
    pending_indices = list(range(len(review_inputs)))

    for attempt in range(1, total_attempts + 1):
        if not pending_indices:
            break
        current_inputs = [review_inputs[index] for index in pending_indices]
        current_artifacts = [primary_artifacts[index] for index in pending_indices]
        batch_results = run_case_facts_extraction_batch(
            agent,
            current_inputs,
            current_artifacts,
            batch_size=batch_size,
            config={"recursion_limit": max_agent_steps},
            show_progress=False,
        )
        next_pending: list[int] = []
        for original_index, batch_result in zip(pending_indices, batch_results):
            if isinstance(batch_result, Exception):
                errors_by_index[original_index] = f"{type(batch_result).__name__}: {batch_result}"
                next_pending.append(original_index)
                continue
            try:
                raw_results[original_index] = batch_result
                parsed_payloads[original_index] = extract_review_artifact(batch_result)
                errors_by_index.pop(original_index, None)
            except Exception as exc:
                errors_by_index[original_index] = f"{type(exc).__name__}: {exc}"
                next_pending.append(original_index)
        if next_pending and attempt < total_attempts:
            logger.warning(
                "Case-facts stage retrying %s docket(s) after attempt %s/%s.",
                len(next_pending),
                attempt,
                total_attempts,
            )
        pending_indices = next_pending

    return raw_results, parsed_payloads, errors_by_index


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_stdio()
    configure_logging(args.log_level)
    logger = logging.getLogger(__name__)
    if args.batch_size < 1:
        raise RuntimeError("--batch-size must be greater than 0")
    if args.max_retries < 0:
        raise RuntimeError("--max-retries must be at least 0")
    if args.review_version < 0:
        raise RuntimeError("--review-version must be at least 0")
    execution_env = args.execution_env or resolve_execution_env()
    spark_app_name = "review-pipeline-v1-generate-review-dataset"

    def refresh_review_store_spark() -> Any:
        return resolve_review_store_spark(app_name=spark_app_name)

    spark = refresh_review_store_spark()
    model_name = args.model_name or resolve_model_name()
    mlflow_experiment = args.mlflow_experiment or resolve_mlflow_experiment()
    run_name = args.mlflow_run_name or f"review_pipeline_v1_generate_review_dataset_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    spark, _ = run_with_spark_session_retry(
        spark,
        lambda active_spark: ensure_review_tables(
            active_spark,
            artifact_table=args.artifact_table,
            final_dataset_table=args.final_dataset_table,
        ),
        refresh_spark=refresh_review_store_spark,
        logger=logger,
        context="initial review table setup",
    )
    spark, docket_ids = run_with_spark_session_retry(
        spark,
        lambda active_spark: list(iter_docket_ids(args, active_spark)),
        refresh_spark=refresh_review_store_spark,
        logger=logger,
        context="loading docket ids",
    )

    existing_docket_ids: set[str] = set()
    existing_artifacts_by_docket_id: dict[str, dict[str, Any]] = {}
    if not args.overwrite_existing:
        spark, existing_docket_ids = run_with_spark_session_retry(
            spark,
            lambda active_spark: {
                str(row["docket_id"])
                for row in active_spark.sql(
                    f"SELECT docket_id FROM {args.artifact_table} "
                    f"WHERE COALESCE(review_version, {DEFAULT_REVIEW_VERSION}) = {int(args.review_version)}"
                ).collect()
                if row["docket_id"] is not None
            },
            refresh_spark=refresh_review_store_spark,
            logger=logger,
            context="loading existing artifact docket ids",
        )
    if args.refresh_case_facts_existing:
        spark, existing_artifacts_by_docket_id = run_with_spark_session_retry(
            spark,
            lambda active_spark: load_existing_artifacts_by_docket_id(
                active_spark,
                artifact_table=args.artifact_table,
                review_version=args.review_version,
            ),
            refresh_spark=refresh_review_store_spark,
            logger=logger,
            context="loading existing artifacts",
        )
        existing_docket_ids = set(existing_artifacts_by_docket_id)

    logger.info(
        "Starting streamed docket generation with estimated_total_dockets=%s and known_existing_dockets=%s.",
        args.limit if args.limit is not None else DEFAULT_ESTIMATED_TOTAL_DOCKETS,
        len(existing_docket_ids),
    )

    agent = build_review_agent(model=model_name, tools=build_search_tools(guideline_year=args.guideline_year))
    retrying_agent = agent.with_retry(stop_after_attempt=args.max_retries + 1) if args.max_retries > 0 else agent
    artifact_rows: list[dict[str, object]] = []
    final_dataset_rows: list[dict[str, object]] = []
    batch_rows: list[dict[str, object]] = []
    source_run_id: str | None = None
    source_experiment_id: str | None = None
    discovered_docket_count = 0
    skipped_existing_count = 0

    if not args.disable_mlflow:
        if mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow.set_experiment(mlflow_experiment)
        mlflow.langchain.autolog(disable=True, silent=True)

    def process_dockets() -> None:
        nonlocal source_run_id, discovered_docket_count, skipped_existing_count, spark
        progress_total = args.limit if args.limit is not None else DEFAULT_ESTIMATED_TOTAL_DOCKETS
        progress_bar = tqdm(total=progress_total, desc="review dataset dockets", unit="docket")
        started_at_monotonic = monotonic()

        def process_loaded_batch(
            docket_id_batch: list[str],
            review_inputs: list[Any],
            review_input_payloads: list[dict[str, Any]],
            existing_primary_artifacts: list[dict[str, Any] | None],
        ) -> None:
            nonlocal spark
            if not review_inputs:
                return

            batch_artifact_rows: list[dict[str, object]] = []
            batch_final_dataset_rows: list[dict[str, object]] = []

            primary_raw_results: list[Any | None] = [None] * len(review_inputs)
            primary_artifacts: list[dict[str, Any] | None] = list(existing_primary_artifacts)
            primary_errors: dict[int, str] = {}

            primary_run_indices = [
                index
                for index, existing_artifact in enumerate(existing_primary_artifacts)
                if existing_artifact is None
            ]
            if primary_run_indices:
                primary_run_inputs = [review_inputs[index] for index in primary_run_indices]
                primary_batch_results = run_docket_review_batch(
                    retrying_agent,
                    primary_run_inputs,
                    batch_size=len(primary_run_inputs),
                    config={"recursion_limit": args.max_agent_steps},
                    show_progress=False,
                )
                for local_index, original_index in enumerate(primary_run_indices):
                    batch_result = primary_batch_results[local_index]
                    primary_raw_results[original_index] = batch_result
                    if isinstance(batch_result, Exception):
                        primary_errors[original_index] = f"{type(batch_result).__name__}: {batch_result}"
                        continue
                    try:
                        primary_artifacts[original_index] = extract_review_artifact(batch_result)
                    except Exception as exc:
                        primary_errors[original_index] = f"{type(exc).__name__}: {exc}"

            successful_indices = [index for index, artifact in enumerate(primary_artifacts) if artifact is not None]
            case_facts_raw_results: list[Any | None] = [None] * len(review_inputs)
            case_facts_payloads: list[dict[str, Any] | None] = [None] * len(review_inputs)
            case_facts_errors: dict[int, str] = {}
            if successful_indices:
                case_facts_inputs = [review_inputs[index] for index in successful_indices]
                case_facts_primary_artifacts = [primary_artifacts[index] for index in successful_indices if primary_artifacts[index] is not None]
                case_facts_batch_results = run_case_facts_extraction_batch(
                    retrying_agent,
                    case_facts_inputs,
                    case_facts_primary_artifacts,
                    batch_size=len(case_facts_inputs),
                    config={"recursion_limit": args.max_agent_steps},
                    show_progress=False,
                )
                for local_index, original_index in enumerate(successful_indices):
                    batch_result = case_facts_batch_results[local_index]
                    if isinstance(batch_result, Exception):
                        case_facts_errors[original_index] = f"{type(batch_result).__name__}: {batch_result}"
                        continue
                    case_facts_raw_results[original_index] = batch_result
                    try:
                        case_facts_payloads[original_index] = extract_review_artifact(batch_result)
                    except Exception as exc:
                        case_facts_errors[original_index] = f"{type(exc).__name__}: {exc}"

            for batch_index, docket_id in enumerate(docket_id_batch):
                handled_index = progress_bar.n + 1
                review_input_payload = review_input_payloads[batch_index]
                primary_artifact = primary_artifacts[batch_index]
                case_facts_payload = case_facts_payloads[batch_index]
                docket_error = primary_errors.get(batch_index) or case_facts_errors.get(batch_index)
                used_existing_primary_artifact = existing_primary_artifacts[batch_index] is not None

                serialized_messages = serialize_agent_messages(
                    (primary_raw_results[batch_index] or {}).get("messages", [])
                ) if isinstance(primary_raw_results[batch_index], Mapping) else []
                case_facts_messages = serialize_agent_messages(
                    (case_facts_raw_results[batch_index] or {}).get("messages", [])
                ) if isinstance(case_facts_raw_results[batch_index], Mapping) else []

                artifact = None
                final_dataset_row = None
                if primary_artifact is not None and case_facts_payload is not None:
                    artifact = merge_case_facts_into_artifact(primary_artifact, case_facts_payload)
                    artifact_record = build_artifact_record(
                        artifact,
                        guideline_year=args.guideline_year,
                        review_version=args.review_version,
                        source_run_id=source_run_id,
                    )
                    artifact_rows.append(artifact_record)
                    batch_artifact_rows.append(artifact_record)
                    final_dataset_row = build_final_dataset_record(
                        artifact,
                        guideline_year=args.guideline_year,
                        review_version=args.review_version,
                        source_run_id=source_run_id,
                    )
                    if final_dataset_row is not None:
                        final_dataset_rows.append(final_dataset_row)
                        batch_final_dataset_rows.append(final_dataset_row)

                if not args.disable_mlflow:
                    log_docket_trace(
                        experiment_id=source_experiment_id,
                        docket_id=str(docket_id),
                        guideline_year=args.guideline_year,
                        model_name=model_name,
                        max_agent_steps=args.max_agent_steps,
                        batch_size=args.batch_size,
                        max_retries=args.max_retries,
                        review_input_payload=review_input_payload,
                        primary_artifact=primary_artifact,
                        case_facts_payload=case_facts_payload,
                        artifact=artifact,
                        docket_error=docket_error,
                        serialized_messages=serialized_messages,
                        case_facts_messages=case_facts_messages,
                    )
                    if artifact is not None:
                        mlflow.log_dict(
                            to_jsonable(
                                {
                                    "run_id": source_run_id,
                                    "docket_id": str(docket_id),
                                    "review_input": review_input_payload,
                                    "primary_artifact": primary_artifact,
                                    "case_facts_payload": case_facts_payload,
                                    "artifact": artifact,
                                    "message_count": len(serialized_messages),
                                    "case_facts_message_count": len(case_facts_messages),
                                }
                            ),
                            f"batch/{docket_id}/review_pipeline_v1_summary.json",
                        )
                        mlflow.log_dict(
                            to_jsonable({"messages": serialized_messages}),
                            f"batch/{docket_id}/review_pipeline_v1_messages.json",
                        )
                        mlflow.log_dict(
                            to_jsonable({"messages": case_facts_messages}),
                            f"batch/{docket_id}/review_pipeline_v1_case_facts_messages.json",
                        )
                    else:
                        mlflow.log_dict(
                            to_jsonable(
                                {
                                    "run_id": source_run_id,
                                    "docket_id": str(docket_id),
                                    "review_input": review_input_payload,
                                    "status": "error",
                                    "error": docket_error,
                                }
                            ),
                            f"batch/{docket_id}/review_pipeline_v1_error.json",
                        )

                logger.info(
                    "Processed docket_id=%s (%s/~%s) status=%s final_total_offense_level=%s final_dataset_included=%s reused_primary_artifact=%s",
                    docket_id,
                    handled_index,
                    progress_total,
                    "ok" if artifact is not None else "error",
                    None if artifact is None else artifact.get("final_total_offense_level"),
                    final_dataset_row is not None,
                    used_existing_primary_artifact,
                )

                batch_rows.append(
                    {
                        "docket_id": str(docket_id),
                        "status": "ok" if artifact is not None else "error",
                        "error": docket_error,
                        "final_total_offense_level": None if artifact is None else artifact.get("final_total_offense_level"),
                        "case_facts_count": 0 if artifact is None else len(artifact.get("case_facts") or []),
                        "final_dataset_case_fact_count": (
                            final_dataset_row.get("input_case_fact_count") if final_dataset_row is not None else None
                        ),
                        "source_run_id": source_run_id,
                    }
                )

                print(
                    json.dumps(
                        {
                            "progress_index": handled_index,
                            "estimated_total_dockets": progress_total,
                            "docket_id": docket_id,
                            "status": "ok" if artifact is not None else "error",
                            "error": docket_error,
                            "final_total_offense_level": None if artifact is None else artifact.get("final_total_offense_level"),
                            "case_facts_count": 0 if artifact is None else len(artifact.get("case_facts") or []),
                            "final_dataset_case_fact_count": (
                                final_dataset_row.get("input_case_fact_count") if final_dataset_row is not None else None
                            ),
                        },
                        ensure_ascii=False,
                    )
                )
                progress_bar.update(1)
                eta_payload = build_eta_payload(
                    started_at_monotonic=started_at_monotonic,
                    processed_count=progress_bar.n,
                    estimated_total_count=progress_total,
                )
                logger.info(
                    "Progress %s/%s elapsed=%s eta_remaining=%s eta_completion_utc=%s",
                    eta_payload["processed_count"],
                    eta_payload["estimated_total_count"],
                    eta_payload["elapsed"],
                    eta_payload["eta_remaining"],
                    eta_payload["estimated_completion_utc"],
                )

            if batch_artifact_rows:
                spark, _ = run_with_spark_session_retry(
                    spark,
                    lambda active_spark: upsert_artifact_records(
                        active_spark,
                        batch_artifact_rows,
                        artifact_table=args.artifact_table,
                    ),
                    refresh_spark=refresh_review_store_spark,
                    logger=logger,
                    context=f"upserting {len(batch_artifact_rows)} artifact rows",
                )
            if batch_final_dataset_rows:
                spark, _ = run_with_spark_session_retry(
                    spark,
                    lambda active_spark: upsert_final_dataset_records(
                        active_spark,
                        batch_final_dataset_rows,
                        final_dataset_table=args.final_dataset_table,
                    ),
                    refresh_spark=refresh_review_store_spark,
                    logger=logger,
                    context=f"upserting {len(batch_final_dataset_rows)} final dataset rows",
                )
            if batch_artifact_rows or batch_final_dataset_rows:
                logger.info(
                    "Flushed batch to catalog tables: artifact_rows=%s final_dataset_rows=%s",
                    len(batch_artifact_rows),
                    len(batch_final_dataset_rows),
                )

        pending_docket_ids: list[str] = []
        pending_review_inputs: list[Any] = []
        pending_payloads: list[dict[str, Any]] = []
        pending_existing_primary_artifacts: list[dict[str, Any] | None] = []
        for docket_id in docket_ids:
            discovered_docket_count += 1
            if not args.overwrite_existing and not args.refresh_case_facts_existing and docket_id in existing_docket_ids:
                skipped_existing_count += 1
                logger.info("Skipping existing docket_id=%s (%s/~%s)", docket_id, progress_bar.n + 1, progress_total)
                progress_bar.update(1)
                eta_payload = build_eta_payload(
                    started_at_monotonic=started_at_monotonic,
                    processed_count=progress_bar.n,
                    estimated_total_count=progress_total,
                )
                logger.info(
                    "Progress %s/%s elapsed=%s eta_remaining=%s eta_completion_utc=%s",
                    eta_payload["processed_count"],
                    eta_payload["estimated_total_count"],
                    eta_payload["elapsed"],
                    eta_payload["eta_remaining"],
                    eta_payload["estimated_completion_utc"],
                )
                continue

            spark, review_input = run_with_spark_session_retry(
                spark,
                lambda active_spark: build_review_input_from_docintel(
                    docket_id=docket_id,
                    execution_env=execution_env,
                    docintel_root=args.docintel_root,
                    spark=active_spark,
                    guideline_year=args.guideline_year,
                    max_documents=None,
                    max_chars_per_document=None,
                    max_case_summary_chars=None,
                ),
                refresh_spark=refresh_review_store_spark,
                logger=logger,
                context=f"loading docintel bundle for docket_id={docket_id}",
            )
            pending_docket_ids.append(docket_id)
            pending_review_inputs.append(review_input)
            pending_payloads.append(review_input_to_dict(review_input))
            pending_existing_primary_artifacts.append(existing_artifacts_by_docket_id.get(docket_id))

            if len(pending_docket_ids) >= args.batch_size:
                process_loaded_batch(
                    pending_docket_ids,
                    pending_review_inputs,
                    pending_payloads,
                    pending_existing_primary_artifacts,
                )
                pending_docket_ids = []
                pending_review_inputs = []
                pending_payloads = []
                pending_existing_primary_artifacts = []

        process_loaded_batch(
            pending_docket_ids,
            pending_review_inputs,
            pending_payloads,
            pending_existing_primary_artifacts,
        )
        progress_bar.close()

        if discovered_docket_count == 0:
            raise RuntimeError("No docket ids matched the requested filters.")

    if args.disable_mlflow:
        process_dockets()
    else:
        with mlflow.start_run(run_name=run_name) as run:
            source_run_id = run.info.run_id
            source_experiment_id = run.info.experiment_id
            mlflow.log_params(
                {
                    "pipeline": "review_pipeline_v1_generate_review_dataset",
                    "execution_env": execution_env,
                    "model_name": model_name,
                    "requested_limit": args.limit,
                    "estimated_total_dockets": DEFAULT_ESTIMATED_TOTAL_DOCKETS,
                    "guideline_year": args.guideline_year,
                    "docintel_root": args.docintel_root,
                    "artifact_table": args.artifact_table,
                    "final_dataset_table": args.final_dataset_table,
                    "overwrite_existing": args.overwrite_existing,
                    "refresh_case_facts_existing": args.refresh_case_facts_existing,
                    "review_version": args.review_version,
                    "max_agent_steps": args.max_agent_steps,
                    "batch_size": args.batch_size,
                    "max_retries": args.max_retries,
                }
            )
            mlflow.set_tags(
                {
                    "pipeline": "review_pipeline_v1_generate_review_dataset",
                    "input_source": "catalog_docintel",
                }
            )
            process_dockets()
            mlflow.log_dict(
                to_jsonable(
                    {
                        "run_id": source_run_id,
                        "discovered_docket_count": discovered_docket_count,
                        "generated_count": len(batch_rows),
                        "final_dataset_count": len(final_dataset_rows),
                        "skipped_existing_count": skipped_existing_count,
                        "rows": batch_rows,
                    }
                ),
                "batch/review_pipeline_v1_batch_summary.json",
            )

    logger.info(
        "Generated %s artifact row(s) and %s final dataset row(s); rows were flushed incrementally during the run.",
        len(artifact_rows),
        len(final_dataset_rows),
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "artifact_table": args.artifact_table,
                "final_dataset_table": args.final_dataset_table,
                "generated_count": len(artifact_rows),
                "final_dataset_count": len(final_dataset_rows),
                "discovered_docket_count": discovered_docket_count,
                "skipped_existing_count": skipped_existing_count,
                "mlflow_experiment": None if args.disable_mlflow else mlflow_experiment,
                "processed_docket_ids": [row["docket_id"] for row in batch_rows],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())