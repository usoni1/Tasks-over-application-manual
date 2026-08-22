from __future__ import annotations

import json
import random
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Mapping

from .runtime import resolve_spark_session


DEFAULT_REVIEW_ARTIFACT_TABLE = "federal_sentencing_review_pipeline_v1_artifacts"
DEFAULT_REVIEW_DECISION_TABLE = "federal_sentencing_review_pipeline_v1_verification"
DEFAULT_FINAL_DATASET_TABLE = "federal_sentencing_review_pipeline_v1_final_dataset"
DEFAULT_REVIEW_VERSION = 0
DEFAULT_CLAIM_TTL_MINUTES = 45
DEFAULT_DELTA_WRITE_RETRY_ATTEMPTS = 5
DEFAULT_DELTA_WRITE_RETRY_DELAY_SECONDS = 0.25


ARTIFACT_TABLE_SCHEMA_SQL = """
    docket_id STRING,
    generated_at_utc STRING,
    updated_at_utc STRING,
    guideline_year INT,
    docket_support_status STRING,
    queue_label STRING,
    final_total_offense_level INT,
    selected_document_count INT,
    offense_level_step_count INT,
    case_facts_count INT,
    review_version INT,
    source_run_id STRING,
    artifact_json STRING
""".strip()

REVIEW_TABLE_SCHEMA_SQL = """
    docket_id STRING,
    reviewer_name STRING,
    claim_status STRING,
    claim_token STRING,
    claimed_at_utc STRING,
    updated_at_utc STRING,
    review_version INT,
    review_decision STRING,
    review_notes STRING,
    verification_json STRING
""".strip()

FINAL_DATASET_TABLE_SCHEMA_SQL = """
    docket_id STRING,
    generated_at_utc STRING,
    updated_at_utc STRING,
    guideline_year INT,
    docket_support_status STRING,
    queue_label STRING,
    approved BOOLEAN,
    review_decision STRING,
    reviewed_at_utc STRING,
    approved_by STRING,
    input_case_facts_json STRING,
    input_case_facts_text STRING,
    input_case_fact_count INT,
    ground_truth_offense_level INT,
    review_version INT,
    source_run_id STRING
""".strip()

LEAKY_CASE_FACT_PATTERNS = [
    "offense level",
    "guideline range",
    "guidelines range",
    "u.s.s.g.",
    "acceptance of responsibility",
    "criminal history category",
    "base offense",
    "total offense",
    "reduction will be appropriate",
    "enhancement",
    "tax table",
    "psr computed",
    "guideline calculation",
    "guidelines computation",
]

FINAL_DATASET_REQUIRED_COLUMNS = {
    "approved": "BOOLEAN",
    "review_decision": "STRING",
    "reviewed_at_utc": "STRING",
    "approved_by": "STRING",
}

ARTIFACT_TABLE_REQUIRED_COLUMNS = {
    "review_version": "INT",
}

REVIEW_TABLE_REQUIRED_COLUMNS = {
    "review_version": "INT",
}

FINAL_DATASET_TABLE_REQUIRED_COLUMNS = {
    "review_version": "INT",
}


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _is_delta_concurrency_conflict(exc: Exception) -> bool:
    error_text = f"{type(exc).__name__}: {exc}"
    conflict_markers = (
        "ConcurrentAppendException",
        "DELTA_CONCURRENT_APPEND",
        "DELTA_CONCURRENT_MODIFICATION",
        "Transaction conflict detected",
    )
    return any(marker in error_text for marker in conflict_markers)


def _run_with_delta_retry(operation, *, max_attempts: int = DEFAULT_DELTA_WRITE_RETRY_ATTEMPTS):
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if not _is_delta_concurrency_conflict(exc) or attempt >= max_attempts:
                raise
            last_error = exc
            time.sleep(DEFAULT_DELTA_WRITE_RETRY_DELAY_SECONDS * attempt)
    if last_error is not None:
        raise last_error


def resolve_review_store_spark(app_name: str | None = None) -> Any:
    return resolve_spark_session(app_name=app_name or "review-pipeline-v1-review-store")


def _ensure_table_columns(spark: Any, table_name: str, required_columns: Mapping[str, str]) -> None:
    existing_columns = {str(column).lower() for column in spark.table(table_name).columns}
    missing_columns = [
        f"{column_name} {column_type}"
        for column_name, column_type in required_columns.items()
        if column_name.lower() not in existing_columns
    ]
    if missing_columns:
        spark.sql(f"ALTER TABLE {table_name} ADD COLUMNS ({', '.join(missing_columns)})")


def ensure_review_tables(
    spark: Any,
    *,
    artifact_table: str = DEFAULT_REVIEW_ARTIFACT_TABLE,
    review_table: str = DEFAULT_REVIEW_DECISION_TABLE,
    final_dataset_table: str = DEFAULT_FINAL_DATASET_TABLE,
) -> None:
    spark.sql(f"CREATE TABLE IF NOT EXISTS {artifact_table} ({ARTIFACT_TABLE_SCHEMA_SQL}) USING DELTA")
    spark.sql(f"CREATE TABLE IF NOT EXISTS {review_table} ({REVIEW_TABLE_SCHEMA_SQL}) USING DELTA")
    spark.sql(f"CREATE TABLE IF NOT EXISTS {final_dataset_table} ({FINAL_DATASET_TABLE_SCHEMA_SQL}) USING DELTA")
    _ensure_table_columns(spark, artifact_table, ARTIFACT_TABLE_REQUIRED_COLUMNS)
    _ensure_table_columns(spark, review_table, REVIEW_TABLE_REQUIRED_COLUMNS)
    _ensure_table_columns(spark, final_dataset_table, FINAL_DATASET_TABLE_REQUIRED_COLUMNS)
    _ensure_table_columns(spark, final_dataset_table, FINAL_DATASET_REQUIRED_COLUMNS)


def build_artifact_record(
    artifact: Mapping[str, Any],
    *,
    guideline_year: int,
    review_version: int = DEFAULT_REVIEW_VERSION,
    source_run_id: str | None = None,
) -> dict[str, Any]:
    offense_level_steps = artifact.get("offense_level_steps") or []
    case_facts = artifact.get("case_facts") or []
    timestamp = utc_now_iso()
    return {
        "docket_id": str(artifact.get("docket_id") or "").strip(),
        "generated_at_utc": timestamp,
        "updated_at_utc": timestamp,
        "guideline_year": int(guideline_year),
        "docket_support_status": str(artifact.get("docket_support_status") or ""),
        "queue_label": str(artifact.get("queue_label") or ""),
        "final_total_offense_level": artifact.get("final_total_offense_level"),
        "selected_document_count": len(artifact.get("selected_documents") or []),
        "offense_level_step_count": len(offense_level_steps),
        "case_facts_count": len(case_facts),
        "review_version": int(review_version),
        "source_run_id": str(source_run_id or ""),
        "artifact_json": json.dumps(dict(artifact), ensure_ascii=False),
    }


def _normalize_case_fact_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _case_fact_is_leaky(fact_text: str) -> bool:
    normalized = _normalize_case_fact_text(fact_text).lower()
    if not normalized:
        return True
    return any(pattern in normalized for pattern in LEAKY_CASE_FACT_PATTERNS)


def sanitize_case_facts_for_final_dataset(case_facts: list[Mapping[str, Any]] | None) -> list[str]:
    sanitized_facts: list[str] = []
    seen: set[str] = set()
    for case_fact in case_facts or []:
        fact_text = _normalize_case_fact_text(case_fact.get("fact"))
        if not fact_text or _case_fact_is_leaky(fact_text):
            continue
        lowered = fact_text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        sanitized_facts.append(fact_text)
    return sanitized_facts


def _included_step_has_weak_support(artifact: Mapping[str, Any]) -> bool:
    offense_level_steps = artifact.get("offense_level_steps") or []
    for step in offense_level_steps:
        if not isinstance(step, Mapping):
            continue
        if not bool(step.get("included_in_total_offense_level")):
            continue
        support_strength = str(step.get("support_strength") or "").strip().lower()
        if support_strength == "weak":
            return True
    return False


def build_final_dataset_record(
    artifact: Mapping[str, Any],
    *,
    guideline_year: int,
    review_version: int = DEFAULT_REVIEW_VERSION,
    source_run_id: str | None = None,
) -> dict[str, Any] | None:
    if str(artifact.get("docket_support_status") or "").strip().lower() != "supported":
        return None
    if _included_step_has_weak_support(artifact):
        return None
    ground_truth_offense_level = artifact.get("final_total_offense_level")
    if ground_truth_offense_level is None:
        return None

    sanitized_case_facts = sanitize_case_facts_for_final_dataset(artifact.get("case_facts") or [])
    if not sanitized_case_facts:
        return None

    timestamp = utc_now_iso()
    return {
        "docket_id": str(artifact.get("docket_id") or "").strip(),
        "generated_at_utc": timestamp,
        "updated_at_utc": timestamp,
        "guideline_year": int(guideline_year),
        "docket_support_status": str(artifact.get("docket_support_status") or ""),
        "queue_label": str(artifact.get("queue_label") or ""),
        "approved": None,
        "review_decision": None,
        "reviewed_at_utc": None,
        "approved_by": None,
        "input_case_facts_json": json.dumps(sanitized_case_facts, ensure_ascii=False),
        "input_case_facts_text": "\n".join(f"- {fact}" for fact in sanitized_case_facts),
        "input_case_fact_count": len(sanitized_case_facts),
        "ground_truth_offense_level": int(ground_truth_offense_level),
        "review_version": int(review_version),
        "source_run_id": str(source_run_id or ""),
    }


def update_final_dataset_approval(
    spark: Any,
    *,
    docket_id: str,
    review_decision: str,
    reviewer_name: str,
    review_version: int = DEFAULT_REVIEW_VERSION,
    final_dataset_table: str = DEFAULT_FINAL_DATASET_TABLE,
) -> None:
    normalized_docket_id = str(docket_id).strip()
    if not normalized_docket_id:
        raise ValueError("docket_id must be non-empty.")

    ensure_review_tables(spark, final_dataset_table=final_dataset_table)
    normalized_review_decision = str(review_decision or "").strip()
    approved_value = normalized_review_decision.lower() == "approved"
    timestamp = utc_now_iso()
    _run_with_delta_retry(
        lambda: spark.sql(
            f"""
            UPDATE {final_dataset_table}
            SET
                approved = {str(approved_value).lower()},
                review_decision = {json.dumps(normalized_review_decision)},
                reviewed_at_utc = {json.dumps(timestamp)},
                approved_by = {json.dumps(str(reviewer_name or '').strip())},
                updated_at_utc = {json.dumps(timestamp)}
                        WHERE docket_id = {json.dumps(normalized_docket_id)}
                            AND COALESCE(review_version, {DEFAULT_REVIEW_VERSION}) = {int(review_version)}
            """
        )
    )


def _create_aligned_dataframe(
    spark: Any,
    *,
    table_name: str,
    rows: list[Mapping[str, Any]],
) -> Any:
    table_schema = spark.table(table_name).schema
    ordered_rows = [
        tuple(row.get(field.name) for field in table_schema.fields)
        for row in rows
    ]
    return spark.createDataFrame(ordered_rows, schema=table_schema)


def _review_version_merge_condition(*, target_alias: str = "target", source_alias: str = "source") -> str:
    return (
        f"{target_alias}.docket_id = {source_alias}.docket_id "
        f"AND COALESCE({target_alias}.review_version, {DEFAULT_REVIEW_VERSION}) = "
        f"COALESCE({source_alias}.review_version, {DEFAULT_REVIEW_VERSION})"
    )


def upsert_artifact_records(
    spark: Any,
    rows: list[Mapping[str, Any]],
    *,
    artifact_table: str = DEFAULT_REVIEW_ARTIFACT_TABLE,
) -> None:
    if not rows:
        return
    ensure_review_tables(spark, artifact_table=artifact_table)
    source_df = _create_aligned_dataframe(spark, table_name=artifact_table, rows=rows)
    temp_view = f"review_pipeline_v1_artifact_upsert_{uuid.uuid4().hex}"
    source_df.createOrReplaceTempView(temp_view)
    spark.sql(
        f"""
        MERGE INTO {artifact_table} AS target
        USING {temp_view} AS source
        ON {_review_version_merge_condition()}
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
        """
    )
    spark.catalog.dropTempView(temp_view)


def upsert_final_dataset_records(
    spark: Any,
    rows: list[Mapping[str, Any]],
    *,
    final_dataset_table: str = DEFAULT_FINAL_DATASET_TABLE,
) -> None:
    if not rows:
        return
    ensure_review_tables(spark, final_dataset_table=final_dataset_table)
    source_df = _create_aligned_dataframe(spark, table_name=final_dataset_table, rows=rows)
    temp_view = f"review_pipeline_v1_final_dataset_upsert_{uuid.uuid4().hex}"
    source_df.createOrReplaceTempView(temp_view)
    spark.sql(
        f"""
        MERGE INTO {final_dataset_table} AS target
        USING {temp_view} AS source
        ON {_review_version_merge_condition()}
        WHEN MATCHED THEN UPDATE SET
            target.generated_at_utc = COALESCE(target.generated_at_utc, source.generated_at_utc),
            target.updated_at_utc = source.updated_at_utc,
            target.guideline_year = source.guideline_year,
            target.docket_support_status = source.docket_support_status,
            target.queue_label = source.queue_label,
            target.input_case_facts_json = source.input_case_facts_json,
            target.input_case_facts_text = source.input_case_facts_text,
            target.input_case_fact_count = source.input_case_fact_count,
            target.ground_truth_offense_level = source.ground_truth_offense_level,
            target.review_version = source.review_version,
            target.source_run_id = source.source_run_id
        WHEN NOT MATCHED THEN INSERT *
        """
    )
    spark.catalog.dropTempView(temp_view)


def upsert_review_record(
    spark: Any,
    row: Mapping[str, Any],
    *,
    review_table: str = DEFAULT_REVIEW_DECISION_TABLE,
) -> None:
    ensure_review_tables(spark, review_table=review_table)
    source_df = _create_aligned_dataframe(spark, table_name=review_table, rows=[row])
    temp_view = f"review_pipeline_v1_review_upsert_{uuid.uuid4().hex}"
    try:
        source_df.createOrReplaceTempView(temp_view)
        _run_with_delta_retry(
            lambda: spark.sql(
                f"""
                MERGE INTO {review_table} AS target
                USING {temp_view} AS source
                ON {_review_version_merge_condition()}
                WHEN MATCHED THEN UPDATE SET *
                WHEN NOT MATCHED THEN INSERT *
                """
            )
        )
    finally:
        if spark.catalog.tableExists(temp_view):
            spark.catalog.dropTempView(temp_view)


def load_artifact_rows(
    spark: Any,
    *,
    artifact_table: str = DEFAULT_REVIEW_ARTIFACT_TABLE,
    review_version: int | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    ensure_review_tables(spark, artifact_table=artifact_table)
    query = f"SELECT * FROM {artifact_table}"
    if review_version is not None:
        query = f"{query} WHERE COALESCE(review_version, {DEFAULT_REVIEW_VERSION}) = {int(review_version)}"
    if limit is not None:
        query = f"{query} LIMIT {int(limit)}"
    return [row.asDict(recursive=True) for row in spark.sql(query).collect()]


def load_artifact_by_docket_id(
    spark: Any,
    docket_id: str,
    *,
    artifact_table: str = DEFAULT_REVIEW_ARTIFACT_TABLE,
    review_version: int | None = None,
) -> dict[str, Any] | None:
    normalized_docket_id = str(docket_id).strip()
    if not normalized_docket_id:
        raise ValueError("docket_id must be non-empty.")
    ensure_review_tables(spark, artifact_table=artifact_table)
    version_filter = ""
    if review_version is not None:
        version_filter = f" AND COALESCE(review_version, {DEFAULT_REVIEW_VERSION}) = {int(review_version)}"
    rows = spark.sql(
        f"SELECT artifact_json, COALESCE(review_version, {DEFAULT_REVIEW_VERSION}) AS review_version FROM {artifact_table} WHERE docket_id = {json.dumps(normalized_docket_id)}{version_filter} LIMIT 1"
    ).collect()
    if not rows:
        return None
    artifact = json.loads(str(rows[0]["artifact_json"]))
    artifact["review_version"] = int(rows[0]["review_version"])
    return artifact


def load_review_rows(
    spark: Any,
    *,
    review_table: str = DEFAULT_REVIEW_DECISION_TABLE,
    review_version: int | None = None,
) -> list[dict[str, Any]]:
    ensure_review_tables(spark, review_table=review_table)
    query = f"SELECT * FROM {review_table}"
    if review_version is not None:
        query = f"{query} WHERE COALESCE(review_version, {DEFAULT_REVIEW_VERSION}) = {int(review_version)}"
    return [row.asDict(recursive=True) for row in spark.sql(query).collect()]


def _is_claim_stale(updated_at_utc: str | None, claim_ttl_minutes: int) -> bool:
    if not updated_at_utc:
        return True
    try:
        updated_at = datetime.fromisoformat(updated_at_utc)
    except ValueError:
        return True
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=UTC)
    return updated_at < datetime.now(UTC) - timedelta(minutes=claim_ttl_minutes)


def list_review_queue(
    spark: Any,
    *,
    artifact_table: str = DEFAULT_REVIEW_ARTIFACT_TABLE,
    review_table: str = DEFAULT_REVIEW_DECISION_TABLE,
    review_version: int = DEFAULT_REVIEW_VERSION,
    supported_only: bool = True,
) -> list[dict[str, Any]]:
    artifacts = load_artifact_rows(spark, artifact_table=artifact_table, review_version=review_version)
    current_review_rows = {
        str(row.get("docket_id") or ""): row
        for row in load_review_rows(spark, review_table=review_table, review_version=review_version)
    }
    prior_completed_review_rows = {
        str(row.get("docket_id") or ""): row
        for row in load_review_rows(spark, review_table=review_table)
        if int(row.get("review_version") or DEFAULT_REVIEW_VERSION) != int(review_version)
        and str(row.get("claim_status") or "").strip().lower() == "completed"
    }
    queue_rows: list[dict[str, Any]] = []
    for artifact_row in artifacts:
        if supported_only and str(artifact_row.get("docket_support_status") or "").strip().lower() != "supported":
            continue
        if artifact_row.get("final_total_offense_level") is None:
            continue
        docket_id = str(artifact_row.get("docket_id") or "")
        review_row = current_review_rows.get(docket_id) or prior_completed_review_rows.get(docket_id) or {}
        review_source_version = review_row.get("review_version") if review_row else None
        queue_rows.append(
            {
                **artifact_row,
                "reviewer_name": review_row.get("reviewer_name"),
                "claim_status": review_row.get("claim_status"),
                "claim_token": review_row.get("claim_token"),
                "claimed_at_utc": review_row.get("claimed_at_utc"),
                "review_decision": review_row.get("review_decision"),
                "review_notes": review_row.get("review_notes"),
                "verification_json": review_row.get("verification_json"),
                "review_source_version": review_source_version,
                "review_carried_forward": bool(review_row) and int(review_source_version or DEFAULT_REVIEW_VERSION) != int(review_version),
            }
        )
    return queue_rows


def claim_next_docket(
    spark: Any,
    reviewer_name: str,
    *,
    claim_token: str,
    artifact_table: str = DEFAULT_REVIEW_ARTIFACT_TABLE,
    review_table: str = DEFAULT_REVIEW_DECISION_TABLE,
    review_version: int = DEFAULT_REVIEW_VERSION,
    claim_ttl_minutes: int = DEFAULT_CLAIM_TTL_MINUTES,
) -> dict[str, Any] | None:
    normalized_reviewer = str(reviewer_name).strip()
    if not normalized_reviewer:
        raise ValueError("reviewer_name must be non-empty.")
    normalized_claim_token = str(claim_token).strip()
    if not normalized_claim_token:
        raise ValueError("claim_token must be non-empty.")

    queue_rows = list_review_queue(
        spark,
        artifact_table=artifact_table,
        review_table=review_table,
        review_version=review_version,
    )
    eligible_rows = [
        row
        for row in queue_rows
        if str(row.get("claim_status") or "") != "completed"
        and (
            not row.get("reviewer_name")
            or str(row.get("claim_status") or "") != "claimed"
            or str(row.get("claim_token") or "") == normalized_claim_token
            or _is_claim_stale(row.get("claimed_at_utc"), claim_ttl_minutes)
        )
    ]
    if not eligible_rows:
        return None

    selected_row = random.choice(eligible_rows)
    claim_timestamp = utc_now_iso()
    review_row = {
        "docket_id": str(selected_row.get("docket_id") or ""),
        "reviewer_name": normalized_reviewer,
        "claim_status": "claimed",
        "claim_token": normalized_claim_token,
        "claimed_at_utc": claim_timestamp,
        "updated_at_utc": claim_timestamp,
        "review_version": int(review_version),
        "review_decision": str(selected_row.get("review_decision") or ""),
        "review_notes": str(selected_row.get("review_notes") or ""),
        "verification_json": str(selected_row.get("verification_json") or ""),
    }
    upsert_review_record(spark, review_row, review_table=review_table)
    return load_artifact_by_docket_id(
        spark,
        review_row["docket_id"],
        artifact_table=artifact_table,
        review_version=review_version,
    )


def release_review_claim(
    spark: Any,
    *,
    docket_id: str,
    claim_token: str,
    review_table: str = DEFAULT_REVIEW_DECISION_TABLE,
    review_version: int = DEFAULT_REVIEW_VERSION,
    claim_ttl_minutes: int = DEFAULT_CLAIM_TTL_MINUTES,
) -> bool:
    normalized_docket_id = str(docket_id).strip()
    normalized_claim_token = str(claim_token).strip()
    if not normalized_docket_id:
        raise ValueError("docket_id must be non-empty.")
    if not normalized_claim_token:
        raise ValueError("claim_token must be non-empty.")

    matching_rows = [
        row
        for row in load_review_rows(spark, review_table=review_table, review_version=review_version)
        if str(row.get("docket_id") or "") == normalized_docket_id
    ]
    if not matching_rows:
        return False

    existing_row = matching_rows[0]
    existing_claim_token = str(existing_row.get("claim_token") or "")
    if (
        str(existing_row.get("claim_status") or "") == "claimed"
        and existing_claim_token != normalized_claim_token
        and not _is_claim_stale(existing_row.get("claimed_at_utc"), claim_ttl_minutes)
    ):
        return False

    timestamp = utc_now_iso()
    upsert_review_record(
        spark,
        {
            "docket_id": normalized_docket_id,
            "reviewer_name": "",
            "claim_status": "available",
            "claim_token": "",
            "claimed_at_utc": "",
            "updated_at_utc": timestamp,
            "review_version": int(review_version),
            "review_decision": "",
            "review_notes": "",
            "verification_json": "",
        },
        review_table=review_table,
    )
    return True


def submit_review_decision(
    spark: Any,
    *,
    docket_id: str,
    reviewer_name: str,
    review_decision: str,
    review_notes: str | None = None,
    verification_payload: Mapping[str, Any] | None = None,
    review_table: str = DEFAULT_REVIEW_DECISION_TABLE,
    final_dataset_table: str = DEFAULT_FINAL_DATASET_TABLE,
    review_version: int = DEFAULT_REVIEW_VERSION,
) -> None:
    normalized_docket_id = str(docket_id).strip()
    normalized_reviewer = str(reviewer_name).strip()
    if not normalized_docket_id:
        raise ValueError("docket_id must be non-empty.")
    if not normalized_reviewer:
        raise ValueError("reviewer_name must be non-empty.")

    timestamp = utc_now_iso()
    upsert_review_record(
        spark,
        {
            "docket_id": normalized_docket_id,
            "reviewer_name": normalized_reviewer,
            "claim_status": "completed",
            "claim_token": uuid.uuid4().hex,
            "claimed_at_utc": timestamp,
            "updated_at_utc": timestamp,
            "review_version": int(review_version),
            "review_decision": str(review_decision or ""),
            "review_notes": str(review_notes or ""),
            "verification_json": json.dumps(dict(verification_payload or {}), ensure_ascii=False),
        },
        review_table=review_table,
    )
    update_final_dataset_approval(
        spark,
        docket_id=normalized_docket_id,
        review_decision=review_decision,
        reviewer_name=normalized_reviewer,
        review_version=review_version,
        final_dataset_table=final_dataset_table,
    )


__all__ = [
    "DEFAULT_CLAIM_TTL_MINUTES",
    "DEFAULT_FINAL_DATASET_TABLE",
    "DEFAULT_REVIEW_ARTIFACT_TABLE",
    "DEFAULT_REVIEW_DECISION_TABLE",
    "DEFAULT_REVIEW_VERSION",
    "build_artifact_record",
    "build_final_dataset_record",
    "claim_next_docket",
    "ensure_review_tables",
    "list_review_queue",
    "load_artifact_by_docket_id",
    "load_artifact_rows",
    "load_review_rows",
    "release_review_claim",
    "resolve_review_store_spark",
    "sanitize_case_facts_for_final_dataset",
    "submit_review_decision",
    "update_final_dataset_approval",
    "upsert_artifact_records",
    "upsert_final_dataset_records",
    "upsert_review_record",
]