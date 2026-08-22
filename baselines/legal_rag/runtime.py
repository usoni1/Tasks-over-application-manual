from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .config import PROJECT_ROOT, first_non_empty
from standalone_data import (
    DEFAULT_LOCAL_ACCEPTANCE_TABLE,
    DEFAULT_LOCAL_FINAL_LEGAL_DATASET_TABLE,
    DEFAULT_LOCAL_LEGAL_STRICT_TABLE,
    DEFAULT_LOCAL_SENTENCING_YEAR_TABLE,
    resolve_spark_session as resolve_local_spark_session,
)


DEFAULT_STRICT_TABLE = DEFAULT_LOCAL_LEGAL_STRICT_TABLE
DEFAULT_ACCEPTANCE_TABLE = DEFAULT_LOCAL_ACCEPTANCE_TABLE
DEFAULT_SENTENCING_YEAR_TABLE = DEFAULT_LOCAL_SENTENCING_YEAR_TABLE
DEFAULT_FINAL_LEGAL_DATASET_TABLE = DEFAULT_LOCAL_FINAL_LEGAL_DATASET_TABLE
DEFAULT_MODEL_NAME = "openai:gpt-5"


def load_runtime_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def resolve_execution_env() -> str:
    load_runtime_env()
    execution_env = first_non_empty(os.environ.get("EXECUTION_ENV"), "local") or "local"
    execution_env = execution_env.strip().lower()
    if execution_env not in {"local", "databricks"}:
        raise RuntimeError("EXECUTION_ENV must be either 'local' or 'databricks'")
    return execution_env


def resolve_spark_session(app_name: str | None = None) -> Any:
    load_runtime_env()
    return resolve_local_spark_session(execution_env=resolve_execution_env(), app_name=app_name)


def resolve_legal_prompts_dir() -> Path:
    return PROJECT_ROOT / "baselines" / "legal_rag" / "prompts"


def resolve_strict_table() -> str:
    load_runtime_env()
    return (
        first_non_empty(
            os.environ.get("LEGAL_STRICT_TABLE"),
            os.environ.get("FEDERAL_SENTENCING_STRICT_TABLE"),
            os.environ.get("STRICT_TABLE"),
            DEFAULT_STRICT_TABLE,
        )
        or DEFAULT_STRICT_TABLE
    )


def resolve_acceptance_table() -> str:
    load_runtime_env()
    return (
        first_non_empty(
            os.environ.get("FEDERAL_SENTENCING_ACCEPTANCE_TABLE"),
            os.environ.get("ACCEPTANCE_TABLE"),
            DEFAULT_ACCEPTANCE_TABLE,
        )
        or DEFAULT_ACCEPTANCE_TABLE
    )


def resolve_sentencing_year_table() -> str:
    load_runtime_env()
    return (
        first_non_empty(
            os.environ.get("FEDERAL_SENTENCING_YEAR_TABLE"),
            os.environ.get("LEGAL_SENTENCING_YEAR_TABLE"),
            os.environ.get("SENTENCING_YEAR_TABLE"),
            DEFAULT_SENTENCING_YEAR_TABLE,
        )
        or DEFAULT_SENTENCING_YEAR_TABLE
    )


def resolve_final_legal_dataset_table() -> str:
    load_runtime_env()
    return (
        first_non_empty(
            os.environ.get("LEGAL_FINAL_DATASET_TABLE"),
            os.environ.get("FEDERAL_SENTENCING_FINAL_DATASET_TABLE"),
            os.environ.get("FINAL_LEGAL_DATASET_TABLE"),
            DEFAULT_FINAL_LEGAL_DATASET_TABLE,
        )
        or DEFAULT_FINAL_LEGAL_DATASET_TABLE
    )


def load_sentencing_year_lookup(
    spark: Any,
    docket_ids: list[str],
    table_name: str | None = None,
    logger: Any | None = None,
) -> dict[str, dict[str, Any]]:
    normalized_table_name = str(table_name or resolve_sentencing_year_table()).strip()
    normalized_docket_ids = [str(docket_id).strip() for docket_id in docket_ids if str(docket_id).strip()]
    if not normalized_docket_ids:
        return {}

    if not spark.catalog.tableExists(normalized_table_name):
        if logger is not None:
            logger.warning(
                "Sentencing year table %s does not exist; continuing without catalog-backed year lookup.",
                normalized_table_name,
            )
        return {}

    from pyspark.sql import functions as F

    year_df = spark.table(normalized_table_name)
    available_columns = set(year_df.columns)
    required_columns = {"docket_id", "sentencing_year"}
    missing_columns = sorted(required_columns - available_columns)
    if missing_columns:
        raise RuntimeError(
            f"Table {normalized_table_name} is missing required column(s): {missing_columns}"
        )

    selected_columns = [
        F.col("docket_id").cast("string").alias("lookup_docket_id"),
        F.col("sentencing_year"),
    ]
    for optional_column in [
        "year_lookup_status",
        "selected_document_id",
        "selected_date_filed",
        "sentencing_memo_doc_count",
        "source_document_count",
        "candidate_years_json",
        "sentencing_memo_years_json",
    ]:
        if optional_column in available_columns:
            selected_columns.append(F.col(optional_column))

    rows = [
        row.asDict(recursive=True)
        for row in year_df.select(*selected_columns)
        .filter(F.col("docket_id").cast("string").isin(normalized_docket_ids))
        .collect()
    ]

    return {
        str(row.get("lookup_docket_id") or "").strip(): row
        for row in rows
        if str(row.get("lookup_docket_id") or "").strip()
    }


def prepare_federal_sentencing_source_table(
    spark: Any,
    strict_table: str,
    acceptance_table: str | None = None,
    temp_view_name: str = "federal_sentencing_case_source",
    logger: Any | None = None,
) -> tuple[str, str | None]:
    normalized_acceptance_table = str(acceptance_table or "").strip()
    if not normalized_acceptance_table or normalized_acceptance_table.lower() in {"none", "null", "false", "off", "disabled"}:
        return strict_table, None

    if not spark.catalog.tableExists(normalized_acceptance_table):
        if logger is not None:
            logger.warning(
                "Acceptance table %s does not exist; continuing without acceptance inputs.",
                normalized_acceptance_table,
            )
        return strict_table, None

    from pyspark.sql import functions as F

    strict_df = spark.table(strict_table).alias("strict")
    strict_columns = list(strict_df.columns)

    acceptance_source_df = spark.table(normalized_acceptance_table)
    acceptance_columns = set(acceptance_source_df.columns)
    required_acceptance_columns = {"year", "docket_id", "acceptance_of_responsibility"}
    missing_acceptance_columns = sorted(required_acceptance_columns - acceptance_columns)
    if missing_acceptance_columns:
        raise RuntimeError(
            f"Table {normalized_acceptance_table} is missing required column(s): {missing_acceptance_columns}"
        )

    acceptance_projection = [
        F.col("year"),
        F.col("docket_id"),
        F.col("acceptance_of_responsibility"),
    ]
    acceptance_extra_columns: list[str] = []
    for source_name, target_name in [
        ("classification_status", "acceptance_classification_status"),
        ("classification_reason", "acceptance_classification_reason"),
        ("evidence_snippets_json", "acceptance_evidence_snippets_json"),
        ("error", "acceptance_error"),
    ]:
        if source_name in acceptance_columns:
            acceptance_projection.append(F.col(source_name).alias(target_name))
            acceptance_extra_columns.append(target_name)

    acceptance_df = spark.table(normalized_acceptance_table).select(*acceptance_projection).dropDuplicates(["year", "docket_id"]).alias("acceptance")

    strict_has_acceptance = "acceptance_of_responsibility" in strict_columns
    base_acceptance_column = F.col("strict.acceptance_of_responsibility") if strict_has_acceptance else F.lit(None).cast("boolean")

    selected_columns = [
        F.col(f"strict.{column_name}").alias(column_name)
        for column_name in strict_columns
        if column_name != "acceptance_of_responsibility"
    ]
    selected_columns.append(
        F.coalesce(F.col("acceptance.acceptance_of_responsibility"), base_acceptance_column).alias(
            "acceptance_of_responsibility"
        )
    )
    for column_name in acceptance_extra_columns:
        selected_columns.append(F.col(f"acceptance.{column_name}").alias(column_name))

    augmented_df = strict_df.join(
        acceptance_df,
        on=[
            F.col("strict.year") == F.col("acceptance.year"),
            F.col("strict.docket_id") == F.col("acceptance.docket_id"),
        ],
        how="left",
    ).select(*selected_columns)
    augmented_df.createOrReplaceTempView(temp_view_name)

    if logger is not None:
        logger.info(
            "Using acceptance inputs from %s via temp view %s",
            normalized_acceptance_table,
            temp_view_name,
        )

    return temp_view_name, normalized_acceptance_table


def resolve_model_name() -> str:
    load_runtime_env()
    return (
        first_non_empty(
            os.environ.get("LEGAL_MODEL_NAME"),
            os.environ.get("OPENAI_CHAT_MODEL"),
            os.environ.get("MODEL_NAME"),
            DEFAULT_MODEL_NAME,
        )
        or DEFAULT_MODEL_NAME
    )
