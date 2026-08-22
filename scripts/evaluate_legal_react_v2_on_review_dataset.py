from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import logging
import sys
from pathlib import Path
from typing import Any

import mlflow
import mlflow.genai
from mlflow.genai.scorers import scorer
import pandas as pd
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from standalone_data import (
    DEFAULT_LOCAL_ACCEPTANCE_TABLE,
    DEFAULT_LOCAL_CASE_SOURCE_TABLE,
    DEFAULT_LOCAL_FINAL_LEGAL_DATASET_TABLE,
    DEFAULT_LOCAL_LEGAL_STRICT_TABLE,
    ensure_table_available,
    register_temp_view_from_path,
)


DEFAULT_MLFLOW_EXPERIMENT = "legal-react-v2-review-dataset"
NOISY_LOGGER_NAMES = [
    "databricks",
    "databricks.sdk",
    "databricks.sdk.core",
    "databricks_cli",
    "urllib3",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from baselines.legal_rag.runtime import resolve_sentencing_year_table

    parser = argparse.ArgumentParser(
        description="Evaluate legal_react_v2 on review-pipeline final-dataset rows with mlflow.evaluate."
    )
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    parser.add_argument("--execution-env", choices=["local", "databricks"], default=None)
    parser.add_argument("--final-dataset-table", type=str, default=DEFAULT_LOCAL_FINAL_LEGAL_DATASET_TABLE)
    parser.add_argument(
        "--final-dataset-path",
        type=str,
        default=None,
        help="Optional local CSV or Parquet path to register as the review final-dataset temp view.",
    )
    parser.add_argument("--strict-table", type=str, default=None)
    parser.add_argument(
        "--strict-path",
        type=str,
        default=None,
        help="Optional local CSV or Parquet path to register as the legal strict dataset temp view.",
    )
    parser.add_argument("--acceptance-table", type=str, default=None)
    parser.add_argument(
        "--acceptance-path",
        type=str,
        default=None,
        help="Optional local CSV or Parquet path to register as the acceptance lookup temp view.",
    )
    parser.add_argument(
        "--sentencing-year-table",
        type=str,
        default=resolve_sentencing_year_table(),
        help="Catalog table mapping docket_id to sentencing_year for legal runs that need sentencing year lookup.",
    )
    parser.add_argument(
        "--sentencing-year-path",
        type=str,
        default=None,
        help="Optional local CSV or Parquet path to register as the sentencing-year lookup temp view.",
    )
    parser.add_argument(
        "--case-source-table",
        type=str,
        default=None,
        help="Optional fully prepared legal case source table/view containing at least docket_id and year.",
    )
    parser.add_argument(
        "--case-source-path",
        type=str,
        default=None,
        help="Optional local CSV or Parquet path to register as a prepared legal case source temp view.",
    )
    parser.add_argument("--review-version", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--docket-id",
        dest="docket_ids",
        action="append",
        default=None,
        help="Optional docket id filter. Repeat the flag to run multiple exact dockets.",
    )
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--max-agent-steps", type=int, default=60)
    parser.add_argument(
        "--supported-ussg-years-only",
        action="store_true",
        help="Restrict evaluation to rows whose sentencing year has a local USSG DocIntel export available to legal_react_v2.",
    )
    parser.add_argument("--mlflow-experiment", type=str, default=DEFAULT_MLFLOW_EXPERIMENT)
    parser.add_argument("--mlflow-run-name", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    return parser.parse_args(argv)


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        stream=sys.stdout,
        level=getattr(logging, log_level),
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for logger_name in NOISY_LOGGER_NAMES:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def to_jsonable(value: object) -> object:
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


@scorer(name="offense_level_exact_match")
def offense_level_exact_match(inputs: dict[str, object], outputs: dict[str, object], expectations: dict[str, object]) -> float:
    predicted = str(outputs.get("predicted_offense_level_total") or "").strip()
    expected = str(expectations.get("expected_offense_level_total") or "").strip()
    if not expected:
        return 0.0
    return float(predicted == expected)


def load_review_dataset_cases(
    spark: Any,
    *,
    table_name: str,
    review_version: int,
    limit: int | None,
    docket_ids: list[str] | None,
) -> list[dict[str, Any]]:
    df = spark.table(table_name)
    available_columns = set(df.columns)
    required_columns = {
        "docket_id",
        "guideline_year",
        "input_case_facts_text",
        "ground_truth_offense_level",
        "review_version",
    }
    missing_columns = sorted(required_columns - available_columns)
    if missing_columns:
        raise RuntimeError(f"Table {table_name} is missing required column(s): {missing_columns}")

    selected_columns = [
        column_name
        for column_name in [
            "docket_id",
            "guideline_year",
            "input_case_facts_text",
            "input_case_fact_count",
            "ground_truth_offense_level",
            "queue_label",
            "approved",
            "review_decision",
            "review_version",
            "source_run_id",
        ]
        if column_name in available_columns
    ]

    case_df = (
        df.filter(df["input_case_facts_text"].isNotNull())
        .filter(df["ground_truth_offense_level"].isNotNull())
        .filter(df["review_version"] == int(review_version))
        .select(*selected_columns)
    )

    normalized_docket_ids = [str(docket_id).strip() for docket_id in docket_ids or [] if str(docket_id).strip()]
    if normalized_docket_ids:
        case_df = case_df.filter(case_df["docket_id"].cast("string").isin(normalized_docket_ids))

    case_df = case_df.orderBy("docket_id")
    if limit is not None and not normalized_docket_ids:
        case_df = case_df.limit(int(limit))

    rows = [row.asDict(recursive=True) for row in case_df.collect()]
    if not normalized_docket_ids:
        return rows

    row_by_docket_id = {str(row.get("docket_id") or ""): row for row in rows}
    missing_docket_ids = [docket_id for docket_id in normalized_docket_ids if docket_id not in row_by_docket_id]
    if missing_docket_ids:
        raise RuntimeError(
            f"Requested docket id(s) were not found in {table_name} for review_version={review_version}: {missing_docket_ids}"
        )
    return [row_by_docket_id[docket_id] for docket_id in normalized_docket_ids]


def resolve_case_source_table(spark: Any, args: argparse.Namespace, logger: logging.Logger) -> tuple[str | None, str | None, str | None]:
    from baselines.legal_rag.runtime import prepare_federal_sentencing_source_table, resolve_acceptance_table, resolve_strict_table

    if args.case_source_table:
        return str(args.case_source_table), None, None

    strict_table = args.strict_table or resolve_strict_table()
    acceptance_table = args.acceptance_table if args.acceptance_table is not None else resolve_acceptance_table()
    case_source_table, effective_acceptance_table = prepare_federal_sentencing_source_table(
        spark=spark,
        strict_table=strict_table,
        acceptance_table=acceptance_table,
        temp_view_name="legal_react_v2_review_dataset_case_source",
        logger=logger,
    )
    return case_source_table, strict_table, effective_acceptance_table


def load_case_source_lookup(
    spark: Any,
    *,
    table_name: str,
    docket_ids: list[str],
) -> dict[str, list[dict[str, Any]]]:
    from pyspark.sql import functions as F

    source_df = spark.table(table_name)
    available_columns = set(source_df.columns)
    required_columns = {"docket_id", "year"}
    missing_columns = sorted(required_columns - available_columns)
    if missing_columns:
        raise RuntimeError(f"Table {table_name} is missing required column(s): {missing_columns}")

    selected_columns = [
        F.col("docket_id").cast("string").alias("lookup_docket_id"),
        F.col("year").alias("case_year"),
    ]
    if "government_sm_doc_id" in available_columns:
        selected_columns.append(F.col("government_sm_doc_id"))
    if "acceptance_of_responsibility" in available_columns:
        selected_columns.append(F.col("acceptance_of_responsibility"))

    rows = [
        row.asDict(recursive=True)
        for row in source_df.select(*selected_columns)
        .filter(F.col("docket_id").cast("string").isin(docket_ids))
        .collect()
    ]

    lookup: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        docket_id = str(row.get("lookup_docket_id") or "").strip()
        if not docket_id:
            continue
        lookup.setdefault(docket_id, []).append(row)
    return lookup


def load_acceptance_lookup(
    spark: Any,
    *,
    table_name: str | None,
    docket_ids: list[str],
) -> dict[str, list[dict[str, Any]]]:
    from pyspark.sql import functions as F

    normalized_table_name = str(table_name or "").strip()
    if not normalized_table_name or not spark.catalog.tableExists(normalized_table_name):
        return {}

    acceptance_df = spark.table(normalized_table_name)
    available_columns = set(acceptance_df.columns)
    required_columns = {"docket_id", "year", "acceptance_of_responsibility"}
    missing_columns = sorted(required_columns - available_columns)
    if missing_columns:
        raise RuntimeError(f"Table {normalized_table_name} is missing required column(s): {missing_columns}")

    rows = [
        row.asDict(recursive=True)
        for row in acceptance_df.select(
            F.col("docket_id").cast("string").alias("lookup_docket_id"),
            F.col("year").alias("case_year"),
            F.col("acceptance_of_responsibility"),
        )
        .filter(F.col("docket_id").cast("string").isin(docket_ids))
        .collect()
    ]

    lookup: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        docket_id = str(row.get("lookup_docket_id") or "").strip()
        if not docket_id:
            continue
        lookup.setdefault(docket_id, []).append(row)
    return lookup


def resolve_acceptance_value(
    *,
    docket_id: str,
    case_year: int | None,
    acceptance_lookup: dict[str, list[dict[str, Any]]],
) -> tuple[bool | None, str]:
    candidate_rows = [
        row
        for row in acceptance_lookup.get(docket_id, [])
        if row.get("acceptance_of_responsibility") is not None
    ]
    if not candidate_rows:
        return None, "missing"

    if case_year is not None:
        exact_rows = [
            row
            for row in candidate_rows
            if row.get("case_year") is not None and int(row["case_year"]) == int(case_year)
        ]
        exact_values = {row.get("acceptance_of_responsibility") for row in exact_rows}
        if len(exact_values) == 1:
            return bool(next(iter(exact_values))), "acceptance_table_exact_year"
        if len(exact_values) > 1:
            return None, "acceptance_table_exact_year_conflict"

    docket_values = {row.get("acceptance_of_responsibility") for row in candidate_rows}
    if len(docket_values) == 1:
        return bool(next(iter(docket_values))), "acceptance_table_unique_docket"
    return None, "acceptance_table_conflict"


def enrich_cases_with_case_year(
    cases: list[dict[str, Any]],
    *,
    case_source_lookup: dict[str, list[dict[str, Any]]],
    acceptance_lookup: dict[str, list[dict[str, Any]]],
    sentencing_year_lookup: dict[str, dict[str, Any]],
    logger: logging.Logger,
) -> list[dict[str, Any]]:
    matched_count = 0
    missing_count = 0
    ambiguous_count = 0
    mapped_count = 0
    enriched_cases: list[dict[str, Any]] = []

    for case_row in cases:
        enriched_row = dict(case_row)
        docket_id = str(case_row.get("docket_id") or "").strip()
        source_rows = case_source_lookup.get(docket_id, [])
        unique_years = sorted({int(row.get("case_year")) for row in source_rows if row.get("case_year") is not None})

        enriched_row["case_year"] = None
        enriched_row["government_sm_doc_id"] = None
        enriched_row["acceptance_of_responsibility"] = None
        enriched_row["acceptance_lookup_status"] = "missing"
        enriched_row["case_year_lookup_status"] = "missing"
        enriched_row["sentencing_year_lookup_status"] = None

        if not source_rows:
            year_lookup_row = sentencing_year_lookup.get(docket_id)
            if year_lookup_row and year_lookup_row.get("sentencing_year") is not None:
                mapped_count += 1
                enriched_row["case_year"] = int(year_lookup_row["sentencing_year"])
                enriched_row["case_year_lookup_status"] = str(year_lookup_row.get("year_lookup_status") or "sentencing_year_table")
                enriched_row["sentencing_year_lookup_status"] = str(year_lookup_row.get("year_lookup_status") or "sentencing_year_table")
            else:
                missing_count += 1
                if year_lookup_row:
                    enriched_row["sentencing_year_lookup_status"] = str(year_lookup_row.get("year_lookup_status") or "sentencing_year_missing")
        elif len(unique_years) != 1:
            ambiguous_count += 1
            enriched_row["case_year_lookup_status"] = "ambiguous"
        else:
            matched_count += 1
            chosen_row = sorted(
                source_rows,
                key=lambda row: (
                    0 if row.get("government_sm_doc_id") is not None else 1,
                    int(row.get("government_sm_doc_id") or -1),
                ),
            )[0]
            enriched_row["case_year"] = int(unique_years[0])
            enriched_row["government_sm_doc_id"] = chosen_row.get("government_sm_doc_id")
            if chosen_row.get("acceptance_of_responsibility") is not None:
                enriched_row["acceptance_of_responsibility"] = chosen_row.get("acceptance_of_responsibility")
                enriched_row["acceptance_lookup_status"] = "case_source"
            enriched_row["case_year_lookup_status"] = "matched"

        if enriched_row.get("acceptance_of_responsibility") is None:
            acceptance_value, acceptance_status = resolve_acceptance_value(
                docket_id=docket_id,
                case_year=enriched_row.get("case_year"),
                acceptance_lookup=acceptance_lookup,
            )
            if acceptance_value is not None:
                enriched_row["acceptance_of_responsibility"] = acceptance_value
            enriched_row["acceptance_lookup_status"] = acceptance_status

        enriched_cases.append(enriched_row)

    logger.info(
        "Case-year lookup results: matched=%s mapped=%s missing=%s ambiguous=%s",
        matched_count,
        mapped_count,
        missing_count,
        ambiguous_count,
    )
    return enriched_cases


def build_eval_dataframe(cases: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "inputs": [
                {
                    "docket_id": str(row.get("docket_id") or "").strip(),
                    "case_year": row.get("case_year"),
                    "guideline_year": row.get("guideline_year"),
                    "input_case_facts_text": str(row.get("input_case_facts_text") or ""),
                    "input_case_fact_count": row.get("input_case_fact_count"),
                    "queue_label": row.get("queue_label"),
                    "review_version": row.get("review_version"),
                    "ground_truth_offense_level": row.get("ground_truth_offense_level"),
                    "government_sm_doc_id": row.get("government_sm_doc_id"),
                    "acceptance_of_responsibility": row.get("acceptance_of_responsibility"),
                    "acceptance_lookup_status": row.get("acceptance_lookup_status"),
                    "case_year_lookup_status": row.get("case_year_lookup_status"),
                    "sentencing_year_lookup_status": row.get("sentencing_year_lookup_status"),
                    "review_decision": row.get("review_decision"),
                    "approved": row.get("approved"),
                    "source_run_id": row.get("source_run_id"),
                }
                for row in cases
            ],
            "expectations": [
                {
                    "expected_offense_level_total": str(row.get("ground_truth_offense_level") or "").strip(),
                }
                for row in cases
            ],
        }
    )


def list_supported_ussg_years(config: Any) -> list[int]:
    root = getattr(config, "ussg_docintel_text_root", None)
    if root is None:
        return []

    root_path = Path(root)
    if not root_path.exists():
        return []

    supported_years: list[int] = []
    for child in root_path.iterdir():
        if not child.is_dir() or not child.name.isdigit():
            continue
        if (child / "GLMFull.docintel.json").exists():
            supported_years.append(int(child.name))
    return sorted(set(supported_years))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    from baselines.legal_rag.runtime import load_sentencing_year_lookup, resolve_acceptance_table, resolve_model_name, resolve_sentencing_year_table
    from baselines.legal_react_v2 import fetch_case_record, load_config, run_single_case_prediction, score_prediction
    from review_pipeline_v1.review_store import resolve_review_store_spark

    logger = logging.getLogger(__name__)
    config = load_config(args)
    spark = resolve_review_store_spark(app_name="legal-react-v2-review-dataset")
    final_dataset_table = args.final_dataset_table or DEFAULT_LOCAL_FINAL_LEGAL_DATASET_TABLE
    if args.final_dataset_path:
        final_dataset_table = register_temp_view_from_path(
            spark,
            table_name=final_dataset_table,
            path_value=args.final_dataset_path,
            description="review final dataset",
        )
    else:
        final_dataset_table = ensure_table_available(
            spark,
            table_name=final_dataset_table,
            description="review final dataset",
            path_flag="--final-dataset-path",
        )
    args.final_dataset_table = final_dataset_table

    if args.case_source_path:
        args.case_source_table = register_temp_view_from_path(
            spark,
            table_name=args.case_source_table or DEFAULT_LOCAL_CASE_SOURCE_TABLE,
            path_value=args.case_source_path,
            description="prepared legal case source dataset",
        )

    if args.strict_path:
        args.strict_table = register_temp_view_from_path(
            spark,
            table_name=args.strict_table or DEFAULT_LOCAL_LEGAL_STRICT_TABLE,
            path_value=args.strict_path,
            description="legal strict dataset",
        )

    if args.acceptance_path:
        args.acceptance_table = register_temp_view_from_path(
            spark,
            table_name=args.acceptance_table or DEFAULT_LOCAL_ACCEPTANCE_TABLE,
            path_value=args.acceptance_path,
            description="acceptance lookup dataset",
        )

    sentencing_year_table = str(args.sentencing_year_table).strip()
    if args.sentencing_year_path:
        sentencing_year_table = register_temp_view_from_path(
            spark,
            table_name=sentencing_year_table,
            path_value=args.sentencing_year_path,
            description="sentencing year lookup dataset",
        )

    if args.case_source_table:
        args.case_source_table = ensure_table_available(
            spark,
            table_name=str(args.case_source_table).strip(),
            description="prepared legal case source dataset",
            path_flag="--case-source-path",
        )

    if args.case_source_table is None:
        strict_table_name = args.strict_table or DEFAULT_LOCAL_LEGAL_STRICT_TABLE
        args.strict_table = ensure_table_available(
            spark,
            table_name=strict_table_name,
            description="legal strict dataset",
            path_flag="--strict-path",
        )

    model_name = args.model_name or resolve_model_name()
    run_name = args.mlflow_run_name or f"legal_react_v2_review_dataset_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    initial_limit = None if args.supported_ussg_years_only and not args.docket_ids else args.limit

    review_cases = load_review_dataset_cases(
        spark,
        table_name=final_dataset_table,
        review_version=args.review_version,
        limit=initial_limit,
        docket_ids=args.docket_ids,
    )
    if not review_cases:
        raise RuntimeError("No collected review-dataset rows matched the requested selection.")

    case_source_table, strict_table, effective_acceptance_table = resolve_case_source_table(spark, args, logger)
    sentencing_year_table = str(sentencing_year_table or resolve_sentencing_year_table()).strip()
    case_source_lookup = load_case_source_lookup(
        spark,
        table_name=case_source_table,
        docket_ids=[str(row.get("docket_id") or "").strip() for row in review_cases],
    )
    acceptance_lookup_table = effective_acceptance_table or args.acceptance_table or resolve_acceptance_table()
    acceptance_lookup = load_acceptance_lookup(
        spark,
        table_name=acceptance_lookup_table,
        docket_ids=[str(row.get("docket_id") or "").strip() for row in review_cases],
    )
    sentencing_year_lookup = load_sentencing_year_lookup(
        spark,
        docket_ids=[str(row.get("docket_id") or "").strip() for row in review_cases],
        table_name=sentencing_year_table,
        logger=logger,
    )
    cases = enrich_cases_with_case_year(
        review_cases,
        case_source_lookup=case_source_lookup,
        acceptance_lookup=acceptance_lookup,
        sentencing_year_lookup=sentencing_year_lookup,
        logger=logger,
    )

    supported_ussg_years = list_supported_ussg_years(config)
    if args.supported_ussg_years_only:
        if not supported_ussg_years:
            raise RuntimeError(
                "--supported-ussg-years-only was requested, but no local USSG DocIntel years were found for legal_react_v2."
            )
        original_case_count = len(cases)
        cases = [
            case_row
            for case_row in cases
            if case_row.get("case_year") is not None and int(case_row["case_year"]) in supported_ussg_years
        ]
        logger.info(
            "Filtered review-dataset cases to supported USSG years %s: kept=%s dropped=%s",
            supported_ussg_years,
            len(cases),
            original_case_count - len(cases),
        )
        if not cases:
            raise RuntimeError("No review-dataset rows remain after filtering to supported USSG years.")

    if args.limit is not None and not args.docket_ids:
        cases = cases[: int(args.limit)]

    logger.info(
        "Loaded %s review-dataset case(s) from %s for review_version=%s",
        len(cases),
        final_dataset_table,
        args.review_version,
    )
    eval_df = build_eval_dataframe(cases)

    if mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_experiment(args.mlflow_experiment)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "baseline": "legal_react_v2_review_dataset",
                "final_dataset_table": final_dataset_table,
                "case_source_table": case_source_table,
                "sentencing_year_table": sentencing_year_table,
                "strict_table": strict_table,
                "acceptance_table": args.acceptance_table,
                "acceptance_table_used": effective_acceptance_table,
                "model_name": model_name,
                "selector_count": len(cases),
                "execution_env": config.execution_env,
                "review_version": args.review_version,
                "supported_ussg_years_only": args.supported_ussg_years_only,
                "supported_ussg_years": json.dumps(supported_ussg_years),
                "limit": args.limit,
                "docket_ids": json.dumps(args.docket_ids or []),
                "max_agent_steps": args.max_agent_steps,
            }
        )
        prediction_rows_by_key: dict[str, dict[str, Any]] = {}
        completed_keys: set[str] = set()
        progress_bar = tqdm(total=len(eval_df), desc="legal react v2 review dataset", unit="case")

        def predict_case(
            docket_id: str,
            input_case_facts_text: str,
            ground_truth_offense_level: Any,
            case_year: Any = None,
            guideline_year: Any = None,
            input_case_fact_count: Any = None,
            queue_label: Any = None,
            review_version: Any = None,
            government_sm_doc_id: Any = None,
            acceptance_of_responsibility: Any = None,
            acceptance_lookup_status: Any = None,
            case_year_lookup_status: Any = None,
            sentencing_year_lookup_status: Any = None,
            review_decision: Any = None,
            approved: Any = None,
            source_run_id: Any = None,
        ) -> dict[str, object]:
            row: dict[str, Any] = {
                "docket_id": docket_id,
                "case_year": case_year,
                "guideline_year": guideline_year,
                "case_year_lookup_status": case_year_lookup_status,
                "sentencing_year_lookup_status": sentencing_year_lookup_status,
                "input_case_fact_count": input_case_fact_count,
                "queue_label": queue_label,
                "review_version": review_version,
                "ground_truth_offense_level": ground_truth_offense_level,
                "government_sm_doc_id": government_sm_doc_id,
                "acceptance_of_responsibility": acceptance_of_responsibility,
                "acceptance_lookup_status": acceptance_lookup_status,
                "review_decision": review_decision,
                "approved": approved,
                "source_run_id": source_run_id,
                "predicted_offense_level_total": "",
                "status": "error",
                "error": "",
            }
            selected_year = case_year if case_year is not None else guideline_year
            try:
                with mlflow.start_span(name=f"docket_{docket_id}") as case_span:
                    case_span.set_inputs(
                        {
                            "docket_id": docket_id,
                            "case_year": case_year,
                            "guideline_year": guideline_year,
                            "selected_year": selected_year,
                            "case_year_lookup_status": case_year_lookup_status,
                            "sentencing_year_lookup_status": sentencing_year_lookup_status,
                            "acceptance_of_responsibility": acceptance_of_responsibility,
                            "acceptance_lookup_status": acceptance_lookup_status,
                            "model_name": model_name,
                            "max_agent_steps": args.max_agent_steps,
                        }
                    )
                    try:
                        if case_year is not None and str(case_year_lookup_status or "") == "matched":
                            case_record = fetch_case_record(
                                spark=spark,
                                table_name=case_source_table,
                                docket_id=int(docket_id) if docket_id is not None else None,
                                year=int(case_year),
                                government_sm_doc_id=(
                                    int(government_sm_doc_id)
                                    if government_sm_doc_id is not None and str(government_sm_doc_id).strip()
                                    else None
                                ),
                            )
                            if acceptance_of_responsibility is not None and case_record.get("acceptance_of_responsibility") is None:
                                case_record["acceptance_of_responsibility"] = acceptance_of_responsibility
                            summary_text = str(case_record.get("case_summary") or input_case_facts_text or "")
                            selected_year = case_record.get("year") or selected_year
                        else:
                            case_record = {
                                "docket_id": docket_id,
                                "year": selected_year,
                                "government_sm_doc_id": government_sm_doc_id,
                                "expected_offense_level_total": "" if ground_truth_offense_level is None else str(ground_truth_offense_level),
                                "acceptance_of_responsibility": acceptance_of_responsibility,
                            }
                            summary_text = str(input_case_facts_text or "")
                        result = run_single_case_prediction(
                            config=config,
                            summary_text=summary_text,
                            model_name=model_name,
                            year=selected_year,
                            case_record=case_record,
                            max_agent_steps=args.max_agent_steps,
                        )
                        predicted_offense_level = result["prediction"].get("offense_level")
                        metrics = score_prediction(result["prediction"], case_record)
                        row.update(
                            {
                                "predicted_offense_level_total": "" if predicted_offense_level is None else str(predicted_offense_level),
                                "predicted_offense_level": predicted_offense_level,
                                "offense_level_total_exact_match": metrics.get("offense_level_total_exact_match", 0),
                                "exact_match_rate": metrics.get("exact_match_rate", 0.0),
                                "status": "ok",
                                "error": "",
                                "prediction": result["prediction"],
                            }
                        )
                        case_span.set_outputs(
                            {
                                "status": "ok",
                                "selected_year": selected_year,
                                "expected_offense_level_total": case_record.get("expected_offense_level_total"),
                                "predicted_offense_level_total": row["predicted_offense_level_total"],
                                "offense_level_total_exact_match": row["offense_level_total_exact_match"],
                            }
                        )
                    except Exception as exc:
                        logger.exception(
                            "Legal ReAct v2 review-dataset evaluation failed for docket_id=%s selected_year=%s",
                            docket_id,
                            selected_year,
                        )
                        row["error"] = f"{type(exc).__name__}: {exc}"
                        case_span.set_outputs({"status": "error", "error": row["error"]})
            finally:
                prediction_rows_by_key[str(docket_id)] = row
                if str(docket_id) not in completed_keys:
                    completed_keys.add(str(docket_id))
                    progress_bar.update(1)

            return {
                "predicted_offense_level_total": row.get("predicted_offense_level_total") or "",
                "status": row.get("status", "error"),
                "error": row.get("error", ""),
            }

        try:
            evaluation = mlflow.genai.evaluate(
                data=eval_df,
                predict_fn=predict_case,
                scorers=[offense_level_exact_match],
            )
        finally:
            progress_bar.close()

        evaluation_df = pd.DataFrame(prediction_rows_by_key.values())
        if evaluation_df.empty:
            raise RuntimeError("No legal ReAct v2 review-dataset rows were produced by the prediction function.")

        summary = to_jsonable(
            {
                "run_id": run.info.run_id,
                "row_count": len(evaluation_df),
                "scored_row_count": len(eval_df),
                "error_count": int((evaluation_df["status"] != "ok").sum()),
                "metrics": evaluation.metrics,
            }
        )
        mlflow.log_dict(summary, "evaluation/legal_react_v2_review_dataset_summary.json")
        mlflow.log_dict(
            to_jsonable({"rows": evaluation_df.to_dict(orient="records")}),
            "evaluation/legal_react_v2_review_dataset_rows.json",
        )
        if "eval_results" in evaluation.tables:
            mlflow.log_dict(
                to_jsonable({"rows": evaluation.tables["eval_results"].to_dict(orient="records")}),
                "evaluation/legal_react_v2_review_dataset_genai_eval.json",
            )

        if args.output_path:
            output_path = Path(args.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_payload = {
                "summary": summary,
                "rows": evaluation_df.to_dict(orient="records"),
                "eval_results": (
                    evaluation.tables["eval_results"].to_dict(orient="records")
                    if "eval_results" in evaluation.tables
                    else []
                ),
            }
            output_path.write_text(json.dumps(to_jsonable(output_payload), indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info("Wrote results to %s", output_path)

        print(json.dumps(summary, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
