from __future__ import annotations

import argparse
from datetime import UTC, datetime
import inspect
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import mlflow
import mlflow.genai
from mlflow.genai.scorers import scorer
import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm


def resolve_project_root() -> Path:
    script_path = globals().get("__file__")
    if script_path:
        return Path(script_path).resolve().parents[1]

    frame = inspect.currentframe()
    try:
        code_filename = frame.f_code.co_filename if frame is not None else ""
    finally:
        del frame

    if code_filename:
        return Path(code_filename).resolve().parents[1]

    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "scripts").exists() and (candidate / "pyproject.toml").exists():
            return candidate
    return cwd.parent if cwd.name == "scripts" else cwd


def load_eval_logging_helpers():
    from scripts.eval_logging import EvaluationProgressLogger, suppress_noisy_loggers

    return EvaluationProgressLogger, suppress_noisy_loggers


PROJECT_ROOT = resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from standalone_data import (
    DEFAULT_LOCAL_FINAL_LEGAL_DATASET_TABLE,
    ensure_table_available,
    register_temp_view_from_path,
)


EvaluationProgressLogger, suppress_noisy_loggers = load_eval_logging_helpers()


DEFAULT_MLFLOW_EXPERIMENT = "legal-agentic-rag-evaluate"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from baselines.legal_rag.runtime import resolve_sentencing_year_table

    parser = argparse.ArgumentParser(description="Evaluate the legal agentic RAG baseline with mlflow.evaluate.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    parser.add_argument("--execution-env", choices=["local", "databricks"], default=None)
    parser.add_argument("--search-service", choices=["service_1", "service_2"], default=None)
    parser.add_argument("--index-name", type=str, default=None)
    parser.add_argument(
        "--dataset-table",
        type=str,
        default=None,
        help="Evaluation dataset table. Defaults to the approved final legal dataset.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Optional local CSV or Parquet path to register as the legal evaluation dataset temp view.",
    )
    parser.add_argument(
        "--strict-table",
        type=str,
        default=None,
        help="Deprecated alias for --dataset-table when running against the old strict-table schema.",
    )
    parser.add_argument(
        "--strict-path",
        type=str,
        default=None,
        help="Deprecated alias for --dataset-path when running against the old strict-table schema.",
    )
    parser.add_argument(
        "--sentencing-year-table",
        type=str,
        default=resolve_sentencing_year_table(),
        help="Catalog table mapping docket_id to sentencing_year for final-dataset evaluation runs.",
    )
    parser.add_argument(
        "--sentencing-year-path",
        type=str,
        default=None,
        help="Optional local CSV or Parquet path to register as the sentencing-year lookup temp view.",
    )
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-agent-steps", type=int, default=30)
    parser.add_argument(
        "--eval-max-workers",
        type=int,
        default=None,
        help="Optional override for worker threads used by mlflow.genai.evaluate predict_fn execution. If omitted, uses the inherited environment or .env setting.",
    )
    parser.add_argument(
        "--eval-max-scorer-workers",
        type=int,
        default=None,
        help="Optional override for scorer worker threads used by mlflow.genai.evaluate. If omitted, uses the inherited environment or .env setting.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional case limit for smoke tests such as --limit 10.")
    parser.add_argument("--year", type=int, default=None, help="Optional single-year filter for the evaluation set.")
    parser.add_argument("--mlflow-experiment", type=str, default=DEFAULT_MLFLOW_EXPERIMENT)
    parser.add_argument("--mlflow-run-name", type=str, default=None)
    return parser.parse_args(argv)


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        stream=sys.stdout,
        level=getattr(logging, log_level),
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    suppress_noisy_loggers()


def configure_mlflow_genai_workers(max_workers: int | None, max_scorer_workers: int | None) -> None:
    if max_workers is not None:
        if max_workers <= 0:
            raise RuntimeError("--eval-max-workers must be greater than 0")
        os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = str(max_workers)
    if max_scorer_workers is not None:
        if max_scorer_workers <= 0:
            raise RuntimeError("--eval-max-scorer-workers must be greater than 0")
        os.environ["MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS"] = str(max_scorer_workers)


def resolve_effective_mlflow_worker_settings() -> tuple[str | None, str | None]:
    return os.environ.get("MLFLOW_GENAI_EVAL_MAX_WORKERS"), os.environ.get("MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS")


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


@scorer(name="non_empty_prediction")
def non_empty_prediction(inputs: dict[str, object], outputs: dict[str, object], expectations: dict[str, object]) -> float:
    predicted = str(outputs.get("predicted_offense_level_total") or "").strip()
    return float(predicted != "")


def normalize_docket_id(value: object) -> str:
    return str(value or "").strip()


def build_case_key(year: object, docket_id: object) -> tuple[int | None, str]:
    normalized_year = None if year in (None, "") else int(year)
    return normalized_year, normalize_docket_id(docket_id)


def load_case_selectors(
    spark: object,
    table_name: str,
    limit: int | None,
    year: int | None,
    sentencing_year_table: str,
    logger: logging.Logger,
) -> list[dict[str, object]]:
    from baselines.legal_rag.runtime import load_sentencing_year_lookup

    df = spark.table(table_name)
    available_columns = set(df.columns)

    strict_required_columns = {"year", "docket_id", "case_facts_summary", "offense_level_total"}
    if strict_required_columns.issubset(available_columns):
        selector_columns = [
            column_name
            for column_name in ["year", "docket_id", "government_sm_doc_id", "case_facts_summary", "offense_level_total"]
            if column_name in available_columns
        ]
        case_df = df.filter(df["case_facts_summary"].isNotNull()).filter(df["offense_level_total"].isNotNull()).select(*selector_columns)
        if year is not None:
            case_df = case_df.filter(case_df["year"] == year)
        case_df = case_df.dropDuplicates(["year", "docket_id"]).orderBy("year", "docket_id")
        if limit is not None:
            case_df = case_df.limit(limit)
        return [
            {
                "year": row.get("year"),
                "docket_id": normalize_docket_id(row.get("docket_id")),
                "government_sm_doc_id": row.get("government_sm_doc_id"),
                "case_summary": str(row.get("case_facts_summary") or "").strip(),
                "expected_offense_level_total": str(row.get("offense_level_total") or "").strip(),
            }
            for row in (row.asDict(recursive=True) for row in case_df.collect())
        ]

    final_required_columns = {"docket_id", "input", "output"}
    missing_columns = sorted(final_required_columns - available_columns)
    if missing_columns:
        raise RuntimeError(f"Table {table_name} is missing required column(s): {missing_columns}")

    case_rows = [
        row.asDict(recursive=True)
        for row in (
            df.filter(df["input"].isNotNull())
            .filter(df["output"].isNotNull())
            .select("docket_id", "input", "output")
            .dropDuplicates(["docket_id"])
            .orderBy("docket_id")
            .collect()
        )
    ]
    year_lookup = load_sentencing_year_lookup(
        spark=spark,
        docket_ids=[normalize_docket_id(row.get("docket_id")) for row in case_rows],
        table_name=sentencing_year_table,
        logger=logger,
    )

    selectors: list[dict[str, object]] = []
    missing_year_count = 0
    for row in case_rows:
        docket_id = normalize_docket_id(row.get("docket_id"))
        lookup_row = year_lookup.get(docket_id) or {}
        case_year = lookup_row.get("sentencing_year")
        if case_year is None:
            missing_year_count += 1
        if year is not None and case_year is not None and int(case_year) != int(year):
            continue
        if year is not None and case_year is None:
            continue
        selectors.append(
            {
                "year": None if case_year is None else int(case_year),
                "docket_id": docket_id,
                "case_summary": str(row.get("input") or "").strip(),
                "expected_offense_level_total": str(row.get("output") or "").strip(),
                "case_year_lookup_status": str(lookup_row.get("year_lookup_status") or "missing"),
            }
        )

    selectors.sort(key=lambda row: ((row.get("year") is None), row.get("year") or 0, normalize_docket_id(row.get("docket_id"))))
    if limit is not None:
        selectors = selectors[:limit]
    logger.info(
        "Normalized %s approved final legal rows from %s; %s rows had no sentencing year lookup",
        len(selectors),
        table_name,
        missing_year_count,
    )
    return selectors


def build_eval_dataframe(selectors: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "inputs": [
                {
                    "year": row.get("year"),
                    "docket_id": row.get("docket_id"),
                }
                for row in selectors
            ],
            "expectations": [
                {
                    "expected_offense_level_total": str(row.get("expected_offense_level_total") or "").strip(),
                }
                for row in selectors
            ],
        }
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    configure_mlflow_genai_workers(args.eval_max_workers, args.eval_max_scorer_workers)

    from baselines.legal_agentic_rag import load_config, resolve_model_name, resolve_spark_session
    from baselines.legal_agentic_rag.single_case import run_single_case_prediction, score_prediction

    config = load_config(args)
    spark = resolve_spark_session(app_name="legal-agentic-rag-evaluate")
    dataset_table = args.dataset_table or args.strict_table or DEFAULT_LOCAL_FINAL_LEGAL_DATASET_TABLE
    dataset_path = args.dataset_path or args.strict_path
    if dataset_path:
        dataset_table = register_temp_view_from_path(
            spark,
            table_name=dataset_table,
            path_value=dataset_path,
            description="legal evaluation dataset",
        )
    else:
        dataset_table = ensure_table_available(
            spark,
            table_name=dataset_table,
            description="legal evaluation dataset",
            path_flag="--dataset-path",
        )
    sentencing_year_table = args.sentencing_year_table
    if args.sentencing_year_path:
        sentencing_year_table = register_temp_view_from_path(
            spark,
            table_name=sentencing_year_table,
            path_value=args.sentencing_year_path,
            description="sentencing year lookup dataset",
        )
    model_name = args.model_name or resolve_model_name()
    run_name = args.mlflow_run_name or f"legal_agentic_rag_eval_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    logger = logging.getLogger(__name__)
    effective_eval_workers, effective_scorer_workers = resolve_effective_mlflow_worker_settings()

    selectors = load_case_selectors(
        spark=spark,
        table_name=dataset_table,
        limit=args.limit,
        year=args.year,
        sentencing_year_table=sentencing_year_table,
        logger=logger,
    )
    if not selectors:
        raise RuntimeError("No legal cases matched the requested evaluation selection.")

    selector_by_key = {build_case_key(row.get("year"), row.get("docket_id")): dict(row) for row in selectors}
    logger.info("Loaded %s legal case selectors from %s", len(selectors), dataset_table)
    logger.info(
        "Configured MLflow evaluation workers: predict=%s scorer=%s",
        effective_eval_workers,
        effective_scorer_workers,
    )
    eval_df = build_eval_dataframe(selectors)

    if mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_experiment(args.mlflow_experiment)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "baseline": "legal_agentic_rag",
                "search_service": config.search_service.name,
                "search_index": config.index_name,
                "dataset_table": dataset_table,
                "sentencing_year_table": sentencing_year_table,
                "model_name": model_name,
                "top_k": args.top_k,
                "max_agent_steps": args.max_agent_steps,
                "eval_max_workers": effective_eval_workers,
                "eval_max_scorer_workers": effective_scorer_workers,
                "selector_count": len(selectors),
                "execution_env": config.execution_env,
                "year_filter": args.year,
            }
        )
        prediction_rows_by_key: dict[tuple[Any, Any], dict[str, Any]] = {}
        completed_keys: set[tuple[Any, Any]] = set()
        progress_bar = tqdm(total=len(eval_df), desc="legal agentic RAG eval cases", unit="case")
        progress_logger = EvaluationProgressLogger(logger=logger, label="Legal agentic RAG eval", total=len(eval_df))
        progress_logger.log_start()

        def predict_case(year: Any, docket_id: Any) -> dict[str, object]:
            row: dict[str, Any] = {
                "year": year,
                "docket_id": docket_id,
                "expected_offense_level_total": "",
                "predicted_offense_level_total": "",
                "tool_call_count": 0,
                "status": "error",
                "error": "",
            }
            try:
                with mlflow.start_span(name=f"docket_{docket_id}") as case_span:
                    case_span.set_inputs(
                        {
                            "year": year,
                            "docket_id": docket_id,
                            "top_k": args.top_k,
                            "search_index": config.index_name,
                        }
                    )
                    try:
                        case_record = dict(selector_by_key[build_case_key(year, docket_id)])
                        result = run_single_case_prediction(
                            config=config,
                            summary_text=case_record["case_summary"],
                            model_name=model_name,
                            top_k=args.top_k,
                            max_agent_steps=args.max_agent_steps,
                            source_year=case_record.get("year"),
                            case_record=case_record,
                        )
                        metrics = score_prediction(result["prediction"], case_record)
                        row.update(
                            {
                                "year": case_record.get("year"),
                                "docket_id": case_record.get("docket_id"),
                                "expected_offense_level_total": case_record.get("expected_offense_level_total") or "",
                                "predicted_offense_level_total": result["prediction"].get("predicted_offense_level_total") or "",
                                "tool_call_count": result.get("tool_call_count", 0),
                                "status": "ok",
                                "error": "",
                                "offense_level_total_exact_match": metrics.get("offense_level_total_exact_match", 0),
                            }
                        )
                        case_span.set_outputs(
                            {
                                "status": "ok",
                                "tool_call_count": row["tool_call_count"],
                                "expected_offense_level_total": row["expected_offense_level_total"],
                                "predicted_offense_level_total": row["predicted_offense_level_total"],
                                "offense_level_total_exact_match": row["offense_level_total_exact_match"],
                            }
                        )
                    except Exception as exc:
                        logger.exception("Legal agentic RAG evaluation failed for docket_id=%s year=%s", docket_id, year)
                        row["error"] = f"{type(exc).__name__}: {exc}"
                        case_span.set_outputs({"status": "error", "error": row["error"]})
            finally:
                row_key = build_case_key(row.get("year"), row.get("docket_id"))
                prediction_rows_by_key[row_key] = row
                if row_key not in completed_keys:
                    completed_keys.add(row_key)
                    progress_bar.update(1)
                    progress_logger.update(status=str(row.get("status") or "error"))

            return {
                "predicted_offense_level_total": row.get("predicted_offense_level_total") or "",
                "expected_offense_level_total": row.get("expected_offense_level_total") or "",
                "tool_call_count": row.get("tool_call_count", 0),
                "status": row.get("status", "error"),
                "error": row.get("error", ""),
            }

        try:
            evaluation = mlflow.genai.evaluate(
                data=eval_df,
                predict_fn=predict_case,
                scorers=[offense_level_exact_match, non_empty_prediction],
            )
        finally:
            progress_bar.close()

        evaluation_df = pd.DataFrame(prediction_rows_by_key.values())
        if evaluation_df.empty:
            raise RuntimeError("No legal evaluation rows were produced by the prediction function.")

        summary = to_jsonable(
            {
                "run_id": run.info.run_id,
                "row_count": len(evaluation_df),
                "scored_row_count": len(eval_df),
                "error_count": int((evaluation_df["status"] != "ok").sum()),
                "metrics": evaluation.metrics,
            }
        )
        mlflow.log_dict(summary, "evaluation/legal_agentic_rag_summary.json")
        mlflow.log_dict(to_jsonable({"rows": evaluation_df.to_dict(orient="records")}), "evaluation/legal_agentic_rag_rows.json")
        if "eval_results" in evaluation.tables:
            mlflow.log_dict(
                to_jsonable({"rows": evaluation.tables["eval_results"].to_dict(orient="records")}),
                "evaluation/legal_agentic_rag_genai_eval.json",
            )
        print(json.dumps(summary, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())