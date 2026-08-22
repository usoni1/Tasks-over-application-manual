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
from pyspark.sql.functions import rand
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
    DEFAULT_LOCAL_ICD_STRICT_TABLE,
    ensure_table_available,
    register_temp_view_from_path,
    resolve_spark_session as resolve_local_spark_session,
)


EvaluationProgressLogger, suppress_noisy_loggers = load_eval_logging_helpers()


DEFAULT_STRICT_TABLE = DEFAULT_LOCAL_ICD_STRICT_TABLE
DEFAULT_MODEL_NAME = "openai:gpt-5"
DEFAULT_MLFLOW_EXPERIMENT = "icd-agentic-rag-evaluate"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the ICD agentic RAG baseline with mlflow.evaluate.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    parser.add_argument("--execution-env", choices=["local", "databricks"], default=None)
    parser.add_argument("--search-service", choices=["service_1", "service_2"], default=None)
    parser.add_argument("--index-name", type=str, default=None)
    parser.add_argument("--strict-table", type=str, default=DEFAULT_STRICT_TABLE)
    parser.add_argument(
        "--strict-path",
        type=str,
        default=None,
        help="Optional local CSV or Parquet path to register as the ICD strict dataset temp view.",
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
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Number of ICD cases to evaluate. Defaults to the first 1000 cases from the seed-42 sample.",
    )
    parser.add_argument("--sample-mode", choices=["ordered", "random"], default="random")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--mlflow-experiment", type=str, default=resolve_mlflow_experiment())
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


def first_non_empty(*values: str | None) -> str | None:
    for value in values:
        if value and value.strip():
            return value.strip()
    return None


def resolve_model_name() -> str:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    return (
        first_non_empty(
            os.environ.get("ICD_MODEL_NAME"),
            os.environ.get("OPENAI_CHAT_MODEL"),
            os.environ.get("MODEL_NAME"),
            DEFAULT_MODEL_NAME,
        )
        or DEFAULT_MODEL_NAME
    )


def resolve_mlflow_experiment() -> str:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    return (
        first_non_empty(
            os.environ.get("ICD_AGENTIC_RAG_MLFLOW_EXPERIMENT"),
            os.environ.get("ICD_RAG_MLFLOW_EXPERIMENT"),
            os.environ.get("MLFLOW_EXPERIMENT"),
            DEFAULT_MLFLOW_EXPERIMENT,
        )
        or DEFAULT_MLFLOW_EXPERIMENT
    )


def resolve_spark_session(execution_env: str, app_name: str) -> Any:
    return resolve_local_spark_session(execution_env=execution_env, app_name=app_name)


def load_case_selectors(
    spark: object,
    table_name: str,
    limit: int | None,
    sample_mode: str,
    sample_seed: int,
) -> list[dict[str, object]]:
    df = spark.table(table_name)
    available_columns = set(df.columns)
    required_columns = {"hadm_id", "subject_id", "note_id", "input_text", "output_icd_codes"}
    missing_columns = sorted(required_columns - available_columns)
    if missing_columns:
        raise RuntimeError(f"Table {table_name} is missing required column(s): {missing_columns}")

    selector_columns = ["hadm_id", "subject_id", "note_id", "output_icd_codes"]
    selector_keys = ["hadm_id", "subject_id", "note_id"]
    case_df = df.filter(df["input_text"].isNotNull()).filter(df["output_icd_codes"].isNotNull()).select(*selector_columns)
    case_df = case_df.dropDuplicates(selector_keys)
    if sample_mode == "random":
        case_df = case_df.orderBy(rand(sample_seed), *selector_keys)
    else:
        case_df = case_df.orderBy(*selector_keys)
    if limit is not None:
        case_df = case_df.limit(limit)
    return [row.asDict(recursive=True) for row in case_df.collect()]


def build_eval_dataframe(selectors: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "inputs": [
                {
                    "hadm_id": row.get("hadm_id"),
                    "subject_id": row.get("subject_id"),
                    "note_id": row.get("note_id"),
                }
                for row in selectors
            ],
            "expectations": [
                {
                    "expected_codes_json": json.dumps(parse_code_list_json(row.get("output_icd_codes"))),
                }
                for row in selectors
            ],
        }
    )


def parse_code_list_json(value: object) -> list[str]:
    from baselines.icd_rag.single_case import coerce_icd_code_list

    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            decoded = stripped
        return list(coerce_icd_code_list(decoded))
    return list(coerce_icd_code_list(value))


def parse_code_json(value: object) -> set[str]:
    return set(parse_code_list_json(value))


def first_code_json(value: object) -> str:
    ordered_codes = parse_code_list_json(value)
    return ordered_codes[0] if ordered_codes else ""


@scorer(name="exact_set_match")
def exact_set_match(inputs: dict[str, object], outputs: dict[str, object], expectations: dict[str, object]) -> float:
    predicted_set = parse_code_json(outputs.get("predicted_codes_json"))
    expected_set = parse_code_json(expectations.get("expected_codes_json"))
    return float(predicted_set == expected_set)


@scorer(name="precision")
def precision_scorer(inputs: dict[str, object], outputs: dict[str, object], expectations: dict[str, object]) -> float:
    predicted_set = parse_code_json(outputs.get("predicted_codes_json"))
    expected_set = parse_code_json(expectations.get("expected_codes_json"))
    true_positive = len(predicted_set & expected_set)
    return float((true_positive / len(predicted_set)) if predicted_set else 0.0)


@scorer(name="recall")
def recall_scorer(inputs: dict[str, object], outputs: dict[str, object], expectations: dict[str, object]) -> float:
    predicted_set = parse_code_json(outputs.get("predicted_codes_json"))
    expected_set = parse_code_json(expectations.get("expected_codes_json"))
    true_positive = len(predicted_set & expected_set)
    return float((true_positive / len(expected_set)) if expected_set else 0.0)


@scorer(name="primary_diagnosis_accuracy")
def primary_diagnosis_accuracy_scorer(
    inputs: dict[str, object], outputs: dict[str, object], expectations: dict[str, object]
) -> float:
    predicted_primary = first_code_json(outputs.get("predicted_codes_json"))
    expected_primary = first_code_json(expectations.get("expected_codes_json"))
    if not expected_primary:
        return 0.0
    return float(predicted_primary == expected_primary)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    configure_mlflow_genai_workers(args.eval_max_workers, args.eval_max_scorer_workers)

    from baselines.icd_agentic_rag import load_config
    from baselines.icd_agentic_rag.single_case import fetch_case_record, run_single_case_prediction, score_prediction

    config = load_config(args)
    execution_env = args.execution_env or os.environ.get("EXECUTION_ENV", "local")
    spark = resolve_spark_session(execution_env=execution_env, app_name="icd-agentic-rag-evaluate")
    model_name = args.model_name or resolve_model_name()
    run_name = args.mlflow_run_name or f"icd_agentic_rag_eval_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    effective_eval_workers, effective_scorer_workers = resolve_effective_mlflow_worker_settings()
    strict_table = args.strict_table
    if args.strict_path:
        strict_table = register_temp_view_from_path(
            spark,
            table_name=strict_table,
            path_value=args.strict_path,
            description="ICD strict dataset",
        )
    else:
        strict_table = ensure_table_available(
            spark,
            table_name=strict_table,
            description="ICD strict dataset",
            path_flag="--strict-path",
        )

    selectors = load_case_selectors(
        spark=spark,
        table_name=strict_table,
        limit=args.limit,
        sample_mode=args.sample_mode,
        sample_seed=args.sample_seed,
    )
    if not selectors:
        raise RuntimeError("No ICD cases matched the requested evaluation selection.")

    logger = logging.getLogger(__name__)
    logger.info("Loaded %s ICD case selectors from %s", len(selectors), strict_table)
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
                "baseline": "icd_agentic_rag",
                "search_service": config.search_service.name,
                "search_index": config.index_name,
                "strict_table": strict_table,
                "model_name": model_name,
                "top_k": args.top_k,
                "max_agent_steps": args.max_agent_steps,
                "eval_max_workers": effective_eval_workers,
                "eval_max_scorer_workers": effective_scorer_workers,
                "selector_count": len(selectors),
                "execution_env": execution_env,
                "sample_mode": args.sample_mode,
                "sample_seed": args.sample_seed,
            }
        )
        prediction_rows_by_key: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
        completed_keys: set[tuple[Any, Any, Any]] = set()
        progress_bar = tqdm(total=len(eval_df), desc="ICD agentic RAG eval cases", unit="case")
        progress_logger = EvaluationProgressLogger(
            logger=logger,
            label="ICD agentic RAG eval",
            total=len(eval_df),
            log_every=10,
        )
        progress_logger.log_start()

        def predict_case(hadm_id: Any, subject_id: Any, note_id: Any) -> dict[str, object]:
            row: dict[str, Any] = {
                "hadm_id": hadm_id,
                "subject_id": subject_id,
                "note_id": note_id,
                "expected_codes_json": "[]",
                "predicted_codes_json": "[]",
                "expected_icd_codes": [],
                "predicted_icd_codes": [],
                "expected_primary_code": None,
                "predicted_primary_code": None,
                "primary_diagnosis_accuracy": 0.0,
                "tool_call_count": 0,
                "status": "error",
                "error": "",
            }
            try:
                with mlflow.start_span(name=f"hadm_{hadm_id}") as case_span:
                    case_span.set_inputs(
                        {
                            "hadm_id": hadm_id,
                            "subject_id": subject_id,
                            "note_id": note_id,
                            "top_k": args.top_k,
                            "search_index": config.index_name,
                        }
                    )
                    try:
                        case_record = fetch_case_record(
                            spark=spark,
                            table_name=strict_table,
                            hadm_id=int(hadm_id) if hadm_id is not None else None,
                            subject_id=int(subject_id) if subject_id is not None else None,
                            note_id=str(note_id) if note_id is not None else None,
                        )
                        result = run_single_case_prediction(
                            config=config,
                            summary_text=case_record["case_summary"],
                            model_name=model_name,
                            top_k=args.top_k,
                            max_agent_steps=args.max_agent_steps,
                        )
                        metrics = score_prediction(
                            result["prediction"].get("predicted_icd_codes", []),
                            case_record.get("expected_icd_codes", []),
                        )
                        row.update(
                            {
                                "hadm_id": case_record.get("hadm_id"),
                                "subject_id": case_record.get("subject_id"),
                                "note_id": case_record.get("note_id"),
                                "expected_codes_json": json.dumps(list(case_record.get("expected_icd_codes", []))),
                                "predicted_codes_json": json.dumps(list(result["prediction"].get("predicted_icd_codes", []))),
                                "expected_icd_codes": list(case_record.get("expected_icd_codes", [])),
                                "predicted_icd_codes": list(result["prediction"].get("predicted_icd_codes", [])),
                                "expected_primary_code": metrics.get("expected_primary_code"),
                                "predicted_primary_code": metrics.get("predicted_primary_code"),
                                "primary_diagnosis_accuracy": metrics.get("primary_diagnosis_accuracy", 0.0),
                                "tool_call_count": result.get("tool_call_count", 0),
                                "status": "ok",
                                "error": "",
                                "precision": metrics.get("precision", 0.0),
                                "recall": metrics.get("recall", 0.0),
                                "true_positive_count": metrics.get("true_positive_count", 0),
                            }
                        )
                        case_span.set_outputs(
                            {
                                "status": "ok",
                                "tool_call_count": row["tool_call_count"],
                                "predicted_icd_codes": row["predicted_icd_codes"],
                                "expected_icd_codes": row["expected_icd_codes"],
                                "predicted_code_count": len(result["prediction"].get("predicted_icd_codes", [])),
                                "expected_code_count": len(case_record.get("expected_icd_codes", [])),
                                "precision": row["precision"],
                                "recall": row["recall"],
                                "primary_diagnosis_accuracy": row["primary_diagnosis_accuracy"],
                            }
                        )
                    except Exception as exc:
                        logger.exception("ICD agentic RAG evaluation failed for hadm_id=%s note_id=%s", hadm_id, note_id)
                        row["error"] = f"{type(exc).__name__}: {exc}"
                        case_span.set_outputs({"status": "error", "error": row["error"]})
            finally:
                row_key = (row.get("hadm_id"), row.get("subject_id"), row.get("note_id"))
                prediction_rows_by_key[row_key] = row
                if row_key not in completed_keys:
                    completed_keys.add(row_key)
                    progress_bar.update(1)
                    progress_logger.update(status=str(row.get("status") or "error"))

            return {
                "predicted_codes_json": row.get("predicted_codes_json") or "[]",
                "predicted_icd_codes": row.get("predicted_icd_codes", []),
                "expected_icd_codes": row.get("expected_icd_codes", []),
                "tool_call_count": row.get("tool_call_count", 0),
                "status": row.get("status", "error"),
                "error": row.get("error", ""),
            }

        try:
            evaluation = mlflow.genai.evaluate(
                data=eval_df,
                predict_fn=predict_case,
                scorers=[exact_set_match, precision_scorer, recall_scorer, primary_diagnosis_accuracy_scorer],
            )
        finally:
            progress_bar.close()

        evaluation_df = pd.DataFrame(prediction_rows_by_key.values())
        if evaluation_df.empty:
            raise RuntimeError("No ICD evaluation rows were produced by the prediction function.")

        summary = to_jsonable(
            {
                "run_id": run.info.run_id,
                "row_count": len(evaluation_df),
                "scored_row_count": len(eval_df),
                "error_count": int((evaluation_df["status"] != "ok").sum()),
                "metrics": evaluation.metrics,
            }
        )
        mlflow.log_dict(summary, "evaluation/icd_agentic_rag_summary.json")
        mlflow.log_dict(to_jsonable({"rows": evaluation_df.to_dict(orient="records")}), "evaluation/icd_agentic_rag_rows.json")
        if "eval_results" in evaluation.tables:
            mlflow.log_dict(
                to_jsonable({"rows": evaluation.tables["eval_results"].to_dict(orient="records")}),
                "evaluation/icd_agentic_rag_genai_eval.json",
            )
        print(json.dumps(summary, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())