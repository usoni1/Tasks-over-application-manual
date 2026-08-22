from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import mlflow
import mlflow.genai
import mlflow.langchain
from mlflow.genai.scorers import scorer
import pandas as pd
from pyspark.sql.functions import rand
from tqdm.auto import tqdm

def load_eval_logging_helpers():
    from scripts.eval_logging import EvaluationProgressLogger, suppress_noisy_loggers

    return EvaluationProgressLogger, suppress_noisy_loggers


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from standalone_data import (
    DEFAULT_LOCAL_ICD_STRICT_TABLE,
    ensure_table_available,
    register_temp_view_from_path,
)


EvaluationProgressLogger, suppress_noisy_loggers = load_eval_logging_helpers()


DEFAULT_MLFLOW_EXPERIMENT = "icd-react-v2-evaluate"
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the ICD ReAct v2 baseline with mlflow.evaluate.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    parser.add_argument("--execution-env", choices=["local", "databricks"], default=None)
    parser.add_argument("--strict-table", type=str, default=DEFAULT_LOCAL_ICD_STRICT_TABLE)
    parser.add_argument(
        "--strict-path",
        type=str,
        default=None,
        help="Optional local CSV or Parquet path to register as the ICD strict dataset temp view.",
    )
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Number of ICD cases to evaluate. Defaults to the first 1000 cases from the seed-42 sample.",
    )
    parser.add_argument("--sample-mode", choices=["ordered", "random"], default="random")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--prompt-variant", type=str, default="active_first_no_repeat")
    parser.add_argument("--hadm-id", type=int, default=None, help="Optional single HADM filter for targeted smoke tests.")
    parser.add_argument("--subject-id", type=int, default=None, help="Optional single subject filter used with HADM/note filters.")
    parser.add_argument("--note-id", type=str, default=None, help="Optional single note filter for targeted smoke tests.")
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        default=None,
        help="Optional exact case selector HADM_ID:SUBJECT_ID:NOTE_ID. Repeat for mixed samples.",
    )
    parser.add_argument("--max-agent-steps", type=int, default=60, help="Maximum LangGraph recursion limit for each case.")
    parser.add_argument(
        "--eval-max-workers",
        type=int,
        default=None,
        help="Optional override for worker threads used by mlflow.genai.evaluate. If omitted, MLflow uses the inherited environment setting.",
    )
    parser.add_argument(
        "--eval-max-scorer-workers",
        type=int,
        default=None,
        help="Optional override for scorer worker threads used by mlflow.genai.evaluate. If omitted, MLflow uses the inherited environment setting.",
    )
    parser.add_argument(
        "--prompt-suffix-file",
        type=str,
        default=None,
        help="Optional UTF-8 text file appended to the v2 single-case prompt for all evaluated cases.",
    )
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


def read_prompt_suffix_file(path_value: str | None) -> str | None:
    if path_value is None:
        return None
    path = Path(path_value).expanduser()
    return path.read_text(encoding="utf-8")


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


def parse_case_selector(raw_value: str) -> tuple[int, int, str]:
    hadm_text, first_separator, remainder = raw_value.partition(":")
    subject_text, second_separator, note_text = remainder.partition(":")
    if first_separator != ":" or second_separator != ":":
        raise RuntimeError(f"Invalid --case value {raw_value!r}. Expected HADM_ID:SUBJECT_ID:NOTE_ID")
    if not hadm_text.strip().isdigit() or not subject_text.strip().isdigit() or not note_text.strip():
        raise RuntimeError(f"Invalid --case value {raw_value!r}. Expected HADM_ID:SUBJECT_ID:NOTE_ID")
    return int(hadm_text.strip()), int(subject_text.strip()), note_text.strip()


def resolve_requested_case_selectors(args: argparse.Namespace) -> list[tuple[int, int, str]] | None:
    selectors: list[tuple[int, int, str]] = []

    for raw_case in args.cases or []:
        selectors.append(parse_case_selector(raw_case))

    if args.hadm_id is not None or args.subject_id is not None or args.note_id is not None:
        if args.hadm_id is None or args.subject_id is None or args.note_id is None:
            raise RuntimeError("--hadm-id, --subject-id, and --note-id must be provided together unless you use --case.")
        selectors.append((int(args.hadm_id), int(args.subject_id), str(args.note_id).strip()))

    if not selectors:
        return None

    return list(dict.fromkeys(selectors))


def load_case_selectors(
    spark: Any,
    table_name: str,
    limit: int | None,
    requested_case_selectors: list[tuple[int, int, str]] | None,
    sample_mode: str,
    sample_seed: int,
) -> list[dict[str, object]]:
    df = spark.table(table_name)
    available_columns = set(df.columns)
    required_columns = {"hadm_id", "subject_id", "note_id", "input_text", "output_icd_codes"}
    missing_columns = sorted(required_columns - available_columns)
    if missing_columns:
        raise RuntimeError(f"Table {table_name} is missing required column(s): {missing_columns}")

    selector_columns = [
        column_name
        for column_name in [
            "hadm_id",
            "subject_id",
            "note_id",
            "input_text",
            "output_icd_codes",
            "real_discharge_year_min",
            "real_discharge_year_max",
        ]
        if column_name in available_columns
    ]
    selector_keys = [column_name for column_name in ["hadm_id", "subject_id", "note_id"] if column_name in available_columns]
    case_df = df.filter(df["input_text"].isNotNull()).filter(df["output_icd_codes"].isNotNull()).select(*selector_columns)

    if requested_case_selectors:
        selector_filter = None
        for requested_hadm_id, requested_subject_id, requested_note_id in requested_case_selectors:
            clause = (
                (case_df["hadm_id"] == requested_hadm_id)
                & (case_df["subject_id"] == requested_subject_id)
                & (case_df["note_id"] == requested_note_id)
            )
            selector_filter = clause if selector_filter is None else (selector_filter | clause)
        case_df = case_df.filter(selector_filter)

    case_df = case_df.dropDuplicates(selector_keys)
    if requested_case_selectors:
        case_df = case_df.orderBy(*selector_keys)
    elif sample_mode == "random":
        case_df = case_df.orderBy(rand(sample_seed))
    else:
        case_df = case_df.orderBy(*selector_keys)

    if limit is not None and not requested_case_selectors:
        case_df = case_df.limit(limit)

    selectors = []
    for row in case_df.collect():
        payload = row.asDict(recursive=True)
        payload["expected_icd_codes"] = parse_code_list_json(payload.get("output_icd_codes"))
        payload["case_summary"] = str(payload.get("input_text") or "").strip()
        selectors.append(payload)

    if not requested_case_selectors:
        return selectors

    selector_by_key = {
        (int(row["hadm_id"]), int(row["subject_id"]), str(row["note_id"])): row
        for row in selectors
    }
    missing_keys = [
        {
            "hadm_id": requested_hadm_id,
            "subject_id": requested_subject_id,
            "note_id": requested_note_id,
        }
        for requested_hadm_id, requested_subject_id, requested_note_id in requested_case_selectors
        if (requested_hadm_id, requested_subject_id, requested_note_id) not in selector_by_key
    ]
    if missing_keys:
        raise RuntimeError(f"Requested case selector(s) were not found in {table_name}: {missing_keys}")
    return [selector_by_key[(requested_hadm_id, requested_subject_id, requested_note_id)] for requested_hadm_id, requested_subject_id, requested_note_id in requested_case_selectors]


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
                    "expected_codes_json": json.dumps(list(row.get("expected_icd_codes", []))),
                }
                for row in selectors
            ],
        }
    )


def parse_code_list_json(value: object) -> list[str]:
    from baselines.icd_react.single_case import coerce_icd_code_list

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


def summarize_prediction_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    row_count = len(rows)
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    error_count = row_count - len(ok_rows)
    if not ok_rows:
        return {
            "row_count": row_count,
            "ok_row_count": 0,
            "error_count": error_count,
            "mean_precision": 0.0,
            "mean_recall": 0.0,
            "exact_set_match_rate": 0.0,
            "mean_primary_diagnosis_accuracy": 0.0,
        }

    exact_matches = 0
    precision_total = 0.0
    recall_total = 0.0
    primary_diagnosis_accuracy_total = 0.0
    for row in ok_rows:
        predicted_set = set(row.get("predicted_icd_codes", []))
        expected_set = set(row.get("expected_icd_codes", []))
        if predicted_set == expected_set:
            exact_matches += 1
        precision_total += float(row.get("precision", 0.0) or 0.0)
        recall_total += float(row.get("recall", 0.0) or 0.0)
        primary_diagnosis_accuracy_total += float(row.get("primary_diagnosis_accuracy", 0.0) or 0.0)

    return {
        "row_count": row_count,
        "ok_row_count": len(ok_rows),
        "error_count": error_count,
        "mean_precision": round(precision_total / len(ok_rows), 6),
        "mean_recall": round(recall_total / len(ok_rows), 6),
        "exact_set_match_rate": round(exact_matches / len(ok_rows), 6),
        "mean_primary_diagnosis_accuracy": round(primary_diagnosis_accuracy_total / len(ok_rows), 6),
    }


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

    from baselines.icd_react.runtime import resolve_model_name, resolve_spark_session, resolve_strict_table
    from baselines.icd_react_v2 import load_config, run_single_case_prediction, score_prediction

    config = load_config(args)
    spark = resolve_spark_session(app_name="icd-react-v2-evaluate")
    strict_table = args.strict_table or resolve_strict_table()
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
    model_name = args.model_name or resolve_model_name()
    run_name = args.mlflow_run_name or f"icd_react_v2_eval_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    requested_case_selectors = resolve_requested_case_selectors(args)
    prompt_suffix_text = read_prompt_suffix_file(args.prompt_suffix_file)
    configure_mlflow_genai_workers(args.eval_max_workers, args.eval_max_scorer_workers)
    effective_eval_workers, effective_scorer_workers = resolve_effective_mlflow_worker_settings()

    selectors = load_case_selectors(
        spark=spark,
        table_name=strict_table,
        limit=args.limit,
        requested_case_selectors=requested_case_selectors,
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
    case_records_by_key = {
        (row.get("hadm_id"), row.get("subject_id"), row.get("note_id")): row
        for row in selectors
    }

    if mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_experiment(args.mlflow_experiment)
    mlflow.langchain.autolog(log_traces=True, silent=True)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "baseline": "icd_react_v2",
                "strict_table": strict_table,
                "model_name": model_name,
                "selector_count": len(selectors),
                "execution_env": config.execution_env,
                "hadm_id_filter": args.hadm_id,
                "subject_id_filter": args.subject_id,
                "note_id_filter": args.note_id,
                "case_selectors": json.dumps(requested_case_selectors) if requested_case_selectors else None,
                "sample_mode": args.sample_mode,
                "sample_seed": args.sample_seed,
                "prompt_variant": args.prompt_variant,
                "max_agent_steps": args.max_agent_steps,
                "eval_max_workers": effective_eval_workers,
                "eval_max_scorer_workers": effective_scorer_workers,
                "prompt_suffix_file": args.prompt_suffix_file,
            }
        )
        prediction_rows_by_key: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
        completed_keys: set[tuple[Any, Any, Any]] = set()
        progress_bar = tqdm(total=len(eval_df), desc="icd react v2 cases", unit="case")
        progress_logger = EvaluationProgressLogger(logger=logger, label="ICD ReAct v2 eval", total=len(eval_df))
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
                            "model_name": model_name,
                            "max_agent_steps": args.max_agent_steps,
                        }
                    )
                    try:
                        case_key = (hadm_id, subject_id, note_id)
                        case_record = case_records_by_key.get(case_key)
                        if case_record is None:
                            raise RuntimeError(f"Prefetched ICD case not found for key={case_key}")

                        logger.debug("Starting case hadm_id=%s note_id=%s", hadm_id, note_id)
                        result = run_single_case_prediction(
                            config=config,
                            summary_text=str(case_record.get("case_summary") or ""),
                            model_name=model_name,
                            prompt_variant=args.prompt_variant,
                            prompt_suffix=prompt_suffix_text,
                            max_agent_steps=args.max_agent_steps,
                        )
                        metrics = score_prediction(
                            predicted_codes=result["prediction"].get("predicted_icd_codes", []),
                            expected_codes=case_record.get("expected_icd_codes", []),
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
                                "status": "ok",
                                "error": "",
                                "precision": metrics.get("precision", 0.0),
                                "recall": metrics.get("recall", 0.0),
                                "true_positive_count": metrics.get("true_positive_count", 0),
                                "predicted_count": metrics.get("predicted_count", 0),
                                "expected_count": metrics.get("expected_count", 0),
                                "rationale": result["prediction"].get("rationale", ""),
                                "supporting_evidence": result["prediction"].get("supporting_evidence", []),
                                "confidence": result["prediction"].get("confidence"),
                            }
                        )
                        logger.debug(
                            "Finished case hadm_id=%s note_id=%s status=ok predicted_count=%s",
                            row.get("hadm_id"),
                            row.get("note_id"),
                            row.get("predicted_count"),
                        )
                        case_span.set_outputs(
                            {
                                "status": "ok",
                                "predicted_icd_codes": row["predicted_icd_codes"],
                                "expected_icd_codes": row["expected_icd_codes"],
                                "predicted_code_count": row["predicted_count"],
                                "expected_code_count": row["expected_count"],
                                "precision": row["precision"],
                                "recall": row["recall"],
                                "primary_diagnosis_accuracy": row["primary_diagnosis_accuracy"],
                            }
                        )
                    except Exception as exc:
                        logger.exception("ICD ReAct v2 evaluation failed for hadm_id=%s note_id=%s", hadm_id, note_id)
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
            raise RuntimeError("No ICD ReAct v2 evaluation rows were produced by the prediction function.")

        summary = to_jsonable(
            {
                "run_id": run.info.run_id,
                "row_count": len(evaluation_df),
                "scored_row_count": len(eval_df),
                "error_count": int((evaluation_df["status"] != "ok").sum()),
                "prompt_variant": args.prompt_variant,
                "metrics": evaluation.metrics,
                "aggregate_metrics": summarize_prediction_rows(evaluation_df.to_dict(orient="records")),
            }
        )
        mlflow.log_dict(summary, "evaluation/icd_react_v2_summary.json")
        mlflow.log_dict(to_jsonable({"rows": evaluation_df.to_dict(orient="records")}), "evaluation/icd_react_v2_rows.json")
        if "eval_results" in evaluation.tables:
            mlflow.log_dict(to_jsonable({"rows": evaluation.tables["eval_results"].to_dict(orient="records")}), "evaluation/icd_react_v2_genai_eval.json")
        print(json.dumps(summary, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())