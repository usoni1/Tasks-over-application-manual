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
import mlflow.langchain
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


PROJECT_ROOT = resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_icd_react_v2 import (
    build_eval_dataframe,
    configure_logging,
    configure_mlflow_genai_workers,
    exact_set_match,
    load_case_selectors,
    precision_scorer,
    primary_diagnosis_accuracy_scorer,
    read_prompt_suffix_file,
    recall_scorer,
    resolve_effective_mlflow_worker_settings,
    resolve_requested_case_selectors,
    summarize_prediction_rows,
    to_jsonable,
)

DEFAULT_MLFLOW_EXPERIMENT = "/Users/sonutka@mfcgd.com/icd-deepagents-evaluate"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the ICD Deep Agents skills baseline with mlflow.evaluate.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    parser.add_argument("--execution-env", choices=["local", "databricks"], default=None)
    parser.add_argument("--strict-table", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--model-provider", choices=["default", "azure_ad", "azure_key", "gateway"], default=None)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--sample-mode", choices=["ordered", "random"], default="random")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--prompt-variant", type=str, default="active_first_no_repeat")
    parser.add_argument("--hadm-id", type=int, default=None)
    parser.add_argument("--subject-id", type=int, default=None)
    parser.add_argument("--note-id", type=str, default=None)
    parser.add_argument("--case", dest="cases", action="append", default=None)
    parser.add_argument("--max-agent-steps", type=int, default=60)
    parser.add_argument("--eval-max-workers", type=int, default=None)
    parser.add_argument("--eval-max-scorer-workers", type=int, default=None)
    parser.add_argument("--progress-log-every", type=int, default=1, help="Log progress after this many completed cases.")
    parser.add_argument("--prompt-suffix-file", type=str, default=None)
    parser.add_argument("--mlflow-experiment", type=str, default=DEFAULT_MLFLOW_EXPERIMENT)
    parser.add_argument("--mlflow-run-name", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    if args.progress_log_every <= 0:
        raise RuntimeError("--progress-log-every must be greater than 0")

    from baselines.icd_deepagents import run_single_case_prediction, score_prediction
    from baselines.icd_react.runtime import resolve_model_name, resolve_spark_session, resolve_strict_table
    from baselines.icd_react_v2 import load_config
    from scripts.eval_logging import EvaluationProgressLogger

    config = load_config(args)
    spark = resolve_spark_session(app_name="icd-deepagents-evaluate")
    strict_table = args.strict_table or resolve_strict_table()
    model_name = args.model_name or resolve_model_name()
    run_name = args.mlflow_run_name or f"icd_deepagents_eval_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    requested_case_selectors = resolve_requested_case_selectors(args)
    prompt_suffix_text = read_prompt_suffix_file(args.prompt_suffix_file)
    configure_mlflow_genai_workers(args.eval_max_workers, args.eval_max_scorer_workers)
    effective_eval_workers, effective_scorer_workers = resolve_effective_mlflow_worker_settings()

    if mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_experiment(args.mlflow_experiment)
    mlflow.langchain.autolog(log_traces=True, silent=True)

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
    eval_df = build_eval_dataframe(selectors)
    case_records_by_key = {(row.get("hadm_id"), row.get("subject_id"), row.get("note_id")): row for row in selectors}

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "baseline": "icd_deepagents_skills_subagents",
                "strict_table": strict_table,
                "model_name": model_name,
                "model_provider": args.model_provider or os.environ.get("ICD_DEEPAGENTS_MODEL_PROVIDER") or "auto",
                "selector_count": len(selectors),
                "execution_env": config.execution_env,
                "case_selectors": json.dumps(requested_case_selectors) if requested_case_selectors else None,
                "sample_mode": args.sample_mode,
                "sample_seed": args.sample_seed,
                "prompt_variant": args.prompt_variant,
                "max_agent_steps": args.max_agent_steps,
                "eval_max_workers": effective_eval_workers,
                "eval_max_scorer_workers": effective_scorer_workers,
                "prompt_suffix_file": args.prompt_suffix_file,
                "architecture": "coordinator_plus_role_specialized_skills_subagents_full_harness",
            }
        )
        prediction_rows_by_key: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
        completed_keys: set[tuple[Any, Any, Any]] = set()
        progress_bar = tqdm(total=len(eval_df), desc="icd deepagents cases", unit="case")
        progress_logger = EvaluationProgressLogger(
            logger=logger,
            label="ICD Deep Agents eval",
            total=len(eval_df),
            log_every=args.progress_log_every,
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
                "status": "error",
                "error": "",
            }
            try:
                with mlflow.start_span(name=f"hadm_{hadm_id}") as case_span:
                    case_span.set_inputs({"hadm_id": hadm_id, "subject_id": subject_id, "note_id": note_id, "model_name": model_name})
                    try:
                        case_record = case_records_by_key.get((hadm_id, subject_id, note_id))
                        if case_record is None:
                            raise RuntimeError(f"Prefetched ICD case not found for key={(hadm_id, subject_id, note_id)}")
                        result = run_single_case_prediction(
                            config=config,
                            summary_text=str(case_record.get("case_summary") or ""),
                            model_name=model_name,
                            prompt_variant=args.prompt_variant,
                            prompt_suffix=prompt_suffix_text,
                            max_agent_steps=args.max_agent_steps,
                            model_provider=args.model_provider,
                        )
                        metrics = score_prediction(result["prediction"].get("predicted_icd_codes", []), case_record.get("expected_icd_codes", []))
                        row.update(
                            {
                                "hadm_id": case_record.get("hadm_id"),
                                "subject_id": case_record.get("subject_id"),
                                "note_id": case_record.get("note_id"),
                                "expected_codes_json": json.dumps(list(case_record.get("expected_icd_codes", []))),
                                "predicted_codes_json": json.dumps(list(result["prediction"].get("predicted_icd_codes", []))),
                                "expected_icd_codes": list(case_record.get("expected_icd_codes", [])),
                                "predicted_icd_codes": list(result["prediction"].get("predicted_icd_codes", [])),
                                "status": "ok",
                                "precision": metrics.get("precision", 0.0),
                                "recall": metrics.get("recall", 0.0),
                                "primary_diagnosis_accuracy": metrics.get("primary_diagnosis_accuracy", 0.0),
                                "rationale": result["prediction"].get("rationale", ""),
                                "supporting_evidence": result["prediction"].get("supporting_evidence", []),
                                "confidence": result["prediction"].get("confidence"),
                            }
                        )
                        case_span.set_outputs({"status": "ok", "predicted_icd_codes": row["predicted_icd_codes"], "expected_icd_codes": row["expected_icd_codes"]})
                    except Exception as exc:
                        logger.exception("ICD Deep Agents evaluation failed for hadm_id=%s note_id=%s", hadm_id, note_id)
                        row["error"] = f"{type(exc).__name__}: {exc}"
                        case_span.set_outputs({"status": "error", "error": row["error"]})
            finally:
                row_key = (row.get("hadm_id"), row.get("subject_id"), row.get("note_id"))
                prediction_rows_by_key[row_key] = row
                if row_key not in completed_keys:
                    completed_keys.add(row_key)
                    progress_bar.update(1)
                    progress_logger.update(status=str(row.get("status") or "error"))
            return {"predicted_codes_json": row["predicted_codes_json"], "predicted_icd_codes": row["predicted_icd_codes"], "status": row["status"], "error": row["error"]}

        try:
            evaluation = mlflow.genai.evaluate(
                data=eval_df,
                predict_fn=predict_case,
                scorers=[exact_set_match, precision_scorer, recall_scorer, primary_diagnosis_accuracy_scorer],
            )
        finally:
            progress_bar.close()

        evaluation_df = pd.DataFrame(prediction_rows_by_key.values())
        summary = to_jsonable(
            {
                "run_id": run.info.run_id,
                "row_count": len(evaluation_df),
                "scored_row_count": len(eval_df),
                "error_count": int((evaluation_df["status"] != "ok").sum()),
                "metrics": evaluation.metrics,
                "aggregate_metrics": summarize_prediction_rows(evaluation_df.to_dict(orient="records")),
            }
        )
        mlflow.log_dict(summary, "evaluation/icd_deepagents_summary.json")
        mlflow.log_dict(to_jsonable({"rows": evaluation_df.to_dict(orient="records")}), "evaluation/icd_deepagents_rows.json")
        if "eval_results" in evaluation.tables:
            mlflow.log_dict(to_jsonable({"rows": evaluation.tables["eval_results"].to_dict(orient="records")}), "evaluation/icd_deepagents_genai_eval.json")
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
