from __future__ import annotations

import argparse
from datetime import UTC, datetime
import inspect
import json
import logging
import os
import sys
from pathlib import Path
from time import monotonic
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

from scripts.evaluate_legal_react_v2 import (
    build_case_key,
    build_eval_dataframe,
    configure_logging,
    load_case_selectors,
    offense_level_exact_match,
    resolve_requested_case_selectors,
    to_jsonable,
)


DEFAULT_MLFLOW_EXPERIMENT = "/Users/sonutka@mfcgd.com/legal-deepagents-evaluate"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from baselines.legal_rag.runtime import resolve_final_legal_dataset_table, resolve_sentencing_year_table

    parser = argparse.ArgumentParser(description="Evaluate the legal Deep Agents skills baseline with mlflow.evaluate.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    parser.add_argument("--execution-env", choices=["local", "databricks"], default=None)
    parser.add_argument("--dataset-table", type=str, default=resolve_final_legal_dataset_table())
    parser.add_argument("--strict-table", type=str, default=None)
    parser.add_argument("--acceptance-table", type=str, default=None)
    parser.add_argument("--sentencing-year-table", type=str, default=resolve_sentencing_year_table())
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--model-provider", choices=["default", "azure_ad", "azure_key", "gateway"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--docket-id", type=int, default=None)
    parser.add_argument("--docket-ids", type=str, default=None)
    parser.add_argument("--case", dest="cases", action="append", default=None)
    parser.add_argument("--max-agent-steps", type=int, default=60)
    parser.add_argument("--eval-max-workers", type=int, default=None)
    parser.add_argument("--eval-max-scorer-workers", type=int, default=None)
    parser.add_argument("--progress-log-every", type=int, default=1, help="Log progress after this many completed cases.")
    parser.add_argument("--mlflow-experiment", type=str, default=DEFAULT_MLFLOW_EXPERIMENT)
    parser.add_argument("--mlflow-run-name", type=str, default=None)
    return parser.parse_args(argv)


def configure_mlflow_genai_workers(max_workers: int | None, max_scorer_workers: int | None) -> tuple[str | None, str | None]:
    if max_workers is not None:
        if max_workers <= 0:
            raise RuntimeError("--eval-max-workers must be greater than 0")
        os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = str(max_workers)
    if max_scorer_workers is not None:
        if max_scorer_workers <= 0:
            raise RuntimeError("--eval-max-scorer-workers must be greater than 0")
        os.environ["MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS"] = str(max_scorer_workers)
    return os.environ.get("MLFLOW_GENAI_EVAL_MAX_WORKERS"), os.environ.get("MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    if args.progress_log_every <= 0:
        raise RuntimeError("--progress-log-every must be greater than 0")

    from baselines.legal_deepagents import run_single_case_prediction, score_prediction
    from baselines.legal_rag.runtime import (
        prepare_federal_sentencing_source_table,
        resolve_acceptance_table,
        resolve_final_legal_dataset_table,
        resolve_model_name,
        resolve_spark_session,
        resolve_strict_table,
    )
    from baselines.legal_react_v2 import load_config
    from scripts.eval_logging import EvaluationProgressLogger

    config = load_config(args)
    logger = logging.getLogger(__name__)
    logger.info("Creating Databricks Spark session for legal Deep Agents evaluation")
    spark = resolve_spark_session(app_name="legal-deepagents-evaluate")
    dataset_table = args.dataset_table or resolve_final_legal_dataset_table()
    strict_table = args.strict_table or resolve_strict_table()
    acceptance_table = args.acceptance_table if args.acceptance_table is not None else resolve_acceptance_table()
    logger.info("Preparing legal acceptance source view from %s", strict_table)
    case_source_table, effective_acceptance_table = prepare_federal_sentencing_source_table(
        spark=spark,
        strict_table=strict_table,
        acceptance_table=acceptance_table,
        temp_view_name="legal_deepagents_case_source",
        logger=logger,
    )
    model_name = args.model_name or resolve_model_name()
    run_name = args.mlflow_run_name or f"legal_deepagents_eval_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    requested_case_selectors = resolve_requested_case_selectors(args)
    effective_eval_workers, effective_scorer_workers = configure_mlflow_genai_workers(
        args.eval_max_workers, args.eval_max_scorer_workers
    )

    if mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_experiment(args.mlflow_experiment)
    mlflow.langchain.autolog(log_traces=True, silent=True)

    logger.info("Selecting legal evaluation cases from %s", dataset_table)
    selectors = load_case_selectors(
        spark=spark,
        table_name=dataset_table,
        limit=args.limit,
        year=args.year,
        requested_case_selectors=requested_case_selectors,
        sentencing_year_table=args.sentencing_year_table,
        case_source_table=case_source_table,
        logger=logger,
    )
    if not selectors:
        raise RuntimeError("No legal cases matched the requested evaluation selection.")

    from baselines.legal_react_v2.tools import available_legal_manual_years, prewarm_legal_manual_indexes

    supported_manual_years = set(available_legal_manual_years(config))
    selector_years = {int(row["year"]) for row in selectors if row.get("year") is not None}
    unsupported_years = sorted(selector_years - supported_manual_years)
    unsupported_selector_count = sum(
        row.get("year") is None or int(row["year"]) not in supported_manual_years for row in selectors
    )
    if unsupported_selector_count:
        logger.warning(
            "%s legal selector(s) use unavailable complete manual editions and may produce per-case errors; supported_years=%s unsupported_years=%s",
            unsupported_selector_count,
            sorted(supported_manual_years),
            unsupported_years,
        )

    selector_by_key = {build_case_key(row.get("year"), row.get("docket_id")): dict(row) for row in selectors}
    logger.info("Loaded %s legal case selectors from %s", len(selectors), dataset_table)

    manual_years = selector_years & supported_manual_years
    logger.info("Prewarming legal manual indexes for year(s): %s", sorted(manual_years))
    warmup_started_at = monotonic()
    prewarm_legal_manual_indexes(config=config, years=manual_years)
    logger.info("Legal manual index warmup complete in %.1fs", monotonic() - warmup_started_at)
    eval_df = build_eval_dataframe(selectors)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "baseline": "legal_deepagents_skills_subagents",
                "architecture": "coordinator_plus_role_specialized_skills_subagents_full_harness",
                "dataset_table": dataset_table,
                "strict_table": strict_table,
                "case_source_table": case_source_table,
                "acceptance_table": acceptance_table,
                "acceptance_table_used": effective_acceptance_table,
                "model_name": model_name,
                "model_provider": args.model_provider or os.environ.get("ICD_DEEPAGENTS_MODEL_PROVIDER") or "auto",
                "selector_count": len(selectors),
                "unsupported_manual_year_selector_count": unsupported_selector_count,
                "supported_manual_years": json.dumps(sorted(supported_manual_years)),
                "execution_env": config.execution_env,
                "year_filter": args.year,
                "case_selectors": json.dumps(requested_case_selectors) if requested_case_selectors else None,
                "max_agent_steps": args.max_agent_steps,
                "eval_max_workers": effective_eval_workers,
                "eval_max_scorer_workers": effective_scorer_workers,
            }
        )
        prediction_rows_by_key: dict[tuple[Any, Any], dict[str, Any]] = {}
        completed_keys: set[tuple[Any, Any]] = set()
        progress_bar = tqdm(total=len(eval_df), desc="legal deepagents cases", unit="case")
        progress_logger = EvaluationProgressLogger(
            logger=logger,
            label="Legal Deep Agents eval",
            total=len(eval_df),
            log_every=args.progress_log_every,
        )
        progress_logger.log_start()

        def predict_case(year: Any, docket_id: Any) -> dict[str, object]:
            row: dict[str, Any] = {
                "year": year,
                "docket_id": docket_id,
                "expected_offense_level_total": "",
                "predicted_offense_level_total": "",
                "status": "error",
                "error": "",
            }
            try:
                with mlflow.start_span(name=f"docket_{docket_id}") as case_span:
                    case_span.set_inputs({"year": year, "docket_id": docket_id, "model_name": model_name})
                    try:
                        case_record = dict(selector_by_key[build_case_key(year, docket_id)])
                        result = run_single_case_prediction(
                            config=config,
                            summary_text=str(case_record.get("case_summary") or ""),
                            model_name=model_name,
                            year=case_record.get("year"),
                            case_record=case_record,
                            max_agent_steps=args.max_agent_steps,
                            model_provider=args.model_provider,
                        )
                        predicted_offense_level = result["prediction"].get("offense_level")
                        metrics = score_prediction(result["prediction"], case_record)
                        row.update(
                            {
                                "year": case_record.get("year"),
                                "docket_id": case_record.get("docket_id"),
                                "government_sm_doc_id": case_record.get("government_sm_doc_id"),
                                "expected_offense_level_total": case_record.get("expected_offense_level_total") or "",
                                "predicted_offense_level_total": "" if predicted_offense_level is None else str(predicted_offense_level),
                                "status": "ok",
                                "offense_level_total_exact_match": metrics.get("offense_level_total_exact_match", 0),
                                "exact_match_rate": metrics.get("exact_match_rate", 0.0),
                                "justifications": result["prediction"].get("justifications", []),
                            }
                        )
                        case_span.set_outputs({"status": "ok", "predicted_offense_level_total": row["predicted_offense_level_total"]})
                    except Exception as exc:
                        logger.exception("Legal Deep Agents evaluation failed for docket_id=%s year=%s", docket_id, year)
                        row["error"] = f"{type(exc).__name__}: {exc}"
                        case_span.set_outputs({"status": "error", "error": row["error"]})
            finally:
                row_key = build_case_key(row.get("year"), row.get("docket_id"))
                prediction_rows_by_key[row_key] = row
                if row_key not in completed_keys:
                    completed_keys.add(row_key)
                    progress_bar.update(1)
                    progress_logger.update(status=str(row.get("status") or "error"))
            return {"predicted_offense_level_total": row["predicted_offense_level_total"], "status": row["status"], "error": row["error"]}

        try:
            logger.info(
                "Dispatching %s legal cases to mlflow.genai.evaluate with predict_workers=%s scorer_workers=%s",
                len(eval_df),
                effective_eval_workers,
                effective_scorer_workers,
            )
            evaluation = mlflow.genai.evaluate(data=eval_df, predict_fn=predict_case, scorers=[offense_level_exact_match])
        finally:
            progress_bar.close()

        evaluation_df = pd.DataFrame(prediction_rows_by_key.values())
        if evaluation_df.empty:
            raise RuntimeError("No legal Deep Agents evaluation rows were produced.")
        summary = to_jsonable(
            {
                "run_id": run.info.run_id,
                "row_count": len(evaluation_df),
                "scored_row_count": len(eval_df),
                "error_count": int((evaluation_df["status"] != "ok").sum()),
                "metrics": evaluation.metrics,
            }
        )
        mlflow.log_dict(summary, "evaluation/legal_deepagents_summary.json")
        mlflow.log_dict(to_jsonable({"rows": evaluation_df.to_dict(orient="records")}), "evaluation/legal_deepagents_rows.json")
        if "eval_results" in evaluation.tables:
            mlflow.log_dict(to_jsonable({"rows": evaluation.tables["eval_results"].to_dict(orient="records")}), "evaluation/legal_deepagents_genai_eval.json")
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
