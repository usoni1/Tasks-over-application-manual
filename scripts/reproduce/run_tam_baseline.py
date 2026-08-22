from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ICD_DATASET_PATH = REPO_ROOT / "data" / "mimic" / "mimic_icd10_note_dataset_2017_2019_strict.parquet"
DEFAULT_LEGAL_FINAL_DATASET_CSV = (
    REPO_ROOT / "data" / "final-approved-200" / "federal_sentencing_legal_final_dataset_approved.csv"
)
DEFAULT_REVIEW_VERSION = 4
DEFAULT_ICD_VIEW_NAME = "tam_icd_strict_local"
DEFAULT_LEGAL_DATASET_VIEW_NAME = "tam_legal_final_dataset_local"
DEFAULT_LEGAL_YEAR_VIEW_NAME = "tam_legal_sentencing_years_local"
DEFAULT_LEGAL_REVIEW_VIEW_NAME = "tam_legal_review_dataset_local"
DEFAULT_LEGAL_CASE_SOURCE_VIEW_NAME = "tam_legal_case_source_local"

EXPERIMENTS: dict[str, dict[str, str]] = {
    "icd-single-pass-rag": {
        "script": "scripts/evaluate_icd_rag.py",
        "kind": "icd",
        "paper_label": "Single-pass RAG",
    },
    "icd-agentic-rag": {
        "script": "scripts/evaluate_icd_agentic_rag.py",
        "kind": "icd",
        "paper_label": "Agentic RAG",
    },
    "icd-react-style-tool-use": {
        "script": "scripts/evaluate_icd_react_v2.py",
        "kind": "icd",
        "paper_label": "ReAct-style tool use",
    },
    "icd-deepagents": {
        "script": "scripts/evaluate_icd_deepagents.py",
        "kind": "icd",
        "paper_label": "Deep Agents skills",
    },
    "legal-single-pass-rag": {
        "script": "scripts/evaluate_legal_rag.py",
        "kind": "legal_final",
        "paper_label": "Single-pass RAG",
    },
    "legal-agentic-rag": {
        "script": "scripts/evaluate_legal_agentic_rag.py",
        "kind": "legal_final",
        "paper_label": "Agentic RAG",
    },
    "legal-react-style-tool-use": {
        "script": "scripts/evaluate_legal_react_v2_on_review_dataset.py",
        "kind": "legal_review",
        "paper_label": "ReAct-style tool use",
    },
    "legal-deepagents": {
        "script": "scripts/evaluate_legal_deepagents.py",
        "kind": "legal_final",
        "paper_label": "Deep Agents skills",
    },
}


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run one TAM paper baseline using the standalone evaluators vendored into this repository."
        ),
        epilog=(
            "Forward any evaluator-specific arguments after --. Example: "
            "python scripts/reproduce/run_tam_baseline.py --experiment icd-single-pass-rag "
            "-- --limit 25"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument(
        "--icd-dataset-path",
        type=Path,
        default=DEFAULT_ICD_DATASET_PATH,
        help="Local parquet dataset produced by the MIMIC ICD build helper.",
    )
    parser.add_argument(
        "--legal-final-dataset-csv",
        type=Path,
        default=DEFAULT_LEGAL_FINAL_DATASET_CSV,
        help="Released legal approved-slice CSV with docket_id, input, and output columns.",
    )
    parser.add_argument(
        "--legal-sentencing-year-map-csv",
        type=Path,
        default=None,
        help=(
            "CSV with docket_id and one of sentencing_year, guideline_year, or year. "
            "Required for local legal runs unless you pass explicit table/view arguments through to the evaluator."
        ),
    )
    parser.add_argument(
        "--review-version",
        type=int,
        default=DEFAULT_REVIEW_VERSION,
        help="Review version injected when synthesizing the legal review-dataset view locally.",
    )
    parser.add_argument("--icd-view-name", default=DEFAULT_ICD_VIEW_NAME)
    parser.add_argument("--legal-dataset-view-name", default=DEFAULT_LEGAL_DATASET_VIEW_NAME)
    parser.add_argument("--legal-year-view-name", default=DEFAULT_LEGAL_YEAR_VIEW_NAME)
    parser.add_argument("--legal-review-view-name", default=DEFAULT_LEGAL_REVIEW_VIEW_NAME)
    parser.add_argument("--legal-case-source-view-name", default=DEFAULT_LEGAL_CASE_SOURCE_VIEW_NAME)
    args, forwarded_args = parser.parse_known_args(argv)
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]
    return args, forwarded_args


def require_existing_file(path: Path, description: str) -> Path:
    candidate = path.expanduser()
    if not candidate.is_file():
        raise SystemExit(f"Missing {description}: {candidate}")
    return candidate.resolve()


def forwarded_has_option(forwarded_args: list[str], option_names: list[str]) -> bool:
    for token in forwarded_args:
        for option_name in option_names:
            if token == option_name or token.startswith(f"{option_name}="):
                return True
    return False


def append_option_if_missing(
    forwarded_args: list[str],
    option_names: list[str],
    option_name: str,
    option_value: str,
) -> list[str]:
    if forwarded_has_option(forwarded_args, option_names):
        return forwarded_args
    return [*forwarded_args, option_name, option_value]


def build_spark() -> SparkSession:
    active_session = SparkSession.getActiveSession()
    if active_session is not None:
        return active_session

    return (
        SparkSession.builder.appName("tam-artifact-reproduction")
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def read_csv(spark: SparkSession, csv_path: Path, *, multiline: bool) -> Any:
    reader = spark.read.option("header", True)
    if multiline:
        reader = reader.option("multiLine", True).option("quote", '"').option("escape", '"')
    return reader.csv(str(csv_path))


def resolve_column_name(columns: list[str], candidates: list[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def register_icd_view(spark: SparkSession, dataset_path: Path, view_name: str) -> None:
    resolved_path = require_existing_file(dataset_path, "ICD dataset parquet directory") if dataset_path.is_file() else dataset_path.expanduser().resolve()
    if not resolved_path.exists():
        raise SystemExit(
            f"Missing ICD dataset parquet directory: {resolved_path}. Build it first with scripts/mimic/build_mimic_icd10_dataset_2017_2019.py."
        )
    dataset_df = spark.read.parquet(str(resolved_path))
    required_columns = {"hadm_id", "subject_id", "note_id", "input_text", "output_icd_codes"}
    missing_columns = sorted(required_columns - set(dataset_df.columns))
    if missing_columns:
        raise SystemExit(f"ICD dataset is missing required column(s): {missing_columns}")
    dataset_df.createOrReplaceTempView(view_name)


def register_legal_final_dataset_view(spark: SparkSession, csv_path: Path, view_name: str) -> None:
    resolved_csv = require_existing_file(csv_path, "approved legal CSV")
    dataset_df = read_csv(spark, resolved_csv, multiline=True)
    docket_column = resolve_column_name(dataset_df.columns, ["docket_id"])
    input_column = resolve_column_name(dataset_df.columns, ["input", "input_case_facts_text"])
    output_column = resolve_column_name(dataset_df.columns, ["output", "ground_truth_offense_level"])
    if docket_column is None or input_column is None or output_column is None:
        raise SystemExit(
            "Approved legal CSV must contain docket_id, input, and output columns "
            "or compatible aliases."
        )

    normalized_df = (
        dataset_df.select(
            F.col(docket_column).cast("long").alias("docket_id"),
            F.col(input_column).cast("string").alias("input"),
            F.col(output_column).cast("string").alias("output"),
        )
        .filter(F.col("docket_id").isNotNull())
        .filter(F.col("input").isNotNull())
        .filter(F.col("output").isNotNull())
        .dropDuplicates(["docket_id"])
    )
    normalized_df.createOrReplaceTempView(view_name)


def register_legal_year_view(spark: SparkSession, csv_path: Path, view_name: str) -> None:
    resolved_csv = require_existing_file(csv_path, "legal sentencing year map CSV")
    year_df = read_csv(spark, resolved_csv, multiline=False)
    docket_column = resolve_column_name(year_df.columns, ["docket_id"])
    year_column = resolve_column_name(year_df.columns, ["sentencing_year", "guideline_year", "year"])
    status_column = resolve_column_name(year_df.columns, ["year_lookup_status", "lookup_status", "source"])
    if docket_column is None or year_column is None:
        raise SystemExit(
            "Legal sentencing year map CSV must contain docket_id and one of sentencing_year, guideline_year, or year."
        )

    normalized_df = (
        year_df.select(
            F.col(docket_column).cast("long").alias("docket_id"),
            F.col(year_column).cast("int").alias("sentencing_year"),
            (
                F.col(status_column).cast("string")
                if status_column is not None
                else F.lit("local_csv")
            ).alias("year_lookup_status"),
        )
        .filter(F.col("docket_id").isNotNull())
        .filter(F.col("sentencing_year").isNotNull())
        .dropDuplicates(["docket_id"])
    )
    normalized_df.createOrReplaceTempView(view_name)


def register_legal_review_dataset_view(
    spark: SparkSession,
    *,
    dataset_view_name: str,
    year_view_name: str,
    review_view_name: str,
    review_version: int,
) -> None:
    dataset_df = spark.table(dataset_view_name)
    year_df = spark.table(year_view_name)
    review_df = (
        dataset_df.alias("dataset")
        .join(year_df.alias("years"), on="docket_id", how="inner")
        .select(
            F.col("docket_id").cast("long").alias("docket_id"),
            F.col("years.sentencing_year").cast("int").alias("guideline_year"),
            F.col("dataset.input").cast("string").alias("input_case_facts_text"),
            F.expr(
                "size(filter(split(regexp_replace(input, '\\r', ''), '\\n'), x -> trim(x) <> ''))"
            ).alias("input_case_fact_count"),
            F.col("dataset.output").cast("int").alias("ground_truth_offense_level"),
            F.lit("approved_release").alias("queue_label"),
            F.lit(True).alias("approved"),
            F.lit("approved").alias("review_decision"),
            F.lit(int(review_version)).alias("review_version"),
            F.lit("approved_release_csv").alias("source_run_id"),
        )
        .filter(F.col("ground_truth_offense_level").isNotNull())
    )
    review_df.createOrReplaceTempView(review_view_name)


def register_legal_case_source_view(
    spark: SparkSession,
    *,
    year_view_name: str,
    case_source_view_name: str,
) -> None:
    case_source_df = spark.table(year_view_name).select(
        F.col("docket_id").cast("long").alias("docket_id"),
        F.col("sentencing_year").cast("int").alias("year"),
        F.lit(None).cast("long").alias("government_sm_doc_id"),
        F.lit(None).cast("boolean").alias("acceptance_of_responsibility"),
    )
    case_source_df.createOrReplaceTempView(case_source_view_name)


def load_script_module(script_relative_path: str):
    script_path = REPO_ROOT / script_relative_path
    if not script_path.is_file():
        raise SystemExit(f"Evaluator script not found: {script_path}")
    module_name = f"tam_repro_{script_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load evaluator module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def install_shared_spark(spark: SparkSession) -> None:
    for module_name in ["baselines.legal_rag.runtime", "review_pipeline_v1.runtime"]:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        setattr(module, "spark", spark)


def prepare_forwarded_args(
    args: argparse.Namespace,
    forwarded_args: list[str],
    spark: SparkSession,
) -> list[str]:
    experiment = EXPERIMENTS[args.experiment]
    kind = experiment["kind"]
    prepared_args = list(forwarded_args)
    prepared_args = append_option_if_missing(prepared_args, ["--execution-env"], "--execution-env", "local")

    if kind == "icd":
        if not forwarded_has_option(prepared_args, ["--strict-table"]):
            register_icd_view(spark, args.icd_dataset_path, args.icd_view_name)
            prepared_args = append_option_if_missing(
                prepared_args,
                ["--strict-table"],
                "--strict-table",
                args.icd_view_name,
            )
        return prepared_args

    if kind == "legal_final":
        has_dataset_table = forwarded_has_option(prepared_args, ["--dataset-table", "--strict-table"])
        has_year_table = forwarded_has_option(prepared_args, ["--sentencing-year-table"])
        if not has_dataset_table:
            register_legal_final_dataset_view(spark, args.legal_final_dataset_csv, args.legal_dataset_view_name)
            prepared_args = append_option_if_missing(
                prepared_args,
                ["--dataset-table", "--strict-table"],
                "--dataset-table",
                args.legal_dataset_view_name,
            )
        if not has_year_table:
            if args.legal_sentencing_year_map_csv is None:
                raise SystemExit(
                    "Local legal RAG runs need --legal-sentencing-year-map-csv unless you pass "
                    "--sentencing-year-table and dataset-table arguments through to the evaluator."
                )
            register_legal_year_view(spark, args.legal_sentencing_year_map_csv, args.legal_year_view_name)
            prepared_args = append_option_if_missing(
                prepared_args,
                ["--sentencing-year-table"],
                "--sentencing-year-table",
                args.legal_year_view_name,
            )
        return prepared_args

    has_review_table = forwarded_has_option(prepared_args, ["--final-dataset-table"])
    has_case_source = forwarded_has_option(prepared_args, ["--case-source-table", "--strict-table"])
    has_year_table = forwarded_has_option(prepared_args, ["--sentencing-year-table"])

    if not has_review_table or not has_case_source or not has_year_table:
        if args.legal_sentencing_year_map_csv is None:
            raise SystemExit(
                "Local legal ReAct runs need --legal-sentencing-year-map-csv unless you pass "
                "--final-dataset-table, --case-source-table or --strict-table, and --sentencing-year-table."
            )
        register_legal_final_dataset_view(spark, args.legal_final_dataset_csv, args.legal_dataset_view_name)
        register_legal_year_view(spark, args.legal_sentencing_year_map_csv, args.legal_year_view_name)

    if not has_review_table:
        register_legal_review_dataset_view(
            spark,
            dataset_view_name=args.legal_dataset_view_name,
            year_view_name=args.legal_year_view_name,
            review_view_name=args.legal_review_view_name,
            review_version=args.review_version,
        )
        prepared_args = append_option_if_missing(
            prepared_args,
            ["--final-dataset-table"],
            "--final-dataset-table",
            args.legal_review_view_name,
        )

    if not has_case_source:
        register_legal_case_source_view(
            spark,
            year_view_name=args.legal_year_view_name,
            case_source_view_name=args.legal_case_source_view_name,
        )
        prepared_args = append_option_if_missing(
            prepared_args,
            ["--case-source-table", "--strict-table"],
            "--case-source-table",
            args.legal_case_source_view_name,
        )

    if not has_year_table:
        prepared_args = append_option_if_missing(
            prepared_args,
            ["--sentencing-year-table"],
            "--sentencing-year-table",
            args.legal_year_view_name,
        )

    prepared_args = append_option_if_missing(
        prepared_args,
        ["--review-version"],
        "--review-version",
        str(args.review_version),
    )
    return prepared_args


def main(argv: list[str] | None = None) -> int:
    args, forwarded_args = parse_args(argv)
    os.environ.setdefault("EXECUTION_ENV", "local")

    experiment = EXPERIMENTS[args.experiment]
    if forwarded_has_option(forwarded_args, ["-h", "--help"]):
        evaluator_module = load_script_module(experiment["script"])
        try:
            evaluator_module.main(["--help"])
        except SystemExit as exc:
            return int(exc.code or 0)
        return 0

    spark = build_spark()
    install_shared_spark(spark)

    prepared_args = prepare_forwarded_args(args, forwarded_args, spark)
    evaluator_module = load_script_module(experiment["script"])
    setattr(evaluator_module, "spark", spark)

    print(
        f"Running {args.experiment} ({experiment['paper_label']}) via {experiment['script']}",
        flush=True,
    )
    print(f"Evaluator arguments: {prepared_args}", flush=True)

    result = evaluator_module.main(prepared_args)
    if result is None:
        return 0
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())