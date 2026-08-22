from __future__ import annotations

from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LOCAL_ICD_STRICT_TABLE = "mimic_icd10_note_dataset_2017_2019_strict"
DEFAULT_LOCAL_LEGAL_STRICT_TABLE = "federal_sentencing_strict_docintel_script"
DEFAULT_LOCAL_ACCEPTANCE_TABLE = "federal_sentencing_acceptance_of_responsibility"
DEFAULT_LOCAL_SENTENCING_YEAR_TABLE = "federal_sentencing_docket_sentencing_year_map"
DEFAULT_LOCAL_FINAL_LEGAL_DATASET_TABLE = "federal_sentencing_legal_final_dataset_approved"
DEFAULT_LOCAL_CASE_SOURCE_TABLE = "federal_sentencing_case_source"


def resolve_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def resolve_spark_session(*, execution_env: str = "local", app_name: str | None = None) -> Any:
    existing_spark = globals().get("spark")
    if existing_spark is not None:
        return existing_spark

    normalized_env = str(execution_env or "local").strip().lower()
    if normalized_env == "databricks":
        try:
            from databricks.connect import DatabricksSession

            builder = DatabricksSession.builder
            if app_name and hasattr(builder, "appName"):
                builder = builder.appName(app_name)
            return builder.getOrCreate()
        except ImportError:
            pass

    from pyspark.sql import SparkSession

    builder = SparkSession.builder.master("local[*]")
    if app_name and hasattr(builder, "appName"):
        builder = builder.appName(app_name)
    return builder.getOrCreate()


def infer_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return "csv"
    if suffix == ".json":
        return "json"
    return "parquet"


def read_dataframe(spark: Any, *, path: Path, format_hint: str | None = None) -> Any:
    normalized_format = (format_hint or infer_format(path)).strip().lower()
    if normalized_format == "csv":
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        return (
            spark.read.option("header", True)
            .option("inferSchema", True)
            .option("multiLine", True)
            .option("escape", '"')
            .option("delimiter", delimiter)
            .csv(str(path))
        )
    if normalized_format == "json":
        return spark.read.option("multiLine", True).json(str(path))
    return spark.read.parquet(str(path))


def register_temp_view_from_path(
    spark: Any,
    *,
    table_name: str,
    path_value: str,
    description: str,
    logger: Any | None = None,
    format_hint: str | None = None,
) -> str:
    path = resolve_path(path_value)
    if not path.exists():
        raise RuntimeError(f"Missing {description} path: {path}")

    dataframe = read_dataframe(spark, path=path, format_hint=format_hint)
    dataframe.createOrReplaceTempView(table_name)
    if logger is not None:
        logger.info("Registered %s as temp view %s", path, table_name)
    return table_name


def ensure_table_available(
    spark: Any,
    *,
    table_name: str,
    description: str,
    path_flag: str,
) -> str:
    if spark.catalog.tableExists(table_name):
        return table_name
    raise RuntimeError(
        f"{description} table/view {table_name!r} was not found. "
        f"Pass {path_flag} with a local CSV/Parquet path or register the table before running the script."
    )