from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from review_pipeline_v1.review_store import (
    DEFAULT_FINAL_DATASET_TABLE,
    DEFAULT_REVIEW_DECISION_TABLE,
    resolve_review_store_spark,
)


DEFAULT_TARGET_TABLE = (
    "usdo_aa_catalog.research_tam_datasets.federal_sentencing_legal_final_dataset_approved"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a clean approved-only legal final dataset table with exactly docket_id, input, and output columns."
        )
    )
    parser.add_argument("--review-table", type=str, default=DEFAULT_REVIEW_DECISION_TABLE)
    parser.add_argument("--source-table", type=str, default=DEFAULT_FINAL_DATASET_TABLE)
    parser.add_argument("--target-table", type=str, default=DEFAULT_TARGET_TABLE)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_review_approval_filter(available_columns: set[str]):
    from pyspark.sql import functions as F

    required_columns = {"docket_id", "claim_status", "review_decision"}
    missing_columns = sorted(required_columns - available_columns)
    if missing_columns:
        raise RuntimeError(f"Review table is missing required column(s): {missing_columns}")

    return (
        (F.length(F.trim(F.coalesce(F.col("docket_id").cast("string"), F.lit("")))) > 0)
        & (F.lower(F.trim(F.coalesce(F.col("claim_status").cast("string"), F.lit("")))) == F.lit("completed"))
        & (F.lower(F.trim(F.coalesce(F.col("review_decision").cast("string"), F.lit("")))) == F.lit("approved"))
    )


def main() -> int:
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    args = parse_args()
    spark = resolve_review_store_spark(app_name="review-pipeline-v1-approved-final-legal-table")

    source_df = spark.table(args.source_table)
    review_df = spark.table(args.review_table)
    available_columns = set(source_df.columns)
    required_columns = {"docket_id", "input_case_facts_text", "ground_truth_offense_level"}
    missing_columns = sorted(required_columns - available_columns)
    if missing_columns:
        raise RuntimeError(f"Source table {args.source_table} is missing required column(s): {missing_columns}")

    approved_review_df = (
        review_df.filter(build_review_approval_filter(set(review_df.columns)))
        .select(F.col("docket_id").cast("string").alias("docket_id"))
        .distinct()
        .cache()
    )
    approved_review_docket_count = approved_review_df.count()

    usable_final_df = (
        source_df.join(approved_review_df, on="docket_id", how="inner")
        .filter(F.col("docket_id").isNotNull())
        .filter(F.col("input_case_facts_text").isNotNull())
        .filter(F.length(F.trim(F.col("input_case_facts_text"))) > 0)
        .filter(F.col("ground_truth_offense_level").isNotNull())
        .select(
            F.col("docket_id").cast("string").alias("docket_id"),
            F.col("input_case_facts_text").cast("string").alias("input"),
            F.col("ground_truth_offense_level").cast("int").alias("output"),
            F.col("review_version").cast("int").alias("review_version") if "review_version" in available_columns else F.lit(None).cast("int").alias("review_version"),
            F.col("reviewed_at_utc").cast("string").alias("reviewed_at_utc") if "reviewed_at_utc" in available_columns else F.lit(None).cast("string").alias("reviewed_at_utc"),
            F.col("updated_at_utc").cast("string").alias("updated_at_utc") if "updated_at_utc" in available_columns else F.lit(None).cast("string").alias("updated_at_utc"),
            F.col("generated_at_utc").cast("string").alias("generated_at_utc") if "generated_at_utc" in available_columns else F.lit(None).cast("string").alias("generated_at_utc"),
            F.col("source_run_id").cast("string").alias("source_run_id") if "source_run_id" in available_columns else F.lit(None).cast("string").alias("source_run_id"),
        )
        .cache()
    )

    usable_final_row_count = usable_final_df.count()
    usable_final_distinct_docket_count = usable_final_df.select("docket_id").distinct().count()

    missing_usable_final_rows = approved_review_df.join(
        usable_final_df.select("docket_id").distinct(),
        on="docket_id",
        how="left_anti",
    ).orderBy("docket_id")
    missing_usable_final_dockets = [row["docket_id"] for row in missing_usable_final_rows.collect()]

    ranking_window = Window.partitionBy("docket_id").orderBy(
        F.col("review_version").desc_nulls_last(),
        F.col("reviewed_at_utc").desc_nulls_last(),
        F.col("updated_at_utc").desc_nulls_last(),
        F.col("generated_at_utc").desc_nulls_last(),
        F.col("source_run_id").desc_nulls_last(),
    )
    final_df = (
        usable_final_df.withColumn("_row_number", F.row_number().over(ranking_window))
        .filter(F.col("_row_number") == 1)
        .select("docket_id", "input", "output")
        .cache()
    )
    final_row_count = final_df.count()

    if not args.dry_run:
        final_df.write.mode("overwrite").format("delta").saveAsTable(args.target_table)

    summary = {
        "approved_review_docket_count": approved_review_docket_count,
        "approved_review_missing_usable_final_row_count": len(missing_usable_final_dockets),
        "approved_review_missing_usable_final_rows": missing_usable_final_dockets,
        "approved_review_with_usable_final_row_count": usable_final_distinct_docket_count,
        "source_table": args.source_table,
        "review_table": args.review_table,
        "target_table": args.target_table,
        "dry_run": bool(args.dry_run),
        "usable_final_row_count": usable_final_row_count,
        "usable_final_distinct_docket_count": usable_final_distinct_docket_count,
        "duplicate_usable_final_row_count": usable_final_row_count - usable_final_distinct_docket_count,
        "final_row_count": final_row_count,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())