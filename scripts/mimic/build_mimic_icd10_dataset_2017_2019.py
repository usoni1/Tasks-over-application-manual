from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


DEFAULT_MIMIC_ROOT_PLACEHOLDER = Path("PUT_PATH_TO_MIMIC_IV_ROOT_HERE")
DEFAULT_ADMISSIONS_PATH = DEFAULT_MIMIC_ROOT_PLACEHOLDER / "hosp" / "admissions.csv.gz"
DEFAULT_DIAGNOSES_PATH = DEFAULT_MIMIC_ROOT_PLACEHOLDER / "hosp" / "diagnoses_icd.csv.gz"
DEFAULT_PATIENTS_PATH = DEFAULT_MIMIC_ROOT_PLACEHOLDER / "hosp" / "patients.csv.gz"
DEFAULT_NOTES_PATH = DEFAULT_MIMIC_ROOT_PLACEHOLDER / "note" / "discharge.csv.gz"
DEFAULT_OUTPUT_PATH = Path("data/mimic/mimic_icd10_note_dataset_2017_2019_strict.parquet")
DEFAULT_SUMMARY_PATH = Path("data/mimic/mimic_icd10_note_dataset_2017_2019_strict_summary.json")
DEFAULT_ADDENDA_DIR = Path("data/reference-manuals/ICD-addendums")
ADDENDA_FILENAMES = [
    "icd10cm_codes_addenda_2018.txt",
    "icd10cm_codes_addenda_2019.txt",
    "icd10cm_codes_addenda_2020.txt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the strict MIMIC ICD-10 note dataset for the 2017-2019 discharge-year window "
            "from local MIMIC-IV CSV files."
        ),
        epilog=(
            "Replace PUT_PATH_TO_MIMIC_IV_ROOT_HERE in the default input paths with the directory "
            "that contains your local MIMIC-IV hosp/ and note/ files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--admissions-path",
        type=Path,
        default=DEFAULT_ADMISSIONS_PATH,
        help="Path to the local MIMIC-IV hosp/admissions.csv.gz file.",
    )
    parser.add_argument(
        "--diagnoses-path",
        type=Path,
        default=DEFAULT_DIAGNOSES_PATH,
        help="Path to the local MIMIC-IV hosp/diagnoses_icd.csv.gz file.",
    )
    parser.add_argument(
        "--patients-path",
        type=Path,
        default=DEFAULT_PATIENTS_PATH,
        help="Path to the local MIMIC-IV hosp/patients.csv.gz file.",
    )
    parser.add_argument(
        "--notes-path",
        type=Path,
        default=DEFAULT_NOTES_PATH,
        help="Path to the local MIMIC-IV note/discharge.csv.gz file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path where the output parquet dataset directory will be written.",
    )
    parser.add_argument("--addenda-dir", type=Path, default=DEFAULT_ADDENDA_DIR)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument(
        "--write-mode",
        choices=["overwrite", "errorifexists", "ignore", "append"],
        default="overwrite",
        help="Spark write mode for the output dataset.",
    )
    parser.add_argument(
        "--skip-write",
        action="store_true",
        help="Run the full build and write the local summary without saving the output dataset.",
    )
    return parser.parse_args()


def resolve_spark():
    active_session = SparkSession.getActiveSession()
    if active_session is not None:
        return active_session, False

    spark = (
        SparkSession.builder.appName("build_mimic_icd10_dataset_2017_2019")
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    return spark, True


def require_source_file(file_path: Path, flag_name: str) -> Path:
    candidate = file_path.expanduser()
    if candidate.is_file():
        return candidate.resolve()

    raise SystemExit(
        f"Set --{flag_name} to your local MIMIC file. Replace the placeholder path "
        f"{file_path.as_posix()} with the actual file location."
    )


def read_mimic_csv(spark: SparkSession, file_path: Path, *, multiline: bool = False):
    reader = spark.read.option("header", True).option("inferSchema", True)
    if multiline:
        reader = reader.option("multiLine", True).option("quote", '"').option("escape", '"')
    return reader.csv(str(file_path))


def normalize_code(code: str) -> str:
    return str(code or "").strip().replace(".", "").upper()


def extract_code_from_payload(payload: str) -> str:
    normalized_payload = str(payload or "").strip()
    if not normalized_payload:
        raise ValueError(f"Could not parse addenda payload: {payload!r}")
    return normalize_code(normalized_payload.split(None, 1)[0])


def load_ignore_codes(addenda_dir: Path) -> list[str]:
    touched_codes: set[str] = set()
    for filename in ADDENDA_FILENAMES:
        file_path = addenda_dir / filename
        if not file_path.is_file():
            raise SystemExit(f"Missing addenda file: {file_path}")
        pending_revise_from: str | None = None
        with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("Add:"):
                    touched_codes.add(extract_code_from_payload(line[len("Add:") :]))
                    continue
                if line.startswith("Delete:"):
                    touched_codes.add(extract_code_from_payload(line[len("Delete:") :]))
                    continue
                if line.startswith("Revise from:"):
                    pending_revise_from = extract_code_from_payload(line[len("Revise from:") :])
                    touched_codes.add(pending_revise_from)
                    continue
                if line.startswith("Revise to:"):
                    revise_to = extract_code_from_payload(line[len("Revise to:") :])
                    touched_codes.add(revise_to)
                    if pending_revise_from is not None and revise_to != pending_revise_from:
                        raise ValueError(
                            f"Mismatched revise pair in {file_path}: {pending_revise_from} vs {revise_to}"
                        )
                    pending_revise_from = None
        if pending_revise_from is not None:
            raise ValueError(f"Unclosed revise block in {file_path} for code {pending_revise_from}")
    return sorted(touched_codes)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_dataset(args: argparse.Namespace) -> dict[str, object]:
    spark, owns_session = resolve_spark()

    try:
        admissions_path = require_source_file(args.admissions_path, "admissions-path")
        diagnoses_path = require_source_file(args.diagnoses_path, "diagnoses-path")
        patients_path = require_source_file(args.patients_path, "patients-path")
        notes_path = require_source_file(args.notes_path, "notes-path")
        output_path = args.output_path.expanduser()

        ignore_codes = load_ignore_codes(args.addenda_dir.resolve())
        ignore_codes_df = spark.createDataFrame(
            [{"icd_code_norm": code} for code in ignore_codes]
        ).dropDuplicates(["icd_code_norm"])

        diagnoses_icd_df = (
            read_mimic_csv(spark, diagnoses_path)
            .select(
                F.col("subject_id").cast("long").alias("subject_id"),
                F.col("hadm_id").cast("long").alias("hadm_id"),
                F.col("seq_num").cast("int").alias("seq_num"),
                F.col("icd_code").cast("string").alias("icd_code"),
                F.col("icd_version").cast("int").alias("icd_version"),
            )
            .filter(F.col("icd_version") == 10)
            .filter(F.col("hadm_id").isNotNull())
            .withColumn("icd_code_norm", F.upper(F.regexp_replace(F.col("icd_code"), r"\.", "")))
        )
        hadm_ids_with_ignored_codes_df = (
            diagnoses_icd_df.join(ignore_codes_df, on="icd_code_norm", how="inner")
            .select("hadm_id")
            .dropna()
            .dropDuplicates()
        )
        diagnoses_icd_kept_df = diagnoses_icd_df.join(
            hadm_ids_with_ignored_codes_df, on="hadm_id", how="left_anti"
        )

        patients_anchor_df = read_mimic_csv(spark, patients_path).select(
            F.col("subject_id").cast("long").alias("subject_id"),
            F.col("anchor_year").cast("int").alias("anchor_year"),
            F.col("anchor_year_group").cast("string").alias("anchor_year_group"),
        )
        admissions_with_anchor_df = (
            read_mimic_csv(spark, admissions_path)
            .select(
                F.col("subject_id").cast("long").alias("subject_id"),
                F.col("hadm_id").cast("long").alias("hadm_id"),
                F.col("dischtime").cast("string").alias("dischtime"),
            )
            .withColumn("dischtime", F.to_timestamp("dischtime"))
            .filter(F.col("dischtime").isNotNull())
            .join(patients_anchor_df, on="subject_id", how="inner")
            .withColumn("shifted_discharge_year", F.year("dischtime"))
            .withColumn(
                "anchor_year_group_start",
                F.expr("try_cast(regexp_extract(anchor_year_group, '(\\\\d{4})', 1) as int)"),
            )
            .withColumn(
                "anchor_year_group_end",
                F.expr("try_cast(regexp_extract(anchor_year_group, '.*(\\\\d{4})', 1) as int)"),
            )
            .withColumn("year_offset_from_anchor", F.col("shifted_discharge_year") - F.col("anchor_year"))
            .withColumn("real_discharge_year_min", F.col("anchor_year_group_start") + F.col("year_offset_from_anchor"))
            .withColumn("real_discharge_year_max", F.col("anchor_year_group_end") + F.col("year_offset_from_anchor"))
        )
        strict_discharge_2017_2019_df = (
            admissions_with_anchor_df.filter(F.col("real_discharge_year_min") >= 2017)
            .filter(F.col("real_discharge_year_max") <= 2019)
            .select("subject_id", "hadm_id", "dischtime", "real_discharge_year_min", "real_discharge_year_max")
            .dropDuplicates(["hadm_id"])
        )

        kept_hadm_ids_df = diagnoses_icd_kept_df.select("hadm_id").dropna().dropDuplicates()
        final_time_and_code_hadm_ids_df = (
            strict_discharge_2017_2019_df.join(kept_hadm_ids_df, on="hadm_id", how="inner")
            .dropDuplicates(["hadm_id"])
        )

        discharge_notes_df = (
            read_mimic_csv(spark, notes_path, multiline=True)
            .select(
                F.col("note_id").cast("string").alias("note_id"),
                F.col("subject_id").cast("long").alias("subject_id"),
                F.col("hadm_id").cast("long").alias("hadm_id"),
                F.col("note_seq").cast("int").alias("note_seq"),
                F.col("charttime").cast("string").alias("charttime"),
                F.col("storetime").cast("string").alias("storetime"),
                F.col("text").cast("string").alias("text"),
            )
            .filter(F.col("hadm_id").isNotNull())
        )
        notes_per_hadm_df = discharge_notes_df.groupBy("hadm_id").agg(F.count("*").alias("n_notes"))
        cohort_note_coverage_df = (
            final_time_and_code_hadm_ids_df.join(notes_per_hadm_df, on="hadm_id", how="left")
            .withColumn("has_discharge_note", F.col("n_notes").isNotNull())
        )
        note_coverage_row = cohort_note_coverage_df.agg(
            F.count("*").alias("cohort_admissions"),
            F.sum(F.when(F.col("has_discharge_note"), 1).otherwise(0)).alias("with_discharge_note"),
            F.sum(F.when(~F.col("has_discharge_note"), 1).otherwise(0)).alias("without_discharge_note"),
        ).collect()[0]

        cohort_with_notes_df = (
            final_time_and_code_hadm_ids_df.join(
                discharge_notes_df.select("hadm_id", "subject_id", "note_id", "text"),
                on="hadm_id",
                how="inner",
            )
            .select(
                final_time_and_code_hadm_ids_df["hadm_id"],
                final_time_and_code_hadm_ids_df["subject_id"],
                "note_id",
                "text",
                "dischtime",
                "real_discharge_year_min",
                "real_discharge_year_max",
            )
            .dropDuplicates(["hadm_id"])
        )

        icd_codes_per_hadm_df = (
            diagnoses_icd_kept_df.select("hadm_id", "seq_num", "icd_code_norm")
            .groupBy("hadm_id")
            .agg(
                F.sort_array(
                    F.collect_list(
                        F.struct(F.col("seq_num").alias("seq_num"), F.col("icd_code_norm").alias("icd_code"))
                    )
                ).alias("icd_code_structs")
            )
            .withColumn("output_icd_codes", F.expr("transform(icd_code_structs, x -> x.icd_code)"))
            .drop("icd_code_structs")
        )

        final_df = (
            cohort_with_notes_df.join(icd_codes_per_hadm_df, on="hadm_id", how="inner")
            .select(
                "hadm_id",
                "subject_id",
                "note_id",
                F.col("text").alias("input_text"),
                "output_icd_codes",
                "dischtime",
                "real_discharge_year_min",
                "real_discharge_year_max",
            )
        )

        final_row_count = final_df.count()
        unique_hadm_id_count = final_df.select("hadm_id").dropDuplicates().count()

        if not args.skip_write:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            (
                final_df.write.mode(args.write_mode)
                .format("parquet")
                .save(str(output_path))
            )

        summary = {
            "generated_at_utc": utc_now_iso(),
            "output_path": str(output_path.resolve()),
            "output_format": "parquet",
            "write_mode": args.write_mode,
            "dataset_written": not args.skip_write,
            "source_files": {
                "admissions": str(admissions_path),
                "diagnoses_icd": str(diagnoses_path),
                "patients": str(patients_path),
                "discharge_notes": str(notes_path),
            },
            "addenda_dir": str(args.addenda_dir.resolve()),
            "ignore_code_count": len(ignore_codes),
            "cohort_counts": {
                "strict_discharge_2017_2019": int(strict_discharge_2017_2019_df.count()),
                "admissions_after_ignore_filter": int(kept_hadm_ids_df.count()),
                "admissions_after_strict_time_and_ignore_filter": int(final_time_and_code_hadm_ids_df.count()),
                "cohort_with_discharge_note": int(note_coverage_row["with_discharge_note"] or 0),
                "cohort_without_discharge_note": int(note_coverage_row["without_discharge_note"] or 0),
                "final_rows": int(final_row_count),
                "unique_hadm_id_count": int(unique_hadm_id_count),
            },
            "final_columns": final_df.columns,
            "filtering_rule_sentence": (
                "Start from all MIMIC ICD-10 admissions, exclude any admission if any assigned diagnosis code appears in the 2018-2020 ICD-10-CM addenda, keep only admissions whose deidentified discharge-year range is fully within 2017-2019, and retain one discharge-note-backed row per surviving admission."
            ),
        }
        args.summary_path.parent.mkdir(parents=True, exist_ok=True)
        args.summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary
    finally:
        if owns_session:
            spark.stop()


def main() -> None:
    args = parse_args()
    summary = build_dataset(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()