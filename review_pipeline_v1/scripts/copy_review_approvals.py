from __future__ import annotations

import argparse
import json

from review_pipeline_v1.review_store import (
    DEFAULT_FINAL_DATASET_TABLE,
    DEFAULT_REVIEW_DECISION_TABLE,
    resolve_review_store_spark,
    load_review_rows,
    update_final_dataset_approval,
    upsert_review_record,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy approved review decisions from one review version to another.")
    parser.add_argument("--from-version", type=int, required=True)
    parser.add_argument("--to-version", type=int, required=True)
    parser.add_argument("--review-table", type=str, default=DEFAULT_REVIEW_DECISION_TABLE)
    parser.add_argument("--final-dataset-table", type=str, default=DEFAULT_FINAL_DATASET_TABLE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.from_version < 0 or args.to_version < 0:
        raise RuntimeError("versions must be non-negative")
    if args.from_version == args.to_version:
        raise RuntimeError("from-version and to-version must differ")

    spark = resolve_review_store_spark(app_name="review-pipeline-v1-copy-review-approvals")
    source_rows = load_review_rows(
        spark,
        review_table=args.review_table,
        review_version=args.from_version,
    )

    approved_source_rows = [
        row
        for row in source_rows
        if str(row.get("claim_status") or "").strip().lower() == "completed"
        and str(row.get("review_decision") or "").strip().lower() == "approved"
    ]

    target_docket_ids = {
        str(row["docket_id"] or "")
        for row in spark.sql(
            f"SELECT docket_id FROM {args.final_dataset_table} WHERE COALESCE(review_version, 0) = {int(args.to_version)}"
        ).collect()
        if row["docket_id"] is not None
    }

    copied_count = 0
    for row in approved_source_rows:
        docket_id = str(row.get("docket_id") or "").strip()
        if not docket_id or docket_id not in target_docket_ids:
            continue

        verification_payload = {}
        raw_verification = str(row.get("verification_json") or "").strip()
        if raw_verification:
            try:
                verification_payload = json.loads(raw_verification)
            except json.JSONDecodeError:
                verification_payload = {"copied_from_raw": raw_verification}
        verification_payload["copied_from_review_version"] = int(args.from_version)
        verification_payload["review_version"] = int(args.to_version)

        upsert_review_record(
            spark,
            {
                "docket_id": docket_id,
                "reviewer_name": str(row.get("reviewer_name") or "anonymous_reviewer"),
                "claim_status": "completed",
                "claim_token": "copied_approval",
                "claimed_at_utc": str(row.get("claimed_at_utc") or ""),
                "updated_at_utc": str(row.get("updated_at_utc") or ""),
                "review_version": int(args.to_version),
                "review_decision": "approved",
                "review_notes": str(row.get("review_notes") or ""),
                "verification_json": json.dumps(verification_payload, ensure_ascii=False),
            },
            review_table=args.review_table,
        )
        update_final_dataset_approval(
            spark,
            docket_id=docket_id,
            review_decision="approved",
            reviewer_name=str(row.get("reviewer_name") or "anonymous_reviewer"),
            review_version=args.to_version,
            final_dataset_table=args.final_dataset_table,
        )
        copied_count += 1

    print(f"Copied {copied_count} approved review(s) from version {args.from_version} to version {args.to_version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())