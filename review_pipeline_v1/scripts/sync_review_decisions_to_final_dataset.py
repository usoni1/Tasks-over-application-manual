from __future__ import annotations

import argparse

from review_pipeline_v1.review_store import (
    DEFAULT_FINAL_DATASET_TABLE,
    DEFAULT_REVIEW_DECISION_TABLE,
    load_review_rows,
    resolve_review_store_spark,
    update_final_dataset_approval,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay completed review decisions from the verification table into the final dataset for one review version."
    )
    parser.add_argument("--review-version", type=int, required=True)
    parser.add_argument("--review-table", type=str, default=DEFAULT_REVIEW_DECISION_TABLE)
    parser.add_argument("--final-dataset-table", type=str, default=DEFAULT_FINAL_DATASET_TABLE)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.review_version < 0:
        raise RuntimeError("review-version must be non-negative")

    spark = resolve_review_store_spark(app_name="review-pipeline-v1-sync-review-decisions")
    review_rows = load_review_rows(
        spark,
        review_table=args.review_table,
        review_version=args.review_version,
    )
    completed_rows = [
        row
        for row in review_rows
        if str(row.get("claim_status") or "").strip().lower() == "completed"
        and str(row.get("review_decision") or "").strip()
    ]

    synced_count = 0
    for row in completed_rows:
        docket_id = str(row.get("docket_id") or "").strip()
        if not docket_id:
            continue
        if not args.dry_run:
            update_final_dataset_approval(
                spark,
                docket_id=docket_id,
                review_decision=str(row.get("review_decision") or ""),
                reviewer_name=str(row.get("reviewer_name") or "anonymous_reviewer"),
                review_version=args.review_version,
                final_dataset_table=args.final_dataset_table,
            )
        synced_count += 1

    action = "Would sync" if args.dry_run else "Synced"
    print(f"{action} {synced_count} completed review decision(s) into final dataset version {args.review_version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())