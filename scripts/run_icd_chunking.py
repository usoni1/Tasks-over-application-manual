from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk the ICD manuals and upload them to Azure Search.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Log verbosity for the run.")
    parser.add_argument("--query-text", type=str, default=None, help="Run a retrieval query against the configured Azure Search index instead of chunking.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of search hits to return for --query-text.")
    parser.add_argument("--execution-env", choices=["local", "databricks"], default=None, help="Override EXECUTION_ENV for this run.")
    parser.add_argument("--search-service", choices=["service_1", "service_2"], default=None, help="Override the target Azure Search service for this run.")
    parser.add_argument("--index-name", type=str, default=None, help="Override the Azure Search index name for this run.")
    parser.add_argument("--limit", type=int, default=None, help="Optional chunk limit for a small test run.")
    parser.add_argument("--per-source-limit", type=int, default=None, help="Optional balanced source cap applied before the global limit.")
    parser.add_argument("--recreate-index", action="store_true", help="Delete and recreate the Azure Search index before upload.")
    parser.add_argument("--skip-upload", action="store_true", help="Build chunks and print run stats without embedding or upload.")
    parser.add_argument("--max-chunk-chars", type=int, default=None, help="Maximum characters allowed in a semantic chunk before child-branch splitting kicks in.")
    parser.add_argument("--embedding-batch-size", type=int, default=None, help="Embedding batch size for Azure OpenAI requests.")
    parser.add_argument("--upload-batch-size", type=int, default=None, help="Upload batch size for Azure Search document writes.")
    parser.add_argument("--chars-per-token-estimate", type=float, default=None, help="Approximate characters-per-token ratio used for cost estimation.")
    parser.add_argument("--embedding-cost-per-1m-tokens", type=float, default=None, help="Optional cost model for embedding estimation output.")
    parser.add_argument("--include-guidelines", action=argparse.BooleanOptionalAction, default=None, help="Include guideline chunks in this run.")
    parser.add_argument("--include-tabular", action=argparse.BooleanOptionalAction, default=None, help="Include tabular chunks in this run.")
    parser.add_argument("--include-index", action=argparse.BooleanOptionalAction, default=None, help="Include alphabetic index chunks in this run.")
    return parser.parse_args(argv)


def configure_logging(log_level: str) -> None:
    logger.remove()
    logger.add(sys.stdout, level=log_level, format="[{time:YYYY-MM-DD HH:mm:ss}] {level} {message}")


def main(argv: list[str] | None = None) -> int:
    from baselines.icd_rag import load_config
    from baselines.icd_rag.pipeline import run_balanced_chunk_and_upload, run_query, summary_to_json

    args = parse_args(argv)
    configure_logging(args.log_level)
    config = load_config(args)
    if args.query_text:
        summary = run_query(config=config, query_text=args.query_text, top_k=args.top_k)
        print(summary_to_json(summary))
        return 0

    summary = run_balanced_chunk_and_upload(
        config=config,
        limit=args.limit,
        per_source_limit=args.per_source_limit,
        recreate_index=args.recreate_index,
        skip_upload=args.skip_upload,
    )
    print(summary_to_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())