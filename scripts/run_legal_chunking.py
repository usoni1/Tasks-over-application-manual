from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk the legal manuals and upload them to Azure Search.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Log verbosity for the run.")
    parser.add_argument("--query-text", type=str, default=None, help="Run a retrieval query against the configured Azure Search index instead of chunking.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of search hits to return for --query-text.")
    parser.add_argument("--years", type=int, nargs="+", default=None, help="Optional source years to process for a smaller test run, for example --years 2024.")
    parser.add_argument("--execution-env", choices=["local", "databricks"], default=None, help="Override EXECUTION_ENV for this run.")
    parser.add_argument("--search-service", choices=["service_1", "service_2"], default=None, help="Override the target Azure Search service for this run.")
    parser.add_argument("--index-name", type=str, default=None, help="Override the Azure Search index name for this run.")
    parser.add_argument("--limit", type=int, default=None, help="Optional chunk limit for a small test run.")
    parser.add_argument("--per-source-limit", type=int, default=None, help="Optional balanced source cap applied before the global limit.")
    parser.add_argument("--recreate-index", action="store_true", help="Delete and recreate the Azure Search index before upload.")
    parser.add_argument("--skip-upload", action="store_true", help="Build chunks and print run stats without embedding or upload.")
    parser.add_argument("--max-chunk-chars", type=int, default=None, help="Maximum characters allowed in a semantic chunk before splitting.")
    parser.add_argument("--embedding-batch-size", type=int, default=None, help="Embedding batch size for Azure OpenAI requests.")
    parser.add_argument("--upload-batch-size", type=int, default=None, help="Upload batch size for Azure Search document writes.")
    parser.add_argument("--chars-per-token-estimate", type=float, default=None, help="Approximate characters-per-token ratio used for cost estimation.")
    parser.add_argument("--embedding-cost-per-1m-tokens", type=float, default=None, help="Optional cost model for embedding estimation output.")
    parser.add_argument("--docintel-endpoint", type=str, default=None, help="Optional override for the Azure Document Intelligence endpoint.")
    parser.add_argument("--docintel-key", type=str, default=None, help="Optional override for the Azure Document Intelligence key.")
    parser.add_argument("--docintel-model", type=str, default=None, help="Optional override for the Azure Document Intelligence model.")
    parser.add_argument("--ussg-docintel-text-root", type=str, default=None, help="Optional root directory for pre-extracted USSG Doc Intelligence JSON files.")
    parser.add_argument("--include-ussg", action=argparse.BooleanOptionalAction, default=None, help="Include USSG manual chunks in this run.")
    parser.add_argument("--include-title18", action=argparse.BooleanOptionalAction, default=None, help="Include Title 18 chunks in this run.")
    parser.add_argument("--use-docintel-for-ussg", action=argparse.BooleanOptionalAction, default=None, help="Use Azure Document Intelligence for USSG extraction when configured.")
    return parser.parse_args(argv)


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        stream=sys.stdout,
        level=getattr(logging, log_level),
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    from baselines.legal_rag import load_config
    from baselines.legal_rag.pipeline import run_balanced_chunk_and_upload, run_query, summary_to_json

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
        years=set(args.years) if args.years else None,
    )
    print(summary_to_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())