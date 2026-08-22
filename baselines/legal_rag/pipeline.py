from __future__ import annotations

import json

from .azure_search import create_clients, embed_and_prepare_documents, ensure_index, search_index, upload_documents
from .chunking import ChunkRecord, build_all_chunks
from .config import LegalRAGConfig


def summarize_chunks(config: LegalRAGConfig, chunks: list[ChunkRecord]) -> dict[str, object]:
    source_counts: dict[str, int] = {}
    year_counts: dict[str, int] = {}
    total_tokens = 0
    for chunk in chunks:
        source_counts[chunk.source_type] = source_counts.get(chunk.source_type, 0) + 1
        year_key = str(chunk.source_year)
        year_counts[year_key] = year_counts.get(year_key, 0) + 1
        total_tokens += chunk.estimated_tokens

    summary: dict[str, object] = {
        "chunk_count": len(chunks),
        "chunk_counts_by_source": source_counts,
        "chunk_counts_by_year": year_counts,
        "estimated_embedding_tokens": total_tokens,
    }
    if config.embedding_cost_per_1m_tokens is not None:
        summary["estimated_embedding_cost"] = round(
            (total_tokens / 1_000_000.0) * config.embedding_cost_per_1m_tokens,
            6,
        )
    return summary


def apply_source_limit(chunks: list[ChunkRecord], per_source_limit: int | None) -> list[ChunkRecord]:
    if per_source_limit is None:
        return chunks

    counts: dict[str, int] = {}
    selected: list[ChunkRecord] = []
    for chunk in chunks:
        current = counts.get(chunk.source_type, 0)
        if current >= per_source_limit:
            continue
        selected.append(chunk)
        counts[chunk.source_type] = current + 1
    return selected


def run_balanced_chunk_and_upload(
    config: LegalRAGConfig,
    limit: int | None,
    per_source_limit: int | None,
    recreate_index: bool,
    skip_upload: bool,
    years: set[int] | None = None,
) -> dict[str, object]:
    chunks = build_all_chunks(config, years=years)
    chunks = apply_source_limit(chunks, per_source_limit)
    if limit is not None:
        chunks = chunks[:limit]
    if not chunks:
        raise RuntimeError("No legal chunks were produced.")

    summary = summarize_chunks(config, chunks)
    summary["index_name"] = config.index_name
    summary["search_service"] = config.search_service.name
    summary["manuals_root"] = str(config.manuals_root)
    summary["ussg_root"] = str(config.ussg_root)
    summary["title18_root"] = str(config.title18_root)
    if years is not None:
        summary["years"] = sorted(years)
    if per_source_limit is not None:
        summary["per_source_limit"] = per_source_limit

    if skip_upload:
        return summary

    index_client, search_client = create_clients(config)
    documents, vector_dimensions = embed_and_prepare_documents(config, chunks)
    ensure_index(index_client, config.index_name, vector_dimensions, recreate=recreate_index)
    upload_documents(search_client, config, documents)
    summary["uploaded_documents"] = len(documents)
    summary["vector_dimensions"] = vector_dimensions
    return summary


def run_query(config: LegalRAGConfig, query_text: str, top_k: int) -> dict[str, object]:
    result = {
        "index_name": config.index_name,
        "search_service": config.search_service.name,
        "query": query_text,
        "top_k": top_k,
        "results": search_index(config, query_text, top_k),
    }
    return result


def summary_to_json(summary: dict[str, object]) -> str:
    return json.dumps(summary, indent=2, sort_keys=True)