from __future__ import annotations

from typing import Any

from baselines.legal_rag.azure_search import VECTOR_FIELD_NAME, build_embeddings_client, create_clients

from .config import LegalAgenticRAGConfig, load_config


VALID_SOURCE_TYPES = {"ussg", "usc_title18"}
MAX_TOOL_TOP_K = 8


def tool_error_response(tool_name: str, error: Exception, **inputs: Any) -> dict[str, Any]:
    return {
        "tool_name": tool_name,
        "status": "error",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "inputs": inputs,
    }


def _quote_filter_value(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _build_search_filter(source_year: int | None, source_type: str | None) -> str | None:
    clauses: list[str] = []
    if source_year is not None:
        clauses.append(f"source_year eq {int(source_year)}")
    if source_type:
        clauses.append(f"source_type eq {_quote_filter_value(source_type)}")
    if not clauses:
        return None
    return " and ".join(clauses)


def _normalize_query_text(value: Any, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


def _normalize_top_k(default_top_k: int, top_k: int | None) -> int:
    return min(max(int(default_top_k if top_k is None else top_k), 1), MAX_TOOL_TOP_K)


def _normalize_source_type(source_type: str | None) -> str | None:
    normalized_source_type = None if source_type is None else str(source_type).strip().lower()
    if normalized_source_type and normalized_source_type not in VALID_SOURCE_TYPES:
        raise ValueError(f"source_type must be one of {sorted(VALID_SOURCE_TYPES)}, got {normalized_source_type!r}")
    return normalized_source_type


def _execute_legal_search(
    *,
    search_client: Any,
    embeddings: Any,
    query_text: str,
    top_k: int,
    source_year: int | None,
    source_type: str | None,
) -> list[dict[str, Any]]:
    query_vector = embeddings.embed_query(query_text)
    search_filter = _build_search_filter(source_year=source_year, source_type=source_type)
    raw_results = list(
        search_client.search(
            search_text=query_text,
            vector_queries=[
                {
                    "kind": "vector",
                    "vector": query_vector,
                    "fields": VECTOR_FIELD_NAME,
                    "k": top_k,
                }
            ],
            top=top_k,
            filter=search_filter,
            select=[
                "chunk_id",
                "source_type",
                "source_year",
                "document_title",
                "chunk_title",
                "semantic_path",
                "citation",
                "source_path",
                "text",
            ],
        )
    )

    results: list[dict[str, Any]] = []
    for item in raw_results:
        results.append(
            {
                "chunk_id": item.get("chunk_id"),
                "source_type": item.get("source_type"),
                "source_year": item.get("source_year"),
                "document_title": item.get("document_title"),
                "chunk_title": item.get("chunk_title"),
                "semantic_path": item.get("semantic_path"),
                "citation": item.get("citation"),
                "source_path": item.get("source_path"),
                "score": item.get("@search.score"),
                "text": item.get("text") or "",
            }
        )
    return results


def build_legal_search_tools(
    config: LegalAgenticRAGConfig,
    default_top_k: int = 8,
    source_year: int | None = None,
) -> list[Any]:
    _, search_client = create_clients(config)
    embeddings = build_embeddings_client(config)
    normalized_default_top_k = min(max(int(default_top_k), 1), MAX_TOOL_TOP_K)
    tool_state = {"call_count": 0}

    def search_legal_manuals_tool(
        query_text: str,
        top_k: int | None = None,
        source_type: str | None = None,
    ) -> dict[str, Any]:
        """Search the indexed legal manuals semantically.

        Input:
            query_text: Focused retrieval request describing the sentencing issue,
                guideline concept, enhancement, reduction, statute, or fact pattern.
            top_k: Optional number of chunks to return. Use a small number for
                focused lookups. The tool caps this at 8.
            source_type: Optional source filter. Allowed values are `ussg` or
                `usc_title18`.

        Returns:
            A dictionary with the normalized query and a `results` list. Each
            result includes `source_type`, `source_year`, `document_title`,
            `chunk_title`, optional `citation`, retrieval `score`, and full `text`.

        Notes:
            - The tool is already scoped to the case year when a year is
              available, so repeated searches stay on the same manual edition.
            - Use repeated focused searches instead of one broad search.
            - If you need a broad first-pass retrieval for the whole case, use
              `search_legal_manuals_full_case_tool` instead.
            - Use `ussg` for guideline calculations and commentary, and
              `usc_title18` when you need statutory text.
        """
        try:
            normalized_query = _normalize_query_text(query_text, "query_text")
            effective_top_k = _normalize_top_k(normalized_default_top_k, top_k)
            normalized_source_type = _normalize_source_type(source_type)

            tool_state["call_count"] += 1
            results = _execute_legal_search(
                search_client=search_client,
                embeddings=embeddings,
                query_text=normalized_query,
                top_k=effective_top_k,
                source_year=source_year,
                source_type=normalized_source_type,
            )

            return {
                "query_text": normalized_query,
                "query_mode": "focused",
                "top_k": effective_top_k,
                "source_type": normalized_source_type,
                "source_year": source_year,
                "status": "ok",
                "call_count": tool_state["call_count"],
                "count": len(results),
                "results": results,
            }
        except Exception as error:
            return tool_error_response(
                "search_legal_manuals_tool",
                error,
                query_text=query_text,
                top_k=top_k,
                source_type=source_type,
                source_year=source_year,
            )

    def search_legal_manuals_full_case_tool(
        case_summary: str,
        top_k: int | None = None,
        source_type: str | None = None,
    ) -> dict[str, Any]:
        """Search the indexed legal manuals using the entire case summary.

        Input:
            case_summary: The full case summary text. The tool embeds the whole
                summary directly for a broad first-pass retrieval.
            top_k: Optional number of chunks to return. The tool caps this at 8.
            source_type: Optional source filter. Allowed values are `ussg` or
                `usc_title18`.

        Returns:
            A dictionary with the normalized case summary and a `results` list.
            Each result includes `source_type`, `source_year`,
            `document_title`, `chunk_title`, optional `citation`, retrieval
            `score`, and full `text`.

        Notes:
            - Use this as a broad first-pass retrieval when the focused issue is
              still unclear from the case facts.
            - After this tool, prefer targeted follow-up searches with
              `search_legal_manuals_tool`.
        """
        try:
            normalized_case_summary = _normalize_query_text(case_summary, "case_summary")
            effective_top_k = _normalize_top_k(normalized_default_top_k, top_k)
            normalized_source_type = _normalize_source_type(source_type)

            tool_state["call_count"] += 1
            results = _execute_legal_search(
                search_client=search_client,
                embeddings=embeddings,
                query_text=normalized_case_summary,
                top_k=effective_top_k,
                source_year=source_year,
                source_type=normalized_source_type,
            )

            return {
                "case_summary": normalized_case_summary,
                "query_text": normalized_case_summary,
                "query_mode": "full_case",
                "top_k": effective_top_k,
                "source_type": normalized_source_type,
                "source_year": source_year,
                "status": "ok",
                "call_count": tool_state["call_count"],
                "count": len(results),
                "results": results,
            }
        except Exception as error:
            return tool_error_response(
                "search_legal_manuals_full_case_tool",
                error,
                case_summary=case_summary,
                top_k=top_k,
                source_type=source_type,
                source_year=source_year,
            )

    return [search_legal_manuals_tool, search_legal_manuals_full_case_tool]


def search_legal_manuals(
    query_text: str,
    top_k: int = 8,
    source_type: str | None = None,
    source_year: int | None = None,
    config: LegalAgenticRAGConfig | None = None,
) -> dict[str, Any]:
    resolved_config = config or load_config()
    tool = build_legal_search_tools(resolved_config, default_top_k=top_k, source_year=source_year)[0]
    return tool(query_text=query_text, top_k=top_k, source_type=source_type)


def search_legal_manuals_full_case(
    case_summary: str,
    top_k: int = 8,
    source_type: str | None = None,
    source_year: int | None = None,
    config: LegalAgenticRAGConfig | None = None,
) -> dict[str, Any]:
    resolved_config = config or load_config()
    tool = build_legal_search_tools(resolved_config, default_top_k=top_k, source_year=source_year)[1]
    return tool(case_summary=case_summary, top_k=top_k, source_type=source_type)


__all__ = [
    "VALID_SOURCE_TYPES",
    "build_legal_search_tools",
    "search_legal_manuals",
    "search_legal_manuals_full_case",
    "tool_error_response",
]