from __future__ import annotations

from typing import Any

from baselines.icd_rag.azure_search import VECTOR_FIELD_NAME, build_embeddings_client, create_clients

from .config import ICDAgenticRAGConfig, load_config


VALID_SOURCE_TYPES = {"guidelines", "index", "tabular"}
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


def _build_search_filter(source_type: str | None) -> str | None:
    if not source_type:
        return None
    return f"source_type eq {_quote_filter_value(source_type)}"


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


def _execute_icd_search(
    *,
    search_client: Any,
    embeddings: Any,
    query_text: str,
    top_k: int,
    source_type: str | None,
) -> list[dict[str, Any]]:
    query_vector = embeddings.embed_query(query_text)
    search_filter = _build_search_filter(source_type)
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
                "document_title",
                "chunk_title",
                "semantic_path",
                "code",
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
                "document_title": item.get("document_title"),
                "chunk_title": item.get("chunk_title"),
                "semantic_path": item.get("semantic_path"),
                "code": item.get("code"),
                "source_path": item.get("source_path"),
                "score": item.get("@search.score"),
                "text": item.get("text") or "",
            }
        )
    return results


def build_icd_search_tools(config: ICDAgenticRAGConfig, default_top_k: int = 8) -> list[Any]:
    _, search_client = create_clients(config)
    embeddings = build_embeddings_client(config)
    normalized_default_top_k = min(max(int(default_top_k), 1), MAX_TOOL_TOP_K)
    tool_state = {"call_count": 0}

    def search_icd_manuals_tool(
        query_text: str,
        top_k: int | None = None,
        source_type: str | None = None,
    ) -> dict[str, Any]:
        """Search the indexed ICD manuals with semantic retrieval.

        Input:
            query_text: Focused search request describing the diagnosis, rule,
                lead term, code family, or verification question you need help with.
            top_k: Optional number of chunks to return. Use a small number for
                focused lookups. The tool caps this at 8.
            source_type: Optional manual surface filter. Allowed values are
                `guidelines`, `index`, or `tabular`.

        Returns:
            A dictionary with the normalized query and a `results` list. Each
            result includes `source_type`, `document_title`, `chunk_title`,
            `semantic_path`, optional `code`, retrieval `score`, and full `text`.

        Notes:
            - Use repeated focused searches instead of one broad search.
            - If you need a broad first-pass retrieval for the whole case, use
              `search_icd_manuals_full_case_tool` instead.
            - Use `index` for lead-term discovery, `tabular` for code
              verification, and `guidelines` only when a coding rule may change
              inclusion, specificity, sequencing, or exclusions.
        """
        try:
            normalized_query = _normalize_query_text(query_text, "query_text")
            effective_top_k = _normalize_top_k(normalized_default_top_k, top_k)
            normalized_source_type = _normalize_source_type(source_type)

            tool_state["call_count"] += 1
            results = _execute_icd_search(
                search_client=search_client,
                embeddings=embeddings,
                query_text=normalized_query,
                top_k=effective_top_k,
                source_type=normalized_source_type,
            )

            return {
                "query_text": normalized_query,
                "query_mode": "focused",
                "top_k": effective_top_k,
                "source_type": normalized_source_type,
                "status": "ok",
                "call_count": tool_state["call_count"],
                "count": len(results),
                "results": results,
            }
        except Exception as error:
            return tool_error_response(
                "search_icd_manuals_tool",
                error,
                query_text=query_text,
                top_k=top_k,
                source_type=source_type,
            )

    def search_icd_manuals_full_case_tool(
        case_summary: str,
        top_k: int | None = None,
        source_type: str | None = None,
    ) -> dict[str, Any]:
        """Search the indexed ICD manuals using the entire clinical case summary.

        Input:
            case_summary: The full clinical case summary. The tool embeds the
                whole summary directly for a broad first-pass retrieval.
            top_k: Optional number of chunks to return. The tool caps this at 8.
            source_type: Optional manual surface filter. Allowed values are
                `guidelines`, `index`, or `tabular`.

        Returns:
            A dictionary with the normalized case summary and a `results` list.
            Each result includes `source_type`, `document_title`, `chunk_title`,
            `semantic_path`, optional `code`, retrieval `score`, and full `text`.

        Notes:
            - Use this when you want a broad first-pass retrieval over the whole
              case before switching to focused follow-up searches.
            - After this tool, prefer targeted follow-up searches with
              `search_icd_manuals_tool`.
        """
        try:
            normalized_case_summary = _normalize_query_text(case_summary, "case_summary")
            effective_top_k = _normalize_top_k(normalized_default_top_k, top_k)
            normalized_source_type = _normalize_source_type(source_type)

            tool_state["call_count"] += 1
            results = _execute_icd_search(
                search_client=search_client,
                embeddings=embeddings,
                query_text=normalized_case_summary,
                top_k=effective_top_k,
                source_type=normalized_source_type,
            )

            return {
                "case_summary": normalized_case_summary,
                "query_text": normalized_case_summary,
                "query_mode": "full_case",
                "top_k": effective_top_k,
                "source_type": normalized_source_type,
                "status": "ok",
                "call_count": tool_state["call_count"],
                "count": len(results),
                "results": results,
            }
        except Exception as error:
            return tool_error_response(
                "search_icd_manuals_full_case_tool",
                error,
                case_summary=case_summary,
                top_k=top_k,
                source_type=source_type,
            )

    return [search_icd_manuals_tool, search_icd_manuals_full_case_tool]


def search_icd_manuals(
    query_text: str,
    top_k: int = 8,
    source_type: str | None = None,
    config: ICDAgenticRAGConfig | None = None,
) -> dict[str, Any]:
    resolved_config = config or load_config()
    tool = build_icd_search_tools(resolved_config, default_top_k=top_k)[0]
    return tool(query_text=query_text, top_k=top_k, source_type=source_type)


def search_icd_manuals_full_case(
    case_summary: str,
    top_k: int = 8,
    source_type: str | None = None,
    config: ICDAgenticRAGConfig | None = None,
) -> dict[str, Any]:
    resolved_config = config or load_config()
    tool = build_icd_search_tools(resolved_config, default_top_k=top_k)[1]
    return tool(case_summary=case_summary, top_k=top_k, source_type=source_type)


__all__ = [
    "VALID_SOURCE_TYPES",
    "build_icd_search_tools",
    "search_icd_manuals",
    "search_icd_manuals_full_case",
    "tool_error_response",
]