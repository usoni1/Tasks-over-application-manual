from __future__ import annotations

import re
from typing import Any

from baselines.legal_react_v2.tools import open_ussg_section


DEFAULT_GUIDELINE_YEAR = 2024
GUIDELINE_CITATION_PATTERN = re.compile(r"(?i)(?:USSG\s*)?(?:§\s*)?([0-9][A-Z][0-9A-Z]*\.[0-9A-Z]+(?:\([a-z0-9]+\))*)")
GUIDELINE_SECTION_PATTERN = re.compile(r"(?i)^([0-9][A-Z][0-9A-Z]*\.[0-9A-Z]+)")


def extract_guideline_citation(query: str | None) -> str | None:
    match = GUIDELINE_CITATION_PATTERN.search(str(query or ""))
    if not match:
        return None
    return match.group(1).upper()


def extract_guideline_section_citation(query: str | None) -> str | None:
    citation = extract_guideline_citation(query)
    if citation is None:
        return None
    match = GUIDELINE_SECTION_PATTERN.match(citation)
    if not match:
        return citation
    return match.group(1).upper()


def lookup_guideline_section(query: str, *, year: int = DEFAULT_GUIDELINE_YEAR) -> dict[str, Any]:
    normalized_query = str(query or "").strip()
    if not normalized_query:
        raise ValueError("query must be non-empty.")

    requested_citation = extract_guideline_citation(normalized_query)
    if requested_citation is None:
        return {
            "result_type": "local_guideline_lookup",
            "query": normalized_query,
            "year": year,
            "status": "unsupported_query",
            "message": "Use a citation-like query such as '2K2.1' or 'USSG §2K2.1(b)(1)'.",
        }

    section_citation = extract_guideline_section_citation(requested_citation) or requested_citation
    requested_subsection = requested_citation[len(section_citation) :].strip() if requested_citation.startswith(section_citation) else ""

    section = open_ussg_section(year=year, section_citation=section_citation)
    if section.get("status") == "not_found":
        return {
            "result_type": "local_guideline_lookup",
            "query": normalized_query,
            "requested_citation": requested_citation,
            "section_citation": section_citation,
            "requested_subsection": requested_subsection or None,
            "year": year,
            "status": "not_found",
            "message": section.get("message") or f"USSG section not found: {section_citation}",
        }

    text = str(section.get("text") or "").strip()
    subsection_context = (
        _extract_citation_context(text, requested_citation, fallback_citation=section_citation)
        if requested_subsection
        else None
    )
    return {
        "result_type": "local_guideline_lookup",
        "query": normalized_query,
        "requested_citation": requested_citation,
        "section_citation": section_citation,
        "requested_subsection": requested_subsection or None,
        "year": year,
        "title": str(section.get("section_heading") or f"USSG §{section_citation}"),
        "entry_id": section.get("entry_id"),
        "source_type": section.get("source_type"),
        "blocks": section.get("blocks") or [],
        "block_count": len(section.get("blocks") or []),
        "text": text,
        "text_length": len(text),
        "subsection_context": subsection_context,
        "preview_excerpt": text[:4000],
    }


def guideline_lookup_tool(query: str, year: int = DEFAULT_GUIDELINE_YEAR) -> dict[str, Any]:
    """Open one USSG guideline section from local manual data.

    Inputs:
        query: A citation-like query such as `2T4.1`, `3E1.1`, or `USSG §2K2.1(b)(1)`.
        year: Guidelines manual year to inspect.

    Returns:
        A dictionary with the requested citation, normalized section citation,
        section title, optional subsection context, and a section text excerpt.
    """
    return lookup_guideline_section(query=query, year=year)


def build_search_tools(*, guideline_year: int = DEFAULT_GUIDELINE_YEAR) -> list[Any]:
    def configured_guideline_lookup_tool(query: str) -> dict[str, Any]:
        """Open one USSG guideline section from local manual data for the configured guideline year.

        Input:
            query: A citation-like query such as `2T4.1`, `3E1.1`, or `USSG §2K2.1(b)(1)`.

        Returns:
            A dictionary with the requested citation, normalized section citation,
            section title, optional subsection context, and a section text excerpt.
        """
        return lookup_guideline_section(query=query, year=guideline_year)

    configured_guideline_lookup_tool.__name__ = "guideline_lookup_tool"
    return [configured_guideline_lookup_tool]


def _extract_citation_context(text: str, citation: str, fallback_citation: str | None = None, window: int = 260) -> str:
    normalized_text = str(text or "")
    primary = str(citation or "").strip().upper().removeprefix("§").strip()
    fallback = str(fallback_citation or "").strip().upper().removeprefix("§").strip()
    upper_text = normalized_text.upper()
    tokens = [token for token in [f"§{primary}" if primary else "", primary, f"§{fallback}" if fallback else "", fallback] if token]
    index = -1
    for token in tokens:
        index = upper_text.find(token)
        if index != -1:
            break
    if index == -1:
        return normalized_text[:window]

    start = max(0, index - 80)
    end = min(len(normalized_text), index + window)
    return normalized_text[start:end].strip()


__all__ = [
    "DEFAULT_GUIDELINE_YEAR",
    "build_search_tools",
    "extract_guideline_citation",
    "extract_guideline_section_citation",
    "guideline_lookup_tool",
    "lookup_guideline_section",
]