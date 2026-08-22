from __future__ import annotations
from slugify import slugify

import json
from functools import lru_cache
from pathlib import Path
import re
import sys
from typing import Any

from baselines.icd_react.config import ICDReactConfig, load_config
from baselines.icd_react.tools import (
    list_index_main_terms,
    list_tabular_chapters as base_list_tabular_chapters,
    open_index_term,
    open_tabular_chapter as base_open_tabular_chapter,
    open_tabular_entry as base_open_tabular_entry,
    open_tabular_section as base_open_tabular_section,
)


INDEX_MANUAL_NAME = "ICD-10-CM Alphabetic Index"
TABULAR_MANUAL_NAME = "ICD-10-CM Tabular List of Diseases and Injuries"
GUIDELINES_MANUAL_NAME = "FY 2019 ICD-10-CM Official Guidelines for Coding and Reporting"
GUIDELINES_DOCINTEL_EXPORT_NAME = "2019-icd10-coding-guidelines-.docintel.json"
INDEX_HIERARCHY_MAX_DIRECT_CHILDREN = 40
INDEX_HIERARCHY_MAX_DESCENDANT_MATCHES = 80


def list_index_letter_headings(
    letter: str,
    prefix: str | None = None,
    limit: int | None = 500,
    start_index: int = 0,
    config: ICDReactConfig | None = None,
) -> dict[str, Any]:
    """List top-level Alphabetic Index headings for one letter.

    This is the first keyword-exploration step. It exposes the ICD Index
    main-term headings under a single starting letter without opening any
    nested hierarchy for free.

    Args:
        letter: One alphabetic letter such as `A`.
        prefix: Optional heading prefix filter within that letter.
        limit: Maximum number of headings to return. Use `None` to return all
            headings from `start_index` onward.
        start_index: Zero-based offset into the ordered heading list.
        config: Optional preloaded runtime configuration.

    Returns:
        A dictionary with `letter`, `count`, and `results`. Each result row
        includes:
        - `entry_id`: Stable identifier for the follow-up opener.
        - `title`: Main-term heading text.
        - `code`: Direct code attached to the heading when present.
        - `see`: Direct `see` cross-reference when present.
        - `see_also`: Direct `see also` cross-reference when present.
        - `child_count`: Number of nested child terms beneath the heading.

    Raises:
        ValueError: If `letter` is blank or not a single alphabetic character.
    """
    normalized_letter = str(letter or "").strip().upper()
    if len(normalized_letter) != 1 or not normalized_letter.isalpha():
        raise ValueError("letter must be a single alphabetic character such as 'A'")
    if start_index < 0:
        raise ValueError("start_index must be greater than or equal to 0")
    if limit is not None and limit < 0:
        raise ValueError("limit must be greater than or equal to 0")

    resolved_config = config or load_config()
    upstream_limit = sys.maxsize if limit is None else start_index + limit
    result = list_index_main_terms(
        config=resolved_config,
        letter=normalized_letter,
        prefix=prefix,
        limit=max(upstream_limit, 1),
    )
    slice_end = None if limit is None else start_index + limit
    sliced_results = result["results"][start_index:slice_end]
    return {
        "manual": INDEX_MANUAL_NAME,
        "letter": result["letter"],
        "prefix": result["prefix"],
        "start_index": start_index,
        "limit": limit,
        "count": len(sliced_results),
        "results": sliced_results,
    }


def open_index_heading_hierarchy(entry_id: str, config: ICDReactConfig | None = None) -> dict[str, Any]:
    """Open one top-level Alphabetic Index heading and return its full subtree.

    This is the second keyword-exploration step. It resolves one exact heading
    returned by `list_index_letter_headings` and exposes the nested ICD Index
    hierarchy beneath that heading.

    Args:
        entry_id: Exact heading identifier returned by `list_index_letter_headings`.
        config: Optional preloaded runtime configuration.

    Returns:
        A dictionary with the selected heading metadata and its nested
        `children` hierarchy.
    """
    resolved_config = config or load_config()
    result = open_index_term(config=resolved_config, entry_id=entry_id)

    direct_children = result["children"][:INDEX_HIERARCHY_MAX_DIRECT_CHILDREN]
    descendant_matches: list[dict[str, Any]] = []

    def collect_descendant_matches(nodes: list[dict[str, Any]]) -> None:
        for node in nodes:
            if len(descendant_matches) >= INDEX_HIERARCHY_MAX_DESCENDANT_MATCHES:
                return
            if node.get("code") or node.get("see") or node.get("see_also"):
                descendant_matches.append(
                    {
                        "path": node.get("path"),
                        "code": node.get("code"),
                        "see": node.get("see"),
                        "see_also": node.get("see_also"),
                        "child_count": len(node.get("children") or []),
                    }
                )
            collect_descendant_matches(node.get("children") or [])
            if len(descendant_matches) >= INDEX_HIERARCHY_MAX_DESCENDANT_MATCHES:
                return

    collect_descendant_matches(result["children"])
    return {
        "manual": INDEX_MANUAL_NAME,
        "entry_id": result["entry_id"],
        "letter": result["letter"],
        "title": result["title"],
        "code": result["code"],
        "see": result["see"],
        "see_also": result["see_also"],
        "child_count": len(result["children"]),
        "children": [
            {
                "title": child.get("title"),
                "path": child.get("path"),
                "code": child.get("code"),
                "see": child.get("see"),
                "see_also": child.get("see_also"),
                "child_count": len(child.get("children") or []),
            }
            for child in direct_children
        ],
        "children_truncated": len(result["children"]) > len(direct_children),
        "descendant_match_count": len(descendant_matches),
        "descendant_matches_truncated": len(descendant_matches) >= INDEX_HIERARCHY_MAX_DESCENDANT_MATCHES,
        "descendant_matches": descendant_matches,
    }


def list_tabular_chapters(code_prefix: str | None = None, config: ICDReactConfig | None = None) -> dict[str, Any]:
    """List ICD Tabular chapter headings.

    This is the top-level Tabular browser. It returns the chapter table of
    contents only, so the agent has to choose one chapter before seeing the
    underlying block ranges.

    Args:
        code_prefix: Optional code-family filter such as `A`, `A00`, or `C`.
        config: Optional preloaded runtime configuration.

    Returns:
                A dictionary with `chapter_count` and `chapters`. Each chapter includes:
        - `chapter_id`: Stable chapter identifier such as `1`.
        - `chapter_heading`: Display heading such as `Chapter 1`.
        - `description`: Chapter description with its code range.
        - `code_range`: High-level code range when available.
    """
    resolved_config = config or load_config()
    chapter_rows = base_list_tabular_chapters(
        config=resolved_config,
        code_prefix=code_prefix,
        limit=sys.maxsize,
    )["results"]

    chapters: list[dict[str, Any]] = []
    for chapter_row in chapter_rows:
        chapter_id = str(chapter_row["chapter_id"])
        chapters.append(
            {
                "chapter_id": chapter_id,
                "chapter_heading": f"Chapter {chapter_id}",
                "description": chapter_row["description"],
                "code_range": chapter_row.get("code_range"),
            }
        )

    return {
        "manual": TABULAR_MANUAL_NAME,
        "document_title": TABULAR_MANUAL_NAME,
        "code_prefix": code_prefix or None,
        "chapter_count": len(chapters),
        "chapters": chapters,
    }


def open_tabular_chapter(chapter_id: str, config: ICDReactConfig | None = None) -> dict[str, Any]:
    """Open one ICD Tabular chapter and expose its blocks.

    Args:
        chapter_id: Exact chapter identifier returned by `list_tabular_chapters`.
        config: Optional preloaded runtime configuration.

    Returns:
        A dictionary with the selected chapter metadata, chapter note groups,
        and `blocks`. Each block row includes only block metadata so the caller
        can choose one exact block to open next.
    """
    resolved_config = config or load_config()
    result = base_open_tabular_chapter(config=resolved_config, chapter_id=chapter_id)

    blocks: list[dict[str, Any]] = []
    for block in result["sections"]:
        blocks.append(
            {
                "section_id": str(block["section_id"]),
                "first_code": block.get("first_code"),
                "last_code": block.get("last_code"),
                "description": block.get("description"),
            }
        )

    return {
        "manual": TABULAR_MANUAL_NAME,
        "chapter_id": result["chapter_id"],
        "chapter_heading": f"Chapter {result['chapter_id']}",
        "description": result["description"],
        "note_groups": result["note_groups"],
        "block_count": len(blocks),
        "blocks": blocks,
    }


def open_tabular_block(section_id: str, config: ICDReactConfig | None = None) -> dict[str, Any]:
    """Open one ICD Tabular block and list the direct codes beneath it.

    Args:
        section_id: Exact block identifier returned by `list_tabular_chapters`.
        config: Optional preloaded runtime configuration.

    Returns:
        A dictionary with the selected block metadata and its direct `codes`.
        Each code row includes `code`, `description`, `child_count`, and any
        local note groups attached at that top level.
    """
    resolved_config = config or load_config()
    result = base_open_tabular_section(config=resolved_config, section_id=section_id)
    return {
        "manual": TABULAR_MANUAL_NAME,
        "section_id": result["section_id"],
        "description": result["description"],
        "chapter_id": result["chapter_id"],
        "chapter_heading": None if result["chapter_id"] is None else f"Chapter {result['chapter_id']}",
        "chapter_description": result["chapter_description"],
        "note_groups": result["note_groups"],
        "code_count": len(result["top_level_codes"]),
        "codes": result["top_level_codes"],
    }


def open_tabular_code(code: str, config: ICDReactConfig | None = None) -> dict[str, Any]:
    """Open one exact ICD Tabular code entry.

    Args:
        code: Exact code to inspect such as `A00`.
        config: Optional preloaded runtime configuration.

    Returns:
        A dictionary with chapter context, section context, ancestor codes,
        child codes, and chapter, section, and entry note groups for the exact
        requested code.
    """
    resolved_config = config or load_config()
    result = base_open_tabular_entry(config=resolved_config, code=code)
    return {
        **result,
        "manual": TABULAR_MANUAL_NAME,
    }


def normalize_guideline_text(value: str) -> str:
    text = str(value or "")
    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", " ", text)
    return text.strip()


def guideline_heading_level(title: str) -> int:
    stripped = normalize_guideline_text(title)
    if stripped.startswith("Section "):
        return 1
    if stripped.startswith("Appendix "):
        return 1
    if re.match(r"^[A-Z]\.", stripped):
        return 2
    if re.match(r"^\d+\.", stripped):
        return 3
    if re.match(r"^[a-z]\.", stripped):
        return 4
    if re.match(r"^\d+\)", stripped):
        return 5
    if re.match(r"^[a-z]\)", stripped):
        return 6
    return 0


def extract_guideline_marker(title: str) -> str | None:
    stripped = normalize_guideline_text(title)
    if not stripped:
        return None
    section_match = re.match(r"^Section\s+([IVXLC]+)\b", stripped)
    if section_match:
        return section_match.group(1)
    appendix_match = re.match(r"^Appendix\s+([A-Z])\b", stripped)
    if appendix_match:
        return appendix_match.group(1)
    dotted_match = re.match(r"^([A-Z]|\d+|[a-z])[\.)]", stripped)
    if dotted_match:
        return dotted_match.group(1)
    return None


def require_guidelines_docintel_path(config: ICDReactConfig) -> Path:
    path = config.manuals_root / "_docintel_text" / "icd" / GUIDELINES_DOCINTEL_EXPORT_NAME
    if not path.exists():
        raise RuntimeError(
            "ICD guidelines DocIntel export not found. "
            f"Run notebooks/icd_docintel_export.ipynb first: {path}"
        )
    return path


def is_guideline_page_artifact(line: str) -> bool:
    stripped = normalize_guideline_text(line)
    if not stripped:
        return True
    if stripped in {"FY 2019", "ICD-10-CM Official Guidelines for Coding and Reporting", "ICD-10-CM Official Guidelines for Coding and Reporting."}:
        return True
    if re.match(r"^Page \d+ of \d+$", stripped):
        return True
    return False


def is_guideline_toc_heading(title: str) -> bool:
    stripped = normalize_guideline_text(title)
    if not stripped:
        return False
    if stripped == GUIDELINES_MANUAL_NAME:
        return False
    return bool(
        stripped.startswith("Section ")
        or stripped.startswith("Appendix ")
        or re.match(r"^[A-Z]\.", stripped)
        or re.match(r"^\d+\.", stripped)
        or re.match(r"^[a-z]\.", stripped)
        or re.match(r"^\d+\)", stripped)
        or re.match(r"^[a-z]\)", stripped)
    )


def clean_guideline_title(title: str) -> str:
    return normalize_guideline_text(title).rstrip(".")


def is_guideline_toc_marker(line: str) -> bool:
    stripped = normalize_guideline_text(line)
    return bool(
        re.match(r"^\d+[\.)]$", stripped)
        or re.match(r"^[A-Z][\.)]$", stripped)
        or re.match(r"^[a-z][\.)]$", stripped)
        or re.match(r"^(Section|Appendix) [IVXLC]+\.?$", stripped)
    )


def is_guideline_chapter_title(line: str) -> bool:
    return normalize_guideline_text(line).startswith("Chapter ")


def compose_guideline_toc_title(prefix: str | None, title_lines: list[str]) -> str:
    if prefix:
        return clean_guideline_title(" ".join([prefix, *title_lines]))
    return clean_guideline_title(" ".join(title_lines))


def parse_guideline_toc_group(lines: list[str], deferred_prefixes: list[str]) -> tuple[str | None, list[str]]:
    stripped_lines = [normalize_guideline_text(line) for line in lines if normalize_guideline_text(line)]
    if not stripped_lines:
        return None, deferred_prefixes

    chapter_index = next((index for index, line in enumerate(stripped_lines) if is_guideline_chapter_title(line)), None)
    if chapter_index is not None:
        prefix_candidates = [line for line in stripped_lines[:chapter_index] if is_guideline_toc_marker(line)]
        title_lines = [line for line in stripped_lines[chapter_index:] if not is_guideline_toc_marker(line)]
        if not title_lines:
            return None, [*deferred_prefixes, *prefix_candidates]

        prefix = prefix_candidates[0] if prefix_candidates else (deferred_prefixes[0] if deferred_prefixes else None)
        remaining_deferred_prefixes = [*deferred_prefixes[1:], *prefix_candidates[1:]] if prefix_candidates else deferred_prefixes[1:]
        return compose_guideline_toc_title(prefix, title_lines), remaining_deferred_prefixes

    remaining_deferred_prefixes = list(deferred_prefixes)
    prefix: str | None = None
    title_lines = list(stripped_lines)

    if title_lines and is_guideline_toc_marker(title_lines[0]):
        prefix = title_lines[0]
        title_lines = title_lines[1:]
    elif remaining_deferred_prefixes:
        prefix = remaining_deferred_prefixes.pop(0)

    title_lines = [line for line in title_lines if not is_guideline_toc_marker(line)]
    if not title_lines:
        if prefix is not None:
            remaining_deferred_prefixes.insert(0, prefix)
        return None, remaining_deferred_prefixes

    return compose_guideline_toc_title(prefix, title_lines), remaining_deferred_prefixes


def title_lookup_variants(title: str) -> list[str]:
    cleaned = clean_guideline_title(title)
    variants: list[str] = []
    for candidate in [cleaned, cleaned.replace('"', ""), cleaned.rstrip(":"), cleaned.rstrip(".")]:
        normalized = normalize_guideline_text(candidate)
        if normalized and normalized not in variants:
            variants.append(normalized)
    return variants


def join_guideline_lines(lines: list[str]) -> str:
    return normalize_guideline_text(" ".join(lines))


@lru_cache(maxsize=2)
def _load_guidelines_docintel_pages_cached(docintel_path: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(docintel_path).read_text(encoding="utf-8"))
    raw_pages = payload.get("pages")
    if not isinstance(raw_pages, list):
        return []

    pages: list[dict[str, Any]] = []
    for raw_page in raw_pages:
        if not isinstance(raw_page, dict):
            continue
        page_number = int(raw_page.get("page_number") or 0)
        raw_text = str(raw_page.get("text") or "")
        lines = [normalize_guideline_text(line) for line in raw_text.splitlines() if not is_guideline_page_artifact(line)]
        lines = [line for line in lines if line]
        pages.append(
            {
                "page_number": page_number,
                "text": raw_text,
                "lines": lines,
            }
        )
    return pages


@lru_cache(maxsize=2)
def _build_guideline_toc_cached(docintel_path: str) -> list[dict[str, Any]]:
    pages = _load_guidelines_docintel_pages_cached(docintel_path)
    entries: list[dict[str, Any]] = []
    current_title_lines: list[str] = []
    deferred_prefixes: list[str] = []

    def flush_entry(target_page: int) -> None:
        nonlocal current_title_lines, deferred_prefixes
        title, deferred_prefixes = parse_guideline_toc_group(current_title_lines, deferred_prefixes)
        current_title_lines = []
        if not title:
            return
        if not is_guideline_toc_heading(title):
            return
        entry_id = f"guidelines|{target_page}|{slugify(title)}"
        entries.append(
            {
                "section_id": entry_id,
                "title": title,
                "level": guideline_heading_level(title),
                "target_page": target_page,
            }
        )

    for page in pages:
        page_number = int(page.get("page_number") or 0)
        if page_number < 2 or page_number > 7:
            continue
        for line in page.get("lines", []):
            if re.match(r"^\d+$", line):
                if current_title_lines:
                    flush_entry(int(line))
                continue
            current_title_lines.append(line)

    return entries


def find_guideline_heading_line_range(lines: list[str], title: str) -> tuple[int, int] | None:
    variants = title_lookup_variants(title)
    if not variants:
        return None
    max_span = min(4, len(lines))
    for start_index in range(len(lines)):
        for span in range(1, max_span + 1):
            end_index = start_index + span
            if end_index > len(lines):
                break
            candidate = join_guideline_lines(lines[start_index:end_index]).rstrip(".")
            if candidate in variants:
                return start_index, end_index
    return None


def locate_guideline_entry(docintel_path: str, entry: dict[str, Any]) -> tuple[int, int, int] | None:
    pages = _load_guidelines_docintel_pages_cached(docintel_path)
    target_page = int(entry["target_page"])
    title = str(entry["title"])
    for page_index, page in enumerate(pages):
        if int(page.get("page_number") or 0) < target_page:
            continue
        match = find_guideline_heading_line_range(page.get("lines", []), title)
        if match is not None:
            start_line, end_line = match
            return page_index, start_line, end_line
    return None


def resolve_guideline_section_entry(toc_entries: list[dict[str, Any]], section_id: str) -> dict[str, Any] | None:
    requested_id = str(section_id or "").strip()
    if not requested_id:
        return None

    exact_entry = next((entry for entry in toc_entries if entry["section_id"] == requested_id), None)
    if exact_entry is not None:
        return exact_entry

    normalized_requested = requested_id.replace(" ", "").rstrip(".").upper()
    marker_stack: dict[int, str] = {}
    for entry in toc_entries:
        level = int(entry.get("level") or 0)
        marker = extract_guideline_marker(str(entry.get("title") or ""))
        if level > 0 and marker:
            marker_stack[level] = marker.upper()
            marker_stack = {current_level: value for current_level, value in marker_stack.items() if current_level <= level}
            alias = ".".join(marker_stack[current_level] for current_level in sorted(marker_stack))
            if alias.rstrip(".").upper() == normalized_requested:
                return entry
    return None


def slice_guideline_text(docintel_path: str, current_entry: dict[str, Any], next_boundary_entry: dict[str, Any] | None) -> tuple[str, int, int]:
    pages = _load_guidelines_docintel_pages_cached(docintel_path)
    current_location = locate_guideline_entry(docintel_path, current_entry)
    if current_location is None:
        raise KeyError(f"Could not locate guideline section in body text: {current_entry['title']}")

    start_page_index, start_line_index, _ = current_location
    end_page_index = len(pages) - 1
    end_line_index: int | None = None

    if next_boundary_entry is not None:
        next_location = locate_guideline_entry(docintel_path, next_boundary_entry)
        if next_location is not None:
            end_page_index = next_location[0]
            end_line_index = next_location[1]

    text_lines: list[str] = []
    for page_index in range(start_page_index, end_page_index + 1):
        page_lines = list(pages[page_index].get("lines", []))
        page_start = start_line_index if page_index == start_page_index else 0
        page_end = end_line_index if end_line_index is not None and page_index == end_page_index else len(page_lines)
        text_lines.extend(page_lines[page_start:page_end])

    text = "\n".join(text_lines).strip()
    return text, int(pages[start_page_index]["page_number"]), int(pages[end_page_index]["page_number"])


def list_guideline_toc(section_prefix: str | None = None, limit: int | None = None, config: ICDReactConfig | None = None) -> dict[str, Any]:
    """List the ICD coding-guidelines table of contents.

    This is the top-level guidelines browser. It uses the exported DocIntel text
    to return a flat TOC with simple titles and stable ids.

    Args:
        section_prefix: Optional title-prefix filter over the extracted
            guideline headings.
        limit: Maximum number of TOC rows to return. Use `None` to return the
            full matching TOC.
        config: Optional preloaded runtime configuration.

    Returns:
        A dictionary with `count` and `results`. Each TOC row includes
        `section_id`, `title`, and `level`.
    """
    if limit is not None and limit < 0:
        raise ValueError("limit must be greater than or equal to 0")

    resolved_config = config or load_config()
    docintel_path = require_guidelines_docintel_path(resolved_config)
    toc_entries = _build_guideline_toc_cached(str(docintel_path))
    prefix_filter = normalize_guideline_text(section_prefix or "").lower()
    filtered_entries = [
        {
            "section_id": entry["section_id"],
            "title": entry["title"],
            "level": entry["level"],
        }
        for entry in toc_entries
        if not prefix_filter or normalize_guideline_text(entry["title"]).lower().startswith(prefix_filter)
    ]
    if limit is not None:
        filtered_entries = filtered_entries[:limit]
    return {
        "manual": GUIDELINES_MANUAL_NAME,
        "section_prefix": section_prefix or None,
        "limit": limit,
        "count": len(filtered_entries),
        "results": filtered_entries,
    }


def open_guideline_section(section_id: str, config: ICDReactConfig | None = None) -> dict[str, Any]:
    """Open one exact ICD coding-guidelines section by section_id.

    Args:
        section_id: Exact section identifier returned by `list_guideline_toc`.
        config: Optional preloaded runtime configuration.

    Returns:
        A dictionary with the guideline section title and extracted text.
    """
    resolved_config = config or load_config()
    docintel_path = require_guidelines_docintel_path(resolved_config)
    toc_entries = _build_guideline_toc_cached(str(docintel_path))
    current_entry = resolve_guideline_section_entry(toc_entries, section_id)
    if current_entry is None:
        raise KeyError(f"Guideline section not found: {section_id}")

    current_index = next(index for index, entry in enumerate(toc_entries) if entry["section_id"] == current_entry["section_id"])
    next_boundary_entry = next(
        (entry for entry in toc_entries[current_index + 1 :] if int(entry["level"]) <= int(current_entry["level"])),
        None,
    )
    text, start_page, end_page = slice_guideline_text(str(docintel_path), current_entry, next_boundary_entry)
    return {
        "manual": GUIDELINES_MANUAL_NAME,
        "section_id": current_entry["section_id"],
        "title": current_entry["title"],
        "level": current_entry["level"],
        "page_start": start_page,
        "page_end": end_page,
        "text": text,
    }


def build_icd_manual_tools(config: ICDReactConfig) -> list[Any]:
    def tool_error_response(tool_name: str, error: Exception, **inputs: Any) -> dict[str, Any]:
        return {
            "ok": False,
            "tool": tool_name,
            "error_type": type(error).__name__,
            "error": str(error),
            "inputs": inputs,
            "guidance": "Use the returned error details to choose a nearby valid identifier or a different navigation step instead of stopping the case.",
        }

    def list_index_letter_headings_tool(
        letter: str,
        prefix: str | None = None,
        limit: int | None = 500,
        start_index: int = 0,
    ) -> dict[str, Any]:
        """List top-level ICD Alphabetic Index headings for one letter.

        Input:
            letter: One alphabetic letter such as `A`.
            prefix: Optional heading prefix filter within that letter.
            limit: Maximum number of headings to return. Use `None` to return
                all headings from `start_index` onward.
            start_index: Zero-based offset into the ordered heading list.

        Returns:
            A dictionary with `letter`, `start_index`, `limit`, `count`, and `results`. Each item in
            `results` includes:
            - `entry_id`: Stable identifier for the follow-up opener.
            - `title`: Main-term heading text.
            - `code`: Direct code attached to the heading when present.
            - `see`: Direct `see` cross-reference when present.
            - `see_also`: Direct `see also` cross-reference when present.
            - `child_count`: Number of nested child terms beneath the heading.

        Notes:
            - Use this as the first ICD keyword-exploration step.
                        - Use `start_index` with `limit` to page through long letter buckets.
                        - Set `limit` to `None` when the agent needs the full remaining list.
            - This tool is structural only. It does not open the nested hierarchy
              or choose the correct lead term for the case.
        """
        try:
            return list_index_letter_headings(
                letter=letter,
                prefix=prefix,
                limit=limit,
                start_index=start_index,
                config=config,
            )
        except Exception as error:
            return tool_error_response(
                "list_index_letter_headings_tool",
                error,
                letter=letter,
                prefix=prefix,
                limit=limit,
                start_index=start_index,
            )

    def open_index_heading_hierarchy_tool(entry_id: str) -> dict[str, Any]:
        """Open one ICD Alphabetic Index heading and return its nested hierarchy.

        Input:
            entry_id: Exact heading identifier returned by
                list_index_letter_headings_tool.

        Returns:
            A dictionary with the selected heading metadata and nested `children`.
            The response preserves direct `code`, `see`, and `see_also` fields
            and all child terms beneath the selected heading.

        Notes:
            - Use an `entry_id` returned for the same manual edition.
            - This tool exposes the hierarchy under one exact heading. It does not
              verify the final code in the Tabular List or apply guideline logic.
        """
        try:
            return open_index_heading_hierarchy(entry_id=entry_id, config=config)
        except Exception as error:
            return tool_error_response("open_index_heading_hierarchy_tool", error, entry_id=entry_id)

    def list_tabular_chapters_tool(code_prefix: str | None = None) -> dict[str, Any]:
        """List ICD Tabular chapter headings.

        Input:
            code_prefix: Optional code-family filter such as `A`, `A00`, or `C`.

        Returns:
            A dictionary with `chapter_count` and `chapters`. Each chapter row
            includes `chapter_id`, `chapter_heading`, `description`, and
            `code_range`.

        Notes:
            - Use this as the top-level Tabular navigation step.
            - This tool is structural only. It does not open one chapter, one
              block, or one exact code entry.
        """
        try:
            return list_tabular_chapters(code_prefix=code_prefix, config=config)
        except Exception as error:
            return tool_error_response("list_tabular_chapters_tool", error, code_prefix=code_prefix)

    def open_tabular_chapter_tool(chapter_id: str) -> dict[str, Any]:
        """Open one ICD Tabular chapter and show its blocks.

        Input:
            chapter_id: Exact chapter identifier returned by list_tabular_chapters_tool.

        Returns:
            A dictionary with chapter metadata, chapter note groups, and
            `blocks`. Each block row includes `section_id`, code range fields,
            and the block description.

        Notes:
            - Use this after choosing a top-level chapter such as `1`.
            - Use open_tabular_block_tool after selecting one exact block such as `A00-A09`.
            - This tool still does not open one exact Tabular code entry.
        """
        try:
            return open_tabular_chapter(chapter_id=chapter_id, config=config)
        except Exception as error:
            return tool_error_response("open_tabular_chapter_tool", error, chapter_id=chapter_id)

    def open_tabular_block_tool(section_id: str) -> dict[str, Any]:
        """Open one ICD Tabular block and list the direct codes under it.

        Input:
            section_id: Exact block identifier returned by open_tabular_chapter_tool.

        Returns:
            A dictionary with the block metadata, block note groups, and direct
            `codes` under that block.

        Notes:
            - Use this after choosing a block such as `A00-A09`.
            - This tool does not open one exact code entry yet.
        """
        try:
            return open_tabular_block(section_id=section_id, config=config)
        except Exception as error:
            return tool_error_response("open_tabular_block_tool", error, section_id=section_id)

    def open_tabular_code_tool(code: str) -> dict[str, Any]:
        """Open one exact ICD Tabular code entry.

        Input:
            code: Exact code to inspect such as `A00`.

        Returns:
            A dictionary with ancestor codes, child codes, and chapter, section,
            and entry note groups for that exact code.

        Notes:
            - Use this after choosing a code from a block open step.
            - This tool is the authority surface for inspecting one exact ICD
              code entry in the Tabular List.
        """
        try:
            return open_tabular_code(code=code, config=config)
        except Exception as error:
            return tool_error_response("open_tabular_code_tool", error, code=code)

    def list_guideline_toc_tool(section_prefix: str | None = None, limit: int | None = None) -> dict[str, Any]:
        """List the ICD coding-guidelines table of contents.

        Input:
            section_prefix: Optional title-prefix filter over the extracted
                guideline headings.
            limit: Maximum number of TOC rows to return. Use `None` to return
                the full matching TOC.

        Returns:
            A dictionary with `count` and `results`. Each TOC row includes
            `section_id`, `title`, and `level`.

        Notes:
            - Use this as the top-level coding-guidelines navigation step.
            - This tool does not open any section text yet.
        """
        try:
            return list_guideline_toc(section_prefix=section_prefix, limit=limit, config=config)
        except Exception as error:
            return tool_error_response(
                "list_guideline_toc_tool",
                error,
                section_prefix=section_prefix,
                limit=limit,
            )

    def open_guideline_section_tool(section_id: str) -> dict[str, Any]:
        """Open one ICD coding-guidelines section by exact section_id.

        Input:
            section_id: Exact section identifier returned by list_guideline_toc_tool.

        Returns:
            A dictionary with the selected guideline section's title,
            level, page bounds, and extracted text.

        Notes:
            - Use this after selecting one exact TOC row.
        """
        try:
            return open_guideline_section(section_id=section_id, config=config)
        except Exception as error:
            return tool_error_response("open_guideline_section_tool", error, section_id=section_id)

    return [
        list_index_letter_headings_tool,
        open_index_heading_hierarchy_tool,
        list_tabular_chapters_tool,
        open_tabular_chapter_tool,
        open_tabular_block_tool,
        open_tabular_code_tool,
        list_guideline_toc_tool,
        open_guideline_section_tool,
    ]
