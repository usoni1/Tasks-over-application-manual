from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any

from baselines.legal_rag.chunking import (
    HEADING_PATTERN,
    clean_heading_text,
    extract_title18_citation,
    extract_ussg_citation,
    is_subheading,
    normalize_whitespace,
    read_text,
    strip_tags,
)
from baselines.legal_rag.title18_paths import list_title18_years, resolve_title18_manual_path

from .config import LegalReactV2Config, load_config


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def build_title18_chapter_id(year: int, chapter_heading: str) -> str:
    return f"title18-chapter|{year}|{slugify(chapter_heading)}"


def build_title18_section_id(year: int, citation: str | None, section_heading: str) -> str:
    key = citation or slugify(section_heading)
    return f"title18-section|{year}|{key}"


def build_appendix_a_entry_id(year: int, statute_citation: str) -> str:
    return f"appendix-a|{year}|{slugify(statute_citation)}"


def build_ussg_chapter_id(year: int, chapter_heading: str) -> str:
    return f"ussg-chapter|{year}|{slugify(chapter_heading)}"


def build_ussg_subheading_id(year: int, chapter_heading: str, part_heading: str, subheading_heading: str) -> str:
    prefix = build_ussg_subheading_prefix(chapter_heading, part_heading, subheading_heading)
    if prefix is not None:
        return f"ussg-subheading|{year}|{slugify(prefix.rstrip('.'))}"
    return f"ussg-subheading|{year}|{slugify(chapter_heading)}|{slugify(part_heading)}|{slugify(subheading_heading)}"


def build_ussg_section_id(year: int, section_citation: str) -> str:
    return f"ussg|{year}|{slugify(section_citation)}"


def is_bare_ussg_citation_line(line: str) -> bool:
    return bool(re.match(r"^§\s*[0-9A-Z]+[A-Z0-9\.-]*\.?$", line.strip()))


def extract_normalized_ussg_citation(line: str) -> str | None:
    citation = extract_ussg_citation(line)
    return citation.rstrip(".") if citation else None


def is_probable_ussg_section_start(line: str) -> bool:
    citation = extract_ussg_citation(line)
    if citation is None:
        return False
    remainder = line[len(citation) :].strip()
    if not remainder:
        return True
    return bool(re.match(r"^[A-Z\[]", remainder))


def is_ussg_page_artifact(line: str, year: int) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped in {"=", "Il", "ǁ", "|", "*", "**"}:
        return True
    if re.match(r"^\d+$", stripped):
        return True
    if stripped == f"Guidelines Manual (November 1, {year})":
        return True
    return False


def is_probable_ussg_chapter_heading(line: str) -> bool:
    return bool(re.match(r"^(CHAPTER|Chapter)\s+[A-Z0-9IVX]+\b", line))


def is_probable_ussg_part_heading(line: str) -> bool:
    return bool(re.match(r"^(PART|Part)\s+[A-Z0-9]+\s*[-—]", line))


def is_probable_ussg_toc_chapter_heading(line: str) -> bool:
    return bool(re.match(r"^CHAPTER\s+[A-Z0-9IVX]+$", line, re.IGNORECASE))


def is_probable_ussg_toc_subheading(line: str) -> bool:
    return bool(re.match(r"^\d+\.($|\s+.*)", line))


def is_ussg_toc_page_number(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if re.match(r"^[.|]*\d+$", stripped):
        return True
    if re.match(r"^\|?\s*[ivxlcdm]+$", stripped, re.IGNORECASE):
        return True
    return False


def strip_ussg_toc_page_number(line: str) -> str:
    if re.match(r"^\d+\.$", line.strip()):
        return line.strip()
    return re.sub(r"\s*[.|]*\d+\s*$", "", line).rstrip(" .")


def is_probable_ussg_toc_end(line: str) -> bool:
    stripped = line.strip()
    return bool(re.match(r"^Ch\.\s*\d+\b", stripped)) or stripped.startswith("§")


def collect_ussg_toc_heading(lines: list[str], start_index: int) -> tuple[str, int]:
    pieces = [strip_ussg_toc_page_number(lines[start_index])]
    index = start_index + 1

    while index < len(lines):
        candidate = lines[index]
        if (
            not candidate
            or candidate == "TABLE OF CONTENTS"
            or is_ussg_page_artifact(candidate, 0)
            or is_ussg_toc_page_number(candidate)
            or is_probable_ussg_toc_chapter_heading(candidate)
            or is_probable_ussg_part_heading(candidate)
            or is_probable_ussg_toc_subheading(candidate)
            or is_probable_ussg_toc_end(candidate)
        ):
            break
        pieces.append(strip_ussg_toc_page_number(candidate))
        index += 1

    heading = normalize_whitespace(" ".join(piece for piece in pieces if piece))
    return heading, index


def normalize_lookup_token(value: str | None) -> str:
    text = str(value or "").replace("\x00", "").replace("§", "")
    text = text.replace("\u00a7", "").replace("\xa7", "")
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def build_statute_lookup_variants(value: str | None) -> list[str]:
    raw = str(value or "").strip()
    variants: list[str] = []

    def add_variant(candidate: str | None) -> None:
        token = normalize_lookup_token(candidate)
        if token and token not in variants:
            variants.append(token)

    add_variant(raw)
    add_variant(re.sub(r"\([^)]*\)", "", raw))

    usc_match = re.search(
        r"(?P<title>\d+)\s*u\.?s\.?c\.?\s*§*\s*(?P<section>\d+[a-zA-Z]*)",
        raw,
        re.IGNORECASE,
    )
    if usc_match:
        title = usc_match.group("title")
        section = usc_match.group("section")
        add_variant(f"{title} U.S.C. § {section}")
        add_variant(section)

    return variants


def is_structured_statute_query(value: str | None) -> bool:
    text = str(value or "")
    return bool(re.search(r"u\.?s\.?c\.?|§", text, re.IGNORECASE))


def matches_parent_statute_token(entry_value: str, variant: str) -> bool:
    if not entry_value.startswith(variant):
        return False
    if len(entry_value) == len(variant):
        return True
    next_char = entry_value[len(variant)]
    return not next_char.isdigit()


def extract_ussg_chapter_number(chapter_heading: str) -> str | None:
    match = re.match(r"^CHAPTER\s+([A-Z0-9IVX]+)$", chapter_heading.strip(), re.IGNORECASE)
    if not match:
        return None
    token = match.group(1).upper()
    word_to_number = {
        "ONE": "1",
        "TWO": "2",
        "THREE": "3",
        "FOUR": "4",
        "FIVE": "5",
        "SIX": "6",
        "SEVEN": "7",
        "EIGHT": "8",
        "NINE": "9",
        "TEN": "10",
    }
    if token in word_to_number:
        return word_to_number[token]
    if token.isdigit():
        return token
    return None


def extract_ussg_part_code(part_heading: str) -> str | None:
    match = re.match(r"^PART\s+([A-Z0-9]+)\b", part_heading.strip(), re.IGNORECASE)
    return match.group(1).upper() if match else None


def extract_ussg_subheading_number(subheading_heading: str) -> str | None:
    match = re.match(r"^(\d+)\.", subheading_heading.strip())
    return match.group(1) if match else None


def build_ussg_subheading_prefix(chapter_heading: str, part_heading: str, subheading_heading: str) -> str | None:
    chapter_number = extract_ussg_chapter_number(chapter_heading)
    part_code = extract_ussg_part_code(part_heading)
    subheading_number = extract_ussg_subheading_number(subheading_heading)
    if chapter_number is None or part_code is None or subheading_number is None:
        return None
    return f"{chapter_number}{part_code}{subheading_number}."


def available_title18_years(config: LegalReactV2Config) -> list[int]:
    return list_title18_years(config.title18_root)


def require_title18_html_path(config: LegalReactV2Config, year: int) -> Path:
    path = resolve_title18_manual_path(config.title18_root / str(year))
    if path is None:
        raise KeyError(
            f"Title 18 manual for year {year} not found. "
            f"Available Title 18 years: {available_title18_years(config)}"
        )
    return path


def require_ussg_docintel_path(config: LegalReactV2Config, year: int) -> Path:
    if config.ussg_docintel_text_root is None:
        raise RuntimeError("USSG DocIntel export root is not configured.")
    path = config.ussg_docintel_text_root / str(year) / "GLMFull.docintel.json"
    if not path.exists():
        raise KeyError(f"USSG DocIntel export for year {year} not found: {path}")
    return path


@lru_cache(maxsize=16)
def _build_appendix_a_entries_cached(docintel_path: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(docintel_path).read_text(encoding="utf-8"))
    raw_pages = payload.get("pages")
    if not isinstance(raw_pages, list):
        return []

    statute_line_pattern = re.compile(r"^\d+\s+U\.S\.C\.\s+§{1,2}\s+.+$")
    guideline_pattern = re.compile(r"\b\d[A-Z]\d+\.\d+\b")

    entries: list[dict[str, Any]] = []
    in_appendix_a = False
    started_entries = False
    seen_keys: set[tuple[str, tuple[str, ...]]] = set()

    for raw_page in raw_pages:
        page_text = raw_page.get("text") if isinstance(raw_page, dict) else raw_page
        text = str(page_text or "")
        if not in_appendix_a and "APPENDIX A" not in text:
            continue
        in_appendix_a = True
        if started_entries and "INDEX TO GUIDELINES MANUAL" in text:
            break

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        index = 0
        while index < len(lines):
            line = lines[index]
            if line == "APPENDIX A":
                index += 1
                continue
            if not statute_line_pattern.match(line):
                index += 1
                continue

            guideline_sections: list[str] = []
            lookahead = index + 1
            while lookahead < len(lines):
                next_line = lines[lookahead]
                if statute_line_pattern.match(next_line) or next_line in {"APPENDIX A", "INDEX TO GUIDELINES MANUAL"}:
                    break
                guideline_sections.extend(guideline_pattern.findall(next_line))
                if guideline_sections:
                    break
                lookahead += 1

            if guideline_sections:
                unique_guidelines = list(dict.fromkeys(guideline_sections))
                key = (line, tuple(unique_guidelines))
                if key not in seen_keys:
                    seen_keys.add(key)
                    started_entries = True
                    year_match = re.search(r"(20\d{2})", docintel_path)
                    source_year = int(year_match.group(1)) if year_match else None
                    entries.append(
                        {
                            "entry_id": build_appendix_a_entry_id(source_year or 0, line),
                            "source_type": "ussg_appendix_a",
                            "source_year": source_year,
                            "document_title": "USSG Appendix A (Statutory Index)",
                            "statute_citation": line,
                            "guideline_sections": unique_guidelines,
                        }
                    )
            index = lookahead if lookahead > index else index + 1

    return entries


@lru_cache(maxsize=16)
def _build_ussg_section_index_cached(docintel_path: str, year: int) -> list[dict[str, Any]]:
    payload = json.loads(Path(docintel_path).read_text(encoding="utf-8"))
    raw_pages = payload.get("pages")
    if not isinstance(raw_pages, list):
        return []

    pages: list[str] = []
    for raw_page in raw_pages:
        page_text = raw_page.get("text") if isinstance(raw_page, dict) else raw_page
        pages.append(normalize_whitespace(str(page_text or "")))

    document_title = f"USSG Guidelines Manual ({year})"
    chapter_heading = ""
    part_heading = ""
    section_heading = ""
    subheading = ""
    section_lines: list[str] = []
    sections: dict[str, dict[str, Any]] = {}

    def flush_current() -> None:
        nonlocal section_lines
        if not section_heading or not section_lines:
            section_lines = []
            return
        body = normalize_whitespace("\n".join(section_lines))
        if not body:
            section_lines = []
            return
        citation = extract_normalized_ussg_citation(section_heading)
        if citation is None:
            section_lines = []
            return
        section_payload = sections.setdefault(
            citation,
            {
                "entry_id": build_ussg_section_id(year, citation),
                "source_type": "ussg",
                "source_year": year,
                "document_title": document_title,
                "chapter_heading": chapter_heading or None,
                "part_heading": part_heading or None,
                "section_heading": section_heading,
                "citation": citation,
                "blocks": [],
            },
        )
        if section_heading and not section_payload.get("section_heading"):
            section_payload["section_heading"] = section_heading
        section_payload["blocks"].append(
            {
                "subheading": subheading or None,
                "text": body,
            }
        )
        section_lines = []

    for page_text in pages:
        page_lines = [
            normalize_whitespace(raw_line)
            for raw_line in page_text.split("\n")
            if normalize_whitespace(raw_line) and not is_ussg_page_artifact(normalize_whitespace(raw_line), year)
        ]
        line_index = 0
        while line_index < len(page_lines):
            line = page_lines[line_index]
            if line_index == 0 and section_heading and is_bare_ussg_citation_line(line):
                line_index += 1
                continue
            if is_probable_ussg_chapter_heading(line):
                flush_current()
                chapter_heading = line
                part_heading = ""
                section_heading = ""
                subheading = ""
                line_index += 1
                continue
            if is_probable_ussg_part_heading(line):
                flush_current()
                part_heading = line
                section_heading = ""
                subheading = ""
                line_index += 1
                continue
            if is_probable_ussg_section_start(line):
                flush_current()
                citation_text = extract_ussg_citation(line) or line
                citation = extract_normalized_ussg_citation(line) or line.rstrip(".")
                remainder = line[len(citation_text) :].strip()
                title_lines: list[str] = [remainder] if remainder and re.match(r"^[A-Z\[]", remainder) else []
                lookahead = line_index + 1
                while lookahead < len(page_lines):
                    next_line = page_lines[lookahead]
                    if (
                        is_probable_ussg_chapter_heading(next_line)
                        or is_probable_ussg_part_heading(next_line)
                        or is_probable_ussg_section_start(next_line)
                        or is_subheading(next_line)
                        or re.match(r"^\([a-z]\)", next_line, re.IGNORECASE)
                    ):
                        break
                    title_lines.append(next_line)
                    lookahead += 1

                if title_lines:
                    section_heading = f"{citation}. {' '.join(title_lines)}"
                    line_index = lookahead
                else:
                    section_heading = citation
                    line_index += 1
                subheading = ""
                section_lines = []
                continue
            if is_subheading(line) and section_heading:
                flush_current()
                subheading = line
                section_lines = []
                line_index += 1
                continue
            if section_heading and line not in {document_title, str(year), "United States Sentencing Commission"}:
                section_lines.append(line)
            line_index += 1

    flush_current()
    return list(sections.values())


@lru_cache(maxsize=16)
def _build_ussg_toc_index_cached(docintel_path: str, year: int) -> dict[str, Any]:
    payload = json.loads(Path(docintel_path).read_text(encoding="utf-8"))
    raw_pages = payload.get("pages")
    if not isinstance(raw_pages, list):
        return {
            "source_type": "ussg_toc",
            "source_year": year,
            "document_title": f"USSG Guidelines Manual ({year}) Table of Contents",
            "chapter_count": 0,
            "chapters": [],
        }

    chapters: list[dict[str, Any]] = []
    current_chapter: dict[str, Any] | None = None
    current_part: dict[str, Any] | None = None
    in_toc = False

    for raw_page in raw_pages:
        page_text = raw_page.get("text") if isinstance(raw_page, dict) else raw_page
        text = str(page_text or "")
        if not in_toc and "TABLE OF CONTENTS" not in text:
            continue
        in_toc = True

        page_lines = [normalize_whitespace(line) for line in text.splitlines() if normalize_whitespace(line)]
        if any(is_probable_ussg_toc_end(line) for line in page_lines):
            break

        index = 0
        while index < len(page_lines):
            line = page_lines[index]
            if (
                line == "TABLE OF CONTENTS"
                or is_ussg_page_artifact(line, year)
                or is_ussg_toc_page_number(line)
            ):
                index += 1
                continue

            if is_probable_ussg_toc_chapter_heading(line):
                chapter_title, next_index = collect_ussg_toc_heading(page_lines, index + 1) if index + 1 < len(page_lines) else ("", index + 1)
                current_chapter = {
                    "chapter_id": build_ussg_chapter_id(year, line),
                    "source_type": "ussg_toc",
                    "source_year": year,
                    "document_title": f"USSG Guidelines Manual ({year}) Table of Contents",
                    "chapter_heading": line,
                    "chapter_title": chapter_title,
                    "display_heading": f"{line}. {chapter_title}" if chapter_title else line,
                    "part_count": 0,
                    "subheading_count": 0,
                    "parts": [],
                }
                chapters.append(current_chapter)
                current_part = None
                index = next_index
                continue

            if current_chapter and is_probable_ussg_part_heading(line):
                part_heading, next_index = collect_ussg_toc_heading(page_lines, index)
                current_part = {
                    "part_heading": part_heading,
                    "subheadings": [],
                }
                current_chapter["parts"].append(current_part)
                current_chapter["part_count"] += 1
                index = next_index
                continue

            if current_part and is_probable_ussg_toc_subheading(line):
                subheading, next_index = collect_ussg_toc_heading(page_lines, index)
                current_part["subheadings"].append(
                    {
                        "subheading_id": build_ussg_subheading_id(
                            year,
                            current_chapter["chapter_heading"],
                            current_part["part_heading"],
                            subheading,
                        ),
                        "subheading_heading": subheading,
                        "section_prefix": build_ussg_subheading_prefix(
                            current_chapter["chapter_heading"],
                            current_part["part_heading"],
                            subheading,
                        ),
                    }
                )
                current_chapter["subheading_count"] += 1
                index = next_index
                continue

            index += 1

    return {
        "source_type": "ussg_toc",
        "source_year": year,
        "document_title": f"USSG Guidelines Manual ({year}) Table of Contents",
        "chapter_count": len(chapters),
        "chapters": chapters,
    }


@lru_cache(maxsize=16)
def _build_title18_structure_cached(html_path: str, year: int, execution_env: str, manuals_root: str) -> dict[str, Any]:
    config = LegalReactV2Config(
        execution_env=execution_env,
        manuals_root=Path(manuals_root),
        title18_root=Path(manuals_root),
        ussg_docintel_text_root=None,
    )
    source_html = read_text(Path(html_path), config, encoding="utf-8", errors="ignore")
    headings = list(HEADING_PATTERN.finditer(source_html))

    document_title = f"Title 18 United States Code ({year})"
    part_heading = ""
    chapter_heading = ""
    chapters: dict[str, dict[str, Any]] = {}
    sections: list[dict[str, Any]] = []

    for index, match in enumerate(headings):
        heading_class = match.group("class").lower()
        heading_text = clean_heading_text(match.group("html"))
        next_start = headings[index + 1].start() if index + 1 < len(headings) else len(source_html)

        if heading_class == "usc-title-head":
            document_title = heading_text
            continue
        if heading_class == "part-head":
            part_heading = heading_text
            chapter_heading = ""
            continue
        if heading_class == "chapter-head":
            chapter_heading = heading_text
            chapter_id = build_title18_chapter_id(year, heading_heading := chapter_heading)
            chapters.setdefault(
                chapter_id,
                {
                    "chapter_id": chapter_id,
                    "source_type": "usc_title18",
                    "source_year": year,
                    "document_title": document_title,
                    "part_heading": part_heading or None,
                    "chapter_heading": heading_heading,
                    "section_count": 0,
                    "sections": [],
                },
            )
            continue
        if heading_class != "section-head":
            continue

        body_html = source_html[match.end() : next_start]
        body_text = strip_tags(body_html)
        if not body_text:
            continue
        citation = extract_title18_citation(heading_text)
        entry_id = build_title18_section_id(year, citation, heading_text)
        chapter_id = build_title18_chapter_id(year, chapter_heading or "unscoped")
        chapter_payload = chapters.setdefault(
            chapter_id,
            {
                "chapter_id": chapter_id,
                "source_type": "usc_title18",
                "source_year": year,
                "document_title": document_title,
                "part_heading": part_heading or None,
                "chapter_heading": chapter_heading or None,
                "section_count": 0,
                "sections": [],
            },
        )
        section_payload = {
            "entry_id": entry_id,
            "source_type": "usc_title18",
            "source_year": year,
            "document_title": document_title,
            "part_heading": part_heading or None,
            "chapter_heading": chapter_heading or None,
            "section_heading": heading_text,
            "citation": citation,
            "text": body_text,
        }
        sections.append(section_payload)
        chapter_payload["sections"].append(
            {
                "entry_id": entry_id,
                "citation": citation,
                "section_heading": heading_text,
            }
        )
        chapter_payload["section_count"] += 1

    return {
        "source_type": "usc_title18",
        "source_year": year,
        "document_title": document_title,
        "chapter_count": len(chapters),
        "chapters": list(chapters.values()),
        "sections": sections,
    }


def list_title18_chapters(year: int, config: LegalReactV2Config | None = None) -> dict[str, Any]:
    """Programmatic helper for browsing Title 18 chapter headings by year.

    This helper is the implementation behind the eventual ReAct tool. It resolves
    the configured Title 18 source, loads the requested edition, and returns a
    structural chapter index without exposing any statute section text.

    Args:
        year: Title 18 edition year to inspect.
        config: Optional preloaded runtime configuration. This is for local code
            paths and should not be exposed to the language model as a tool input.

    Returns:
        A dictionary with the edition metadata and a `chapters` list. Each chapter
        row contains `chapter_id`, `part_heading`, and `chapter_heading` so an
        agent can decide what to inspect next.

    Raises:
        KeyError: If the requested year is not available under the configured
            Title 18 source root.
    """
    resolved_config = config or load_config()
    html_path = require_title18_html_path(resolved_config, year)
    structure = _build_title18_structure_cached(
        str(html_path),
        year,
        resolved_config.execution_env,
        str(resolved_config.manuals_root),
    )
    return {
        "source_type": structure["source_type"],
        "source_year": structure["source_year"],
        "document_title": structure["document_title"],
        "chapter_count": structure["chapter_count"],
        "chapters": [
            {
                "chapter_id": chapter["chapter_id"],
                "part_heading": chapter["part_heading"],
                "chapter_heading": chapter["chapter_heading"],
                "section_count": chapter["section_count"],
            }
            for chapter in structure["chapters"]
        ],
    }


def open_title18_chapter(year: int, chapter_id: str, config: LegalReactV2Config | None = None) -> dict[str, Any]:
    """Programmatic helper for opening one Title 18 chapter by chapter_id.

    This helper resolves one exact chapter returned by `list_title18_chapters`
    for the same year and returns the immediate section list beneath it without
    exposing statute body text.

    Args:
        year: Title 18 edition year to inspect.
        chapter_id: Exact chapter identifier returned by `list_title18_chapters`.
        config: Optional preloaded runtime configuration. This is for local code
            paths and should not be exposed to the language model as a tool input.

    Returns:
        A dictionary for the selected chapter with `part_heading`,
        `chapter_heading`, `section_count`, and `sections`. Each section row
        includes `entry_id`, `citation`, and `section_heading`.

    Raises:
        KeyError: If the requested year is unavailable.
    """
    resolved_config = config or load_config()
    html_path = require_title18_html_path(resolved_config, year)
    structure = _build_title18_structure_cached(
        str(html_path),
        year,
        resolved_config.execution_env,
        str(resolved_config.manuals_root),
    )
    for chapter in structure["chapters"]:
        if chapter["chapter_id"] == chapter_id:
            return chapter
    return {
        "source_type": "usc_title18",
        "source_year": year,
        "chapter_id": chapter_id,
        "status": "not_found",
        "message": f"Title 18 chapter not found for year {year}: {chapter_id}",
    }


def open_title18_section(year: int, entry_id: str, config: LegalReactV2Config | None = None) -> dict[str, Any]:
    """Programmatic helper for opening one Title 18 section by entry_id.

    This helper resolves one exact section returned from a chapter open step for
    the same year and returns the full statute text with its chapter context.

    Args:
        year: Title 18 edition year to inspect.
        entry_id: Exact section identifier returned by `open_title18_chapter`.
        config: Optional preloaded runtime configuration. This is for local code
            paths and should not be exposed to the language model as a tool input.

    Returns:
        A dictionary with the selected section metadata and full extracted text.

    Raises:
        KeyError: If the requested year is unavailable.
    """
    resolved_config = config or load_config()
    html_path = require_title18_html_path(resolved_config, year)
    structure = _build_title18_structure_cached(
        str(html_path),
        year,
        resolved_config.execution_env,
        str(resolved_config.manuals_root),
    )
    for section in structure["sections"]:
        if section["entry_id"] == entry_id:
            return section
    return {
        "source_type": "usc_title18",
        "source_year": year,
        "entry_id": entry_id,
        "status": "not_found",
        "message": f"Title 18 section not found for year {year}: {entry_id}",
    }


def list_appendix_a_entries(year: int, statute_prefix: str | None = None, limit: int = 50, config: LegalReactV2Config | None = None) -> dict[str, Any]:
    """Programmatic helper for browsing USSG Appendix A statutory index entries.

    This helper exposes the statute-to-guideline mapping in Appendix A for one
    Guidelines edition year. It is the bridge between the statute of conviction
    and the candidate Chapter Two guideline sections.

    Args:
        year: USSG edition year to inspect.
        statute_prefix: Optional statute search string such as `18 U.S.C. § 1546`,
            `18 U.S.C. 1546`, `1546`, or `18 U.S.C. § 793(g)`. The lookup is
            normalized and used as a search filter over the Appendix A statute
            citations, so it can return multiple matching rows.
        limit: Maximum number of rows to return.
        config: Optional preloaded runtime configuration. This is for local code
            paths and should not be exposed to the language model as a tool input.

    Returns:
        A dictionary with Appendix A search results. Each row includes
        `entry_id`, `statute_citation`, and `guideline_sections`.
    """
    resolved_config = config or load_config()
    docintel_path = require_ussg_docintel_path(resolved_config, year)
    statute_filters = build_statute_lookup_variants(statute_prefix)
    structured_query = is_structured_statute_query(statute_prefix)

    entries = _build_appendix_a_entries_cached(str(docintel_path))

    def entry_token(entry: dict[str, Any]) -> str:
        return normalize_lookup_token(entry.get("statute_citation"))

    filtered_entries = entries
    if statute_filters:
        matched_entries: list[dict[str, Any]] = []
        for variant in statute_filters:
            exact_matches = [entry for entry in entries if entry_token(entry) == variant]
            if exact_matches:
                matched_entries = exact_matches
                break
        if not matched_entries:
            for variant in statute_filters:
                prefix_matches = [entry for entry in entries if matches_parent_statute_token(entry_token(entry), variant)]
                if prefix_matches:
                    matched_entries = prefix_matches
                    break
        if not matched_entries and not structured_query:
            for variant in statute_filters:
                contains_matches = [entry for entry in entries if variant in entry_token(entry)]
                if contains_matches:
                    matched_entries = contains_matches
                    break
        filtered_entries = matched_entries

    results: list[dict[str, Any]] = []
    for entry in filtered_entries:
        results.append(
            {
                "entry_id": entry["entry_id"],
                "statute_citation": entry["statute_citation"],
                "guideline_sections": entry["guideline_sections"],
            }
        )
        if len(results) >= limit:
            break

    return {
        "source_type": "ussg_appendix_a",
        "source_year": year,
        "count": len(results),
        "results": results,
    }


def open_ussg_section(year: int, section_citation: str, config: LegalReactV2Config | None = None) -> dict[str, Any]:
    """Programmatic helper for opening one USSG section by citation.

    This helper resolves one exact guideline section in the USSG manual for the
    requested year and returns the section text together with any commentary or
    application-note blocks parsed under that section.

    Args:
        year: USSG edition year to inspect.
        section_citation: Exact or normalized guideline citation such as
            `§2L2.2`, `2L2.2`, or `§2B1.1`.
        config: Optional preloaded runtime configuration. This is for local code
            paths and should not be exposed to the language model as a tool input.

    Returns:
        A dictionary with the section metadata and `blocks`. Each item in
        `blocks` includes a `subheading` and the extracted `text` for that block.
    """
    resolved_config = config or load_config()
    docintel_path = require_ussg_docintel_path(resolved_config, year)
    requested = normalize_lookup_token(section_citation)

    for section in _build_ussg_section_index_cached(str(docintel_path), year):
        if requested in {
            normalize_lookup_token(section.get("citation")),
            normalize_lookup_token(section.get("section_heading")),
            normalize_lookup_token(section.get("entry_id")),
        }:
            combined_text = "\n\n".join(
                block["text"] if not block.get("subheading") else f"{block['subheading']}\n{block['text']}"
                for block in section["blocks"]
            )
            return {
                **section,
                "text": combined_text,
            }

    return {
        "source_type": "ussg",
        "source_year": year,
        "section_citation": section_citation,
        "status": "not_found",
        "message": f"USSG section not found for year {year}: {section_citation}",
    }


def list_ussg_chapters(year: int, config: LegalReactV2Config | None = None) -> dict[str, Any]:
    """Programmatic helper for browsing the full USSG table of contents by year.

    This helper exposes the USSG table of contents for one edition year in one
    call, including chapters, their parts, and the subheadings listed under
    each part.

    Args:
        year: Guidelines edition year to inspect.
        config: Optional preloaded runtime configuration. This is for local code
            paths and should not be exposed to the language model as a tool input.

    Returns:
        A dictionary with the edition metadata and a nested `chapters` list.
        Each chapter row includes its `parts`, and each part row includes
        `subheadings` with stable `subheading_id` values for later follow-up.
    """
    resolved_config = config or load_config()
    docintel_path = require_ussg_docintel_path(resolved_config, year)
    structure = _build_ussg_toc_index_cached(str(docintel_path), year)
    return {
        "source_type": structure["source_type"],
        "source_year": structure["source_year"],
        "document_title": structure["document_title"],
        "chapter_count": structure["chapter_count"],
        "chapters": [
            {
                "chapter_id": chapter["chapter_id"],
                "chapter_heading": chapter["chapter_heading"],
                "chapter_title": chapter["chapter_title"],
                "part_count": chapter["part_count"],
                "subheading_count": chapter["subheading_count"],
                "parts": [
                    {
                        "part_heading": part["part_heading"],
                        "subheadings": [
                            {
                                "subheading_id": subheading["subheading_id"],
                                "subheading_heading": subheading["subheading_heading"],
                            }
                            for subheading in part["subheadings"]
                        ],
                    }
                    for part in chapter["parts"]
                ],
            }
            for chapter in structure["chapters"]
        ],
    }


def open_ussg_subheading(year: int, subheading_id: str, config: LegalReactV2Config | None = None) -> dict[str, Any]:
    """Programmatic helper for opening one USSG TOC subheading into its sections.

    This helper resolves one exact subheading returned by `list_ussg_chapters`
    for the same year and returns the guideline sections that belong under that
    subheading.

    Args:
        year: Guidelines edition year to inspect.
        subheading_id: Exact subheading identifier returned by
            `list_ussg_chapters`.
        config: Optional preloaded runtime configuration. This is for local code
            paths and should not be exposed to the language model as a tool input.

    Returns:
        A dictionary for the selected subheading with chapter and part context,
        plus a `sections` list. Each section row includes `entry_id`, `citation`,
        and `section_heading`.
    """
    resolved_config = config or load_config()
    docintel_path = require_ussg_docintel_path(resolved_config, year)
    toc = _build_ussg_toc_index_cached(str(docintel_path), year)

    for chapter in toc["chapters"]:
        for part in chapter["parts"]:
            for subheading in part["subheadings"]:
                if normalize_lookup_token(subheading["subheading_id"]) not in {
                    normalize_lookup_token(subheading_id),
                    normalize_lookup_token(subheading.get("section_prefix")),
                    normalize_lookup_token(subheading.get("subheading_heading")),
                }:
                    continue

                prefix = subheading.get("section_prefix")
                sections = []
                for section in _build_ussg_section_index_cached(str(docintel_path), year):
                    citation = str(section.get("citation") or "")
                    if prefix and normalize_lookup_token(citation).startswith(normalize_lookup_token(prefix)):
                        sections.append(
                            {
                                "entry_id": section["entry_id"],
                                "citation": section["citation"],
                                "section_heading": section["section_heading"],
                            }
                        )

                return {
                    "source_type": "ussg_toc",
                    "source_year": year,
                    "chapter_heading": chapter["chapter_heading"],
                    "chapter_title": chapter["chapter_title"],
                    "part_heading": part["part_heading"],
                    "subheading_id": subheading["subheading_id"],
                    "subheading_heading": subheading["subheading_heading"],
                    "section_prefix": prefix,
                    "section_count": len(sections),
                    "sections": sections,
                }

    return {
        "source_type": "ussg_toc",
        "source_year": year,
        "subheading_id": subheading_id,
        "status": "not_found",
        "message": f"USSG subheading not found for year {year}: {subheading_id}",
    }


def open_ussg_chapter(year: int, chapter_id: str, config: LegalReactV2Config | None = None) -> dict[str, Any]:
    """Programmatic helper for opening one USSG chapter from the table of contents.

    This helper resolves one exact chapter returned by `list_ussg_chapters`
    for the same year and returns the part headings and subheadings listed
    beneath it in the table of contents.

    Args:
        year: Guidelines edition year to inspect.
        chapter_id: Exact chapter identifier returned by `list_ussg_chapters`.
        config: Optional preloaded runtime configuration. This is for local code
            paths and should not be exposed to the language model as a tool input.

    Returns:
        A dictionary for the selected chapter with `chapter_heading`,
        `chapter_title`, and `parts`. Each part row includes `part_heading`
        and its TOC `subheadings`.
    """
    resolved_config = config or load_config()
    structure = _build_ussg_toc_index_cached(str(require_ussg_docintel_path(resolved_config, year)), year)
    for chapter in structure["chapters"]:
        if chapter["chapter_id"] == chapter_id:
            return chapter
    return {
        "source_type": "ussg_toc",
        "source_year": year,
        "chapter_id": chapter_id,
        "status": "not_found",
        "message": f"USSG chapter not found for year {year}: {chapter_id}",
    }


def build_legal_manual_tools(config: LegalReactV2Config) -> list[Any]:
    def list_title18_chapters_tool(year: int) -> dict[str, Any]:
        """Browse Title 18 chapter headings for one edition year.

        Input:
            year: Edition year of Title 18 to inspect.

        Returns:
            A dictionary with `document_title`, `source_year`, `chapter_count`, and
            `chapters`. Each item in `chapters` includes:
            - `chapter_id`: Stable identifier for later follow-up tools.
            - `part_heading`: Enclosing part heading when present.
            - `chapter_heading`: The Title 18 chapter heading text.

        Notes:
            - Use the exact case year when possible.
            - This tool is structural only. It does not open chapters, return section text,
              identify the controlling statute, or perform sentencing reasoning.
        """
        return list_title18_chapters(year=year, config=config)

    def open_title18_chapter_tool(year: int, chapter_id: str) -> dict[str, Any]:
        """Open one Title 18 chapter and list the section headings directly under it.

        Input:
            year: Edition year of Title 18 to inspect.
            chapter_id: Exact chapter identifier returned by list_title18_chapters_tool.

        Returns:
            A dictionary with the selected chapter metadata and a `sections` list.
            Each item in `sections` includes:
            - `entry_id`: Stable identifier for later section-level tools.
            - `citation`: Statute citation when one is detected.
            - `section_heading`: The Title 18 section heading text.

        Notes:
            - Use a chapter_id returned for the same year.
            - This tool is structural only. It does not return statute body text,
              identify the controlling section, or perform sentencing reasoning.
        """
        return open_title18_chapter(year=year, chapter_id=chapter_id, config=config)

    def open_title18_section_tool(year: int, entry_id: str) -> dict[str, Any]:
        """Open one Title 18 section and return the full statute text.

        Input:
            year: Edition year of Title 18 to inspect.
            entry_id: Exact section identifier returned by open_title18_chapter_tool.

        Returns:
            A dictionary with the selected section metadata and full extracted text.
            The response includes:
            - `citation`: Statute citation when one is detected.
            - `part_heading`: Enclosing part heading when present.
            - `chapter_heading`: Enclosing chapter heading.
            - `section_heading`: The Title 18 section heading text.
            - `text`: The extracted statute body text.

        Notes:
            - Use an entry_id returned for the same year.
            - This tool exposes the manual text for one exact section. It does not
              identify the controlling statute or perform sentencing reasoning.
        """
        return open_title18_section(year=year, entry_id=entry_id, config=config)

    def list_ussg_chapters_tool(year: int) -> dict[str, Any]:
        """Browse the full USSG table of contents in one call.

        Input:
            year: Guidelines edition year to inspect.

        Returns:
            A dictionary with `document_title`, `source_year`, `chapter_count`, and
            nested `chapters`. Each chapter includes:
            - `chapter_id`: Stable identifier for structural reference.
            - `chapter_heading`: The top-level chapter label such as `CHAPTER TWO`.
            - `chapter_title`: The chapter title text such as `OFFENSE CONDUCT`.
            - `parts`: A list of TOC part rows.

            Each part row includes:
            - `part_heading`: The TOC part heading.
            - `subheadings`: A list of subheading rows.

            Each subheading row includes:
            - `subheading_id`: Stable identifier for the follow-up opener.
            - `subheading_heading`: The TOC subheading text such as `1. HOMICIDE`.
            - `section_prefix`: The citation prefix used by the follow-up opener.

        Notes:
            - This tool is structural only. It does not open section text or perform
              sentencing reasoning.
        """
        return list_ussg_chapters(year=year, config=config)

    def open_ussg_subheading_tool(year: int, subheading_id: str) -> dict[str, Any]:
        """Open one USSG TOC subheading and list its guideline sections.

        Input:
            year: Guidelines edition year to inspect.
            subheading_id: Exact subheading identifier returned by
                list_ussg_chapters_tool.

        Returns:
            A dictionary with the selected subheading metadata and a `sections` list.
            Each item in `sections` includes:
            - `entry_id`: Stable identifier for later exact section opening.
            - `citation`: Guideline citation such as `§2A1.1`.
            - `section_heading`: Full guideline section heading text.

        Notes:
            - Use a subheading_id returned for the same year.
            - This tool is structural only. It does not return full guideline text.
        """
        return open_ussg_subheading(year=year, subheading_id=subheading_id, config=config)

    def list_appendix_a_entries_tool(year: int, statute_prefix: str | None = None, limit: int = 20) -> dict[str, Any]:
        """Search USSG Appendix A by statute citation.

        Input:
            year: Guidelines edition year to inspect.
            statute_prefix: Optional statute citation or partial citation to search
                within Appendix A.
                Example inputs:
                - `18 U.S.C. § 1546`
                - `18 U.S.C. 1546`
                - `1546`
                - `18 U.S.C. § 793(g)`
            limit: Maximum number of matching Appendix A rows to return.

        Returns:
            A dictionary with `count` and `results`. Each item in `results` includes:
            - `entry_id`: Stable identifier for the Appendix A row.
            - `statute_citation`: The statute citation as listed in Appendix A.
            - `guideline_sections`: Candidate Chapter Two guideline sections.

        Notes:
            - Use this after identifying the count-of-conviction statute.
            - This is a search tool, not an exact-id opener. It returns all matching
              Appendix A rows up to `limit` after normalizing the citation text.
            - This tool exposes the Appendix A mapping only. It does not choose among
              multiple guideline sections or perform offense-level reasoning.
        """
        return list_appendix_a_entries(year=year, statute_prefix=statute_prefix, limit=limit, config=config)

    def open_ussg_section_tool(year: int, section_citation: str) -> dict[str, Any]:
        """Open one USSG guideline section by citation.

        Input:
            year: Guidelines edition year to inspect.
            section_citation: Guideline citation returned from Appendix A or nearby
                guideline text.
                Example inputs:
                - `§2L2.2`
                - `2L2.2`
                - `§2B1.1`

        Returns:
            A dictionary with the selected guideline section and its parsed text blocks.
            The response includes:
            - `citation`: The guideline citation.
            - `section_heading`: The guideline section heading text.
            - `chapter_heading`: Enclosing chapter heading.
            - `part_heading`: Enclosing part heading.
            - `blocks`: Parsed blocks under the section, including commentary and
              application notes when present.
            - `text`: Combined text for the full section.

        Notes:
            - Use this after identifying a candidate guideline section from Appendix A.
            - This tool opens one exact guideline section. It does not choose among
              multiple candidate sections or perform offense-level reasoning.
        """
        return open_ussg_section(year=year, section_citation=section_citation, config=config)

    return [
        list_title18_chapters_tool,
        open_title18_chapter_tool,
        open_title18_section_tool,
        list_ussg_chapters_tool,
        open_ussg_subheading_tool,
        list_appendix_a_entries_tool,
        open_ussg_section_tool,
    ]