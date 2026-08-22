from __future__ import annotations

from functools import lru_cache
import re
import xml.etree.ElementTree as ET

from pypdf import PdfReader

from .config import ICDReactConfig


NOTE_TAGS = [
    "includes",
    "inclusionTerm",
    "excludes1",
    "excludes2",
    "codeFirst",
    "useAdditionalCode",
    "codeAlso",
    "sevenChrNote",
]


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def text_from_element(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return normalize_whitespace(" ".join(part.strip() for part in element.itertext() if part and part.strip()))


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def normalize_code_token(value: str | None) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value or "").upper())


def build_main_term_entry_id(letter: str, title: str) -> str:
    return f"{letter}|{title}"


def collect_note_groups(element: ET.Element | None) -> dict[str, list[str]]:
    if element is None:
        return {}
    note_groups: dict[str, list[str]] = {}
    for tag_name in NOTE_TAGS:
        notes = [text_from_element(note) for note in element.findall(f"{tag_name}/note") if text_from_element(note)]
        if notes:
            note_groups[tag_name] = notes
    return note_groups


def serialize_index_term(term: ET.Element, parent_titles: list[str] | None = None) -> dict[str, object]:
    lineage = list(parent_titles or [])
    title = text_from_element(term.find("title"))
    path = lineage + [title] if title else lineage
    children = [serialize_index_term(child, path) for child in term.findall("term")]
    payload: dict[str, object] = {
        "title": title,
        "path": path,
        "code": text_from_element(term.find("code")) or None,
        "see": text_from_element(term.find("see")) or None,
        "see_also": text_from_element(term.find("seeAlso")) or None,
        "children": children,
    }
    return payload


@lru_cache(maxsize=4)
def load_index_root(index_xml_path: str) -> ET.Element:
    return ET.parse(index_xml_path).getroot()


@lru_cache(maxsize=4)
def load_tabular_root(tabular_xml_path: str) -> ET.Element:
    return ET.parse(tabular_xml_path).getroot()


def serialize_tabular_section_ref(section_ref: ET.Element) -> dict[str, object]:
    return {
        "section_id": section_ref.get("id") or None,
        "first_code": section_ref.get("first") or None,
        "last_code": section_ref.get("last") or None,
        "description": text_from_element(section_ref),
    }


def serialize_top_level_diag(diag: ET.Element) -> dict[str, object]:
    return {
        "code": text_from_element(diag.find("name")) or None,
        "description": text_from_element(diag.find("desc")) or None,
        "child_count": len(diag.findall("diag")),
        "note_groups": collect_note_groups(diag),
    }


def tabular_chapter_matches_prefix(chapter_id: str, code_range: str | None, section_refs: list[dict[str, object]], prefix_filter: str) -> bool:
    if not prefix_filter:
        return True
    normalized_prefix = normalize_code_token(prefix_filter)
    if not normalized_prefix:
        return True
    if normalized_prefix == chapter_id.upper():
        return True
    prefix_head = normalized_prefix[0]
    for section_ref in section_refs:
        first_code = normalize_code_token(section_ref.get("first_code"))
        last_code = normalize_code_token(section_ref.get("last_code"))
        if first_code and last_code and first_code[0] <= prefix_head <= last_code[0]:
            return True
    return False


@lru_cache(maxsize=4)
def build_guideline_sections(guidelines_pdf_path: str) -> list[dict[str, object]]:
    reader = PdfReader(guidelines_pdf_path)
    sections: list[dict[str, object]] = []
    top_heading = ""
    chapter_heading = ""
    subsection_heading = ""
    subsection_lines: list[str] = []
    start_page = 1
    end_page = 1

    def flush_current() -> None:
        nonlocal subsection_heading, subsection_lines, start_page, end_page
        if not subsection_heading or not subsection_lines:
            subsection_lines = []
            return
        path_parts = [part for part in [top_heading, chapter_heading, subsection_heading] if part]
        section_text = normalize_whitespace("\n".join(subsection_lines))
        sections.append(
            {
                "section_id": slugify(" > ".join(path_parts)),
                "path": path_parts,
                "top_heading": top_heading or None,
                "chapter_heading": chapter_heading or None,
                "subsection_heading": subsection_heading,
                "page_start": start_page,
                "page_end": end_page,
                "preview": section_text[:300],
                "text": section_text,
            }
        )
        subsection_lines = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        for raw_line in page_text.splitlines():
            line = normalize_whitespace(raw_line)
            if not line:
                continue
            if re.match(r"^Section [IVX]+", line):
                flush_current()
                top_heading = line
                chapter_heading = ""
                subsection_heading = ""
                continue
            if re.match(r"^\d+\.\s+Chapter\s+\d+:", line):
                flush_current()
                chapter_heading = line
                subsection_heading = ""
                continue
            if re.match(r"^[a-z]\.\s+", line):
                flush_current()
                subsection_heading = line
                subsection_lines = []
                start_page = page_number
                end_page = page_number
                continue
            if subsection_heading:
                subsection_lines.append(line)
                end_page = page_number

    flush_current()
    return sections


def list_index_main_terms(config: ICDReactConfig, letter: str | None = None, prefix: str | None = None, limit: int = 50) -> dict[str, object]:
    """Browse high-level ICD Alphabetic Index entries by letter or prefix.

    This is the main entry point for navigating the Index without forcing a coding
    workflow. The result set contains one row per XML `mainTerm`, including any
    direct code or cross-reference fields already attached to that node.
    """
    root = load_index_root(str(config.index_xml_path))
    letter_filter = (letter or "").strip().upper()
    prefix_filter = (prefix or "").strip().lower()
    results: list[dict[str, object]] = []
    for letter_node in root.findall("letter"):
        current_letter = text_from_element(letter_node.find("title")).upper()
        if letter_filter and current_letter != letter_filter:
            continue
        for main_term in letter_node.findall("mainTerm"):
            title = text_from_element(main_term.find("title"))
            if prefix_filter and not title.lower().startswith(prefix_filter):
                continue
            results.append(
                {
                    "entry_id": build_main_term_entry_id(current_letter, title),
                    "letter": current_letter,
                    "title": title,
                    "code": text_from_element(main_term.find("code")) or None,
                    "see": text_from_element(main_term.find("see")) or None,
                    "see_also": text_from_element(main_term.find("seeAlso")) or None,
                    "child_count": len(main_term.findall("term")),
                }
            )
            if len(results) >= limit:
                return {
                    "manual": "ICD-10-CM Index",
                    "letter": letter_filter or None,
                    "prefix": prefix or None,
                    "count": len(results),
                    "results": results,
                }
    return {
        "manual": "ICD-10-CM Index",
        "letter": letter_filter or None,
        "prefix": prefix or None,
        "count": len(results),
        "results": results,
    }


def open_index_term(config: ICDReactConfig, entry_id: str) -> dict[str, object]:
    """Open one ICD Alphabetic Index main term subtree by entry_id.

    The entry_id must come from list_index_main_terms. The returned payload keeps
    the manual structure intact, including nested child terms, direct codes, and
    `see` or `seeAlso` references.
    """
    letter, title = entry_id.split("|", maxsplit=1)
    root = load_index_root(str(config.index_xml_path))
    for letter_node in root.findall("letter"):
        current_letter = text_from_element(letter_node.find("title")).upper()
        if current_letter != letter.upper().strip():
            continue
        for main_term in letter_node.findall("mainTerm"):
            current_title = text_from_element(main_term.find("title"))
            if current_title == title:
                return {
                    "manual": "ICD-10-CM Index",
                    "entry_id": build_main_term_entry_id(current_letter, current_title),
                    "letter": current_letter,
                    "title": current_title,
                    "code": text_from_element(main_term.find("code")) or None,
                    "see": text_from_element(main_term.find("see")) or None,
                    "see_also": text_from_element(main_term.find("seeAlso")) or None,
                    "children": [serialize_index_term(child, [current_title]) for child in main_term.findall("term")],
                }
    raise KeyError(f"Index entry not found: {entry_id}")


def list_tabular_chapters(config: ICDReactConfig, code_prefix: str | None = None, limit: int = 50) -> dict[str, object]:
    """Browse the top-level ICD Tabular List chapter headings.

    Use this to inspect high-level families such as `Certain infectious and
    parasitic diseases (A00-B99)` before narrowing to a section or a specific code.
    The optional code_prefix filter is intended for browsing by code family, not
    free-text semantic matching.
    """
    root = load_tabular_root(str(config.tabular_xml_path))
    prefix_filter = (code_prefix or "").strip().upper()
    results: list[dict[str, object]] = []
    for chapter in root.findall("chapter"):
        chapter_id = text_from_element(chapter.find("name"))
        chapter_description = text_from_element(chapter.find("desc"))
        section_refs = [serialize_tabular_section_ref(section_ref) for section_ref in chapter.findall("sectionIndex/sectionRef")]
        code_range = None
        if section_refs:
            first_code = str(section_refs[0]["first_code"] or "")
            last_code = str(section_refs[-1]["last_code"] or "")
            if first_code and last_code:
                code_range = f"{first_code}-{last_code}"
        if not tabular_chapter_matches_prefix(chapter_id=chapter_id, code_range=code_range, section_refs=section_refs, prefix_filter=prefix_filter):
            continue
        results.append(
            {
                "chapter_id": chapter_id,
                "description": chapter_description,
                "code_range": code_range,
                "section_count": len(section_refs),
            }
        )
        if len(results) >= limit:
            break
    return {
        "manual": "ICD-10-CM Tabular",
        "code_prefix": code_prefix or None,
        "count": len(results),
        "results": results,
    }


def open_tabular_chapter(config: ICDReactConfig, chapter_id: str) -> dict[str, object]:
    """Open one ICD Tabular chapter by its chapter id.

    This exposes chapter-level notes plus the next browse level, namely the
    section headings and code ranges listed under `sectionRef` entries.
    """
    target_chapter_id = str(chapter_id or "").strip()
    if not target_chapter_id:
        raise ValueError("chapter_id is required")

    root = load_tabular_root(str(config.tabular_xml_path))
    for chapter in root.findall("chapter"):
        current_id = text_from_element(chapter.find("name"))
        if current_id != target_chapter_id:
            continue
        section_refs = [serialize_tabular_section_ref(section_ref) for section_ref in chapter.findall("sectionIndex/sectionRef")]
        return {
            "manual": "ICD-10-CM Tabular",
            "chapter_id": current_id,
            "description": text_from_element(chapter.find("desc")),
            "note_groups": collect_note_groups(chapter),
            "sections": section_refs,
        }
    raise KeyError(f"Tabular chapter not found: {chapter_id}")


def open_tabular_section(config: ICDReactConfig, section_id: str) -> dict[str, object]:
    """Open one ICD Tabular section by section id.

    This is the browse step between a chapter heading and a concrete code entry.
    The response includes the section description, any local notes, and the
    top-level codes directly under that section.
    """
    target_section_id = str(section_id or "").strip()
    if not target_section_id:
        raise ValueError("section_id is required")

    root = load_tabular_root(str(config.tabular_xml_path))
    parent_map = {child: parent for parent in root.iter() for child in parent}
    for section in root.iterfind(".//section"):
        current_id = str(section.get("id") or "")
        if current_id != target_section_id:
            continue
        chapter = parent_map.get(section)
        return {
            "manual": "ICD-10-CM Tabular",
            "section_id": current_id,
            "description": text_from_element(section.find("desc")),
            "chapter_id": None if chapter is None else text_from_element(chapter.find("name")),
            "chapter_description": None if chapter is None else text_from_element(chapter.find("desc")),
            "note_groups": collect_note_groups(section),
            "top_level_codes": [serialize_top_level_diag(diag) for diag in section.findall("diag")],
        }
    raise KeyError(f"Tabular section not found: {section_id}")


def open_tabular_entry(config: ICDReactConfig, code: str) -> dict[str, object]:
    """Open one ICD Tabular code entry by exact code.

    This is the authority surface for validating a candidate code found elsewhere.
    The response includes chapter and section context, ancestor and child codes,
    and note groups such as includes, excludes1, excludes2, codeFirst,
    useAdditionalCode, codeAlso, and sevenChrNote.
    """
    requested_code = str(code or "").strip().upper()
    normalized_code = normalize_code_token(requested_code)
    if not normalized_code:
        raise ValueError("code is required")

    root = load_tabular_root(str(config.tabular_xml_path))
    parent_map = {child: parent for parent in root.iter() for child in parent}
    for diag in root.iterfind(".//diag"):
        current_code = text_from_element(diag.find("name")).upper()
        if normalize_code_token(current_code) != normalized_code:
            continue

        diag_parent = parent_map.get(diag)
        section_node = diag_parent if diag_parent is not None and diag_parent.tag == "section" else None
        current_parent = diag_parent
        ancestor_codes: list[dict[str, str]] = []
        while current_parent is not None and current_parent.tag == "diag":
            ancestor_codes.append(
                {
                    "code": text_from_element(current_parent.find("name")),
                    "description": text_from_element(current_parent.find("desc")),
                }
            )
            current_parent = parent_map.get(current_parent)
        if section_node is None and current_parent is not None and current_parent.tag == "section":
            section_node = current_parent
        chapter_node = parent_map.get(section_node) if section_node is not None else None

        return {
            "manual": "ICD-10-CM Tabular",
            "code": current_code,
            "requested_code": requested_code,
            "description": text_from_element(diag.find("desc")),
            "chapter": None if chapter_node is None else text_from_element(chapter_node.find("desc")),
            "section": None if section_node is None else text_from_element(section_node.find("desc")),
            "ancestor_codes": list(reversed(ancestor_codes)),
            "child_codes": [
                {
                    "code": text_from_element(child.find("name")),
                    "description": text_from_element(child.find("desc")),
                    "note_groups": collect_note_groups(child),
                }
                for child in diag.findall("diag")
            ],
            "chapter_notes": collect_note_groups(chapter_node),
            "section_notes": collect_note_groups(section_node),
            "entry_notes": collect_note_groups(diag),
        }

    prefix = normalized_code[: max(3, min(len(normalized_code), 5))]
    suggestions: list[dict[str, str | None]] = []
    for diag in root.iterfind(".//diag"):
        candidate_code = text_from_element(diag.find("name")).upper()
        normalized_candidate = normalize_code_token(candidate_code)
        if not normalized_candidate:
            continue
        if prefix and not normalized_candidate.startswith(prefix):
            continue
        suggestions.append(
            {
                "code": candidate_code,
                "description": text_from_element(diag.find("desc")) or None,
            }
        )
        if len(suggestions) >= 10:
            break

    return {
        "manual": "ICD-10-CM Tabular",
        "requested_code": requested_code,
        "normalized_code": normalized_code,
        "found": False,
        "message": "Tabular code not found. Inspect a parent code or one of the nearby suggestions instead of assuming this code exists in the 2019 ICD manual.",
        "suggestions": suggestions,
    }


def list_guideline_toc(config: ICDReactConfig, section_prefix: str | None = None, limit: int = 50) -> dict[str, object]:
    """Browse the FY2019 ICD guideline table of contents.

    The guideline TOC is derived heuristically from the PDF text, so it should be
    treated as a lightweight manual browser. It is useful when the agent knows it
    needs a rule section but has not yet opened the exact section text.
    """
    prefix_filter = (section_prefix or "").strip().lower()
    sections = build_guideline_sections(str(config.guidelines_pdf_path))
    results: list[dict[str, object]] = []
    for section in sections:
        path_text = " > ".join(str(part) for part in section["path"])
        if prefix_filter and not path_text.lower().startswith(prefix_filter):
            continue
        results.append(
            {
                "section_id": section["section_id"],
                "path": section["path"],
                "page_start": section["page_start"],
                "page_end": section["page_end"],
                "preview": section["preview"],
            }
        )
        if len(results) >= limit:
            break
    return {
        "manual": "FY 2019 ICD-10-CM Official Guidelines for Coding and Reporting",
        "section_prefix": section_prefix or None,
        "count": len(results),
        "results": results,
    }


def open_guideline_section(config: ICDReactConfig, section_id: str) -> dict[str, object]:
    """Open one ICD guideline section by section id.

    The section id must come from list_guideline_toc. The response includes the
    heading path, page bounds, and extracted section text.
    """
    for section in build_guideline_sections(str(config.guidelines_pdf_path)):
        if str(section["section_id"]) == str(section_id).strip():
            return {
                "manual": "FY 2019 ICD-10-CM Official Guidelines for Coding and Reporting",
                "section_id": section["section_id"],
                "path": section["path"],
                "page_start": section["page_start"],
                "page_end": section["page_end"],
                "text": section["text"],
            }
    raise KeyError(f"Guideline section not found: {section_id}")


def build_icd_manual_tools(config: ICDReactConfig) -> list[object]:
    def list_index_main_terms_tool(letter: str | None = None, prefix: str | None = None, limit: int = 50) -> dict[str, object]:
        """Browse ICD Index main terms by letter or prefix.

        Use this to inspect the Alphabetic Index the way a human coder would browse
        a high-level term list. This tool only exposes manual structure; it does not
        identify the correct lead term or apply coding rules.
        """
        return list_index_main_terms(config=config, letter=letter, prefix=prefix, limit=limit)

    def open_index_term_tool(entry_id: str) -> dict[str, object]:
        """Open one ICD Index main term subtree by entry_id.

        The entry_id must come from list_index_main_terms_tool. The response includes
        the selected main term, any direct code, cross-references such as see or
        seeAlso, and nested child terms.
        """
        return open_index_term(config=config, entry_id=entry_id)

    def list_tabular_chapters_tool(code_prefix: str | None = None, limit: int = 50) -> dict[str, object]:
        """List high-level ICD Tabular chapters.

        Use this to browse the top-level Tabular List headings such as chapter-level
        disease families and their code ranges. This is a manual navigation tool, not
        a coding recommendation tool.
        """
        return list_tabular_chapters(config=config, code_prefix=code_prefix, limit=limit)

    def open_tabular_chapter_tool(chapter_id: str) -> dict[str, object]:
        """Open one ICD Tabular chapter by chapter_id.

        The chapter_id must come from list_tabular_chapters_tool. The response includes
        the chapter description, chapter-level note groups, and the next-level section
        headings with their section ids and code ranges.
        """
        return open_tabular_chapter(config=config, chapter_id=chapter_id)

    def open_tabular_section_tool(section_id: str) -> dict[str, object]:
        """Open one ICD Tabular section by section_id.

        The section_id must come from open_tabular_chapter_tool. The response includes
        section notes and the top-level codes that appear directly under that section.
        Use open_tabular_entry_tool after selecting a specific code to inspect.
        """
        return open_tabular_section(config=config, section_id=section_id)

    def open_tabular_entry_tool(code: str) -> dict[str, object]:
        """Open one ICD Tabular entry by exact code.

        Use this after finding a candidate code in the Index or elsewhere. The result
        shows chapter and section context, ancestor and child codes, and local note
        groups such as includes, excludes1, excludes2, codeFirst, useAdditionalCode,
        codeAlso, and sevenChrNote.
        """
        return open_tabular_entry(config=config, code=code)

    def list_guideline_toc_tool(section_prefix: str | None = None, limit: int = 50) -> dict[str, object]:
        """List the guideline table of contents.

        This exposes a lightweight browseable TOC for the FY2019 ICD guidelines.
        Section extraction is heuristic because the source is a PDF, so treat the
        output as a browser over manual sections rather than a perfect canonical map.
        """
        return list_guideline_toc(config=config, section_prefix=section_prefix, limit=limit)

    def open_guideline_section_tool(section_id: str) -> dict[str, object]:
        """Open one guideline section by section_id.

        The section_id must come from list_guideline_toc_tool. The response includes
        the heading path, page bounds, and extracted section text.
        """
        return open_guideline_section(config=config, section_id=section_id)

    return [
        list_index_main_terms_tool,
        open_index_term_tool,
        list_tabular_chapters_tool,
        open_tabular_chapter_tool,
        open_tabular_section_tool,
        open_tabular_entry_tool,
        list_guideline_toc_tool,
        open_guideline_section_tool,
    ]