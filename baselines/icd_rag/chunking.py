from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

from loguru import logger
from pypdf import PdfReader
from tqdm.auto import tqdm

from .config import ICDRAGConfig


NOTE_TAGS = {
    "includes",
    "inclusionTerm",
    "excludes1",
    "excludes2",
    "codeFirst",
    "useAdditionalCode",
    "codeAlso",
    "sevenChrNote",
}


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    source_type: str
    document_title: str
    chunk_title: str
    semantic_path: str
    code: str | None
    source_path: str
    text: str
    estimated_tokens: int

    def to_document(self) -> dict[str, object]:
        return asdict(self)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def estimate_tokens(text: str, chars_per_token: float) -> int:
    return max(1, int((len(text) / chars_per_token) + 0.9999))


def build_chunk_id(source_type: str, source_path: str, semantic_path: str) -> str:
    raw = f"{source_type}|{source_path}|{semantic_path}"
    slug = re.sub(r"[^a-zA-Z0-9_-]", "-", raw).strip("-")
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    max_slug_length = 200 - len(digest) - 1
    return f"{slug[:max_slug_length].rstrip('-')}-{digest}"


def split_large_text(text: str, max_chunk_chars: int) -> list[str]:
    if len(text) <= max_chunk_chars:
        return [text]

    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    pieces: list[str] = []
    current: list[str] = []
    current_len = 0
    for block in blocks:
        block_len = len(block) + 2
        if current and current_len + block_len > max_chunk_chars:
            pieces.append("\n\n".join(current))
            current = [block]
            current_len = len(block)
            continue
        current.append(block)
        current_len += block_len
    if current:
        pieces.append("\n\n".join(current))
    return pieces or [text]


def extract_guidelines_pages(path: Path) -> list[str]:
    reader = PdfReader(str(path))
    return [normalize_whitespace(page.extract_text() or "") for page in reader.pages]


def chunk_guidelines(path: Path, config: ICDRAGConfig) -> list[ChunkRecord]:
    logger.info("Chunking guidelines from {}", path)
    page_texts = extract_guidelines_pages(path)
    document_title = "FY 2019 ICD-10-CM Official Guidelines for Coding and Reporting"
    top_heading = ""
    chapter_heading = ""
    subsection_heading = ""
    subsection_lines: list[str] = []
    chunks: list[ChunkRecord] = []
    source_path = path.relative_to(config.manuals_root).as_posix()

    def flush_current() -> None:
        nonlocal subsection_lines, subsection_heading
        if not subsection_heading or not subsection_lines:
            subsection_lines = []
            return
        body = normalize_whitespace("\n".join(subsection_lines))
        text = normalize_whitespace(
            f"Document: {document_title}\nTop heading: {top_heading}\nChapter: {chapter_heading}\nSubsection: {subsection_heading}\n\n{body}"
        )
        semantic_path = " > ".join(part for part in [top_heading, chapter_heading, subsection_heading] if part)
        for part_index, piece in enumerate(split_large_text(text, config.max_chunk_chars), start=1):
            chunk_title = subsection_heading if len(text) <= config.max_chunk_chars else f"{subsection_heading} (part {part_index})"
            chunks.append(
                ChunkRecord(
                    chunk_id=build_chunk_id("guidelines", source_path, f"{semantic_path}|{part_index}"),
                    source_type="guidelines",
                    document_title=document_title,
                    chunk_title=chunk_title,
                    semantic_path=semantic_path,
                    code=None,
                    source_path=source_path,
                    text=piece,
                    estimated_tokens=estimate_tokens(piece, config.chars_per_token_estimate),
                )
            )
        subsection_lines = []

    for page_text in tqdm(page_texts, desc="guidelines pages", unit="page"):
        for raw_line in page_text.split("\n"):
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
                continue
            if subsection_heading:
                subsection_lines.append(line)

    flush_current()
    logger.info("Built {} guideline chunks", len(chunks))
    return chunks


def text_from_element(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return normalize_whitespace(" ".join(part.strip() for part in element.itertext() if part and part.strip()))


def collect_note_lines(element: ET.Element) -> list[str]:
    lines: list[str] = []
    for child in list(element):
        if child.tag in NOTE_TAGS:
            label = child.tag
            notes = [text_from_element(note) for note in child.findall(".//note") if text_from_element(note)]
            if notes:
                lines.append(f"{label}: {' | '.join(notes)}")
    return lines


def format_diag_subtree(diag: ET.Element, depth: int = 0) -> list[str]:
    code = text_from_element(diag.find("name"))
    desc = text_from_element(diag.find("desc"))
    indent = "  " * depth
    lines = [f"{indent}{code} {desc}".strip()]
    for note_line in collect_note_lines(diag):
        lines.append(f"{indent}{note_line}")
    for child_diag in diag.findall("diag"):
        lines.extend(format_diag_subtree(child_diag, depth + 1))
    return lines


def build_tabular_chunk_text(document_title: str, chapter_desc: str, section_desc: str, diag: ET.Element) -> str:
    lines = [
        f"Document: {document_title}",
        f"Chapter: {chapter_desc}",
        f"Section: {section_desc}",
        "",
    ]
    lines.extend(format_diag_subtree(diag))
    return normalize_whitespace("\n".join(lines))


def chunk_tabular(path: Path, config: ICDRAGConfig) -> list[ChunkRecord]:
    logger.info("Chunking tabular XML from {}", path)
    root = ET.parse(path).getroot()
    document_title = text_from_element(root.find("./introduction/introSection/title")) or "ICD-10-CM TABULAR LIST of DISEASES and INJURIES"
    source_path = path.relative_to(config.manuals_root).as_posix()
    chunks: list[ChunkRecord] = []

    chapters = root.findall("chapter")
    for chapter in tqdm(chapters, desc="tabular chapters", unit="chapter"):
        chapter_desc = text_from_element(chapter.find("desc"))
        for section in chapter.findall("section"):
            section_desc = text_from_element(section.find("desc"))
            for diag in section.findall("diag"):
                code = text_from_element(diag.find("name"))
                desc = text_from_element(diag.find("desc"))
                semantic_path = " > ".join(part for part in [chapter_desc, section_desc, code] if part)
                full_text = build_tabular_chunk_text(document_title, chapter_desc, section_desc, diag)
                child_diags = diag.findall("diag")

                if len(full_text) <= config.max_chunk_chars or not child_diags:
                    chunks.append(
                        ChunkRecord(
                            chunk_id=build_chunk_id("tabular", source_path, semantic_path),
                            source_type="tabular",
                            document_title=document_title,
                            chunk_title=f"{code} {desc}".strip(),
                            semantic_path=semantic_path,
                            code=code or None,
                            source_path=source_path,
                            text=full_text,
                            estimated_tokens=estimate_tokens(full_text, config.chars_per_token_estimate),
                        )
                    )
                    continue

                root_context = normalize_whitespace(
                    f"Document: {document_title}\nChapter: {chapter_desc}\nSection: {section_desc}\nParent code: {code} {desc}"
                )
                for child_diag in child_diags:
                    child_code = text_from_element(child_diag.find("name"))
                    child_desc = text_from_element(child_diag.find("desc"))
                    child_text = normalize_whitespace(f"{root_context}\n\n" + "\n".join(format_diag_subtree(child_diag)))
                    child_path = f"{semantic_path} > {child_code}"
                    chunks.append(
                        ChunkRecord(
                            chunk_id=build_chunk_id("tabular", source_path, child_path),
                            source_type="tabular",
                            document_title=document_title,
                            chunk_title=f"{code} {desc} > {child_code} {child_desc}".strip(),
                            semantic_path=child_path,
                            code=child_code or code or None,
                            source_path=source_path,
                            text=child_text,
                            estimated_tokens=estimate_tokens(child_text, config.chars_per_token_estimate),
                        )
                    )
    return chunks


def format_index_branch(element: ET.Element, depth: int = 0) -> list[str]:
    indent = "  " * depth
    title = text_from_element(element.find("title"))
    code = text_from_element(element.find("code"))
    see = text_from_element(element.find("see"))
    see_also = text_from_element(element.find("seeAlso"))

    suffix_parts = []
    if code:
        suffix_parts.append(f"code={code}")
    if see:
        suffix_parts.append(f"see={see}")
    if see_also:
        suffix_parts.append(f"see_also={see_also}")
    suffix = f" ({', '.join(suffix_parts)})" if suffix_parts else ""
    lines = [f"{indent}{title}{suffix}".strip()]
    for term in element.findall("term"):
        lines.extend(format_index_branch(term, depth + 1))
    return lines


def chunk_index(path: Path, config: ICDRAGConfig) -> list[ChunkRecord]:
    logger.info("Chunking alphabetic index XML from {}", path)
    root = ET.parse(path).getroot()
    document_title = text_from_element(root.find("title")) or "ICD-10-CM INDEX TO DISEASES and INJURIES"
    source_path = path.relative_to(config.manuals_root).as_posix()
    chunks: list[ChunkRecord] = []

    letters = root.findall("letter")
    for letter in tqdm(letters, desc="index letters", unit="letter"):
        letter_title = text_from_element(letter.find("title"))
        for main_term in letter.findall("mainTerm"):
            main_title = text_from_element(main_term.find("title"))
            semantic_path = f"{letter_title} > {main_title}"
            full_text = normalize_whitespace(
                f"Document: {document_title}\nLetter: {letter_title}\nMain term: {main_title}\n\n" + "\n".join(format_index_branch(main_term))
            )
            child_terms = main_term.findall("term")
            if len(full_text) <= config.max_chunk_chars or not child_terms:
                chunks.append(
                    ChunkRecord(
                        chunk_id=build_chunk_id("index", source_path, semantic_path),
                        source_type="index",
                        document_title=document_title,
                        chunk_title=main_title,
                        semantic_path=semantic_path,
                        code=text_from_element(main_term.find("code")) or None,
                        source_path=source_path,
                        text=full_text,
                        estimated_tokens=estimate_tokens(full_text, config.chars_per_token_estimate),
                    )
                )
                continue

            root_context = normalize_whitespace(
                f"Document: {document_title}\nLetter: {letter_title}\nMain term: {main_title}"
            )
            for child_term in child_terms:
                child_title = text_from_element(child_term.find("title"))
                child_text = normalize_whitespace(f"{root_context}\n\n" + "\n".join(format_index_branch(child_term)))
                child_path = f"{semantic_path} > {child_title}"
                chunks.append(
                    ChunkRecord(
                        chunk_id=build_chunk_id("index", source_path, child_path),
                        source_type="index",
                        document_title=document_title,
                        chunk_title=f"{main_title} > {child_title}",
                        semantic_path=child_path,
                        code=text_from_element(child_term.find("code")) or text_from_element(main_term.find("code")) or None,
                        source_path=source_path,
                        text=child_text,
                        estimated_tokens=estimate_tokens(child_text, config.chars_per_token_estimate),
                    )
                )
    return chunks


def build_all_chunks(config: ICDRAGConfig) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    if config.include_guidelines:
        guidelines_path = config.manuals_root / "2019-icd10-coding-guidelines-.pdf"
        chunks.extend(chunk_guidelines(guidelines_path, config))
    if config.include_tabular:
        tabular_path = config.manuals_root / "icd10cm_tabular_2019" / "icd10cm_tabular_2019.xml"
        chunks.extend(chunk_tabular(tabular_path, config))
    if config.include_index:
        index_path = config.manuals_root / "icd10cm_tabular_2019" / "icd10cm_index_2019.xml"
        chunks.extend(chunk_index(index_path, config))
    logger.info("Built {} total chunks across enabled sources", len(chunks))
    return chunks