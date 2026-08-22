from __future__ import annotations

import hashlib
import html
from io import BytesIO
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader
from tqdm.auto import tqdm

from .config import LegalRAGConfig
from .runtime import resolve_spark_session
from .title18_paths import list_title18_manual_paths, resolve_year_directory


logger = logging.getLogger(__name__)
HEADING_PATTERN = re.compile(r"<h(?P<level>[1-4]) class=\"(?P<class>[^\"]+)\">(?P<html>.*?)</h\1>", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    source_type: str
    source_year: int
    document_title: str
    chunk_title: str
    semantic_path: str
    citation: str | None
    source_path: str
    text: str
    estimated_tokens: int

    def to_document(self) -> dict[str, object]:
        return asdict(self)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = html.unescape(text)
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


def relative_volume_path(path: Path, root: Path) -> Path | None:
    path_uri = to_volume_uri(path)
    root_uri = to_volume_uri(root)
    if path_uri is not None and root_uri is not None:
        normalized_root = root_uri.rstrip("/")
        if path_uri == normalized_root:
            return Path()
        prefix = f"{normalized_root}/"
        if path_uri.startswith(prefix):
            return Path(path_uri[len(prefix) :])
    try:
        return path.relative_to(root)
    except ValueError:
        return None


def safe_relative_path(path: Path, root: Path) -> str:
    relative_path = relative_volume_path(path, root)
    if relative_path is not None:
        return relative_path.as_posix()
    return path.as_posix()


def strip_tags(html_text: str) -> str:
    without_comments = re.sub(r"<!--.*?-->", " ", html_text, flags=re.DOTALL)
    with_breaks = re.sub(r"<(br|p|div|tr|li|h[1-6])\b[^>]*>", "\n", without_comments, flags=re.IGNORECASE)
    with_breaks = re.sub(r"</(p|div|tr|li|h[1-6])>", "\n", with_breaks, flags=re.IGNORECASE)
    without_tags = re.sub(r"<[^>]+>", " ", with_breaks)
    return normalize_whitespace(without_tags)


def clean_heading_text(raw_html: str) -> str:
    return normalize_whitespace(strip_tags(raw_html))


def to_volume_uri(path: Path) -> str | None:
    path_str = str(path).replace("\\", "/")
    if path_str.startswith("dbfs:/Volumes/"):
        return path_str[len("dbfs:") :]
    if path_str.startswith("/Volumes/"):
        return path_str

    drive_prefix = re.match(r"^[A-Za-z]:(/Volumes/.*)$", path_str)
    if drive_prefix:
        return drive_prefix.group(1)
    return None


def path_from_spark_path(path_str: str) -> Path:
    normalized = path_str.replace("\\", "/")
    if normalized.startswith("dbfs:/Volumes/"):
        normalized = normalized[len("dbfs:") :]
    return Path(normalized)


def uses_spark_volume_io(path: Path, config: LegalRAGConfig) -> bool:
    return config.execution_env == "local" and to_volume_uri(path) is not None


def read_bytes(path: Path, config: LegalRAGConfig) -> bytes:
    if not uses_spark_volume_io(path, config):
        return path.read_bytes()

    spark = resolve_spark_session(app_name="legal-rag-volume-io")
    rows = spark.read.format("binaryFile").load(to_volume_uri(path)).select("content").limit(1).collect()
    if not rows:
        raise FileNotFoundError(f"File not found through Spark volume access: {path}")
    return bytes(rows[0]["content"])


def read_text(path: Path, config: LegalRAGConfig, encoding: str = "utf-8", errors: str = "ignore") -> str:
    return read_bytes(path, config).decode(encoding, errors=errors)


def path_exists(path: Path, config: LegalRAGConfig) -> bool:
    if not uses_spark_volume_io(path, config):
        return path.exists()

    spark = resolve_spark_session(app_name="legal-rag-volume-io")
    try:
        rows = spark.read.format("binaryFile").load(to_volume_uri(path)).select("path").limit(1).collect()
    except Exception:
        return False
    return bool(rows)


def glob_paths(root: Path, pattern: str, config: LegalRAGConfig) -> list[Path]:
    if not uses_spark_volume_io(root, config):
        return sorted(root.glob(pattern))

    spark = resolve_spark_session(app_name="legal-rag-volume-io")
    volume_uri = to_volume_uri(root)
    glob_pattern = f"{volume_uri.rstrip('/')}/{pattern}"
    rows = spark.read.format("binaryFile").load(glob_pattern).select("path").collect()
    return sorted(path_from_spark_path(str(row["path"])) for row in rows)


def build_ussg_docintel_text_path(path: Path, config: LegalRAGConfig) -> Path | None:
    if config.ussg_docintel_text_root is None:
        return None
    relative_path = relative_volume_path(path, config.ussg_root)
    if relative_path is None:
        relative_path = Path(path.name)
    return (config.ussg_docintel_text_root / relative_path).with_suffix(".docintel.json")


def load_ussg_pages_from_export(path: Path, config: LegalRAGConfig) -> list[str] | None:
    export_path = build_ussg_docintel_text_path(path, config)
    if export_path is None or not path_exists(export_path, config):
        return None

    payload = json.loads(read_text(export_path, config, encoding="utf-8", errors="strict"))
    raw_pages = payload.get("pages")
    if not isinstance(raw_pages, list):
        return None

    pages: list[str] = []
    for raw_page in raw_pages:
        if isinstance(raw_page, dict):
            page_text = normalize_whitespace(str(raw_page.get("text") or ""))
        else:
            page_text = normalize_whitespace(str(raw_page or ""))
        pages.append(page_text)
    if any(page for page in pages):
        logger.info("Using exported Doc Intelligence text for %s from %s", path, export_path)
        return pages
    return None


def extract_ussg_pages_with_docintel(path: Path, config: LegalRAGConfig) -> list[str] | None:
    if not config.use_docintel_for_ussg or not config.docintel_endpoint or not config.docintel_key:
        return None

    try:
        from azure.ai.formrecognizer import DocumentAnalysisClient
        from azure.core.credentials import AzureKeyCredential
    except ImportError:
        return None

    logger.info("Analyzing USSG PDF with Document Intelligence: %s", path)
    try:
        client = DocumentAnalysisClient(
            endpoint=config.docintel_endpoint,
            credential=AzureKeyCredential(config.docintel_key),
        )
        poller = client.begin_analyze_document(config.docintel_model, document=read_bytes(path, config))
        result = poller.result()
    except Exception as exc:
        logger.warning(
            "Document Intelligence failed for %s with %s: %s. Falling back to pypdf extraction.",
            path,
            type(exc).__name__,
            exc,
        )
        return None

    if getattr(result, "pages", None):
        pages: list[str] = []
        for page in result.pages:
            lines = [normalize_whitespace(line.content) for line in getattr(page, "lines", []) if normalize_whitespace(line.content)]
            pages.append("\n".join(lines))
        return pages

    if getattr(result, "content", None):
        return [normalize_whitespace(result.content)]

    return None


def extract_ussg_pages(path: Path, config: LegalRAGConfig) -> list[str]:
    exported_pages = load_ussg_pages_from_export(path, config)
    if exported_pages:
        return exported_pages

    docintel_pages = extract_ussg_pages_with_docintel(path, config)
    if docintel_pages:
        return docintel_pages

    logger.info("Falling back to pypdf extraction for %s", path)
    reader = PdfReader(BytesIO(read_bytes(path, config)))
    return [normalize_whitespace(page.extract_text() or "") for page in reader.pages]


def is_chapter_heading(line: str) -> bool:
    return bool(re.match(r"^(CHAPTER|Chapter)\s+[A-Z0-9IVX]+", line))


def is_part_heading(line: str) -> bool:
    return bool(re.match(r"^(PART|Part)\s+[A-Z0-9]+", line))


def is_guideline_heading(line: str) -> bool:
    return bool(re.match(r"^§\s*[0-9A-Z]+[A-Z0-9\.-]*", line))


def is_subheading(line: str) -> bool:
    normalized = line.lower()
    return normalized in {
        "commentary",
        "application notes",
        "application note",
        "background",
        "introductory commentary",
        "historical note",
        "statutory provisions",
    }


def build_ussg_chunk_text(document_title: str, chapter_heading: str, part_heading: str, section_heading: str, subheading: str, body: str) -> str:
    lines = [f"Document: {document_title}"]
    if chapter_heading:
        lines.append(f"Chapter: {chapter_heading}")
    if part_heading:
        lines.append(f"Part: {part_heading}")
    if section_heading:
        lines.append(f"Section: {section_heading}")
    if subheading:
        lines.append(f"Subsection: {subheading}")
    lines.append("")
    lines.append(body)
    return normalize_whitespace("\n".join(lines))


def extract_ussg_citation(section_heading: str) -> str | None:
    match = re.match(r"^(§\s*[0-9A-Z]+[A-Z0-9\.-]*)", section_heading)
    return match.group(1) if match else None


def chunk_ussg(path: Path, year: int, config: LegalRAGConfig) -> list[ChunkRecord]:
    pages = extract_ussg_pages(path, config)
    source_path = safe_relative_path(path, config.manuals_root)
    document_title = f"USSG Guidelines Manual ({year})"
    chapter_heading = ""
    part_heading = ""
    section_heading = ""
    subheading = ""
    section_lines: list[str] = []
    chunks: list[ChunkRecord] = []

    def flush_current() -> None:
        nonlocal section_lines, subheading
        if not section_heading or not section_lines:
            section_lines = []
            return
        body = normalize_whitespace("\n".join(section_lines))
        if not body:
            section_lines = []
            return
        text = build_ussg_chunk_text(document_title, chapter_heading, part_heading, section_heading, subheading, body)
        semantic_path = " > ".join(part for part in [chapter_heading, part_heading, section_heading, subheading] if part)
        citation = extract_ussg_citation(section_heading)
        for part_index, piece in enumerate(split_large_text(text, config.max_chunk_chars), start=1):
            chunk_title = section_heading if not subheading else f"{section_heading} - {subheading}"
            if len(text) > config.max_chunk_chars:
                chunk_title = f"{chunk_title} (part {part_index})"
            chunks.append(
                ChunkRecord(
                    chunk_id=build_chunk_id("ussg", source_path, f"{semantic_path}|{part_index}"),
                    source_type="ussg",
                    source_year=year,
                    document_title=document_title,
                    chunk_title=chunk_title,
                    semantic_path=semantic_path,
                    citation=citation,
                    source_path=source_path,
                    text=piece,
                    estimated_tokens=estimate_tokens(piece, config.chars_per_token_estimate),
                )
            )
        section_lines = []

    for page_text in tqdm(pages, desc=f"USSG {year} pages", unit="page"):
        for raw_line in page_text.split("\n"):
            line = normalize_whitespace(raw_line)
            if not line:
                continue
            if is_chapter_heading(line):
                flush_current()
                chapter_heading = line
                part_heading = ""
                section_heading = ""
                subheading = ""
                continue
            if is_part_heading(line):
                flush_current()
                part_heading = line
                section_heading = ""
                subheading = ""
                continue
            if is_guideline_heading(line):
                flush_current()
                section_heading = line
                subheading = ""
                section_lines = []
                continue
            if is_subheading(line) and section_heading:
                flush_current()
                subheading = line
                section_lines = []
                continue
            if section_heading:
                if line in {document_title, str(year), "United States Sentencing Commission"}:
                    continue
                section_lines.append(line)

    flush_current()
    return chunks


def extract_title18_citation(section_heading: str) -> str | None:
    match = re.search(r"\[?§\s*([^\.\]]+)", section_heading)
    return f"18 U.S.C. § {match.group(1).strip()}" if match else None


def build_title18_chunk_text(document_title: str, part_heading: str, chapter_heading: str, section_heading: str, body: str) -> str:
    lines = [f"Document: {document_title}"]
    if part_heading:
        lines.append(f"Part: {part_heading}")
    if chapter_heading:
        lines.append(f"Chapter: {chapter_heading}")
    lines.append(f"Section: {section_heading}")
    lines.append("")
    lines.append(body)
    return normalize_whitespace("\n".join(lines))


def chunk_title18(path: Path, year: int, config: LegalRAGConfig) -> list[ChunkRecord]:
    source_html = read_text(path, config, encoding="utf-8", errors="ignore")
    headings = list(HEADING_PATTERN.finditer(source_html))
    source_path = safe_relative_path(path, config.manuals_root)
    document_title = f"Title 18 United States Code ({year})"
    part_heading = ""
    chapter_heading = ""
    chunks: list[ChunkRecord] = []

    for index, match in enumerate(tqdm(headings, desc=f"Title 18 {year} sections", unit="section")):
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
            continue
        if heading_class != "section-head":
            continue

        body_html = source_html[match.end() : next_start]
        body_text = strip_tags(body_html)
        if not body_text:
            continue

        text = build_title18_chunk_text(document_title, part_heading, chapter_heading, heading_text, body_text)
        semantic_path = " > ".join(part for part in [document_title, part_heading, chapter_heading, heading_text] if part)
        citation = extract_title18_citation(heading_text)
        for part_index, piece in enumerate(split_large_text(text, config.max_chunk_chars), start=1):
            chunk_title = heading_text if len(text) <= config.max_chunk_chars else f"{heading_text} (part {part_index})"
            chunks.append(
                ChunkRecord(
                    chunk_id=build_chunk_id("usc_title18", source_path, f"{semantic_path}|{part_index}"),
                    source_type="usc_title18",
                    source_year=year,
                    document_title=document_title,
                    chunk_title=chunk_title,
                    semantic_path=semantic_path,
                    citation=citation,
                    source_path=source_path,
                    text=piece,
                    estimated_tokens=estimate_tokens(piece, config.chars_per_token_estimate),
                )
            )
    return chunks


def build_all_chunks(config: LegalRAGConfig, years: set[int] | None = None) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    if config.include_ussg:
        ussg_paths = glob_paths(config.ussg_root, "*/GLMFull.pdf", config)
        for pdf_path in tqdm(ussg_paths, desc="USSG manuals", unit="manual"):
            try:
                year = int(pdf_path.parent.name)
            except ValueError:
                continue
            if years is not None and year not in years:
                continue
            logger.info("Chunking USSG %s", pdf_path)
            chunks.extend(chunk_ussg(pdf_path, year, config))

    if config.include_title18:
        title18_paths = list_title18_manual_paths(config.title18_root)
        for html_path in tqdm(title18_paths, desc="Title 18 manuals", unit="manual"):
            year_root = resolve_year_directory(html_path)
            if year_root is None:
                continue
            year = int(year_root.name)
            if years is not None and year not in years:
                continue
            logger.info("Chunking Title 18 %s", html_path)
            chunks.extend(chunk_title18(html_path, year, config))

    logger.info("Built %s total legal chunks across enabled sources", len(chunks))
    return chunks