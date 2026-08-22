from __future__ import annotations

from pathlib import Path


TITLE18_MANUAL_PATTERNS = (
    "title18.html",
    "*usc18*.htm*",
    "PRELIMusc18.htm",
    "*.html",
    "*.htm",
)


def resolve_year_directory(path: Path) -> Path | None:
    for candidate in (path, *path.parents):
        if candidate.name.isdigit():
            return candidate
    return None


def resolve_title18_manual_path(year_root: Path) -> Path | None:
    if not year_root.exists() or not year_root.is_dir():
        return None

    for pattern in TITLE18_MANUAL_PATTERNS:
        matches = sorted(candidate for candidate in year_root.rglob(pattern) if candidate.is_file())
        if matches:
            return matches[0]
    return None


def has_title18_manual_years(root: Path) -> bool:
    if not root.exists() or not root.is_dir():
        return False
    return any(
        child.is_dir() and child.name.isdigit() and resolve_title18_manual_path(child) is not None
        for child in root.iterdir()
    )


def list_title18_manual_paths(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []

    manual_paths: list[Path] = []
    for child in sorted(root.iterdir(), key=lambda entry: entry.name):
        if not child.is_dir() or not child.name.isdigit():
            continue
        manual_path = resolve_title18_manual_path(child)
        if manual_path is not None:
            manual_paths.append(manual_path)
    return manual_paths


def list_title18_years(root: Path) -> list[int]:
    years: list[int] = []
    for manual_path in list_title18_manual_paths(root):
        year_root = resolve_year_directory(manual_path)
        if year_root is None:
            continue
        years.append(int(year_root.name))
    return sorted(set(years))