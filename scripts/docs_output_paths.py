"""Naming convention for generated docs/ CSV and Markdown artifacts."""

from __future__ import annotations

from pathlib import Path

DOCS_7CH_SUFFIX = '_7ch'


def docs_7ch_name(filename: str) -> str:
    """Insert ``_7ch`` before the extension (``foo.csv`` → ``foo_7ch.csv``)."""
    p = Path(filename)
    return f'{p.stem}{DOCS_7CH_SUFFIX}{p.suffix}'


def docs_7ch_path(project_root: str | Path, filename: str) -> Path:
    """Return ``{project_root}/docs/{stem}_7ch.{ext}``."""
    return Path(project_root) / 'docs' / docs_7ch_name(filename)
