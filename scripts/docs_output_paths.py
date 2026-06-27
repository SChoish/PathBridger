"""Centralized paths for generated docs artifacts.

This workspace keeps generated docs CSV/MD files under a local suffix so they do
not collide with artifacts produced from other local clones.
"""

from __future__ import annotations

from pathlib import Path

DOCS_SUFFIX = "choi"


def docs_output_path(project_root: str | Path, stem: str, ext: str) -> str:
    """Return ``docs/<stem>_<DOCS_SUFFIX>.<ext>`` as a string path."""
    clean_ext = ext[1:] if ext.startswith(".") else ext
    return str(Path(project_root) / "docs" / f"{stem}_{DOCS_SUFFIX}.{clean_ext}")
