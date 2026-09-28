#!/usr/bin/env python3
"""Shared, environment-configurable project paths for the public code release."""

from __future__ import annotations

import os
from pathlib import Path


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser().resolve()


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = _path_from_env("COATING_PROJECT_ROOT", REPOSITORY_ROOT)
RAW_DATA_DIR = _path_from_env("COATING_DATA_DIR", PROJECT_ROOT / "external_data")
LITERATURE_DIR = _path_from_env("COATING_LITERATURE_DIR", PROJECT_ROOT / "external_literature")
LITERATURE_TEXT_DIR = _path_from_env(
    "COATING_LITERATURE_TEXT_DIR", PROJECT_ROOT / "supplementary" / "literature_text"
)
TEMPLATE_DOCX = _path_from_env(
    "COATING_TEMPLATE_DOCX", PROJECT_ROOT / "external_templates" / "Manuscript_Ti_CN.docx"
)

PRIMER_FILE = os.environ.get("COATING_PRIMER_FILE", "primer.xlsx")
TOPCOAT_FILE = os.environ.get("COATING_TOPCOAT_FILE", "topcoat.xlsx")
SHEET_NAME = os.environ.get("COATING_SHEET_NAME", "coating_samples")
