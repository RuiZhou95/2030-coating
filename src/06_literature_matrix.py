#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate and normalize an externally supplied coating-literature evidence matrix."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from project_paths import PROJECT_ROOT as PROJECT


INPUT = Path(
    os.environ.get(
        "COATING_LITERATURE_MATRIX_INPUT",
        PROJECT / "external_literature" / "literature_matrix.csv",
    )
).expanduser().resolve()
OUTPUT = PROJECT / "supplementary" / "literature_matrix.csv"

REQUIRED_COLUMNS = [
    "year",
    "title",
    "doi",
    "material_system",
    "sample_scale",
    "inputs",
    "targets",
    "models",
    "validation",
    "composition_constraint",
    "uncertainty",
    "independent_validation",
    "main_contribution",
    "limitation",
    "source_file",
]


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Literature matrix not found: {INPUT}. "
            "Set COATING_LITERATURE_MATRIX_INPUT to an external CSV file."
        )

    matrix = pd.read_csv(INPUT, encoding="utf-8-sig")
    missing = [column for column in REQUIRED_COLUMNS if column not in matrix.columns]
    if missing:
        raise ValueError(f"Literature matrix is missing columns: {missing}")

    matrix = matrix[REQUIRED_COLUMNS].copy()
    matrix["doi"] = matrix["doi"].astype(str).str.strip().str.lower()
    if matrix["doi"].eq("").any() or matrix["doi"].duplicated().any():
        raise ValueError("Every literature record must have a unique, non-empty DOI")

    matrix = matrix.sort_values(["year", "title"], ascending=[False, True])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"LITERATURE_MATRIX_OK rows={len(matrix)} unique_doi={matrix['doi'].nunique()}")


if __name__ == "__main__":
    main()
