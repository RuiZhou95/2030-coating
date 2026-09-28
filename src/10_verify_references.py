#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify bibliography metadata through Crossref using an external DOI list."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd

from project_paths import PROJECT_ROOT as PROJECT


MATRIX = PROJECT / "supplementary" / "literature_matrix.csv"
OUTPUT = PROJECT / "supplementary" / "references_verified.csv"
OVERRIDES_PATH = Path(
    os.environ.get(
        "COATING_REFERENCE_OVERRIDES",
        PROJECT / "external_literature" / "reference_overrides.json",
    )
).expanduser().resolve()


def first(value, default=""):
    if isinstance(value, list):
        return value[0] if value else default
    return value if value not in (None, "") else default


def year_of(message):
    for key in ("published-print", "published-online", "issued"):
        parts = message.get(key, {}).get("date-parts", [])
        if parts and parts[0]:
            return parts[0][0]
    return ""


def author_text(authors):
    names = []
    for author in authors or []:
        family = author.get("family", "").strip()
        given = author.get("given", "").strip()
        name = ", ".join(part for part in (family, given) if part)
        if name:
            names.append(name)
    return "; ".join(names)


def query(doi):
    url = "https://api.crossref.org/works/" + quote(doi, safe="")
    contact = os.environ.get("CROSSREF_MAILTO", "")
    user_agent = "coating-code-release/1.0"
    if contact:
        user_agent += f" (mailto:{contact})"
    request = Request(url, headers={"User-Agent": user_agent})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))["message"]


def load_overrides():
    if not OVERRIDES_PATH.exists():
        return {}
    data = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
    return {str(doi).lower(): value for doi, value in data.items()}


def main():
    if not MATRIX.exists():
        raise FileNotFoundError(
            f"Normalized literature matrix not found: {MATRIX}. Run src/06_literature_matrix.py first."
        )

    matrix = pd.read_csv(MATRIX, encoding="utf-8-sig")
    if "doi" not in matrix.columns:
        raise ValueError("The literature matrix must contain a doi column")

    extra_dois = [item.strip().lower() for item in os.environ.get("COATING_EXTRA_DOIS", "").split(",") if item.strip()]
    dois = list(dict.fromkeys(matrix["doi"].dropna().astype(str).str.strip().str.lower().tolist() + extra_dois))
    fallback = {str(row["doi"]).lower(): row for _, row in matrix.iterrows()}
    overrides = load_overrides()
    rows = []

    for number, doi in enumerate(dois, 1):
        verified, error = True, ""
        try:
            message = query(doi)
        except Exception as exc:
            verified, error, message = False, repr(exc), {}
        local = fallback.get(doi, {})
        override = overrides.get(doi, {})
        rows.append(
            {
                "number": number,
                "doi": doi,
                "title": override.get("title", first(message.get("title"), local.get("title", ""))),
                "authors": override.get("authors", author_text(message.get("author"))),
                "journal": override.get("journal", first(message.get("container-title"), "")),
                "year": override.get("year", year_of(message) or local.get("year", "")),
                "volume": first(message.get("volume"), ""),
                "issue": first(message.get("issue"), ""),
                "pages_or_article": first(message.get("page"), first(message.get("article-number"), "")),
                "crossref_verified": verified,
                "error": error,
            }
        )
        time.sleep(0.08)

    output = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"REFERENCES_OK n={len(output)} verified={int(output['crossref_verified'].sum())}")


if __name__ == "__main__":
    main()
