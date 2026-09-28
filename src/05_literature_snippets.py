#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create compact, reviewable evidence snippets from locally available PDFs."""

from __future__ import annotations

import re

from project_paths import LITERATURE_TEXT_DIR as TEXT_DIR, PROJECT_ROOT as PROJECT

OUT = PROJECT / "supplementary/literature_snippets.md"

KEYWORDS = re.compile(
    r"(?i)(sample|dataset|data point|formulation|composition|cross[- ]validation|hold[- ]out|"
    r"random forest|xgboost|support vector|neural network|gaussian process|bayesian|"
    r"uncertaint|experimental validation|experimentally validated|pareto|multi-objective|shap)"
)
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.I)


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def title_from_text(text: str, fallback: str) -> str:
    lines = [compact(x) for x in text.splitlines()[:100] if compact(x)]
    bad = re.compile(r"(?i)^(article|review|open|abstract|www\.|https?://|vol\.|doi|check for updates)")
    candidates = [x for x in lines if 25 <= len(x) <= 240 and not bad.search(x)]
    return candidates[0] if candidates else fallback


def abstract_block(text: str) -> str:
    lower = text.lower()
    starts = [m.end() for m in re.finditer(r"\babstract\b", lower[:20000])]
    start = starts[0] if starts else 0
    tail = text[start:start + 8000]
    stop = re.search(r"(?i)\b(?:introduction|keywords|background)\b", tail[300:])
    if stop:
        tail = tail[:300 + stop.start()]
    return compact(tail)[:3000]


def evidence_sentences(text: str) -> list[str]:
    normalized = compact(text)
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    hits = []
    for sentence in sentences:
        if KEYWORDS.search(sentence) and 35 <= len(sentence) <= 650:
            hits.append(sentence)
        if len(hits) >= 16:
            break
    return hits


def main():
    sections = ["# 文献证据摘录（自动提取，待人工核验）", ""]
    for path in sorted(TEXT_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        dois = []
        for doi in DOI_RE.findall(text[:50000]):
            doi = doi.rstrip(".,;)")
            if doi not in dois:
                dois.append(doi)
        title = title_from_text(text, path.stem)
        sections.extend(
            [
                f"## {path.stem}",
                "",
                f"- 候选题名：{title}",
                f"- DOI：{' | '.join(dois[:4]) if dois else '未自动识别'}",
                f"- 摘要块：{abstract_block(text)}",
                "- 方法/验证证据句：",
            ]
        )
        hits = evidence_sentences(text)
        sections.extend([f"  - {x}" for x in hits] or ["  - 未自动识别，需人工阅读全文。"])
        sections.append("")
    OUT.write_text("\n".join(sections), encoding="utf-8")
    print(f"SNIPPETS_OK papers={len(list(TEXT_DIR.glob('*.txt')))} out={OUT}")


if __name__ == "__main__":
    main()
