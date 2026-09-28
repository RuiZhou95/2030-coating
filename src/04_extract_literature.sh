#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${COATING_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC="${COATING_LITERATURE_DIR:-$PROJECT_ROOT/external_literature}"
OUT="${COATING_LITERATURE_TEXT_DIR:-$PROJECT_ROOT/supplementary/literature_text}"

if [[ ! -d "$SRC" ]]; then
  echo "Literature directory not found: $SRC" >&2
  echo "Set COATING_LITERATURE_DIR to a directory containing PDF files." >&2
  exit 1
fi

mkdir -p "$OUT"

find "$SRC" -maxdepth 1 -type f -iname '*.pdf' -print0 | while IFS= read -r -d '' pdf; do
  base=$(basename "$pdf" .pdf)
  txt="$OUT/${base}.txt"
  pdftotext -layout -enc UTF-8 "$pdf" "$txt"
done

{
  echo 'file|pages|title|subject'
  find "$SRC" -maxdepth 1 -type f -iname '*.pdf' -print0 | while IFS= read -r -d '' pdf; do
    base=$(basename "$pdf")
    pages=$(pdfinfo "$pdf" 2>/dev/null | awk -F: '/^Pages/{gsub(/^[ \t]+/,"",$2); print $2; exit}')
    title=$(pdfinfo "$pdf" 2>/dev/null | awk -F: '/^Title/{sub(/^[^:]*:[ \t]*/,""); print; exit}' | tr '|' '/')
    subject=$(pdfinfo "$pdf" 2>/dev/null | awk -F: '/^Subject/{sub(/^[^:]*:[ \t]*/,""); print; exit}' | tr '|' '/')
    printf '%s|%s|%s|%s\n' "$base" "$pages" "$title" "$subject"
  done
} > "$OUT/pdf_metadata.psv"

echo "EXTRACTED $(find "$OUT" -maxdepth 1 -type f -name '*.txt' | wc -l) PDF texts"
