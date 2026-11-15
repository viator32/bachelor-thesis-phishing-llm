#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

if ! command -v latexmk >/dev/null 2>&1; then
  for tex_bin in "$HOME/Library/TinyTeX/bin/"* "$HOME/.TinyTeX/bin/"* /Library/TeX/texbin; do
    if [[ -x "$tex_bin/latexmk" ]]; then
      export PATH="$tex_bin:$PATH"
      break
    fi
  done
fi

if ! command -v latexmk >/dev/null 2>&1; then
  echo "LaTeX fehlt: TinyTeX oder TeX Live mit latexmk und biber installieren." >&2
  exit 1
fi

drawio="${DRAWIO:-/Applications/draw.io.app/Contents/MacOS/draw.io}"
for source in assets/diagrams/drawio/*.drawio; do
  pdf="assets/diagrams/$(basename "$source" .drawio).pdf"
  if [[ ! -f "$pdf" || "$source" -nt "$pdf" ]]; then
    if [[ -x "$drawio" ]]; then
      "$drawio" -x -f pdf --crop -b 6 -o "$pdf" "$source"
    else
      echo "draw.io fehlt: $pdf ist nicht aktuell (DRAWIO auf die draw.io-CLI setzen)." >&2
    fi
  fi
done

exec latexmk "$@" thesis.tex
