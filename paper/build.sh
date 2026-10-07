#!/usr/bin/env bash
# Compile main.tex to main.pdf with Tectonic (self-contained LaTeX engine,
# fetches packages on demand). Install: binary from
# https://github.com/tectonic-typesetting/tectonic/releases into ~/.local/bin.
# Docker texlive/texlive also works where Docker can start containers.
set -euo pipefail
cd "$(dirname "$0")"
"${TECTONIC:-$HOME/.local/bin/tectonic}" -X compile main.tex
