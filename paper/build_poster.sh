#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

tectonic --keep-logs --outdir . poster.tex
cp poster.pdf output.pdf

rm -rf preview
mkdir -p preview
pdftoppm -png -r 120 output.pdf preview/output

echo "Built paper/output.pdf"
echo "Rendered paper/preview/output-1.png"
