#!/usr/bin/env bash
# build_pdfs.sh -- compile submission PDFs from markdown sources.
#
# Produces two PDFs under report/build/:
#   * onepager.pdf  -- the mandatory single-page submission (from SUBMISSION_ONEPAGER.md)
#   * appendix.pdf  -- optional combined appendix (REPORT.md + DECISIONS.md tail)
#
# Requires: pandoc (brew install pandoc) and xelatex (from a TeX distribution).
#
# Run from repo root:
#     bash scripts/build_pdfs.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/report/build"
mkdir -p "${BUILD_DIR}"

echo "===== Building submission PDFs ====="
echo "  repo root : ${REPO_ROOT}"
echo "  build dir : ${BUILD_DIR}"
echo "  pandoc    : $(pandoc --version | head -1)"
echo "  engine    : $(xelatex --version | head -1)"
echo

# Lucida Grande is on every Mac, has broad Unicode coverage (arrows,
# checkmarks, Greek). STIX Two Text would be the academic-feel alternative.
MAINFONT="Lucida Grande"

# --------------------------------------------------------------------------
# 1. One-page submission PDF
# --------------------------------------------------------------------------
echo "[1/2] Building onepager.pdf from SUBMISSION_ONEPAGER.md ..."

# Pre-process: strip the leading meta-instructions block (everything before
# the real title "# Machine-Learning Factor Investing"), so the DRAFT
# header doesn't end up on the submitted page.
TMP_ONEPAGER=$(mktemp -t onepager-XXXXXX.md)
awk '/^# Machine-Learning Factor Investing/,EOF' \
  "${REPO_ROOT}/report/SUBMISSION_ONEPAGER.md" > "${TMP_ONEPAGER}"

pandoc \
  "${TMP_ONEPAGER}" \
  --pdf-engine=xelatex \
  --resource-path="${REPO_ROOT}" \
  -V geometry:margin=0.9cm \
  -V mainfont="${MAINFONT}" \
  -V fontsize=8pt \
  -V colorlinks=true \
  -V linkcolor=NavyBlue \
  -V urlcolor=NavyBlue \
  -V documentclass=extarticle \
  -V pagestyle=empty \
  -V linestretch=0.95 \
  -o "${BUILD_DIR}/onepager.pdf"

rm -f "${TMP_ONEPAGER}"

PAGE_COUNT_1=$(pdftotext -layout "${BUILD_DIR}/onepager.pdf" - 2>/dev/null | \
               grep -c $'\f' || echo "?")
echo "  -> ${BUILD_DIR}/onepager.pdf  ($(wc -c <"${BUILD_DIR}/onepager.pdf") bytes; pages: ~${PAGE_COUNT_1})"
echo

# --------------------------------------------------------------------------
# 2. Appendix PDF (REPORT.md + DECISIONS.md tail)
# --------------------------------------------------------------------------
echo "[2/2] Building appendix.pdf from REPORT.md + DECISIONS.md ..."

TMP_APPENDIX=$(mktemp -t appendix-source-XXXXXX.md)
trap 'rm -f "${TMP_APPENDIX}"' EXIT

cat > "${TMP_APPENDIX}" <<'EOF'
---
title: "ML Factor Investing on a Survivorship-Free US Equity Universe -- Appendix"
subtitle: "Full report + decision log extract + reproducibility pointers"
author: "Bowen Zuo, Nicolas Couto Mota, Andrea Fontana"
---

EOF

# Append the main report
cat "${REPO_ROOT}/report/REPORT.md" >> "${TMP_APPENDIX}"

# Append the audit-era DECISIONS.md entries (from Phase 11 onwards).
echo "" >> "${TMP_APPENDIX}"
echo "" >> "${TMP_APPENDIX}"
echo "# Appendix B -- Decision log (audit-era extract)" >> "${TMP_APPENDIX}"
echo "" >> "${TMP_APPENDIX}"
echo "> Selected entries from \`DECISIONS.md\` documenting the Phase 11 - Phase 27b lineage and the survivorship-leak / Q-filter / INCLUDE_FEATURES audit. Full chronological log lives in \`DECISIONS.md\` in the repository." >> "${TMP_APPENDIX}"
echo "" >> "${TMP_APPENDIX}"

# Extract from the "Phase 11" entry onwards
awk '/^## 2026-05-23 . Phase 11/{p=1} p' \
  "${REPO_ROOT}/DECISIONS.md" >> "${TMP_APPENDIX}"

pandoc \
  "${TMP_APPENDIX}" \
  --pdf-engine=xelatex \
  --resource-path="${REPO_ROOT}" \
  --toc \
  --toc-depth=2 \
  -V geometry:margin=2cm \
  -V mainfont="${MAINFONT}" \
  -V fontsize=10pt \
  -V colorlinks=true \
  -V linkcolor=NavyBlue \
  -V urlcolor=NavyBlue \
  -V documentclass=article \
  -o "${BUILD_DIR}/appendix.pdf"

PAGE_COUNT_2=$(pdftotext -layout "${BUILD_DIR}/appendix.pdf" - 2>/dev/null | \
               grep -c $'\f' || echo "?")
echo "  -> ${BUILD_DIR}/appendix.pdf  ($(wc -c <"${BUILD_DIR}/appendix.pdf") bytes; pages: ~${PAGE_COUNT_2})"
echo

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# 3. Presentation slides (Beamer)
# --------------------------------------------------------------------------
echo "[3/3] Building slides.pdf from SLIDES.md (Beamer) ..."

pandoc \
  "${REPO_ROOT}/report/SLIDES.md" \
  -t beamer \
  --pdf-engine=xelatex \
  --resource-path="${REPO_ROOT}" \
  --slide-level=2 \
  -V mainfont="${MAINFONT}" \
  -V colorlinks=true \
  -V urlcolor=NavyBlue \
  -o "${BUILD_DIR}/slides.pdf"

PAGE_COUNT_3=$(mdls -name kMDItemNumberOfPages "${BUILD_DIR}/slides.pdf" 2>/dev/null \
              | sed 's/.* = //')
echo "  -> ${BUILD_DIR}/slides.pdf  ($(wc -c <"${BUILD_DIR}/slides.pdf") bytes; pages: ${PAGE_COUNT_3})"
echo

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
echo "===== Build complete ====="
ls -lh "${BUILD_DIR}/"*.pdf
echo
echo "Open with:"
echo "  open ${BUILD_DIR}/onepager.pdf"
echo "  open ${BUILD_DIR}/appendix.pdf"
echo "  open ${BUILD_DIR}/slides.pdf"
