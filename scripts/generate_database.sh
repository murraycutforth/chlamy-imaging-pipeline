#!/bin/bash
# Run the full Chlamy-IMPI pipeline: Stage 0 → Stage 1 → Stage 2a → Stage 2b.
#
# Usage:
#   conda activate chlamy
#   bash scripts/generate_database.sh
#
# All output is written under output/ relative to the project root.
# The script aborts immediately on any stage failure (set -e).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "======================================================"
echo "Chlamy-IMPI pipeline"
echo "Project root: $PROJECT_ROOT"
echo "Python:       $(python --version 2>&1)"
echo "Start time:   $(date)"
echo "======================================================"

echo ""
echo "------ Stage 0: Raw TIF/CSV error correction --------"
python -m chlamy_impi.error_correction.main

echo ""
echo "------ Stage 1: Well segmentation -------------------"
python -m chlamy_impi.well_segmentation_preprocessing.main

echo ""
echo "------ Stage 2a: Image processing (parquets) --------"
python -m chlamy_impi.image_processing.main

echo ""
echo "------ Stage 2b: Database creation ------------------"
python -m chlamy_impi.database_creation.main_v2

echo ""
echo "======================================================"
echo "Pipeline complete: $(date)"
echo "Output: $PROJECT_ROOT/output/database_creation/"
echo "======================================================"
