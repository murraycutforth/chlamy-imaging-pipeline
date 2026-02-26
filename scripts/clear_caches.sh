#!/bin/bash
# Remove all intermediate pipeline caches to force a full re-run.
#
# Caches cleared:
#   Stage 0  output/cleaned_raw_data/                           (cleaned TIF + CSV)
#   Stage 1  output/well_segmentation_cache/                    (.npy well arrays)
#   Stage 2a output/image_processing/plate_cache/               (per-plate parquets)
#   Stage 2a output/image_processing/mask_visualisations/       (mask mosaics + heatmaps)
#   Stage 2a output/image_processing/plates.parquet
#   Stage 2a output/image_processing/wells.parquet
#   Stage 2a output/image_processing/timeseries.parquet
#
# Final outputs (output/database_creation/) are NOT removed.
#
# Usage:
#   bash scripts/clear_caches.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUT="$PROJECT_ROOT/output"

CACHES=(
    "$OUT/cleaned_raw_data"
    "$OUT/well_segmentation_cache"
    "$OUT/image_processing/plate_cache"
    "$OUT/image_processing/mask_visualisations"
    "$OUT/image_processing/plates.parquet"
    "$OUT/image_processing/wells.parquet"
    "$OUT/image_processing/timeseries.parquet"
)

echo "Pipeline cache paths:"
any_found=0
for path in "${CACHES[@]}"; do
    if [ -e "$path" ]; then
        echo "  [exists]     $path"
        any_found=1
    else
        echo "  [not found]  $path"
    fi
done

if [ "$any_found" -eq 0 ]; then
    echo "Nothing to delete."
    exit 0
fi

echo ""
read -r -p "Delete all existing paths listed above? [y/N] " response
if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
for path in "${CACHES[@]}"; do
    if [ -e "$path" ]; then
        rm -rf "$path"
        echo "Deleted: $path"
    fi
done

echo ""
echo "Caches cleared. Run scripts/generate_database.sh to reprocess from scratch."
