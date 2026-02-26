"""
Generate GitHub Pages report assets from the latest pipeline output.

Run this after a full pipeline run (Stages 0–2b) to update docs/assets/images/.
After running, commit the updated docs/ directory and push to trigger a GitHub Pages redeploy.

Usage:
    python scripts/generate_report_assets.py
"""

import shutil
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_IMAGES = REPO_ROOT / "docs" / "assets" / "images"
DB_CREATION_DIR = REPO_ROOT / "output" / "database_creation"
IMAGE_PROCESSING_DIR = REPO_ROOT / "output" / "image_processing"
SEGMENTATION_DIR = REPO_ROOT / "output" / "well_segmentation_cache" / "mosaics"


def find_latest_run_dir() -> Path:
    """Return the most recent dated run directory under output/database_creation/."""
    dated = sorted(
        [d for d in DB_CREATION_DIR.iterdir() if d.is_dir() and d.name[:4].isdigit()],
        key=lambda d: d.name,
    )
    if not dated:
        raise FileNotFoundError(f"No dated run directories found in {DB_CREATION_DIR}")
    latest = dated[-1]
    print(f"Latest run directory: {latest}")
    return latest


def copy_timeseries_plots(run_dir: Path) -> None:
    """Copy timeseries PNGs from the latest run to docs/assets/images/."""
    for name in ("timeseries_y2.png", "timeseries_ynpq.png"):
        src = run_dir / name
        if not src.exists():
            print(f"  WARNING: {src} not found, skipping")
            continue
        dst = DOCS_IMAGES / name
        shutil.copy2(src, dst)
        print(f"  Copied {name}")


def copy_sample_mosaics(plate_name: str) -> None:
    """Copy well mosaic + mask mosaic + mask heatmap for a named plate."""
    copies = {
        SEGMENTATION_DIR / f"{plate_name}_mosaic.png": DOCS_IMAGES / "sample_well_mosaic.png",
        IMAGE_PROCESSING_DIR / "mask_visualisations" / f"{plate_name}_mask_mosaic.png": DOCS_IMAGES / "sample_mask_mosaic.png",
        IMAGE_PROCESSING_DIR / "mask_visualisations" / f"{plate_name}_mask_heatmap.png": DOCS_IMAGES / "sample_mask_heatmap.png",
    }
    for src, dst in copies.items():
        if not src.exists():
            print(f"  WARNING: {src} not found, skipping")
            continue
        shutil.copy2(src, dst)
        print(f"  Copied {src.name} → {dst.name}")


def copy_problem_mosaics(plate_name: str) -> None:
    """Copy well mosaic + mask images for a known-problematic plate."""
    copies = {
        SEGMENTATION_DIR / f"{plate_name}_mosaic.png": DOCS_IMAGES / "problem_well_mosaic.png",
        IMAGE_PROCESSING_DIR / "mask_visualisations" / f"{plate_name}_mask_mosaic.png": DOCS_IMAGES / "problem_mask_mosaic.png",
        IMAGE_PROCESSING_DIR / "mask_visualisations" / f"{plate_name}_mask_heatmap.png": DOCS_IMAGES / "problem_mask_heatmap.png",
    }
    for src, dst in copies.items():
        if not src.exists():
            print(f"  WARNING: {src} not found, skipping")
            continue
        shutil.copy2(src, dst)
        print(f"  Copied {src.name} → {dst.name}")


def generate_fvfm_plots() -> None:
    """Regenerate Fv/Fm distribution and per-plate boxplots from current parquets."""
    wells_path = IMAGE_PROCESSING_DIR / "wells.parquet"
    plates_path = IMAGE_PROCESSING_DIR / "plates.parquet"
    if not wells_path.exists() or not plates_path.exists():
        print("  WARNING: parquets not found, skipping Fv/Fm plots")
        return

    wells = pd.read_parquet(wells_path)
    plates = pd.read_parquet(plates_path)
    wells_m = wells.merge(
        plates[["plate", "measurement", "start_date", "light_regime"]],
        on=["plate", "measurement", "start_date"],
    )
    non_empty = wells_m[wells_m["mask_area"] > 0]

    # ── Distribution plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    regimes = sorted(non_empty["light_regime"].unique())
    colors = plt.cm.Set2(np.linspace(0, 1, len(regimes)))
    for regime, color in zip(regimes, colors):
        data = non_empty.loc[non_empty["light_regime"] == regime, "fv_fm"]
        axes[0].hist(data, bins=80, alpha=0.6, label=regime, color=color, range=(0, 0.85), density=True)
    axes[0].set_xlabel("Fv/Fm")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Fv/Fm Distribution by Light Regime")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].set_xlim(0, 0.85)

    plate_empty = (
        wells_m.groupby(["plate", "measurement", "start_date"])
        .apply(lambda g: (g["mask_area"] == 0).sum())
        .reset_index(name="n_empty")
    )
    median_empty = plate_empty["n_empty"].median()
    axes[1].hist(plate_empty["n_empty"], bins=40, color="steelblue", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("Empty Wells per Plate")
    axes[1].set_ylabel("Count (plates)")
    axes[1].set_title("Distribution of Empty Wells per Plate")
    axes[1].axvline(median_empty, color="red", linestyle="--", label=f"Median: {median_empty:.0f}")
    axes[1].legend()

    plt.tight_layout()
    out = DOCS_IMAGES / "fvfm_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")

    # ── Per-plate boxplot ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    plate_ids = sorted(non_empty["plate"].unique(), key=lambda x: (not str(x).isdigit(), str(x)))
    data_by_plate = [non_empty.loc[non_empty["plate"] == p, "fv_fm"].dropna().values for p in plate_ids]
    bp = ax.boxplot(data_by_plate, tick_labels=plate_ids, showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.6)
    ax.set_xlabel("Plate ID")
    ax.set_ylabel("Fv/Fm")
    ax.set_title("Fv/Fm Distribution by Plate ID (all measurements)")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.set_ylim(0.3, 0.85)
    ax.axhline(0.6, color="red", linestyle=":", alpha=0.5, label="Fv/Fm = 0.6")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = DOCS_IMAGES / "fvfm_by_plate.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")


def main() -> None:
    DOCS_IMAGES.mkdir(parents=True, exist_ok=True)

    run_dir = find_latest_run_dir()

    print("\n--- Timeseries plots ---")
    copy_timeseries_plots(run_dir)

    # Representative good plate (control plate 99-M1)
    print("\n--- Sample mosaics (good plate) ---")
    copy_sample_mosaics("20231201_99-M1_1min-1min")

    # Known-problematic plate (31v2-M2_20h_HL has 30 tiny-mask + 221 empty wells)
    print("\n--- Problem plate mosaics ---")
    copy_problem_mosaics("20240813_31v2-M2_20h_HL")

    print("\n--- Fv/Fm plots ---")
    generate_fvfm_plots()

    print(f"\nDone. Report assets written to {DOCS_IMAGES}")
    print(
        f"\nNext steps:\n"
        f"  1. Update docs/_config.yml: set report_date to {date.today()}, bump report_version if needed\n"
        f"  2. Update summary statistics in docs/index.md if pipeline outputs have changed\n"
        f"  3. git add docs/ && git commit -m 'Update pipeline report - {date.today()}' && git push"
    )


if __name__ == "__main__":
    main()
