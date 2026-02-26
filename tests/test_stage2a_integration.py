"""Integration tests for image_processing/main.py (Stage 2a)

Runs Stage 2a on 3 real plates drawn from WELL_SEGMENTATION_DIR and
CLEANED_RAW_DATA_DIR.  Skips automatically if Stage 1 / Stage 0 output is absent.

Warning investigation
---------------------
Stage 2a emits two categories of warnings during a full run:

1. Logger WARNING from mask_functions.compute_threshold_mask():
   "Well (i,j) has N masked pixel(s) below MIN_MASK_PIXELS=3; treating as empty"
   Cause: a well whose threshold-based mask contains only 1-2 pixels (shot noise).
   These pixels are zeroed out and the well is treated as empty (mask_area=0,
   fv_fm/y2/ynpq=NaN).  Expected and benign.

2. NumPy RuntimeWarnings emitted during photosynthetic-parameter computation:

   a. "invalid value encountered in divide" / "divide by zero encountered"
      Sources: fv_fm_functions.py (fv_array / fm_array),
               y2_functions.py   ((Fm_prime - F) / Fm_prime),
               npq_functions.py  (F / Fm_prime, F / Fm).
      Cause: after background subtraction, the denominator can be zero or very
      close to zero for pixels inside empty wells.  These pixels are masked to NaN
      before averaging, so the division result is discarded — the warning is
      harmless but noisy.

   b. "Mean of empty slice" / "Degrees of freedom <= 0 for slice"
      Source: np.nanmean / np.nanstd called on all-NaN slices in
      y2_functions.compute_masked_mean / compute_masked_std.
      Cause: wells whose mask is entirely False have every pixel set to NaN before
      the reduction, so nanmean receives an all-NaN slice and returns NaN (correct
      for empty wells).  The warning is expected.

All warning types are side-effects of the masking strategy applied to empty wells
and do not indicate incorrect results.  Suppressing them with np.errstate or
warnings.filterwarnings would remove the noise without changing behaviour.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch

from chlamy_impi.paths import WELL_SEGMENTATION_DIR, CLEANED_RAW_DATA_DIR
from chlamy_impi.image_processing.main import main

MODULE = "chlamy_impi.image_processing.main"

# Three plates covering different time regimes
PLATE_STEMS = [
    "20231012_99-M1_1min-1min",   # plate 99, 1min-1min
    "20231016_99-M5_10min-10min", # plate 99, 10min-10min
    "20231018_2-M1_1min-1min",    # plate 2, 1min-1min
]

pytestmark = pytest.mark.skipif(
    not WELL_SEGMENTATION_DIR.exists() or not any(WELL_SEGMENTATION_DIR.glob("*.npy")),
    reason="Well segmentation cache not present — run Stage 1 first",
)


@pytest.fixture(scope="module")
def run_stage2a(tmp_path_factory):
    """Run Stage 2a on 3 real plates, writing output to a temp dir.

    Returns a dict with parquet paths and the tmp_path.
    """
    tmp_path = tmp_path_factory.mktemp("stage2a")

    npy_paths = [WELL_SEGMENTATION_DIR / f"{stem}.npy" for stem in PLATE_STEMS]
    csv_paths = [CLEANED_RAW_DATA_DIR / f"{stem}.csv" for stem in PLATE_STEMS]

    for p in npy_paths + csv_paths:
        if not p.exists():
            pytest.skip(f"Test data not found: {p}")

    plates_path = tmp_path / "plates.parquet"
    wells_path = tmp_path / "wells.parquet"
    ts_path = tmp_path / "timeseries.parquet"

    with (
        patch(f"{MODULE}.get_npy_and_csv_filenames", return_value=(csv_paths, npy_paths)),
        patch(f"{MODULE}.get_image_processing_output_dir", return_value=tmp_path),
        patch(f"{MODULE}.get_plates_parquet_path", return_value=plates_path),
        patch(f"{MODULE}.get_wells_parquet_path", return_value=wells_path),
        patch(f"{MODULE}.get_timeseries_parquet_path", return_value=ts_path),
        patch(f"{MODULE}.mask_mosaic_path",
              side_effect=lambda name: tmp_path / f"{name}_mask_mosaic.png"),
        patch(f"{MODULE}.mask_heatmap_path",
              side_effect=lambda name: tmp_path / f"{name}_mask_heatmap.png"),
    ):
        main()

    return {
        "plates": plates_path,
        "wells": wells_path,
        "timeseries": ts_path,
        "tmp_path": tmp_path,
    }


class TestStage2aIntegration:

    def test_parquets_created(self, run_stage2a):
        assert run_stage2a["plates"].exists(), "plates.parquet not created"
        assert run_stage2a["wells"].exists(), "wells.parquet not created"
        assert run_stage2a["timeseries"].exists(), "timeseries.parquet not created"

    def test_plates_row_count(self, run_stage2a):
        df = pd.read_parquet(run_stage2a["plates"])
        assert len(df) == len(PLATE_STEMS), (
            f"Expected {len(PLATE_STEMS)} rows in plates.parquet, got {len(df)}"
        )

    def test_plates_columns(self, run_stage2a):
        df = pd.read_parquet(run_stage2a["plates"])
        expected = {"plate", "measurement", "start_date", "light_regime",
                    "dark_threshold", "light_threshold", "num_frames"}
        assert expected.issubset(set(df.columns))

    def test_wells_row_count(self, run_stage2a):
        df = pd.read_parquet(run_stage2a["wells"])
        # plate 99 has a non-standard layout (15x25), plate 2 is 16x24
        assert len(df) > 0
        # Each plate contributes at least 300 wells (conservative lower bound)
        assert len(df) >= len(PLATE_STEMS) * 300

    def test_wells_columns(self, run_stage2a):
        df = pd.read_parquet(run_stage2a["wells"])
        expected = {"plate", "measurement", "i", "j", "well_id",
                    "fv_fm", "fv_fm_std", "mask_area", "measurement_time_0"}
        assert expected.issubset(set(df.columns))

    def test_timeseries_long_format(self, run_stage2a):
        df = pd.read_parquet(run_stage2a["timeseries"])
        assert len(df) > 0
        expected = {"plate", "measurement", "i", "j", "time_step",
                    "y2", "y2_std", "ynpq", "measurement_time"}
        assert expected.issubset(set(df.columns))

    def test_time_step_is_one_based(self, run_stage2a):
        df = pd.read_parquet(run_stage2a["timeseries"])
        assert df["time_step"].min() == 1, "time_step should start at 1"

    def test_fv_fm_in_valid_range(self, run_stage2a):
        df = pd.read_parquet(run_stage2a["wells"])
        non_empty = df[df["mask_area"] > 0]["fv_fm"].dropna()
        assert len(non_empty) > 0, "No non-empty wells found"
        assert non_empty.between(-1.0, 1.0).all(), (
            f"Fv/Fm out of [-1, 1]: min={non_empty.min():.3f}, max={non_empty.max():.3f}"
        )

    def test_y2_in_valid_range(self, run_stage2a):
        df = pd.read_parquet(run_stage2a["timeseries"])
        y2_vals = df["y2"].dropna()
        assert len(y2_vals) > 0, "All Y2 values are NaN"
        assert y2_vals.between(-1.0, 1.0).all(), (
            f"Y2 out of [-1, 1]: min={y2_vals.min():.3f}, max={y2_vals.max():.3f}"
        )

    def test_no_nan_in_identifier_columns(self, run_stage2a):
        plates_df = pd.read_parquet(run_stage2a["plates"])
        wells_df = pd.read_parquet(run_stage2a["wells"])
        ts_df = pd.read_parquet(run_stage2a["timeseries"])

        for col in ("plate", "measurement", "start_date"):
            assert plates_df[col].notna().all(), f"plates.{col!r} has NaN"
        for col in ("plate", "measurement", "i", "j", "well_id", "mask_area"):
            assert wells_df[col].notna().all(), f"wells.{col!r} has NaN"
        for col in ("plate", "measurement", "i", "j", "time_step"):
            assert ts_df[col].notna().all(), f"timeseries.{col!r} has NaN"

    def test_mask_visualisations_created(self, run_stage2a):
        tmp_path = run_stage2a["tmp_path"]
        for stem in PLATE_STEMS:
            assert (tmp_path / f"{stem}_mask_mosaic.png").exists(), (
                f"Mask mosaic not created for {stem}"
            )
            assert (tmp_path / f"{stem}_mask_heatmap.png").exists(), (
                f"Mask heatmap not created for {stem}"
            )

    def test_non_empty_wells_have_positive_mask_area(self, run_stage2a):
        df = pd.read_parquet(run_stage2a["wells"])
        # At least half the wells per plate should be non-empty
        non_empty_fraction = (df["mask_area"] > 0).mean()
        assert non_empty_fraction > 0.5, (
            f"Only {non_empty_fraction:.1%} of wells are non-empty"
        )

    def test_empty_wells_have_nan_fv_fm(self, run_stage2a):
        """Wells with mask_area == 0 must produce NaN for photosynthetic params."""
        wells_df = pd.read_parquet(run_stage2a["wells"])
        ts_df = pd.read_parquet(run_stage2a["timeseries"])

        empty_wells = wells_df[wells_df["mask_area"] == 0]
        assert empty_wells["fv_fm"].isna().all(), (
            "Empty wells (mask_area=0) should have NaN fv_fm"
        )

        if len(empty_wells) > 0:
            sample = empty_wells[["plate", "measurement", "i", "j"]].head(10)
            empty_ts = ts_df.merge(sample, on=["plate", "measurement", "i", "j"])
            assert empty_ts["y2"].isna().all(), (
                "Empty wells should have NaN y2 in timeseries"
            )
