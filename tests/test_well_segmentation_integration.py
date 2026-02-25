"""Integration tests for well_segmentation_preprocessing/main.py

Runs the full Stage 1 pipeline on two real cleaned plates from CLEANED_RAW_DATA_DIR.
These tests are intentionally slow (real TIF loading + segmentation) and are kept
to a minimum of two plates to keep runtime manageable.

Skip automatically if cleaned data is not present.
"""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

import pytest

from chlamy_impi.paths import CLEANED_RAW_DATA_DIR
from chlamy_impi.well_segmentation_preprocessing.main import main

MODULE = "chlamy_impi.well_segmentation_preprocessing.main"

# Two representative plates used for integration testing.
# plate 99 has a special well layout (15 row peaks instead of 17).
PLATE_STEMS = [
    "20231012_99-M1_1min-1min",
    "20231016_99-M5_10min-10min",
]


def _cleaned_tif_paths() -> list[Path]:
    return [CLEANED_RAW_DATA_DIR / f"{stem}.tif" for stem in PLATE_STEMS]


# Skip the entire module if cleaned data is absent
pytestmark = pytest.mark.skipif(
    not CLEANED_RAW_DATA_DIR.exists() or not any(CLEANED_RAW_DATA_DIR.glob("*.tif")),
    reason="Cleaned raw data not present — run Stage 0 first",
)


@pytest.fixture
def run_stage1(tmp_path):
    """Run Stage 1 on two real plates, writing output to tmp_path.

    Returns the list of output .npy paths.
    """
    tif_paths = _cleaned_tif_paths()
    for p in tif_paths:
        if not p.exists():
            pytest.skip(f"Test plate not found: {p}")

    npy_paths = {stem: tmp_path / f"{stem}.npy" for stem in PLATE_STEMS}

    def fake_npy_path(name):
        return npy_paths.get(name, tmp_path / f"{name}.npy")

    def fake_outpath(name):
        d = tmp_path / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    with (
        patch(f"{MODULE}.validate_stage1_inputs"),
        patch(f"{MODULE}.find_all_cleaned_tif_images", return_value=tif_paths),
        patch(f"{MODULE}.npy_img_array_path", side_effect=fake_npy_path),
        patch(f"{MODULE}.well_segmentation_output_dir_path", side_effect=fake_outpath),
        patch(f"{MODULE}.get_database_output_dir", return_value=tmp_path),
        patch(f"{MODULE}.get_well_segmentation_processing_results_df_filename",
              return_value=tmp_path / "results.csv"),
    ):
        main()

    return npy_paths, tmp_path


class TestStage1Integration:

    def test_npy_files_created(self, run_stage1):
        npy_paths, _ = run_stage1
        for stem, path in npy_paths.items():
            assert path.exists(), f"Expected output not created: {path}"

    def test_output_is_5d_array(self, run_stage1):
        npy_paths, _ = run_stage1
        for stem, path in npy_paths.items():
            arr = np.load(path)
            assert arr.ndim == 5, f"{stem}: expected 5D array, got shape {arr.shape}"

    def test_output_dtype_is_float32(self, run_stage1):
        npy_paths, _ = run_stage1
        for stem, path in npy_paths.items():
            arr = np.load(path)
            assert arr.dtype == np.float32, f"{stem}: expected float32, got {arr.dtype}"

    def test_frame_count_matches_tif(self, run_stage1):
        """Axis 2 of the .npy array matches the total TIF frame count.
        F0/Fm channel splitting (pairs → measurements) happens in Stage 2, not here.
        """
        npy_paths, _ = run_stage1
        import tifffile
        for stem, npy_path in npy_paths.items():
            tif_path = CLEANED_RAW_DATA_DIR / f"{stem}.tif"
            tif = tifffile.imread(tif_path)
            arr = np.load(npy_path)
            assert arr.shape[2] == tif.shape[0], (
                f"{stem}: expected {tif.shape[0]} frames (axis 2), got {arr.shape[2]}"
            )

    def test_well_grid_dimensions(self, run_stage1):
        """Axis 0 x Axis 1 should be 15x25 for plate 99, or up to 16x24 for standard plates."""
        npy_paths, _ = run_stage1
        for stem, path in npy_paths.items():
            arr = np.load(path)
            n_rows, n_cols = arr.shape[0], arr.shape[1]
            assert 14 <= n_rows <= 17, f"{stem}: unexpected row count {n_rows}"
            assert 23 <= n_cols <= 25, f"{stem}: unexpected col count {n_cols}"

    def test_results_csv_written(self, run_stage1):
        _, tmp_path = run_stage1
        results_csv = tmp_path / "results.csv"
        assert results_csv.exists()

    def test_all_plates_succeeded(self, run_stage1):
        import pandas as pd
        _, tmp_path = run_stage1
        df = pd.read_csv(tmp_path / "results.csv")
        assert len(df) == len(PLATE_STEMS)
        assert (df["status"] == 1).all(), f"Some plates failed:\n{df}"

    def test_no_all_black_wells(self, run_stage1):
        """No well should be entirely zero across all frames."""
        npy_paths, _ = run_stage1
        for stem, path in npy_paths.items():
            arr = np.load(path)  # (rows, cols, frames, h, w)
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    well = arr[i, j]
                    assert well.max() > 0, f"{stem} well ({i},{j}) is all zeros"
