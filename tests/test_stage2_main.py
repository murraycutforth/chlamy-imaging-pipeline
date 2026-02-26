"""Unit tests for database_creation/main.py (shared utilities)

Tests cover:
- get_npy_and_csv_filenames: basic discovery, filtering failed, missing CSV raises
- prepare_img_array_and_df: loads array + Stage-0-format CSV correctly
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from chlamy_impi.database_creation.main import (
    get_npy_and_csv_filenames,
    prepare_img_array_and_df,
)

MODULE = "chlamy_impi.database_creation.main"

VALID_STEM = "20231206_7-M6_30s-30s"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fake_img_array(ni=2, nj=3, n_frames=4, h=5, w=5):
    """Return float32 array (ni, nj, n_frames, h, w) with valid dark/light structure.

    Top-left well (0, 0) is used as a blank background cell (low values).
    All other wells have signal (higher values in light frames).
    Constraint: n_frames must be even; n_frames // 2 == number of meta_df rows.
    """
    assert n_frames % 2 == 0
    arr = np.zeros((ni, nj, n_frames, h, w), dtype=np.float32)
    # Top-left well (blank): very low values so it becomes background
    arr[0, 0, 0::2, :, :] = 0.1  # dark frames
    arr[0, 0, 1::2, :, :] = 0.2  # light frames
    # All other wells: signal with light > dark (required by mask computation)
    for ii in range(ni):
        for jj in range(nj):
            if ii == 0 and jj == 0:
                continue
            arr[ii, jj, 0::2, :, :] = 0.3  # dark frames
            arr[ii, jj, 1::2, :, :] = 0.7  # light frames
    return arr


def write_fake_csv(path: Path, n_rows: int = 2) -> None:
    """Write a Stage 0 compatible semicolon-delimited CSV with trailing empty column."""
    with open(path, "w") as f:
        f.write("Date;Time;\n")
        for i in range(n_rows):
            f.write(f"25.02.26;{10 + i:02d}:00:00;\n")


def save_fake_npy(path: Path, ni=2, nj=3, n_frames=4, h=5, w=5) -> None:
    """Save a synthetic img_array to a .npy file."""
    arr = make_fake_img_array(ni=ni, nj=nj, n_frames=n_frames, h=h, w=w)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


# ---------------------------------------------------------------------------
# Tests: get_npy_and_csv_filenames
# ---------------------------------------------------------------------------

class TestGetNpyAndCsvFilenames:

    def test_basic(self, tmp_path):
        seg_dir = tmp_path / "seg"
        clean_dir = tmp_path / "clean"
        seg_dir.mkdir()
        clean_dir.mkdir()
        (seg_dir / "20231206_7-M6_30s-30s.npy").touch()
        (seg_dir / "20240101_8-M6_30s-30s.npy").touch()
        (clean_dir / "20231206_7-M6_30s-30s.csv").touch()
        (clean_dir / "20240101_8-M6_30s-30s.csv").touch()

        with patch(f"{MODULE}.WELL_SEGMENTATION_DIR", seg_dir), \
             patch(f"{MODULE}.CLEANED_RAW_DATA_DIR", clean_dir):
            filenames_meta, filenames_npy = get_npy_and_csv_filenames()

        assert len(filenames_npy) == 2
        assert len(filenames_meta) == 2
        assert filenames_npy[0].stem == "20231206_7-M6_30s-30s"
        assert filenames_npy[1].stem == "20240101_8-M6_30s-30s"
        assert all(p.suffix == ".npy" for p in filenames_npy)
        assert all(p.suffix == ".csv" for p in filenames_meta)

    def test_filters_failed(self, tmp_path):
        seg_dir = tmp_path / "seg"
        clean_dir = tmp_path / "clean"
        seg_dir.mkdir()
        clean_dir.mkdir()
        (seg_dir / "20231206_7-M6_30s-30s.npy").touch()
        (seg_dir / "20240101_8-M6_30s-30s.npy").touch()
        (clean_dir / "20231206_7-M6_30s-30s.csv").touch()
        (clean_dir / "20240101_8-M6_30s-30s.csv").touch()

        with patch(f"{MODULE}.WELL_SEGMENTATION_DIR", seg_dir), \
             patch(f"{MODULE}.CLEANED_RAW_DATA_DIR", clean_dir):
            filenames_meta, filenames_npy = get_npy_and_csv_filenames(
                failed_filenames=["20231206_7-M6_30s-30s"]
            )

        assert len(filenames_npy) == 1
        assert filenames_npy[0].stem == "20240101_8-M6_30s-30s"

    def test_missing_csv_raises(self, tmp_path):
        seg_dir = tmp_path / "seg"
        clean_dir = tmp_path / "clean"
        seg_dir.mkdir()
        clean_dir.mkdir()
        (seg_dir / "20231206_7-M6_30s-30s.npy").touch()
        # No matching CSV in clean_dir

        with patch(f"{MODULE}.WELL_SEGMENTATION_DIR", seg_dir), \
             patch(f"{MODULE}.CLEANED_RAW_DATA_DIR", clean_dir):
            with pytest.raises(AssertionError, match="No CSV found"):
                get_npy_and_csv_filenames()


# ---------------------------------------------------------------------------
# Tests: prepare_img_array_and_df
# ---------------------------------------------------------------------------

class TestPrepareImgArrayAndDf:

    def test_loads_array_and_df(self, tmp_path):
        npy_path = tmp_path / f"{VALID_STEM}.npy"
        csv_path = tmp_path / f"{VALID_STEM}.csv"
        save_fake_npy(npy_path, ni=2, nj=3, n_frames=4, h=5, w=5)
        write_fake_csv(csv_path, n_rows=2)

        img_array, meta_df = prepare_img_array_and_df(csv_path, npy_path)

        assert img_array.shape == (2, 3, 4, 5, 5)
        assert img_array.dtype == np.float32
        assert "Date" in meta_df.columns
        assert "Time" in meta_df.columns
        assert len(meta_df) == 2
        # Trailing empty column must be stripped by load_csv
        assert not any(c.startswith("Unnamed") for c in meta_df.columns)
