"""Unit tests for database_creation/main.py

Tests cover:
- get_npy_and_csv_filenames: basic discovery, filtering failed, missing CSV raises
- prepare_img_array_and_df: loads array + Stage-0-format CSV correctly
- construct_plate_info_df: returns expected columns, handles errors gracefully
- construct_well_info_df: correct row count, y2_1 not NaN
- merge_plate_and_well_info_dfs: produces well_id column, no NaN in key columns
"""

import datetime
import itertools
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from chlamy_impi.database_creation.main import (
    construct_plate_info_df,
    construct_well_info_df,
    get_npy_and_csv_filenames,
    merge_plate_and_well_info_dfs,
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


def make_fake_meta_df(n_rows=2):
    """Return DataFrame matching Stage 0 CSV format (as loaded by load_csv)."""
    return pd.DataFrame({
        "Date": ["25.02.26"] * n_rows,
        "Time": [f"{10 + i:02d}:00:00" for i in range(n_rows)],
    })


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


# ---------------------------------------------------------------------------
# Tests: construct_plate_info_df
# ---------------------------------------------------------------------------

class TestConstructPlateInfoDf:

    def _make_temp_plate(self, tmp_path, stem=VALID_STEM, n_frames=4, n_meta_rows=2):
        """Create a .npy + .csv pair in tmp_path and return their paths."""
        npy_path = tmp_path / f"{stem}.npy"
        csv_path = tmp_path / f"{stem}.csv"
        save_fake_npy(npy_path, ni=2, nj=3, n_frames=n_frames, h=5, w=5)
        write_fake_csv(csv_path, n_rows=n_meta_rows)
        return csv_path, npy_path

    def test_returns_expected_columns(self, tmp_path):
        csv_path, npy_path = self._make_temp_plate(tmp_path)

        with patch(f"{MODULE}.get_npy_and_csv_filenames", return_value=([csv_path], [npy_path])):
            df, failed = construct_plate_info_df()

        expected_cols = {"plate", "measurement", "start_date", "light_regime",
                         "dark_threshold", "light_threshold", "num_frames"}
        assert expected_cols.issubset(set(df.columns))
        assert len(failed) == 0
        assert len(df) == 1

    def test_handles_error_in_load(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        bad_npy = tmp_path / f"{VALID_STEM}.npy"
        bad_csv.touch()
        good_stem = "20240101_8-M6_30s-30s"
        good_csv, good_npy = self._make_temp_plate(tmp_path, stem=good_stem)

        # Mock prepare_img_array_and_df to raise for the bad file
        original_prepare = __import__(
            "chlamy_impi.database_creation.main", fromlist=["prepare_img_array_and_df"]
        ).prepare_img_array_and_df

        call_count = {"n": 0}

        def side_effect(meta, npy):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise AssertionError("simulated load error")
            return original_prepare(meta, npy)

        with patch(f"{MODULE}.get_npy_and_csv_filenames",
                   return_value=([bad_csv, good_csv], [bad_npy, good_npy])), \
             patch(f"{MODULE}.prepare_img_array_and_df", side_effect=side_effect):
            df, failed = construct_plate_info_df()

        assert len(failed) == 1
        assert len(df) == 1  # Only the good plate


# ---------------------------------------------------------------------------
# Tests: construct_well_info_df
# ---------------------------------------------------------------------------

class TestConstructWellInfoDf:

    def _make_temp_plate(self, tmp_path, stem=VALID_STEM, ni=2, nj=3, n_frames=4, n_meta_rows=2):
        npy_path = tmp_path / f"{stem}.npy"
        csv_path = tmp_path / f"{stem}.csv"
        save_fake_npy(npy_path, ni=ni, nj=nj, n_frames=n_frames, h=5, w=5)
        write_fake_csv(csv_path, n_rows=n_meta_rows)
        return csv_path, npy_path

    def test_row_count(self, tmp_path):
        ni, nj = 2, 3
        csv_path, npy_path = self._make_temp_plate(tmp_path, ni=ni, nj=nj)

        with patch(f"{MODULE}.get_npy_and_csv_filenames", return_value=([csv_path], [npy_path])):
            df, failed = construct_well_info_df(failed_filenames=[])

        assert len(df) == ni * nj
        assert "y2_1" in df.columns
        assert "ynpq_1" in df.columns

    def test_y2_not_all_nan(self, tmp_path):
        csv_path, npy_path = self._make_temp_plate(tmp_path)

        with patch(f"{MODULE}.get_npy_and_csv_filenames", return_value=([csv_path], [npy_path])):
            df, failed = construct_well_info_df(failed_filenames=[])

        assert not df["y2_1"].isna().all(), "y2_1 should have at least some non-NaN values"


# ---------------------------------------------------------------------------
# Tests: merge_plate_and_well_info_dfs
# ---------------------------------------------------------------------------

class TestMergePlateAndWellInfoDfs:

    def _make_full_plate_df(self):
        return pd.DataFrame([{
            "plate": "7",
            "measurement": "M6",
            "start_date": datetime.datetime(2023, 12, 6),
            "light_regime": "30s-30s",
            "dark_threshold": 0.1,
            "light_threshold": 0.5,
            "num_frames": 4,
        }])

    def _make_full_well_df(self):
        """Create a 16×24 well DataFrame (384 rows) to satisfy the sanity check."""
        rows = []
        for i, j in itertools.product(range(16), range(24)):
            rows.append({
                "plate": "7",
                "measurement": "M6",
                "start_date": datetime.datetime(2023, 12, 6),
                "i": i,
                "j": j,
                "fv_fm": 0.6,
                "fv_fm_std": 0.05,
                "mask_area": 10,
            })
        return pd.DataFrame(rows)

    def test_produces_well_id_column(self):
        plate_df = self._make_full_plate_df()
        well_df = self._make_full_well_df()
        merged = merge_plate_and_well_info_dfs(plate_df, well_df)
        assert "well_id" in merged.columns

    def test_no_nan_in_key_columns(self):
        plate_df = self._make_full_plate_df()
        well_df = self._make_full_well_df()
        merged = merge_plate_and_well_info_dfs(plate_df, well_df)
        for col in ("plate", "measurement", "start_date", "i", "j"):
            assert merged[col].notna().all(), f"Column {col!r} has unexpected NaN values"
