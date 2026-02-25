"""Unit tests for well_segmentation_preprocessing/main.py

Tests cover:
- save_img_array: file creation, dtype conversion, directory creation, value preservation
- main(): caching/skip logic, successful processing, per-file error handling,
  output file contents, all-black frame detection, multiple-file behaviour
"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from chlamy_impi.well_segmentation_preprocessing.main import save_img_array, main


MODULE = "chlamy_impi.well_segmentation_preprocessing.main"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fake_tif(n_frames=10, h=20, w=20, seed=42):
    """3-D image stack where every frame has non-zero variance."""
    rng = np.random.default_rng(seed)
    return (rng.random((n_frames, h, w)) + 0.1).astype(np.float32)


def make_fake_filename(stem="20231206_7-M6_30s-30s"):
    f = MagicMock(spec=Path)
    f.stem = stem
    f.name = stem + ".tif"
    return f


# ---------------------------------------------------------------------------
# Fixture: patch all external dependencies of main()
# ---------------------------------------------------------------------------

@pytest.fixture
def mocks(tmp_path):
    fake_tif = make_fake_tif()
    fake_img_array = np.random.default_rng(0).random((16, 24, 10, 5, 5)).astype(np.float32)
    fake_file = make_fake_filename()

    patchers = {
        "validate_stage1_inputs": patch(f"{MODULE}.validate_stage1_inputs"),
        "find_all_cleaned_tif_images": patch(f"{MODULE}.find_all_cleaned_tif_images"),
        "npy_img_array_path": patch(f"{MODULE}.npy_img_array_path"),
        "well_segmentation_output_dir_path": patch(f"{MODULE}.well_segmentation_output_dir_path"),
        "parse_name": patch(f"{MODULE}.parse_name"),
        "load_image": patch(f"{MODULE}.load_image"),
        "segment_multiwell_plate": patch(f"{MODULE}.segment_multiwell_plate"),
        "assert_expected_shape": patch(f"{MODULE}.assert_expected_shape"),
        "save_img_array": patch(f"{MODULE}.save_img_array"),
        "get_database_output_dir": patch(f"{MODULE}.get_database_output_dir"),
        "get_well_segmentation_processing_results_df_filename": patch(
            f"{MODULE}.get_well_segmentation_processing_results_df_filename"
        ),
    }

    m = {name: p.start() for name, p in patchers.items()}

    # --- default mock return values ---
    mock_npy = MagicMock()
    mock_npy.exists.return_value = False
    m["npy_img_array_path"].return_value = mock_npy
    m["well_segmentation_output_dir_path"].return_value = MagicMock()
    m["find_all_cleaned_tif_images"].return_value = [fake_file]
    m["parse_name"].return_value = ("7", "M6", "30s-30s")
    m["load_image"].return_value = fake_tif.copy()
    m["segment_multiwell_plate"].return_value = (
        fake_img_array, None, list(range(17)), list(range(25))
    )
    m["get_database_output_dir"].return_value = tmp_path
    m["get_well_segmentation_processing_results_df_filename"].return_value = (
        tmp_path / "results.csv"
    )

    # Store test data for assertion helpers
    m["_fake_file"] = fake_file
    m["_fake_tif"] = fake_tif
    m["_fake_img_array"] = fake_img_array
    m["_tmp_path"] = tmp_path

    yield m

    for p in patchers.values():
        p.stop()


# ---------------------------------------------------------------------------
# Tests: save_img_array
# ---------------------------------------------------------------------------

class TestSaveImgArray:

    def test_file_is_created(self, tmp_path):
        arr = np.ones((2, 2, 4, 3, 3))
        with patch(f"{MODULE}.npy_img_array_path", return_value=tmp_path / "test.npy"):
            save_img_array(arr, "test")
        assert (tmp_path / "test.npy").exists()

    def test_saved_as_float32(self, tmp_path):
        arr = np.ones((2, 2, 4, 3, 3), dtype=np.float64)
        with patch(f"{MODULE}.npy_img_array_path", return_value=tmp_path / "test.npy"):
            save_img_array(arr, "test")
        assert np.load(tmp_path / "test.npy").dtype == np.float32

    def test_creates_parent_directories(self, tmp_path):
        nested = tmp_path / "a" / "b" / "test.npy"
        arr = np.ones((2, 2, 4, 3, 3))
        with patch(f"{MODULE}.npy_img_array_path", return_value=nested):
            save_img_array(arr, "test")
        assert nested.exists()

    def test_values_preserved(self, tmp_path):
        arr = np.array([[[[[1.5, 2.5]]]]]).astype(np.float32)
        with patch(f"{MODULE}.npy_img_array_path", return_value=tmp_path / "test.npy"):
            save_img_array(arr, "test")
        np.testing.assert_array_equal(arr, np.load(tmp_path / "test.npy"))


# ---------------------------------------------------------------------------
# Tests: main()
# ---------------------------------------------------------------------------

class TestMain:

    # --- happy path ---

    def test_validate_inputs_called(self, mocks):
        main()
        mocks["validate_stage1_inputs"].assert_called_once()

    def test_successful_processing_calls_segmentation(self, mocks):
        main()
        mocks["segment_multiwell_plate"].assert_called_once()

    def test_successful_processing_calls_save(self, mocks):
        main()
        mocks["save_img_array"].assert_called_once()

    def test_assert_expected_shape_called_with_correct_plate_num(self, mocks):
        main()
        mocks["assert_expected_shape"].assert_called_once()
        args = mocks["assert_expected_shape"].call_args.args
        assert args[2] == "7"

    # --- caching / skip logic ---

    def test_skips_cached_file(self, mocks):
        mocks["npy_img_array_path"].return_value.exists.return_value = True
        main()
        mocks["load_image"].assert_not_called()
        mocks["save_img_array"].assert_not_called()

    def test_skipped_file_recorded_as_success(self, mocks, tmp_path):
        mocks["npy_img_array_path"].return_value.exists.return_value = True
        main()
        df = pd.read_csv(tmp_path / "results.csv")
        assert df["status"].iloc[0] == 1

    # --- error handling ---

    def test_exception_does_not_crash_main(self, mocks):
        mocks["segment_multiwell_plate"].side_effect = RuntimeError("segmentation failed")
        main()  # must not raise

    def test_failed_file_has_status_zero_in_csv(self, mocks, tmp_path):
        mocks["segment_multiwell_plate"].side_effect = RuntimeError("segmentation failed")
        main()
        df = pd.read_csv(tmp_path / "results.csv")
        assert df["status"].iloc[0] == 0

    def test_failed_file_error_written_to_txt(self, mocks, tmp_path):
        mocks["segment_multiwell_plate"].side_effect = RuntimeError("segmentation failed")
        main()
        errors = (tmp_path / "well_segmentation_errors.txt").read_text()
        assert "segmentation failed" in errors

    def test_all_black_frame_caught_as_error(self, mocks, tmp_path):
        # A frame with uniform values has std == 0, triggering the assertion in main()
        bad_tif = make_fake_tif()
        bad_tif[3, :, :] = 0.0
        mocks["load_image"].return_value = bad_tif
        main()
        df = pd.read_csv(tmp_path / "results.csv")
        assert df["status"].iloc[0] == 0

    # --- output files ---

    def test_processing_results_csv_written(self, mocks, tmp_path):
        main()
        assert (tmp_path / "results.csv").exists()

    def test_successful_file_has_status_one_in_csv(self, mocks, tmp_path):
        main()
        df = pd.read_csv(tmp_path / "results.csv")
        assert df["status"].iloc[0] == 1

    def test_error_txt_is_empty_on_success(self, mocks, tmp_path):
        main()
        errors = (tmp_path / "well_segmentation_errors.txt").read_text()
        assert errors == ""

    # --- multiple files ---

    def test_multiple_files_mixed_results(self, mocks, tmp_path):
        file_a = make_fake_filename("20231206_7-M6_30s-30s")
        file_b = make_fake_filename("20240223_16-M6_30s-30s")
        mocks["find_all_cleaned_tif_images"].return_value = [file_a, file_b]

        call_count = 0

        def segment_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("second file fails")
            return mocks["_fake_img_array"], None, list(range(17)), list(range(25))

        mocks["segment_multiwell_plate"].side_effect = segment_side_effect
        main()

        df = pd.read_csv(tmp_path / "results.csv")
        assert df["status"].tolist() == [1, 0]

    def test_multiple_files_all_saved_on_success(self, mocks):
        file_a = make_fake_filename("20231206_7-M6_30s-30s")
        file_b = make_fake_filename("20240223_16-M6_30s-30s")
        mocks["find_all_cleaned_tif_images"].return_value = [file_a, file_b]
        main()
        assert mocks["save_img_array"].call_count == 2
