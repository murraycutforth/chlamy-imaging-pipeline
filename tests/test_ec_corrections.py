"""Tests for error_correction/corrections.py"""
import unittest

import numpy as np
import pandas as pd

from chlamy_impi.error_correction.corrections import (
    remove_all_black_frame_pairs,
    remove_duplicate_initial_frame_pair,
    remove_warmup_pair,
)


def _make_tif(*frames: np.ndarray) -> np.ndarray:
    """Stack a sequence of 2D frames into a (n, H, W) array."""
    return np.stack(frames, axis=0)


def _frame(value: int, h: int = 8, w: int = 8) -> np.ndarray:
    return np.full((h, w), value, dtype=np.uint16)


def _meta(n_rows: int) -> pd.DataFrame:
    return pd.DataFrame({"Date": ["01.01.24"] * n_rows, "Time": [f"00:00:{i:02d}" for i in range(n_rows)]})


class TestRemoveWarmupPair(unittest.TestCase):

    def test_removes_first_two_frames(self):
        tif = _make_tif(_frame(10), _frame(20), _frame(0), _frame(0), _frame(10), _frame(20))
        out = remove_warmup_pair(tif)
        self.assertEqual(out.shape[0], 4)
        np.testing.assert_array_equal(out[0], _frame(0))  # was frame 2

    def test_warmup_matches_frames_4_5(self):
        f0, f1 = _frame(10), _frame(20)
        # Proper warmup: frames [0,1] duplicate [4,5]
        tif = _make_tif(f0, f1, _frame(0), _frame(0), f0, f1, _frame(5), _frame(6))
        out = remove_warmup_pair(tif)
        self.assertEqual(out.shape[0], 6)
        np.testing.assert_array_equal(out[0], _frame(0))   # was frame 2 (black)
        np.testing.assert_array_equal(out[2], f0)           # was frame 4


class TestRemoveAllBlackFramePairs(unittest.TestCase):

    def test_removes_full_black_pair_no_csv_change(self):
        # Full-black pair (both frames 0): NOT in CSV, CSV unchanged
        # Layout: pair0(valid), pair1(full-black, no CSV row??)
        # NOTE: in the real data, full-black pairs have NO CSV row.
        # We test that they are removed without touching meta_df.
        # Use meta with 1 row (the valid measurement); full-black pair has no CSV row.
        tif = _make_tif(_frame(1), _frame(2), _frame(0), _frame(0))
        meta = _meta(1)  # only 1 CSV row — for the valid pair
        # Can't call directly (tif.shape[0]=4 but meta has 1 row = mismatch).
        # The full-black pair is NOT in CSV, so before alignment this function
        # is called with the extra frames.  Simulate the post-warmup state.
        tif_out, meta_out = remove_all_black_frame_pairs(tif, meta)
        self.assertEqual(tif_out.shape[0], 2)
        self.assertEqual(len(meta_out), 1)  # CSV unchanged

    def test_removes_half_black_pair_and_csv_row(self):
        # Half-black pair (F0 valid, Fm black): IS in CSV → both TIF frames + CSV row removed
        # After warmup+full-black removal, TIF and CSV are aligned: pair k ↔ CSV row k
        # Here: pair 0 (F0=1, Fm=2) valid, pair 1 (F0=3, Fm=0) half-black, pair 2 (F0=5, Fm=6) valid
        tif = _make_tif(_frame(1), _frame(2), _frame(3), _frame(0), _frame(5), _frame(6))
        meta = _meta(3)
        tif_out, meta_out = remove_all_black_frame_pairs(tif, meta)
        self.assertEqual(tif_out.shape[0], 4)
        self.assertEqual(len(meta_out), 2)
        # Frames from pair 0 and pair 2 remain
        np.testing.assert_array_equal(tif_out[0], _frame(1))
        np.testing.assert_array_equal(tif_out[2], _frame(5))

    def test_no_black_frames_unchanged(self):
        tif = _make_tif(_frame(1), _frame(2), _frame(3), _frame(4))
        meta = _meta(2)
        tif_out, meta_out = remove_all_black_frame_pairs(tif, meta)
        np.testing.assert_array_equal(tif_out, tif)
        self.assertEqual(len(meta_out), 2)

    def test_full_black_pair_does_not_remove_csv_row(self):
        # Full-black pair followed by a valid pair. Full-black has no CSV row.
        tif = _make_tif(_frame(0), _frame(0), _frame(5), _frame(6))
        meta = pd.DataFrame({"Date": ["02.01.24"], "Time": ["00:00:02"]})
        tif_out, meta_out = remove_all_black_frame_pairs(tif, meta)
        self.assertEqual(len(meta_out), 1)
        self.assertEqual(meta_out.iloc[0]["Date"], "02.01.24")

    def test_half_black_csv_row_dropped(self):
        # The CSV row for the half-black pair IS removed
        tif = _make_tif(_frame(1), _frame(2), _frame(3), _frame(0))
        meta = pd.DataFrame({"Date": ["01.01.24", "02.01.24"], "Time": ["00:00:01", "00:00:02"]})
        tif_out, meta_out = remove_all_black_frame_pairs(tif, meta)
        self.assertEqual(len(meta_out), 1)
        self.assertEqual(meta_out.iloc[0]["Date"], "01.01.24")


class TestRemoveDuplicateInitialFramePair(unittest.TestCase):

    def test_removes_duplicate(self):
        f0, f1 = _frame(10), _frame(20)
        tif = _make_tif(f0, f1, f0, f1, _frame(30), _frame(40))
        meta = _meta(3)
        tif_out, meta_out = remove_duplicate_initial_frame_pair(tif, meta)
        self.assertEqual(tif_out.shape[0], 4)
        self.assertEqual(len(meta_out), 2)
        np.testing.assert_array_equal(tif_out[0], f0)
        np.testing.assert_array_equal(tif_out[1], f1)

    def test_no_duplicate_unchanged(self):
        tif = _make_tif(_frame(1), _frame(2), _frame(3), _frame(4))
        meta = _meta(2)
        tif_out, meta_out = remove_duplicate_initial_frame_pair(tif, meta)
        np.testing.assert_array_equal(tif_out, tif)
        self.assertEqual(len(meta_out), 2)

    def test_too_few_frames_unchanged(self):
        tif = _make_tif(_frame(1), _frame(2))
        meta = _meta(1)
        tif_out, meta_out = remove_duplicate_initial_frame_pair(tif, meta)
        np.testing.assert_array_equal(tif_out, tif)


if __name__ == "__main__":
    unittest.main()
