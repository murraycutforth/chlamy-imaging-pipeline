"""Tests for error_correction/validation.py"""
import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from chlamy_impi.error_correction.validation import validate_tif_csv_pair


def _tif(n_frames: int, h: int = 8, w: int = 8, value: int = 100) -> np.ndarray:
    return np.full((n_frames, h, w), value, dtype=np.uint16)


def _meta(n_rows: int, interval_s: float = 62.0, start: datetime = None) -> pd.DataFrame:
    if start is None:
        start = datetime(2024, 1, 1, 10, 0, 0)
    rows = []
    for i in range(n_rows):
        ts = start + timedelta(seconds=i * interval_s)
        rows.append({"Date": ts.strftime("%d.%m.%y"), "Time": ts.strftime("%H:%M:%S")})
    return pd.DataFrame(rows)


class TestValidateTifCsvPair(unittest.TestCase):

    def test_valid_pair_passes(self):
        # 164 frames, 82 rows, 1min-1min → valid
        tif = _tif(164)
        meta = _meta(82)
        # Should not raise
        validate_tif_csv_pair(tif, meta, "test_plate", "1min-1min")

    def test_wrong_frame_count_fails(self):
        # 166 frames is not in valid set for 1min-1min {164, 172, 180}
        tif = _tif(166)
        meta = _meta(83)
        with self.assertRaises(AssertionError):
            validate_tif_csv_pair(tif, meta, "test_plate", "1min-1min")

    def test_all_black_frame_fails(self):
        tif = _tif(164)
        tif[5] = 0  # make one frame all-black
        meta = _meta(82)
        with self.assertRaises(AssertionError):
            validate_tif_csv_pair(tif, meta, "test_plate", "1min-1min")

    def test_non_monotone_timestamps_fails(self):
        tif = _tif(164)
        meta = _meta(82)
        # Break monotonicity: row 10 has an earlier timestamp than row 9
        ts_bad = datetime(2024, 1, 1, 10, 0, 0) + timedelta(seconds=9 * 62.0) - timedelta(seconds=5)
        meta.at[10, "Date"] = ts_bad.strftime("%d.%m.%y")
        meta.at[10, "Time"] = ts_bad.strftime("%H:%M:%S")
        with self.assertRaises(AssertionError):
            validate_tif_csv_pair(tif, meta, "test_plate", "1min-1min")

    def test_frame_csv_mismatch_fails(self):
        # 164 frames but only 80 CSV rows
        tif = _tif(164)
        meta = _meta(80)
        with self.assertRaises(AssertionError):
            validate_tif_csv_pair(tif, meta, "test_plate", "1min-1min")

    def test_odd_frame_count_fails(self):
        tif = _tif(163)
        meta = _meta(82)
        with self.assertRaises(AssertionError):
            validate_tif_csv_pair(tif, meta, "test_plate", "1min-1min")


if __name__ == "__main__":
    unittest.main()
