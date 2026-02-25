"""Tests for error_correction/spurious_frame_strategy.py"""
import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from chlamy_impi.error_correction.spurious_frame_strategy import (
    _find_anomalous_interval_pairs,
    _infer_n_to_remove,
    remove_spurious_frames,
)


def _make_meta(timestamps_dt: list[datetime]) -> pd.DataFrame:
    """Build a metadata DataFrame from a list of datetime objects."""
    rows = []
    for ts in timestamps_dt:
        rows.append({
            "Date": ts.strftime("%d.%m.%y"),
            "Time": ts.strftime("%H:%M:%S"),
        })
    return pd.DataFrame(rows)


def _make_tif(n_pairs: int, h: int = 4, w: int = 4) -> np.ndarray:
    """Return a (2*n_pairs, h, w) non-black TIF."""
    arr = np.ones((2 * n_pairs, h, w), dtype=np.uint16)
    return arr


class TestInferNToRemove(unittest.TestCase):

    def test_already_valid_1min1min(self):
        # 82 rows × 2 = 164 which is in {164, 172, 180}
        meta = pd.DataFrame({"x": range(82)})
        self.assertEqual(_infer_n_to_remove(meta, "1min-1min"), 0)

    def test_one_spurious_1min1min(self):
        # 83 rows × 2 = 166 → target = 164 → remove 1
        meta = pd.DataFrame({"x": range(83)})
        self.assertEqual(_infer_n_to_remove(meta, "1min-1min"), 1)

    def test_two_spurious_1min1min(self):
        # 84 rows × 2 = 168 → target = 164 → remove 2
        meta = pd.DataFrame({"x": range(84)})
        self.assertEqual(_infer_n_to_remove(meta, "1min-1min"), 2)

    def test_already_valid_30s30s(self):
        # 82 rows × 2 = 164 which is in {164, 172, 180}
        meta = pd.DataFrame({"x": range(82)})
        self.assertEqual(_infer_n_to_remove(meta, "30s-30s"), 0)

    def test_one_spurious_30s30s(self):
        # 83 rows × 2 = 166 → target = 164 → remove 1
        meta = pd.DataFrame({"x": range(83)})
        self.assertEqual(_infer_n_to_remove(meta, "30s-30s"), 1)


class TestFindAnomalousIntervalPairs(unittest.TestCase):

    def _expected(self):
        # Intervals from 1min-1min: normal ~60s, ~545s, ~1800s
        return {(59., 76.), (540, 558), (1730., 1860.)}

    def test_no_anomalies(self):
        intervals = np.array([62.0, 65.0, 60.0, 545.0, 70.0])
        result = _find_anomalous_interval_pairs(intervals, self._expected())
        self.assertEqual(result, [])

    def test_single_spurious_row(self):
        # Normal interval = 60s, spurious row at index k=2 creates two short intervals ~30s each
        intervals = np.array([62.0, 30.0, 31.0, 65.0, 545.0])
        result = _find_anomalous_interval_pairs(intervals, self._expected())
        self.assertEqual(result, [2])  # row 2 is spurious (interval index 1 → row 1+1=2)

    def test_two_spurious_rows(self):
        # Two separate spurious rows at CSV indices 2 and 5
        intervals = np.array([62.0, 30.0, 31.0, 60.0, 30.0, 29.0, 545.0])
        result = _find_anomalous_interval_pairs(intervals, self._expected())
        self.assertEqual(result, [2, 5])


class TestRemoveSpuriousFrames(unittest.TestCase):

    def _normal_timestamps(self, n: int, interval_s: float = 62.0) -> list[datetime]:
        base = datetime(2024, 1, 1, 10, 0, 0)
        return [base + timedelta(seconds=i * interval_s) for i in range(n)]

    def test_already_clean_no_change(self):
        # 82 clean rows → 164 frames, no removal needed
        timestamps = self._normal_timestamps(82)
        meta = _make_meta(timestamps)
        tif = _make_tif(82)
        tif_out, meta_out = remove_spurious_frames(tif, meta, "1min-1min")
        self.assertEqual(tif_out.shape[0], 164)
        self.assertEqual(len(meta_out), 82)

    def test_single_spurious_row_removed(self):
        # 83 rows with one spurious row inserted at position 40 (half normal interval)
        normal_ts = self._normal_timestamps(83)
        # Insert a spurious timestamp between rows 39 and 40 by replacing row 40
        # with something 31s after row 39 (making row 41 also 31s after the spurious)
        ts = list(normal_ts)
        base = ts[39]
        ts[40] = base + timedelta(seconds=31)  # spurious — 31s after row 39
        # row 41 was originally 62s after row 39, now it's 31s after spurious row 40
        # rows 41+ keep their original timestamps (shift by ~31s off)
        # Rebuild rest of timestamps so row 41 onward are still 62s-spaced from row 41
        for i in range(41, 83):
            ts[i] = ts[40] + timedelta(seconds=(i - 40) * 62.0)

        meta = _make_meta(ts)
        tif = _make_tif(83)

        tif_out, meta_out = remove_spurious_frames(tif, meta, "1min-1min")
        self.assertEqual(tif_out.shape[0], 164)
        self.assertEqual(len(meta_out), 82)


if __name__ == "__main__":
    unittest.main()
