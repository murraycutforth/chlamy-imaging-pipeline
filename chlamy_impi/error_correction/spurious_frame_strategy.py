"""Automated detection and removal of spurious frame pairs from raw TIF + CSV data.

This module is intentionally decoupled so the detection algorithm can be
iterated without touching the rest of the pipeline.

Algorithm overview
------------------
Each spurious measurement event inserts one extra CSV row (and two extra TIF
frames) into an otherwise clean sequence.  Because the spurious row's timestamp
falls *between* two legitimate rows, it creates a pair of consecutive
anomalously short inter-row intervals whose sum equals roughly one normal
interval.  We locate these pairs, identify the inserted row, and remove it.
"""
import logging

import numpy as np
import pandas as pd

from chlamy_impi.database_creation.constants import (
    get_possible_frame_numbers,
    get_time_regime_to_expected_intervals,
    get_time_regime_to_valid_frame_counts,
)
from chlamy_impi.error_correction.plot_measurement_times import combine_date_and_time

logger = logging.getLogger(__name__)


def remove_spurious_frames(
    tif: np.ndarray, meta_df: pd.DataFrame, time_regime: str
) -> tuple[np.ndarray, pd.DataFrame]:
    """Detect and remove spurious frame pairs.

    Parameters
    ----------
    tif:        (n_frames, H, W) raw TIF array (after black-frame and
                duplicate-initial removal).
    meta_df:    Metadata DataFrame with Date/Time columns.
    time_regime: e.g. ``'1min-1min'``.

    Returns
    -------
    Corrected (tif, meta_df) tuple.  If no spurious rows are found, the
    inputs are returned unchanged.

    Raises
    ------
    ValueError:
        If the expected number of anomalous-interval pairs cannot be found.
    AssertionError:
        If the post-removal frame count is not in the valid set.
    """
    n_to_remove = _infer_n_to_remove(meta_df, time_regime)
    if n_to_remove == 0:
        return tif, meta_df

    logger.info(f"Need to remove {n_to_remove} spurious row(s) for time regime '{time_regime}'")

    if "Date" not in meta_df.columns or "Time" not in meta_df.columns:
        raise ValueError("meta_df has no Date/Time columns; cannot detect spurious frames")

    timestamps = combine_date_and_time(meta_df["Date"].values, meta_df["Time"].values)
    intervals_s = np.diff(timestamps).astype("timedelta64[s]").astype(float)

    expected_intervals = get_time_regime_to_expected_intervals()[time_regime]
    spurious_rows = _find_anomalous_interval_pairs(intervals_s, expected_intervals)

    if len(spurious_rows) != n_to_remove:
        raise ValueError(
            f"Expected to find {n_to_remove} spurious row(s) via anomalous interval pairs "
            f"but found {len(spurious_rows)}: {spurious_rows}"
        )

    logger.warning(f"Removing spurious rows at CSV indices {spurious_rows}")

    for k in reversed(sorted(spurious_rows)):
        tif = np.delete(tif, [2 * k, 2 * k + 1], axis=0)
        meta_df = meta_df.drop(meta_df.index[k]).reset_index(drop=True)

    valid = get_time_regime_to_valid_frame_counts().get(time_regime, get_possible_frame_numbers())
    assert tif.shape[0] in valid, (
        f"Post-removal frame count {tif.shape[0]} not in valid set {valid} "
        f"for time regime '{time_regime}'"
    )

    return tif, meta_df


def _infer_n_to_remove(meta_df: pd.DataFrame, time_regime: str) -> int:
    """Compute how many CSV rows (and frame pairs) need to be removed.

    Logic
    -----
    1. ``csv_rows = len(meta_df)``
    2. If ``2 * csv_rows`` is already in the valid set → 0 to remove.
    3. Otherwise, target = ``max(c for c in valid_set if c ≤ 2 * csv_rows)``.
    4. ``n_to_remove = csv_rows - target // 2``.
    """
    csv_rows = len(meta_df)
    raw_frames = 2 * csv_rows

    valid = get_time_regime_to_valid_frame_counts().get(time_regime, get_possible_frame_numbers())

    if raw_frames in valid:
        return 0

    candidates = [c for c in valid if c <= raw_frames]
    if not candidates:
        raise ValueError(
            f"2 * csv_rows = {raw_frames} is less than all valid frame counts {valid} "
            f"for time regime '{time_regime}'"
        )

    target = max(candidates)
    return csv_rows - target // 2


def _find_anomalous_interval_pairs(
    intervals_s: np.ndarray, expected_intervals: set[tuple]
) -> list[int]:
    """Return CSV row indices of spurious rows identified by anomalous interval pairs.

    A spurious row at index k causes:
      - intervals_s[k-1]  (from row k-1 → row k)   to be anomalously short
      - intervals_s[k]    (from row k   → row k+1)  to be anomalously short

    The function scans for the first anomalous interval in each such pair and
    records k = i + 1 (1-based position of the first anomalous interval).
    """

    def _is_normal(delta: float) -> bool:
        return any(lo <= delta <= hi for lo, hi in expected_intervals)

    spurious_rows: list[int] = []
    skip_next = False

    for i, d in enumerate(intervals_s):
        if skip_next:
            skip_next = False
            continue
        if not _is_normal(d):
            # Row i+1 is the spurious row (it produced the anomalous interval from row i to i+1)
            spurious_rows.append(i + 1)
            skip_next = True  # skip the second anomalous interval (i+1 → i+2)

    return spurious_rows
