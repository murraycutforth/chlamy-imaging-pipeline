"""Post-correction assertions for raw TIF / CSV pairs."""
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


def get_timestamp_check_exempt_plates() -> frozenset[str]:
    """Plates exempt from timestamp monotonicity and interval consistency checks.

    These plates have known, unfixable timestamp anomalies in the raw data.
    Frame count and TIF/CSV alignment checks still apply.  Add new entries
    here only for plates with confirmed data-collection faults; any plate
    *not* listed will still be validated in full.

    Known reasons:
        20231102_4-M4_20h_ML          — 24-hour gap (experiment interrupted overnight)
        20231104_4-M6_10min-10min     — DST clock rollback (US, first Sunday Nov 2023)
        20241102_33v3-M6_10min-10min  — DST clock rollback (US, first Sunday Nov 2024)
        20260307_37v1-M3_20h_ML       — DST spring-forward (US, second Sunday Mar 2026)
    """
    return frozenset({
        '20231102_4-M4_20h_ML',
        '20231104_4-M6_10min-10min',
        '20241102_33v3-M6_10min-10min',
        '20260307_37v1-M3_20h_ML',
    })


def validate_tif_csv_pair(tif: np.ndarray, meta_df: pd.DataFrame, basename: str, time_regime: str) -> None:
    """Assert all post-correction invariants. Raises AssertionError with a clear message on failure."""
    _assert_frame_count_even(tif, basename)
    _assert_frame_csv_alignment(tif, meta_df, basename)
    _assert_valid_frame_count(tif, basename, time_regime)
    _assert_no_black_frames(tif, basename)

    if basename in get_timestamp_check_exempt_plates():
        logger.warning(f"{basename}: skipping timestamp checks (plate is in exempt list)")
        return

    _assert_timestamps_monotone(meta_df, basename)
    _assert_intervals_consistent(meta_df, basename, time_regime)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _assert_frame_count_even(tif: np.ndarray, basename: str) -> None:
    assert tif.shape[0] % 2 == 0, (
        f"{basename}: TIF frame count {tif.shape[0]} is odd; "
        "F0/Fm interleaving requires an even number of frames"
    )


def _assert_frame_csv_alignment(tif: np.ndarray, meta_df: pd.DataFrame, basename: str) -> None:
    expected = len(meta_df) * 2
    assert tif.shape[0] == expected, (
        f"{basename}: TIF has {tif.shape[0]} frames but CSV has {len(meta_df)} rows "
        f"(expected {expected} frames)"
    )


def _assert_valid_frame_count(tif: np.ndarray, basename: str, time_regime: str) -> None:
    n = tif.shape[0]
    regime_map = get_time_regime_to_valid_frame_counts()
    if time_regime in regime_map:
        valid = regime_map[time_regime]
    else:
        valid = get_possible_frame_numbers()
    assert n in valid, (
        f"{basename}: frame count {n} not in valid set {valid} for time regime '{time_regime}'"
    )


def _assert_no_black_frames(tif: np.ndarray, basename: str) -> None:
    max_per_frame = tif.reshape(tif.shape[0], -1).max(axis=1)
    black = np.where(max_per_frame == 0)[0]
    assert len(black) == 0, (
        f"{basename}: {len(black)} all-black frame(s) remain at indices {black.tolist()}"
    )


def _assert_timestamps_monotone(meta_df: pd.DataFrame, basename: str) -> None:
    if "Date" not in meta_df.columns or "Time" not in meta_df.columns:
        logger.warning(f"{basename}: no Date/Time columns; skipping timestamp monotonicity check")
        return
    timestamps = combine_date_and_time(meta_df["Date"].values, meta_df["Time"].values)
    diffs = np.diff(timestamps).astype("timedelta64[s]").astype(float)
    non_pos = np.where(diffs <= 0)[0]
    assert len(non_pos) == 0, (
        f"{basename}: timestamps are not strictly monotone at row intervals {non_pos.tolist()}"
    )


def _assert_intervals_consistent(meta_df: pd.DataFrame, basename: str, time_regime: str) -> None:
    """Check that all inter-row time intervals match an expected interval for this time regime."""
    if "Date" not in meta_df.columns or "Time" not in meta_df.columns:
        logger.warning(f"{basename}: no Date/Time columns; skipping interval consistency check")
        return
    regime_map = get_time_regime_to_expected_intervals()
    if time_regime not in regime_map:
        logger.warning(f"{basename}: unknown time regime '{time_regime}'; skipping interval check")
        return

    expected_intervals = regime_map[time_regime]
    timestamps = combine_date_and_time(meta_df["Date"].values, meta_df["Time"].values)
    diffs = np.diff(timestamps).astype("timedelta64[s]").astype(float)

    bad_rows = []
    for i, d in enumerate(diffs):
        if not any(lo <= d <= hi for lo, hi in expected_intervals):
            bad_rows.append((i, float(d)))

    assert len(bad_rows) == 0, (
        f"{basename}: {len(bad_rows)} interval(s) don't match expected ranges for "
        f"'{time_regime}': {bad_rows}"
    )
