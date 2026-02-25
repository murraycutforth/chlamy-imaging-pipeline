"""Hardcoded manual frame-pair corrections for known problematic plates.

The lookup maps each plate basename to a list of raw-CSV row indices (0-based)
that should be removed.  These are equivalent to the ``meta_df_index`` entries
in ``database_creation/manual_error_correction.py``.

IMPORTANT: the row indices are raw-CSV positions (before any other correction).
Because ``remove_all_black_frame_pairs`` may have already removed rows from the
CSV before ``apply_manual_corrections`` runs, raw-CSV positions may no longer
match post-correction positions.  The function therefore locates target rows by
their original timestamps rather than by positional index.  If a target row was
already removed (e.g. by black-frame detection) the correction is silently
skipped.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_filename_to_erroneous_rows() -> dict[str, list[int]]:
    """Map from plate basename → list of raw-CSV row indices (0-based) to remove."""
    return {
        '20231206_99-M6_30s-30s': [61],
        '20240223_16-M6_30s-30s': [23],
        '20240330_23-M6_30s-30s': [61],
        '20240418_9-M6_30s-30s': [47],
        '20240422_6-M6_30s-30s': [9],
        '20240424_12-M6_30s-30s': [11, 26],
        '20240502_17-M6_30s-30s': [15, 74],
        '20231024_3-M1_1min-1min': [77],
        '20231031_4-M2_1min-1min': [57],
        '20231105_5-M1_1min-1min': [37, 38],
        '20231117_7-M1_1min-1min': [61],
        '20240313_21-M1_1min-1min': [13, 24],
        '20240301_18-M1_1min-1min': [11],
        '20240308_17-M2_1min-1min': [35],
        '20240406_20-M1_1min-1min': [71],
        '20240506_1-M5_10min-10min': [31],
        '20240130_12-M6_10min-10min': [35],
        '20240305_18-M5_10min-10min': [67],
        '20240323_22-M5_10min-10min': [69],
        '20240412_2-M5_10min-10min': [60],
        '20240405_1-M6_10min-10min': [42, 71],
        '20240215_15-M4_2h-2h': [18],
        '20231027_3-M4_2h-2h': [40],
        '20240316_21-M4_2h-2h': [17],
        '20240429_24-M4_2h-2h': [6],
        '20231217_10-M3_20h_ML': [21],
        '20240220_16-M3_20h_ML': [12],
        '20240327_23-M3_20h_ML': [4],
        '20240503_21-M3_20h_ML': [36],
        '20231025_3-M2_20h_HL': [15],
        '20231112_6-M2_20h_HL': [34],
        '20240225_8-M2_20h_HL': [36],
        '20240427_24-M2_20h_HL': [4],
    }


def apply_manual_corrections(
    tif: np.ndarray,
    meta_df: pd.DataFrame,
    basename: str,
    raw_meta_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Apply manual frame-pair removals if this basename is in the lookup.

    Parameters
    ----------
    tif:         Post-warmup, post-black-removal TIF array.
    meta_df:     Post-black-removal CSV DataFrame.
    basename:    Plate basename (without extension).
    raw_meta_df: Original (unmodified) CSV DataFrame.  When provided, target
                 rows are located by their raw timestamp so that index drift
                 caused by earlier black-frame removals is handled correctly.
                 If a target row's timestamp is no longer present in ``meta_df``
                 (it was already removed), the correction is skipped.

    For each CSV row k (applied in reverse order):
      - Remove TIF frames [2k, 2k+1]
      - Drop CSV row k
    """
    lookup = get_filename_to_erroneous_rows()
    if basename not in lookup:
        return tif, meta_df

    raw_indices = sorted(lookup[basename])
    logger.warning(f"{basename}: applying manual corrections for raw CSV rows {raw_indices}")

    for k in reversed(raw_indices):
        # Locate the target row by its original timestamp when possible.
        if raw_meta_df is not None and "Date" in raw_meta_df.columns and "Time" in raw_meta_df.columns:
            target_date = raw_meta_df.iloc[k]["Date"]
            target_time = raw_meta_df.iloc[k]["Time"]
            matches = meta_df[
                (meta_df["Date"] == target_date) & (meta_df["Time"] == target_time)
            ]
            if len(matches) == 0:
                logger.warning(
                    f"{basename}: manual correction raw row {k} ({target_date} {target_time}) "
                    "was already removed; skipping"
                )
                continue
            actual_k = int(meta_df.index.get_loc(matches.index[0]))
        else:
            if k >= len(meta_df):
                raise ValueError(
                    f"{basename}: manual correction row index {k} out of range "
                    f"(CSV has {len(meta_df)} rows)"
                )
            actual_k = k

        tif = np.delete(tif, [2 * actual_k, 2 * actual_k + 1], axis=0)
        meta_df = meta_df.drop(meta_df.index[actual_k]).reset_index(drop=True)

    return tif, meta_df
