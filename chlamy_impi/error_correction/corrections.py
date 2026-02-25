"""Frame-pair corrections applied to raw TIF + CSV data.

Raw TIF structure
-----------------
Every raw TIF begins with two "warmup" frames (frames 0 and 1) that duplicate
the first real measurement (frames 4 and 5).  These warmup frames are **not**
recorded in the CSV.  Immediately after them there is usually a full-black
frame pair (frames 2 and 3, also absent from the CSV) from a failed pre-run
trigger.  The layout is therefore:

    [0,1]   warmup duplicate  — no CSV row
    [2,3]   trigger pair      — no CSV row  (usually full-black; occasionally
                                             half-black with one non-zero frame)
    [4,5]   measurement 0     — CSV row 0
    [6,7]   measurement 1     — CSV row 1
    ...

Occasionally the Fm frame of a measurement is all-black while its F0 partner
is valid ("half-black pair").  That measurement IS recorded in the CSV (with
Fm values = 0).

Correction order (called by main.py)
--------------------------------------
1. remove_warmup_pair        — always, TIF only
2. remove_all_black_frame_pairs — full-black (TIF only) + half-black (TIF+CSV)
3. remove_duplicate_initial_frame_pair — safety check, rarely fires

After steps 1–2 the invariant  tif.shape[0] == 2 * len(meta_df)  holds.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def remove_warmup_pair(tif: np.ndarray) -> np.ndarray:
    """Remove the first 2 warmup frames from the raw TIF.

    The camera always prepends a duplicate of the first real measurement at
    TIF indices [0, 1].  These frames have no corresponding CSV row and must
    be removed before any CSV-aligned pair logic can run.
    """
    assert tif.shape[0] >= 2, "TIF too short to contain a warmup pair"
    if not (np.all(tif[0] == tif[4]) and np.all(tif[1] == tif[5])):
        logger.warning(
            "Warmup frames [0,1] do not match frames [4,5] — removing anyway"
        )
    logger.debug("Removing warmup pair (frames 0 and 1)")
    return tif[2:]


def remove_all_black_frame_pairs(
    tif: np.ndarray, meta_df: pd.DataFrame
) -> tuple[np.ndarray, pd.DataFrame]:
    """Remove frame pairs that contain at least one all-black frame.

    Must be called **after** ``remove_warmup_pair``.

    Two situations are handled:

    Pre-measurement trigger pairs (TIF only, no CSV row)
        After warmup removal the TIF may still have extra pairs at the front
        that have no corresponding CSV row.  These arise from failed pre-run
        camera triggers and may be full-black (both frames 0) or half-black
        (one frame 0).  The count ``n_pre = n_tif_pairs - n_csv_rows`` gives
        the number of such pairs; they are removed from the TIF only.

    Half-black measurement pairs (one frame black; pair IS in CSV)
        Occasionally the Fm frame of a real measurement is all-black while
        its F0 partner is valid.  The measurement IS recorded in the CSV.
        These are removed from both the TIF and the CSV.

    After this function returns ``tif.shape[0] == 2 * len(meta_df)``.
    """
    n_tif_pairs = tif.shape[0] // 2
    n_csv_rows = len(meta_df)
    n_pre = n_tif_pairs - n_csv_rows

    if n_pre < 0:
        raise ValueError(
            f"TIF has fewer pairs ({n_tif_pairs}) than CSV rows ({n_csv_rows}); "
            "cannot align TIF and CSV"
        )

    # --- Step A: remove pre-measurement pairs from TIF only ---
    # These pairs (full-black or half-black trigger frames) precede real measurements
    # and have no corresponding CSV row.
    if n_pre > 0:
        max_pre = tif.reshape(tif.shape[0], -1).max(axis=1)
        pre_types = []
        for k in range(n_pre):
            if max_pre[2 * k] == 0 and max_pre[2 * k + 1] == 0:
                pre_types.append("full-black")
            elif max_pre[2 * k] == 0 or max_pre[2 * k + 1] == 0:
                pre_types.append("half-black")
            else:
                pre_types.append("non-black")
        logger.warning(
            f"Removing {n_pre} pre-measurement pair(s) from TIF only (no CSV row): "
            f"types={pre_types}"
        )
        tif = tif[2 * n_pre:]

    # --- Step B: half-black measurement pairs (one frame black; pair IS in CSV) ---
    # TIF and CSV are now aligned: pair k ↔ CSV row k.
    max_per_frame = tif.reshape(tif.shape[0], -1).max(axis=1)
    n_pairs = tif.shape[0] // 2
    half_black = [
        k for k in range(n_pairs)
        if (max_per_frame[2 * k] == 0) != (max_per_frame[2 * k + 1] == 0)
    ]

    if half_black:
        logger.warning(
            f"Removing {len(half_black)} half-black pair(s) (has CSV row): {half_black}"
        )
        for k in reversed(half_black):
            tif = np.delete(tif, [2 * k, 2 * k + 1], axis=0)
            meta_df = meta_df.drop(meta_df.index[k]).reset_index(drop=True)

    return tif, meta_df


def remove_duplicate_initial_frame_pair(
    tif: np.ndarray, meta_df: pd.DataFrame
) -> tuple[np.ndarray, pd.DataFrame]:
    """Remove the first frame pair if it is a duplicate of the second pair.

    Safety check: after ``remove_warmup_pair`` this should never fire under
    normal conditions.  Kept for robustness.
    """
    if tif.shape[0] < 4:
        return tif, meta_df

    if np.all(tif[0] == tif[2]) and np.all(tif[1] == tif[3]):
        logger.warning("Found duplicate initial frame pair — removing frames 0,1 and CSV row 0")
        tif = tif[2:]
        meta_df = meta_df.iloc[1:].reset_index(drop=True)

    return tif, meta_df
