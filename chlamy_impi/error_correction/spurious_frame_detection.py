"""
In this file, we aim to automatically detect and remove spurious frames by looking at the measurement times.

TODO: currently not working / unfinished
"""
import logging

import numpy as np
import pandas as pd

from chlamy_impi.database_creation.constants import get_time_regime_to_expected_intervals
from chlamy_impi.database_creation.utils import parse_name
from chlamy_impi.error_correction.plot_measurement_times import combine_date_and_time

logger = logging.getLogger(__name__)


def detect_spurious_frames(df: pd.DataFrame, img_arr: np.array, basename: str) -> tuple[list, list]:
    """
    Detects spurious frames in the image array and corresponding rows in the metadata DataFrame.

    Parameters:
    df (pd.DataFrame): Metadata DataFrame.
    img_arr (np.array): Image array.
    basename (str): Base filename.

    Returns:
    tuple[list, list]: Tuple containing lists of spurious frames and corresponding indices in the metadata DataFrame.
    """
    spurious_frame_inds = []
    spurious_df_inds = []

    plate_num, measurement_num, time_regime = parse_name(basename + ".npy")

    expected_time_intervals = get_time_regime_to_expected_intervals()[time_regime]

    times = df["Time"].values
    dates = df["Date"].values
    timestamps = combine_date_and_time(dates, times)

    # time_increments[i] tells us the time difference between the ith and (i+1)th rows of the df
    # these in turn correspond to the 2i-th and (2i+1)th frames of the img_arr, unless there are spurious frames to mess things up
    time_increments = (timestamps[1:] - timestamps[:-1]).astype('timedelta64[s]')

    num_spurious_frames = 0

    # The last time interval is always expected to be approx 15 mins
    assert deltat_in_intervals(time_increments[-1], {(870., 930.)})

    # Use this flag because erroneous time intervals occur in pairs (one before, one after)
    skipnext = False

    for i in range(len(time_increments) - 1):
        if not deltat_in_intervals(time_increments[i], expected_time_intervals):
            if skipnext:
                skipnext = False
                continue

            logger.warning(f"{basename}: Found spurious frame at index {i + 1} with time increment {time_increments[i]}, expected {expected_time_intervals}")
            #print(f"Corresponding timestamps: {timestamps[i]}, {timestamps[i + 1]}")
            #print(f"Row of df: {df.iloc[i + 1]}")

            spurious_df_inds.append(i + 1)
            spurious_frame_inds.append(df_ind_to_frame_ind(i + 1, num_spurious_frames))
            num_spurious_frames += 1

            skipnext = True

    return spurious_df_inds, spurious_frame_inds


def deltat_in_intervals(deltat: np.timedelta64, intervals: set[tuple]) -> bool:
    for interval in intervals:
        if interval[0] <= deltat.astype(float) <= interval[1]:
            return True

    return False

def df_ind_to_frame_ind(df_ind: int, num_spurious: int) -> int:
    return 2 * df_ind - num_spurious


