import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def filename_to_erroneous_frames() -> dict[str, tuple[int, int]]:
    """In this function I have manually recorded the (meta_df_index, frame index) pairs for files which exhibit
    the known camera error in which a single erroneous frame is inserted into the image stack. A new line also appears
    in the corresponding .csv file, which we need to remove.
    """
    filename_to_inds = {
        '20231206_99-M6_30s-30s': [(61, 122)],
        '20240223_16-M6_30s-30s': [(23, 46)],
        '20240330_23-M6_30s-30s': [(61, 122)],
        '20240418_9-M6_30s-30s': [(47, 94)],
        '20240422_6-M6_30s-30s': [(9, 18)],
        '20240424_12-M6_30s-30s': [(11, 22), (26, 51)],
        '20240502_17-M6_30s-30s': [(15, 30), (74, 147)],
        '20231024_3-M1_1min-1min': [(77, 154)],
        '20231031_4-M2_1min-1min': [(57, 114)],
        '20231105_5-M1_1min-1min': [(37, 74), (38, 75)],
        '20231117_7-M1_1min-1min': [(61, 122)],
        '20240313_21-M1_1min-1min': [(13, 26), (24, 47)],
        '20240301_18-M1_1min-1min': [(11, 22)],
        '20240308_17-M2_1min-1min': [(35, 70)],
        '20240406_20-M1_1min-1min': [(71, 142)],
        '20240506_1-M5_10min-10min': [(31, 62)],
        '20240130_12-M6_10min-10min': [(35, 70)],
        '20240305_18-M5_10min-10min': [(67, 134)],
        '20240323_22-M5_10min-10min': [(69, 138)],
        '20240412_2-M5_10min-10min': [(60, 120)],
        '20240405_1-M6_10min-10min': [(42, 84), (71, 141)],
        '20240215_15-M4_2h-2h': [(18, 36)],
        '20231027_3-M4_2h-2h': [(40, 80)],
        '20240316_21-M4_2h-2h': [(17, 34)],
        '20240429_24-M4_2h-2h': [(6, 12)],
        '20231217_10-M3_20h_ML': [(21, 42)],
        '20240220_16-M3_20h_ML': [(12, 24)],
        '20240327_23-M3_20h_ML': [(4, 8)],
        '20240503_21-M3_20h_ML': [(36, 72)],
        '20231025_3-M2_20h_HL': [(15, 30)],
        '20231112_6-M2_20h_HL': [(34, 68)],
        '20240225_8-M2_20h_HL': [(36, 72)],
        '20240427_24-M2_20h_HL': [(4, 8)],
    }

    return filename_to_inds


def manually_fix_erroneous_time_points(meta_df, img_array, base_filename: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Implementation of manual fix for erroneous frames in the image stack. Removes frames from the image array,
    as well as corresponding rows from the meta_df.
    """

    filename_to_inds = filename_to_erroneous_frames()

    if base_filename in filename_to_inds:
        for meta_df_index, frame_num in filename_to_inds[base_filename]:
            logger.warning(f'Fixing erroneous frame {frame_num} for {base_filename}')

            meta_df = meta_df.drop(meta_df.index[meta_df_index])

            img_array = np.delete(img_array, frame_num, axis=2)

    assert img_array.shape[2] in {84, 164}, f"Unexpected number of frames ({img_array.shape[2]}) for {base_filename}"
    assert img_array.shape[2] == len(meta_df) * 2, f"Number of frames ({img_array.shape[2]}) does not match number of rows in meta_df ({len(meta_df)})"

    return meta_df, img_array


def remove_repeated_initial_frame(img_array) -> np.array:
    """Check for a repetition of the first pair of frames
    """
    if (np.all(img_array[:, :, 0, ...] == img_array[:, :, 2, ...]) and
        np.all(img_array[:, :, 3, ...] == img_array[:, :, 1, ...])):
            logger.warning("Found repeated initial frame pair. Removing!")
            img_array = img_array[:, :, 2:, ...]

    return img_array


def remove_repeated_initial_frame_tif(tif: np.array) -> np.array:
    """As above, but for raw tif files which have time step in the first dimension
    """
    if (np.all(tif[0, ...] == tif[2, ...]) and
        np.all(tif[3, ...] == tif[1, ...])):
        logger.warning("Found repeated initial frame pair. Removing!")
        tif = tif[2:, ...]

    return tif


def remove_failed_photos(tif):
    """
    Remove photos that are all black

    Input:
        tif: numpy array (num_images, height, width)
    Output:
        tif: numpy array of shape (num_images, height, width)
    """
    max_per_timestep = tif.max(1).max(1)
    keep_image = max_per_timestep > 0

    logger.warning(f"Discarding {sum(~keep_image)} images (indices {list(np.argwhere(~keep_image))}) which are all black")

    tif = tif[keep_image]
    return tif


# DEPRECATED - use manually_fix_erroneous_time_points instead
# TODO - remove in future commits if manual fix works
def fix_erroneous_time_points(meta_df, img_array):
    """In this function, we handle issues with the fluorescence images. Occasionally there is an error with the data
    (possibly caused by a battery failure for the saturating pulse), which is recorded in the .csv meta data files.
    We need to locate and remove these single frames, and then re-align the time points to the frames by removing the
    appropriate frame from the image stack.

    NOTE:
        This function is very fragile. It tries to deal with a very wide variety of columns in the meta_df.
        Since we cant seem to rely on the meta_df to be consistent, maybe a better approach to detecting these failure
        cases would be to look in the image stack for the anomalous frames, and then delete them there.

    """
    # One possible column layout in the csv files
    if set(meta_df.columns) == {"Date", "Time", "No.", "PAR", "F1", "F2", "F3", "Fm'1", "Fm'2", "Fm'3", "Y(II)1", "Y(II)2", "Y(II)3"}:
        # Look for cases where Fm'1 == Fm'2 == Fm'3 == 0, as these signify failed measurements
        meta_df["failed_measurement"] = (meta_df["Fm'1"] == 0.) & (meta_df["Fm'2"] == 0.) & (meta_df["Fm'3"] == 0.)
    elif set(meta_df.columns) == {'Date', 'Time', 'No.', 'PAR', 'Y(II)1', 'Y(II)2', 'Y(II)3', 'NPQ1', 'NPQ2', 'NPQ3'}:
        # Look for cases where Y(II) == 0 and NPQ == 1, as these signify failed measurements
        meta_df["failed_measurement"] = (meta_df["Y(II)1"] == 0.) & (meta_df["Y(II)2"] == 0.) & (meta_df["Y(II)3"] == 0.) & (meta_df["NPQ1"] == 1.) & (meta_df["NPQ2"] == 1.) & (meta_df["NPQ3"] == 1.)
    elif set(meta_df.columns) == {'Date', 'Time', 'No.', 'PAR', 'F1', "Fm'1", 'Y(II)1'}:
        # Look for cases where Fm'1 == 0 and Y(II)1 == 0, as these signify failed measurements
        meta_df["failed_measurement"] = (meta_df["Fm'1"] == 0.) & (meta_df["Y(II)1"] == 0.)
    elif set(meta_df.columns) == {'Date', 'Time', 'No.', 'PAR', 'Y(II)1', 'Y(II)2', 'Y(II)3'}:
        meta_df["failed_measurement"] = (meta_df["Y(II)1"] == 0.) & (meta_df["Y(II)2"] == 0.) & (meta_df["Y(II)3"] == 0.)
    else:
        #meta_df["failed_measurement"] = (meta_df["Fm'1"] == 0.) & (meta_df["Fm'2"] == 0.) & (meta_df["Fm'3"] == 0.) & (meta_df["Y(II)1"] == 0.) & (meta_df["Y(II)2"] == 0.) & (meta_df["Y(II)3"] == 0.)

        logger.warning(f"Unknown column layout in meta_df: {meta_df.columns}")

        raise NotImplementedError("Unknown column layout in meta_df")

    failed_rows = meta_df[meta_df["failed_measurement"]]
    assert len(failed_rows) <= 3, f"Too many failed measurements: {len(failed_rows)}"

    for i, (ind, row) in enumerate(failed_rows.iterrows()):
        print(f"Found failed measurement at index {ind}. Attempting to fix!")
        print(f"Bad row: {row}")

        meta_df = meta_df.drop(ind)
        bad_frame_index = (ind - i) * 2
        img_array = np.delete(img_array, bad_frame_index, axis=2)

    print(len(failed_rows), "failed measurements found")
    print(flush=True)

    assert img_array.shape[2] % 2 == 0, f"Odd number of frames ({img_array.shape[2]})"
    assert img_array.shape[2] == len(meta_df) * 2, f"Number of frames ({img_array.shape[2]}) does not match number of rows in meta_df ({len(meta_df)})"

    return meta_df, img_array

