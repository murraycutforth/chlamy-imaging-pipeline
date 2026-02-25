import logging
from pathlib import Path

import numpy as np
import pandas as pd

from chlamy_impi.error_correction.automated_spurious_frame_fix import fix_spurious_frames
from chlamy_impi.paths import get_npy_and_csv_filenames, corrected_well_segmentation_output_dir_path, get_database_output_dir

logger = logging.getLogger(__name__)


def correct_img_array_and_df(filename_meta, filename_npy) -> tuple[np.ndarray, pd.DataFrame]:
    """Load the image array (pre segmented into wells) and the metadata dataframe for a given plate,
    and attempt to automatically fix and spurious frame errors
    """
    img_array = np.load(filename_npy)
    meta_df = pd.read_csv(filename_meta, header=0, delimiter=";").iloc[:, :-1]

    meta_df, img_array = fix_spurious_frames(meta_df, img_array, filename_npy.stem)

    return img_array, meta_df


def store_error_corrected_data(img_array: np.ndarray, meta_df: pd.DataFrame, f_np: Path):
    assert img_array.shape[2] % 2 == 0
    assert img_array.shape[2] == len(
        meta_df) * 2, f"Number of frames ({img_array.shape[2]}) does not match number of rows in meta_df ({len(meta_df)})"

    filename = f_np.stem
    outdir = corrected_well_segmentation_output_dir_path(filename)
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / (filename + '.npy'), img_array)
    meta_df.to_csv(outdir / (filename + '.csv'), index=False)


def main():
    logger.info("="*32)
    logger.info("Starting error correction code")
    logger.info("="*32)
    error_messages = []

    filename_meta, filename_npy = get_npy_and_csv_filenames()

    logger.info(f"Found total of {len(filename_npy)} plates")

    for f_np, f_meta in zip(filename_npy, filename_meta):
        try:
            img_array, meta_df = correct_img_array_and_df(f_meta, f_np)
            store_error_corrected_data(img_array, meta_df, f_np)
        except Exception as e:
            logger.error(f"Error processing file {f_np.stem}. Skipping.")
            error_messages.append(f"{f_np.stem}: {e}")
            continue

    num_errors = len(error_messages)
    logger.info(f"{len(filename_npy) - num_errors} plates processed successfully.")
    logger.info(f"Found {num_errors} errors")

    with open(get_database_output_dir() / "frame_correction_errors.txt", "w") as f:
        for msg in error_messages:
            f.write(msg + "\n")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
