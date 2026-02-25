"""Stage 2a: Image Processing → Parquets

Entry point: python -m chlamy_impi.image_processing.main

Reads NPY well arrays (from Stage 1) and cleaned CSV files (from Stage 0).
Computes photosynthetic parameters per well and writes three parquets:

  output/image_processing/plates.parquet     — one row per experiment plate
  output/image_processing/wells.parquet      — one row per (plate × well)
  output/image_processing/timeseries.parquet — one row per (plate × well × time_step)

Key improvements over the original Stage 2 pass:
  - Mask computed once per NPY (original computes it twice)
  - Wells and timeseries DataFrames built vectorised via np.meshgrid
  - Identity spreadsheet not touched here (that is Stage 2b's job)
"""

import logging

import numpy as np
import pandas as pd

from chlamy_impi.database_creation.constants import get_possible_frame_numbers
from chlamy_impi.database_creation.main import (
    get_npy_and_csv_filenames,
    prepare_img_array_and_df,
)
from chlamy_impi.database_creation.utils import (
    parse_name,
    compute_measurement_times,
)
from chlamy_impi.lib.fv_fm_functions import compute_all_fv_fm_averaged
from chlamy_impi.lib.mask_functions import compute_threshold_mask
from chlamy_impi.lib.npq_functions import compute_all_ynpq_averaged
from chlamy_impi.lib.y2_functions import compute_all_y2_averaged
from chlamy_impi.paths import (
    get_image_processing_output_dir,
    get_plates_parquet_path,
    get_timeseries_parquet_path,
    get_wells_parquet_path,
)

logger = logging.getLogger(__name__)

DEV_MODE = False
IGNORE_ERRORS = False


def _well_id(i: int, j: int) -> str:
    """Convert zero-based (i, j) to well ID string, e.g. (0, 0) → 'A01'."""
    return f"{chr(ord('A') + i)}{j + 1:02d}"


def process_plate(filename_npy, filename_meta):
    """Process a single plate.

    Returns (plate_row dict, wells_df, ts_df) or raises on error.
    """
    plate_num, measurement_num, light_regime, start_date = parse_name(
        filename_npy.name, return_date=True
    )

    img_array, meta_df = prepare_img_array_and_df(filename_meta, filename_npy)
    assert img_array.shape[2] % 2 == 0, f"Odd frame count: {img_array.shape[2]}"

    measurement_times = compute_measurement_times(meta_df=meta_df)
    assert img_array.shape[2] // 2 == len(measurement_times), (
        f"Frame/CSV mismatch: {img_array.shape[2]} frames "
        f"but {len(measurement_times)} CSV rows"
    )

    # Compute mask once, extracting thresholds at the same time
    mask_array, (dark_threshold, light_threshold) = compute_threshold_mask(
        img_array, return_thresholds=True
    )

    # Compute photosynthetic parameters
    y2_array, y2_std_array = compute_all_y2_averaged(img_array, mask_array, return_std=True)
    fv_fm_array, fv_fm_std_array = compute_all_fv_fm_averaged(img_array, mask_array, return_std=True)
    ynpq_array = compute_all_ynpq_averaged(img_array, mask_array)

    Ni, Nj = img_array.shape[:2]
    n_steps = y2_array.shape[2]  # number of Y2/NPQ time steps

    # --- plates row (one per experiment) ---
    plate_row = {
        "plate": plate_num,
        "measurement": measurement_num,
        "start_date": start_date,
        "light_regime": light_regime,
        "dark_threshold": dark_threshold,
        "light_threshold": light_threshold,
        "num_frames": img_array.shape[2],
    }

    # --- wells DataFrame (one row per well, vectorised) ---
    I2d, J2d = np.meshgrid(np.arange(Ni), np.arange(Nj), indexing="ij")
    mask_area = np.sum(mask_array.reshape(Ni, Nj, -1), axis=-1)
    well_ids = [_well_id(i, j) for i, j in zip(I2d.ravel(), J2d.ravel())]

    wells_df = pd.DataFrame(
        {
            "plate": plate_num,
            "measurement": measurement_num,
            "start_date": start_date,
            "i": I2d.ravel().astype(np.int16),
            "j": J2d.ravel().astype(np.int16),
            "well_id": well_ids,
            "fv_fm": fv_fm_array.ravel(),
            "fv_fm_std": fv_fm_std_array.ravel(),
            "mask_area": mask_area.ravel().astype(np.int32),
            "measurement_time_0": measurement_times[0],
        }
    )

    # --- timeseries DataFrame (long format, one row per well × time_step) ---
    I3d, J3d, S3d = np.meshgrid(
        np.arange(Ni), np.arange(Nj), np.arange(1, n_steps + 1), indexing="ij"
    )
    # step_times[k] = measurement_times[k+1] (time_step is 1-based)
    step_times = np.array([measurement_times[s] for s in range(1, n_steps + 1)])

    ts_df = pd.DataFrame(
        {
            "plate": plate_num,
            "measurement": measurement_num,
            "start_date": start_date,
            "i": I3d.ravel().astype(np.int16),
            "j": J3d.ravel().astype(np.int16),
            "time_step": S3d.ravel().astype(np.int16),
            "y2": y2_array.ravel(),
            "y2_std": y2_std_array.ravel(),
            "ynpq": ynpq_array.ravel(),
            "measurement_time": step_times[S3d.ravel() - 1],
        }
    )

    return plate_row, wells_df, ts_df


def main():
    get_image_processing_output_dir()  # ensure output dir exists

    filenames_meta, filenames_npy = get_npy_and_csv_filenames()

    if DEV_MODE:
        filenames_npy = filenames_npy[:5]
        filenames_meta = filenames_meta[:5]
        logger.info(f"DEV_MODE: processing only {len(filenames_npy)} files")

    plates_rows = []
    wells_dfs = []
    ts_dfs = []
    failed_files = []

    for filename_npy, filename_meta in zip(filenames_npy, filenames_meta):
        logger.info(f"Processing: {filename_npy.name}")
        try:
            plate_row, wells_df, ts_df = process_plate(filename_npy, filename_meta)
            plates_rows.append(plate_row)
            wells_dfs.append(wells_df)
            ts_dfs.append(ts_df)
        except Exception as e:
            if IGNORE_ERRORS:
                logger.error(f"Error processing {filename_npy.stem}: {e}")
                failed_files.append({"filename": filename_npy.stem, "error": str(e)})
            else:
                raise

    plates_df = pd.DataFrame(plates_rows)
    wells_df = pd.concat(wells_dfs, ignore_index=True)
    ts_df = pd.concat(ts_dfs, ignore_index=True)

    logger.info(f"plates.parquet shape: {plates_df.shape}")
    logger.info(f"wells.parquet shape: {wells_df.shape}")
    logger.info(f"timeseries.parquet shape: {ts_df.shape}")

    plates_df.to_parquet(get_plates_parquet_path(), index=False)
    wells_df.to_parquet(get_wells_parquet_path(), index=False)
    ts_df.to_parquet(get_timeseries_parquet_path(), index=False)

    logger.info("Stage 2a complete — parquets written to output/image_processing/")

    if failed_files:
        logger.error(f"Failed to process {len(failed_files)} files: {failed_files}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG if DEV_MODE else logging.INFO)
    main()
