# Authors: Murray Cutforth, Leron Perez
#
# Stage 1: Well Segmentation Preprocessing
#
# Reads cleaned TIF files from CLEANED_RAW_DATA_DIR (Stage 0 output),
# segments individual wells from each 384-well plate image stack,
# and saves per-plate .npy arrays to WELL_SEGMENTATION_DIR.

import logging

import numpy as np
from segment_multiwell_plate.segment_multiwell_plate import _correct_rotations_l1nn
from skimage import io
from segment_multiwell_plate import segment_multiwell_plate, find_well_centres
from tqdm import tqdm
import pandas as pd

from chlamy_impi.database_creation.utils import parse_name
from chlamy_impi.lib.visualize_well_segmentation import visualise_channels, visualise_well_histograms, \
    visualise_grid_crop, visualise_well_mosaic
from chlamy_impi.paths import find_all_cleaned_tif_images, well_segmentation_output_dir_path, npy_img_array_path, \
    validate_stage1_inputs, well_segmentation_visualisation_dir_path, well_segmentation_histogram_dir_path, \
    get_well_segmentation_processing_results_df_filename, get_database_output_dir, well_segmentation_mosaic_path
from chlamy_impi.well_segmentation_preprocessing.well_segmentation_assertions import assert_expected_shape

logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)


OUTPUT_VISUALISATIONS = False
LOGGING_LEVEL = logging.INFO


def load_image(filename):
    return io.imread(filename)  # open tiff file in read mode


def save_img_array(img_array, name):

    outdir = npy_img_array_path(name).parent
    if not outdir.exists():
        outdir.mkdir(parents=True)

    logger.debug(f"Saving image array of shape {img_array.shape} to {npy_img_array_path(name)}")
    np.save(npy_img_array_path(name), img_array.astype(np.float32))


def generate_all_mosaics(filenames):
    """Generate a well mosaic PNG for every successfully segmented plate.

    Skips plates whose mosaic already exists.  Loads NPY files from disk so
    this step is independent of whether the plate was processed in this run or
    in a previous one.
    """
    logger.info("Generating well mosaics for all segmented plates...")
    for filename in filenames:
        name = filename.stem
        npy_path = npy_img_array_path(name)
        mosaic_path = well_segmentation_mosaic_path(name)

        if not npy_path.exists():
            logger.debug(f"Skipping mosaic for {name}: NPY not found")
            continue
        if mosaic_path.exists():
            logger.debug(f"Skipping mosaic for {name}: already exists")
            continue

        try:
            img_array = np.load(npy_path)
            visualise_well_mosaic(img_array, name, mosaic_path)
        except Exception as e:
            logger.error(f"Failed to generate mosaic for {name}: {e}")


def main():
    logger.info("\n" + "=" * 32 + "\nStarting Stage 1: Well Segmentation\n" + "=" * 32)
    validate_stage1_inputs()
    filenames = find_all_cleaned_tif_images()

    sep = "\n\t"
    logger.info(f"Found a total of {len(filenames)} cleaned tif files: \n\t{sep.join(str(x) for x in filenames)}")

    error_messages = []  # Log all errors and write to file at the end
    processing_results = []

    for filename in tqdm(filenames):
        result = {
            'filename': filename.stem,
            'status': 1,
        }

        try:
            name = filename.stem
            npy_outpath = npy_img_array_path(name)
            outpath = well_segmentation_output_dir_path(name)
            plate_num, measurement_num, time_regime = parse_name(filename.name)

            logger.debug(f"Processing plate_num={plate_num}, measurement_num={measurement_num}, time_regime={time_regime}")

            if npy_outpath.exists():
                logger.debug(f"Skipping {name} as it already exists")
                processing_results.append(result)
                continue

            outpath.mkdir(parents=True, exist_ok=True)

            tif = load_image(filename)

            logger.debug(f"NUM_TIMESTEPS={tif.shape[0]}")

            assert len(tif.shape) == 3
            for frame in tif:
                assert np.std(frame) > 1e-6, f"Image {filename} is all black"

            # Note - if these parameters need to be tuned, see test_rotation_correction.py
            img_array, well_coords, i_vals, j_vals = segment_multiwell_plate(
                tif,
                peak_finder_kwargs={"peak_prominence": 1 / 25, "filter_threshold": 0.2, "width": 2},
                blob_log_kwargs={"threshold": 0.12, "min_sigma": 1, "max_sigma": 2, "exclude_border": 10},
                output_full=True,
            )

            if OUTPUT_VISUALISATIONS:
                visualise_channels(tif, savedir=well_segmentation_visualisation_dir_path(name))
                visualise_well_histograms(img_array, name, savedir=well_segmentation_histogram_dir_path(name))
                visualise_grid_crop(tif, img_array, i_vals, j_vals, well_coords, savedir=well_segmentation_visualisation_dir_path(name))

            assert_expected_shape(i_vals, j_vals, plate_num)
            save_img_array(img_array, name)

            processing_results.append(result)

        except Exception as e:
            logger.error(f"Error in well segmentation processing of {filename}: {e}")
            error_messages.append(f"{filename.stem}: {e}")

            result['status'] = 0
            processing_results.append(result)

    with open(get_database_output_dir() / "well_segmentation_errors.txt", "w") as f:
        for msg in error_messages:
            f.write(msg + "\n")

    df = pd.DataFrame(processing_results)
    df.to_csv(get_well_segmentation_processing_results_df_filename(), index=False)

    logger.info(f'Failed well segmentation in {len(error_messages)} files')
    for error in error_messages:
        logger.error(error)

    generate_all_mosaics(filenames)

    logger.info("Program completed normally")


if __name__ == "__main__":
    logging.basicConfig(level=LOGGING_LEVEL)
    main()
