"""This file is a central place to store all the paths used in the project. The functions here should be used anywhere
that a path is needed, rather than hardcoding the path somewhere else.
"""

import datetime
import re
from pathlib import Path
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).parent.parent

# INPUT DIR should contain .tif and .csv files from the camera data folder on google drive
# https://drive.google.com/drive/folders/1rU8VOIdwBuDX_N6MTn0Bg5SYYb-Ov8zv
INPUT_DIR = PROJECT_ROOT / "data"

# WELL_SEGMENTATION_DIR is where we save the output of the well segmentation as .npy files
WELL_SEGMENTATION_DIR = PROJECT_ROOT / "output" / "well_segmentation_cache"

# CORRECTED_WELL_SEGMENTATION_DIR is where we store well segmentation arrays and meta_dfs after error correction
CORRECTED_WELL_SEGMENTATION_DIR = PROJECT_ROOT / "output" / "corrected_well_segmentation_cache"

# IDENTITY_SPREADSHEET_PATH is the path to the spreadsheet containing the plate identity information
# https://docs.google.com/spreadsheets/d/1_UcLC4jbI04Rnpt2vUkSCObX8oUY6mzl/edit?usp=drive_link&ouid=108504591016316429773&rtpof=true&sd=true
# Update - final sheet is now here:
# https://docs.google.com/spreadsheets/d/1reX1t-C9rwjwhJWRowGZPUV7F4B1Wvvw/edit#gid=1935584839
IDENTITY_SPREADSHEET_PATH = \
    INPUT_DIR / "20251002 Finalized Identities Phase I plates.xlsx"

# DATABASE_DIR is where we save the output of the database creation as .csv files
DATABASE_DIR = PROJECT_ROOT / "output" / "database_creation"

# CLEANED_RAW_DATA_DIR is where we save cleaned TIF + CSV files after Stage 0 error correction
CLEANED_RAW_DATA_DIR = PROJECT_ROOT / "output" / "cleaned_raw_data"

# IMAGE_PROCESSING_DIR is where we save parquet files produced by Stage 2a image processing
IMAGE_PROCESSING_DIR = PROJECT_ROOT / "output" / "image_processing"


def find_all_tif_images():
    return list(INPUT_DIR.glob("*.tif"))


def get_cleaned_raw_data_dir() -> Path:
    return CLEANED_RAW_DATA_DIR


def find_all_cleaned_tif_images() -> list[Path]:
    return sorted(CLEANED_RAW_DATA_DIR.glob("*.tif"))


def find_all_raw_tif_and_csv() -> list[tuple[Path, Path]]:
    """Return sorted list of (tif_path, csv_path) pairs from INPUT_DIR."""
    tif_files = sorted(INPUT_DIR.glob("*.tif"))
    pairs = []
    for tif_path in tif_files:
        csv_path = INPUT_DIR / (tif_path.stem + ".csv")
        if csv_path.exists():
            pairs.append((tif_path, csv_path))
        else:
            logger.warning(f"No CSV found for {tif_path.name}, skipping")
    return pairs


def well_segmentation_output_dir_path(name) -> Path:
    savedir = WELL_SEGMENTATION_DIR / name
    return savedir


def corrected_well_segmentation_output_dir_path(name) -> Path:
    savedir = CORRECTED_WELL_SEGMENTATION_DIR / name
    return savedir


def well_segmentation_visualisation_dir_path(name) -> Path:
    savedir = well_segmentation_output_dir_path(name) / "visualisation_raw"
    return savedir


def well_segmentation_histogram_dir_path(name) -> Path:
    savedir = well_segmentation_output_dir_path(name) / "visualisation_histograms"
    return savedir


def well_segmentation_mosaic_path(name) -> Path:
    return WELL_SEGMENTATION_DIR / "mosaics" / f"{name}_mosaic.png"


def mask_mosaic_path(name) -> Path:
    return IMAGE_PROCESSING_DIR / "mask_visualisations" / f"{name}_mask_mosaic.png"


def mask_heatmap_path(name) -> Path:
    return IMAGE_PROCESSING_DIR / "mask_visualisations" / f"{name}_mask_heatmap.png"


def npy_img_array_path(name):
    return WELL_SEGMENTATION_DIR / f"{name}.npy"


def get_identity_spreadsheet_path():
    return IDENTITY_SPREADSHEET_PATH



def get_database_output_dir():
    if not DATABASE_DIR.exists():
        DATABASE_DIR.mkdir()
    return DATABASE_DIR


def get_image_processing_output_dir() -> Path:
    if not IMAGE_PROCESSING_DIR.exists():
        IMAGE_PROCESSING_DIR.mkdir(parents=True)
    return IMAGE_PROCESSING_DIR


def get_plates_parquet_path() -> Path:
    return IMAGE_PROCESSING_DIR / "plates.parquet"


def get_wells_parquet_path() -> Path:
    return IMAGE_PROCESSING_DIR / "wells.parquet"


def get_timeseries_parquet_path() -> Path:
    return IMAGE_PROCESSING_DIR / "timeseries.parquet"


def get_csv_filename():
    return DATABASE_DIR / "database.csv"


def get_dated_run_dir(date=None) -> Path:
    """Returns e.g. output/database_creation/2026-02-26/, creating it if needed."""
    if date is None:
        date = datetime.date.today()
    run_dir = DATABASE_DIR / str(date)
    if not run_dir.exists():
        run_dir.mkdir(parents=True)
    return run_dir


def get_dated_csv_filename(date=None) -> Path:
    """Returns e.g. output/database_creation/2026-02-26/database_2026-02-26.csv"""
    if date is None:
        date = datetime.date.today()
    return get_dated_run_dir(date) / f"database_{date}.csv"


def find_previous_database() -> Path | None:
    """Scan DATABASE_DIR for YYYY-MM-DD/ subdirs containing database_YYYY-MM-DD.csv; return most recent path."""
    pattern = re.compile(r"^(\d{4}-\d{2}-\d{2})$")
    candidates = []
    if DATABASE_DIR.exists():
        for d in DATABASE_DIR.iterdir():
            if d.is_dir():
                m = pattern.match(d.name)
                if m:
                    try:
                        date = datetime.date.fromisoformat(m.group(1))
                        csv_path = d / f"database_{date}.csv"
                        if csv_path.exists():
                            candidates.append((date, csv_path))
                    except ValueError:
                        pass
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def find_previous_database_excluding_today() -> Path | None:
    """Like find_previous_database but excludes today's dated subdir (for pre-save calls)."""
    today = datetime.date.today()
    pattern = re.compile(r"^(\d{4}-\d{2}-\d{2})$")
    candidates = []
    if DATABASE_DIR.exists():
        for d in DATABASE_DIR.iterdir():
            if d.is_dir():
                m = pattern.match(d.name)
                if m:
                    try:
                        date = datetime.date.fromisoformat(m.group(1))
                        if date != today:
                            csv_path = d / f"database_{date}.csv"
                            if csv_path.exists():
                                candidates.append((date, csv_path))
                    except ValueError:
                        pass
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def get_well_segmentation_processing_results_df_filename():
    return DATABASE_DIR / "well_segmentation_processing_results.csv"


def get_npy_and_csv_filenames(dev_mode: bool = False, failed_filenames: list[str] = None) -> tuple[list[Path], list[Path]]:
    """In this function, we get a list of all the .npy and .csv files in the input directory

    We also check that the two lists of filenames are the same, and that the .csv files exist
    """
    assert WELL_SEGMENTATION_DIR.exists()
    assert INPUT_DIR.exists()

    filenames_npy = list(WELL_SEGMENTATION_DIR.glob("*.npy"))

    if failed_filenames:
        filenames_npy = [x for x in filenames_npy if not x.stem in failed_filenames]

    filenames_npy.sort()

    filenames_meta = [INPUT_DIR / x.with_suffix(".csv").name for x in filenames_npy]

    if dev_mode:
        filenames_npy = filenames_npy[:10]
        filenames_meta = filenames_meta[:10]
        logger.info(f"DEV_MODE: only using {len(filenames_meta)} files")

    # Check that these two lists of filenames are the same
    assert len(filenames_npy) == len(filenames_meta)
    for f1, f2 in zip(filenames_npy, filenames_meta):
        assert f1.stem == f2.stem, f"{f1.stem} != {f2.stem}"
        assert f2.exists(), f"{f2} does not exist"

    logger.info(f"Found {len(filenames_npy)} files in {INPUT_DIR}")

    return filenames_meta, filenames_npy


def get_npy_and_csv_filenames_given_basename(basename: str) -> tuple[pd.DataFrame, np.array]:
    """Given a base name, load the corresponding meta df and image array
    """
    meta_df = pd.read_csv(INPUT_DIR / f"{basename}.csv", header=0, delimiter=";").iloc[:, :-1]
    img_array = np.load(WELL_SEGMENTATION_DIR / f"{basename}.npy")

    return meta_df, img_array


def validate_inputs():
    assert INPUT_DIR.exists()
    assert len(list(INPUT_DIR.glob("*.tif"))) > 0
    assert len(list(INPUT_DIR.glob("*.tif"))) == len(set(INPUT_DIR.glob("*.tif")))


def validate_stage1_inputs():
    assert CLEANED_RAW_DATA_DIR.exists(), f"Cleaned raw data dir not found: {CLEANED_RAW_DATA_DIR}"
    assert len(list(CLEANED_RAW_DATA_DIR.glob("*.tif"))) > 0, "No cleaned TIF files found"
