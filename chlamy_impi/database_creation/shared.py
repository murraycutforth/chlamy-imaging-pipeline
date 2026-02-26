"""Utility functions shared by Stage 2a (image_processing/main.py) and
Stage 2b (database_creation/main_v2.py).
"""

import logging

import numpy as np
import pandas as pd

from chlamy_impi.database_creation.database_sanity_checks import (
    check_all_plates_have_WT,
    check_non_null_num_mutations,
    check_num_frames,
    check_total_number_of_entries_per_plate,
    check_unique_plate_well_startdate,
)
from chlamy_impi.error_correction.tif_io import load_csv
from chlamy_impi.paths import (
    CLEANED_RAW_DATA_DIR,
    WELL_SEGMENTATION_DIR,
    get_database_output_dir,
    get_identity_spreadsheet_path,
)

logger = logging.getLogger(__name__)


def prepare_img_array_and_df(filename_meta, filename_npy):
    """Load the image array (pre-segmented into wells) and the metadata dataframe for a given plate."""
    img_array = np.load(filename_npy)
    meta_df = load_csv(filename_meta)
    return img_array, meta_df


def get_npy_and_csv_filenames(failed_filenames=None):
    npy_files = sorted(WELL_SEGMENTATION_DIR.glob("*.npy"))
    filenames_meta = []
    filenames_npy = []
    for npy_path in npy_files:
        if failed_filenames and npy_path.stem in failed_filenames:
            continue
        csv_path = CLEANED_RAW_DATA_DIR / f"{npy_path.stem}.csv"
        assert csv_path.exists(), f"No CSV found for {npy_path.name}"
        filenames_meta.append(csv_path)
        filenames_npy.append(npy_path)
    return filenames_meta, filenames_npy


def construct_gene_description_dataframe() -> pd.DataFrame:
    """Extract all gene descriptions as a separate dataframe.

    Each gene has one description, but descriptions are long so stored separately.
    """
    id_spreadsheet_path = get_identity_spreadsheet_path()
    assert id_spreadsheet_path.exists()
    df = pd.read_excel(id_spreadsheet_path, header=0, engine="openpyxl")

    df_gene_descriptions = df[["gene", "description", "feature"]]
    df_gene_descriptions = df_gene_descriptions.drop_duplicates(subset=["gene"])

    logger.info(f"Constructed description dataframe. Shape: {df_gene_descriptions.shape}.")
    return df_gene_descriptions


def construct_mutations_dataframe() -> pd.DataFrame:
    """Extract relevant mutant features from the identity spreadsheet.

    Columns: 'mutant_ID', 'gene', 'confidence_level'.
    """
    identity_spreadsheet = get_identity_spreadsheet_path()
    df = pd.read_excel(identity_spreadsheet, header=0, engine="openpyxl")

    df = df[["mutant_ID", "gene", "confidence_level"]]
    df = df.drop_duplicates(ignore_index=True)

    logger.info(f"Constructed mutation dataframe. Shape: {df.shape}. Columns: {df.columns}.")
    return df


def write_dataframe(df: pd.DataFrame, name: str):
    """Write the dataframe to a csv file in the database output directory."""
    output_dir = get_database_output_dir()
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    df.to_csv(output_dir / name)
    logger.info(f"Written dataframe to {output_dir / name}")


def final_df_sanity_checks(df: pd.DataFrame):
    """Final sanity checks applied to the merged dataframe before writing to disk."""
    check_unique_plate_well_startdate(df)
    check_total_number_of_entries_per_plate(df)
    check_num_frames(df)
    check_all_plates_have_WT(df)
    check_non_null_num_mutations(df)
