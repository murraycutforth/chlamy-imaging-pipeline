"""Stage 2b: Parquets + Identity → database.csv

Entry point: python -m chlamy_impi.database_creation.main_v2

Reads the three parquets written by Stage 2a (image_processing/main.py) and the
identity spreadsheet, then produces the same database.csv as the original Stage 2.

Pipeline:
  1. Read plates.parquet, wells.parquet, timeseries.parquet
  2. Pivot timeseries long → wide (reconstruct y2_1..y2_N, ynpq_1..ynpq_N, etc.)
  3. Merge wells + plates + wide timeseries → exptl_data
  4. Merge exptl_data with identity dataframe
  5. Sanity checks + write database.csv
"""

import logging
from itertools import product

import numpy as np
import pandas as pd

from chlamy_impi.database_creation.constants import get_possible_frame_numbers
from chlamy_impi.database_creation.construct_identity_df import construct_identity_dataframe
from chlamy_impi.database_creation.main import (
    construct_gene_description_dataframe,
    construct_mutations_dataframe,
    final_df_sanity_checks,
    write_dataframe,
)
from chlamy_impi.database_creation.utils import (
    parse_name,
    save_df_to_csv,
)
from chlamy_impi.paths import (
    get_plates_parquet_path,
    get_timeseries_parquet_path,
    get_wells_parquet_path,
)

logger = logging.getLogger(__name__)

DEV_MODE = False
IGNORE_ERRORS = False


def build_wide_experimental_df() -> pd.DataFrame:
    """Read the three Stage 2a parquets and reconstruct the wide-format DataFrame.

    The output matches the schema produced by the original Stage 2 before the
    identity merge: one row per (plate × well) with y2_1..y2_N, ynpq_1..ynpq_N,
    y2_std_1..y2_std_N, and measurement_time_0..measurement_time_N columns.
    """
    plates = pd.read_parquet(get_plates_parquet_path())
    wells = pd.read_parquet(get_wells_parquet_path())
    ts = pd.read_parquet(get_timeseries_parquet_path())

    logger.info(
        f"Read parquets — plates: {plates.shape}, wells: {wells.shape}, ts: {ts.shape}"
    )

    # Maximum number of time steps to match original Stage 2 column layout
    tmax = max(get_possible_frame_numbers()) - 2  # = 178
    all_steps = range(1, tmax)  # 1..177

    key_cols = ["plate", "measurement", "start_date", "i", "j"]
    ts_indexed = ts.set_index(key_cols + ["time_step"])

    y2_w = ts_indexed["y2"].unstack("time_step").reindex(columns=all_steps)
    std_w = ts_indexed["y2_std"].unstack("time_step").reindex(columns=all_steps)
    npq_w = ts_indexed["ynpq"].unstack("time_step").reindex(columns=all_steps)
    mt_w = ts_indexed["measurement_time"].unstack("time_step").reindex(columns=all_steps)

    y2_w.columns = [f"y2_{s}" for s in all_steps]
    std_w.columns = [f"y2_std_{s}" for s in all_steps]
    npq_w.columns = [f"ynpq_{s}" for s in all_steps]
    mt_w.columns = [f"measurement_time_{s}" for s in all_steps]

    ts_wide = pd.concat([y2_w, std_w, npq_w, mt_w], axis=1).reset_index()

    # wells already contains measurement_time_0 (fv/fm time)
    exptl = pd.merge(wells, plates, on=["plate", "measurement", "start_date"], how="left")
    exptl = pd.merge(exptl, ts_wide, on=key_cols, how="left")

    logger.info(f"Wide experimental DataFrame shape: {exptl.shape}")
    return exptl


def merge_identity_and_experimental_dfs(
    exptl_data: pd.DataFrame, identity_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge experimental data with identity spreadsheet.

    Plates not present in both DataFrames are dropped with a warning.
    When IGNORE_ERRORS is True, wells missing from the identity spreadsheet are
    silently dropped; otherwise an assertion error is raised.
    """
    exptl_plates = exptl_data.plate.unique()
    identity_plates = identity_df.plate.unique()
    common_plates = set(exptl_plates).intersection(identity_plates)

    removed_exptl = set(exptl_plates) - common_plates
    removed_identity = set(identity_plates) - common_plates
    if removed_exptl:
        logger.error(
            f"Plates in experimental data but not in identity spreadsheet: {removed_exptl}"
        )
    if removed_identity:
        logger.error(
            f"Plates in identity spreadsheet but not in experimental data: {removed_identity}"
        )

    exptl_data = exptl_data[exptl_data.plate.isin(common_plates)]
    identity_df = identity_df[identity_df.plate.isin(common_plates)]

    # Verify all non-blank wells in exptl_data are present in identity_df
    non_blank = exptl_data[exptl_data.well_id != "A01"]
    exptl_plate_n_well = set(product(non_blank.plate, non_blank.well_id))
    identity_plate_n_well = set(product(identity_df.plate, identity_df.well_id))

    if IGNORE_ERRORS:
        exptl_plate_n_well = exptl_plate_n_well.intersection(identity_plate_n_well)
        exptl_data = exptl_data[
            exptl_data[["plate", "well_id"]].apply(tuple, axis=1).isin(exptl_plate_n_well)
        ]
        logger.warning(
            "IGNORE_ERRORS: dropped wells not found in identity spreadsheet"
        )
    else:
        missing = exptl_plate_n_well - identity_plate_n_well
        assert exptl_plate_n_well.issubset(identity_plate_n_well), (
            f"Wells in experimental data but not in identity spreadsheet: {missing}"
        )

    total_df = pd.merge(
        exptl_data, identity_df, on=["plate", "well_id"], how="left", validate="m:1"
    )
    logger.info(
        f"After identity merge — shape: {total_df.shape}, plates: {total_df.plate.unique()}"
    )
    return total_df


def main():
    mutations_df = construct_mutations_dataframe()
    logger.info(f"Mutations dataframe shape: {mutations_df.shape}")

    identity_df = construct_identity_dataframe(mutations_df)
    logger.info(f"Identity dataframe shape: {identity_df.shape}")

    logger.info("Writing gene descriptions...")
    gene_descriptions = construct_gene_description_dataframe()
    write_dataframe(gene_descriptions, "gene_descriptions.csv")

    exptl_data = build_wide_experimental_df()
    logger.info(f"Wide experimental dataframe shape: {exptl_data.shape}")

    total_df = merge_identity_and_experimental_dfs(exptl_data, identity_df)
    logger.info(f"Total dataframe shape: {total_df.shape}")

    logger.info("Running sanity checks...")
    final_df_sanity_checks(total_df)
    logger.info("All sanity checks passed.")

    logger.info("Writing database.csv...")
    save_df_to_csv(total_df)
    logger.info("Stage 2b complete — database.csv written.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG if DEV_MODE else logging.INFO)
    main()
