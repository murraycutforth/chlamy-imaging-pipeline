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
import numpy as np
import pandas as pd

from chlamy_impi.database_creation.constants import get_possible_frame_numbers
from chlamy_impi.database_creation.construct_identity_df import construct_identity_dataframe
from chlamy_impi.database_creation.shared import (
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
    get_database_output_dir,
    get_dated_csv_filename,
    get_plates_parquet_path,
    get_timeseries_parquet_path,
    get_wells_parquet_path,
    find_previous_database_excluding_today,
)

logger = logging.getLogger(__name__)

DEV_MODE = False


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
    exptl_plate_n_well = set(zip(non_blank.plate, non_blank.well_id))
    identity_plate_n_well = set(zip(identity_df.plate, identity_df.well_id))

    missing = exptl_plate_n_well - identity_plate_n_well
    if missing:
        logger.warning(
            f"{len(missing)} (plate, well) combos in experimental data have no identity entry "
            f"and will be dropped: {sorted(missing)}"
        )
        exptl_data = exptl_data[
            ~exptl_data[["plate", "well_id"]].apply(tuple, axis=1).isin(missing)
        ]

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

    # Capture previous database path before writing (excludes today's file)
    prev_path = find_previous_database_excluding_today()

    logger.info("Writing database.csv...")
    save_df_to_csv(total_df)
    logger.info("Stage 2b complete — database.csv written.")

    # Regression comparison against previous dated database
    if prev_path is not None:
        import datetime
        from chlamy_impi.database_creation.database_comparison import (
            compare_databases,
            write_comparison_report,
        )
        try:
            logger.info(f"Comparing against previous database: {prev_path.name}")
            new_path = get_dated_csv_filename()
            result = compare_databases(prev_path, new_path)
            old_date = prev_path.stem.replace("database_", "")
            new_date = str(datetime.date.today())
            report_name = f"comparison_{old_date}_to_{new_date}.md"
            report_path = get_database_output_dir() / report_name
            write_comparison_report(result, prev_path, new_path, report_path)
        except Exception as exc:
            logger.warning(f"Regression comparison failed (non-fatal): {exc}")
    else:
        logger.info("No previous database found — skipping comparison.")

    logger.info("Generating timeseries visualisations...")
    try:
        from chlamy_impi.database_creation.visualize_timeseries import plot_timeseries_mosaic
        plot_timeseries_mosaic(total_df)
    except Exception as exc:
        logger.warning(f"Timeseries visualisation failed (non-fatal): {exc}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG if DEV_MODE else logging.INFO)
    main()
