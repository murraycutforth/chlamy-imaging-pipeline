import logging

import pandas as pd

from chlamy_impi.database_creation.constants import get_possible_frame_numbers

logger = logging.getLogger(__name__)


def sanity_check_merged_plate_info_and_well_info(df: pd.DataFrame, ignore_errors: bool) -> None:
    """Perform some sanity checks on the merged dataframe in the construct_identity_dataframe() function"""

    if ignore_errors:
        # Remove all rows which would otherwise fail these assertions

        logger.info(f"Removing rows which would otherwise fail the assertions in sanity_check_merged_plate_info_and_well_info()")
        logger.info(f"Original number of rows: {len(df)}")
        unique_combinations = df.groupby(["plate", "measurement", "start_date"]).size()
        too_small_combinations = unique_combinations[unique_combinations < 300]
        too_large_combinations = unique_combinations[unique_combinations > 384]

        for idx in too_small_combinations.index:
            df = df[(df.plate != idx[0]) | (df.measurement != idx[1]) | (df.start_date != idx[2])]

        for idx in too_large_combinations.index:
            df = df[(df.plate != idx[0]) | (df.measurement != idx[1]) | (df.start_date != idx[2])]

        logger.info(f"New number of rows: {len(df)}")


    unique_combinations = df.groupby(["plate", "measurement", "start_date"]).size()
    assert unique_combinations.min() >= 300, f"Minimum number of rows for a unique combination is {unique_combinations.min()}"
    assert unique_combinations.max() <= 384, f"Maximum number of rows for a unique combination is {unique_combinations.max()}"


def check_all_mutant_IDs_non_null(df: pd.DataFrame):
    """For all non-null mutant_ID, check that at least 1 measurement time columns is non-null,
    and at least 1 y2 column is non-null

    TODO: Removed - this isn't testing for the right thing. We expect that some mutants will fail to grow.

    """
    mutant_IDs = df.mutant_ID.dropna().unique()
    y2_cols = [col for col in df.columns if col.startswith("y2_")]
    measurement_time_cols = [col for col in df.columns if col.startswith("measurement_time_")]

    for mutant_ID in mutant_IDs:
        mutant_df = df[df.mutant_ID == mutant_ID]
        plates = mutant_df.plate
        well_ids = mutant_df.well_id
        light_treatments = mutant_df.light_regime

        assert mutant_df[measurement_time_cols].notnull().any().any(), f"Found null values in measurement time columns for mutant_ID {mutant_ID} in plates {plates}, well_ids {well_ids}, light {light_treatments}"
        assert mutant_df[y2_cols].notnull().any().any(), f"Found no non-null values in y2 columns for mutant_ID {mutant_ID} in plates {plates}, well_ids {well_ids}, light {light_treatments}"

    logger.info('All mutant_IDs have non-null measurement time columns and at least 1 non-null y2 column')


def check_num_mutations(df: pd.DataFrame):
    assert df["num_mutations"].notnull().all(), f"Found null values in num_mutations column at plate/well: {df[df['num_mutations'].isnull()][['plate', 'well_id']]}"
    assert df["num_mutations"].min() >= 0
    assert df["num_mutations"].max() <= 8
    logger.info('All num_mutations columns are non-null and between 0 and 8')


def check_non_null_num_mutations(df: pd.DataFrame):
    # Check number of null num_mutations for each plate/well_id combination
    null_num_mutations = df[df["num_mutations"].isnull()].groupby(["plate", "well_id"]).size()
    assert null_num_mutations.max() <= 110, f"Max number of null num_mutations for a plate/well_id combination is {null_num_mutations.max()}"
    assert df[df['num_mutations'].notnull()]['num_mutations'].min() >= 0
    assert df[df['num_mutations'].notnull()]['num_mutations'].max() <= 8


def check_unique_plate_well_startdate(df: pd.DataFrame):
    """Check that the plate, well, and start_date columns are enough to uniquely identify a row"""
    unique_combinations = df.groupby(["plate", "well_id", "start_date"]).size()

    # Print out any rows which are not 1
    for idx, count in unique_combinations.items():
        if count != 1:
            print(df[(df.plate == idx[0]) & (df.well_id == idx[1]) & (df.start_date == idx[2])])

    assert unique_combinations.min() == 1, f"Minimum number of rows for a unique combination is {unique_combinations.min()}"
    assert unique_combinations.max() == 1, f"Maximum number of rows for a unique combination is {unique_combinations.max()}"

    logger.info('All plate, well_id, and start_date combinations are unique')


def check_total_number_of_entries_per_plate(df):
    """Check that each plate + light_regime combination has a maximum of 384 entries.
    """
    plate_light_regime_date_combinations = df.groupby(["plate", "light_regime", "start_date"]).size()

    for idx, count in plate_light_regime_date_combinations.items():
        if count > 384:
            print(df[(df.plate == idx[0]) & (df.light_regime == idx[1]) & (df.start_date == idx[2])].iloc[0])

    assert plate_light_regime_date_combinations.max() <= 384, f"Max number of entries for a plate + light regime is {plate_light_regime_date_combinations.max()}"

    logger.info('All plate + light_regime + date combinations have a maximum of 384 entries')

    #plate_light_date_combinations = df.groupby(["plate", "light_regime", "start_date"]).size()
    #assert plate_light_date_combinations.max() <= 384, f"Max number of entries for a plate + light regime + date is {plate_light_date_combinations.max()}"
    #assert plate_light_date_combinations.min() >= 384, f"Min number of entries for a plate + light regime + date is {plate_light_date_combinations.min()}"

    logger.info('All plate + light_regime + date combinations have 384 entries')


def check_plate_and_wells_are_unique(df):
    """This sanity check is all about ensuring that each plate/well combo only has a single mutant_ID.
    """
    error_message = ''
    plates = df.plate
    for plate in plates.unique():
        well_id_with_multiple_mutants = df[df.plate == plate].groupby("well_id").filter(lambda x: len(x) > 1)
        if not well_id_with_multiple_mutants.empty:
            print(well_id_with_multiple_mutants.sort_values("well_id"))
            error_message += '/n' + str(well_id_with_multiple_mutants)
    assert error_message == '', error_message

    logger.info('All plate/well combos verified to have a single mutant_ID')


def check_num_frames(df):
    """Check that the number of frames is as expected
    """
    assert df["num_frames"].notnull().all(), "Found null values in num_frames column"
    assert df["num_frames"].isin(get_possible_frame_numbers()).all(), f"Found num_frames values that are not 84 or 164: {df['num_frames'].unique()}"

    logger.info('All num_frames column values are non-null and are expected')


def check_all_plates_have_WT(df):
    """Check that all plate + light regime + date combinations have at least 3 wells where mutant_ID is WT"""

    # Debug:
    #plate_df = df[(df.plate == 2) & (df.light_regime == '1min-1min')]
    #print(plate_df)

    num_wt_wells = df[df.mutant_ID == 'WT'].groupby(["plate", "light_regime", "start_date"]).size()
    assert num_wt_wells.min() >= 3, f"Found {(num_wt_wells < 3).sum()} plate + light regime + date combinations with less than 3 WT wells"

    logger.info('All plate + light regime + date combinations have at least 3 WT wells')

