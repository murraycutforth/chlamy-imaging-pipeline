import functools
import logging

import pandas as pd

from chlamy_impi.database_creation.database_sanity_checks import check_plate_and_wells_are_unique, check_num_mutations
from chlamy_impi.database_creation.utils import spreadsheet_plate_name_formatting
from chlamy_impi.paths import get_identity_spreadsheet_path, get_npy_and_csv_filenames

logger = logging.getLogger(__name__)


def construct_identity_dataframe(mutation_df: pd.DataFrame, conf_threshold: int = 5) -> pd.DataFrame:
    """Extract all features which allow us to match a well to a mutant

    There is a single row per plate-well ID
    (currently this corresponds also to a unique mutant ID, but won't always)

    We create columns as follows:
        - mutant_ID: unique identifier for each mutant, e.g. 'LMJ.RY0401.001' or 'WT'
        - plate: plate number (currently numeric, will be converted to string)
        - well_id: unique identifier for each well such as 1-A1, 12-F12, etc. (index)
        - feature: comma separated list of features for each well, e.g. nan, intron, CDS
        - mutated_genes: string of comma-separated gene names which were mutated (or nan for WT)
        - num_mutations: number of genes which were mutated (integer)

    We apply as many assertions as possible to this dataframe to sanity check this data, such as:


    """
    identity_spreadsheet = get_identity_spreadsheet_path()
    df = pd.read_excel(identity_spreadsheet, header=0, engine="openpyxl")

    n_null = df["mutant_ID"].isnull().sum()
    if n_null > 0:
        logger.warning(f"Dropping {n_null} rows with null mutant_ID from identity spreadsheet")
        df = df.dropna(subset=["mutant_ID"])

    # NOTE: the finalised identity spreadsheet has non-unique column names. Pandas appends .1, .2, .3, etc. to these

    # In the old spreadsheet, Location was e.g. A10, and New location was e.g. Plate 01. These have been mapped to the
    # columns below.
    # The new spreadsheet has some null values for "New Location" and "New Location.4" which we need to drop
    df = df.dropna(subset=["New Location", "New Location.4"])

    # Drop the final row which just says "Z-END"
    df = df.iloc[:-1]

    # Map all values of "Plate 1" to "Plate 01" in the "New Location" column
    df["New Location"] = df["New Location"].apply(
        lambda x: x.replace("Plate 1", "Plate 01") if x == "Plate 1" else x
    )

    # Check that all entries in the "New Location" column are of the form "Plate XX"
    assert df["New Location"].apply(lambda x: x.startswith("Plate ")).all(), df["New Location"].unique()
    assert df["New Location"].apply(lambda x: len(x) == 8 or len(x) == 10).all(), df["New Location"].unique()
    assert df["New Location"].apply(lambda x: x[6:8].isdigit()).all(), df["New Location"].unique()

    # Check that all entries in the "New Location.4" column are of the form "A01", "B12", etc.
    assert df["New Location.4"].apply(lambda x: len(x) == 3).all(), df["New Location.4"].unique()
    assert df["New Location.4"].apply(lambda x: x[0] in "ABCDEFGHIJKLMNOP").all(), df["New Location.4"].unique()
    assert df["New Location.4"].apply(lambda x: x[1:].isdigit()).all(), df["New Location.4"].unique()

    # Collect columns which we need
    df = df.rename(columns={"New Location": "plate", "New Location.4": "well_id"})
    df["plate"] = df["plate"].apply(
        functools.partial(spreadsheet_plate_name_formatting, filenames_npy=None))
    df_features = df[["mutant_ID", "plate", "well_id", "feature"]]
    df = df[["mutant_ID", "plate", "well_id"]]

    df = df.drop_duplicates(ignore_index=True)
    df_features = df_features.drop_duplicates(ignore_index=True)
    df_features = df_features.dropna(subset=["feature"])

    # Concatenate all features into a single string, and place into feature column
    df_grouped = df_features.groupby(["mutant_ID", "plate", "well_id"]).apply(
        lambda x: ",".join(str(item) for item in set(x.feature))
    )

    # Convert df_grouped back into a dataframe - the index is a multi-index of (mutant_ID, plate, well_id)
    df_grouped = df_grouped.reset_index().rename(columns={0: "feature"})

    # Merge the cleaned features back in
    df = pd.merge(df, df_grouped, on=["mutant_ID", "plate", "well_id"], how="left")

    # Check that all mutant_IDs are unique - print out any duplicates
    check_plate_and_wells_are_unique(df)

    df = add_mutated_genes_col(conf_threshold, df, mutation_df)

    # Add rows for wild type plate 99
    wt_rows = create_wt_rows()
    df_wt = pd.DataFrame(wt_rows)
    df = pd.concat([df, df_wt], axis=0, ignore_index=False)

    check_plate_and_wells_are_unique(df)
    assert (
        df["mutant_ID"].notnull().all()
    ), f'Found a total of {df["mutant_ID"].isnull().sum()} null values in mutant_ID'
    check_num_mutations(df)

    # Group by the plate number (the first part of the index string) to check number of wells per plate
    plates = df.plate
    plate_counts = plates.value_counts()
    for plate, count in plate_counts.items():
        assert (
            count <= 384
        ), f"Plate {plate} has {count} wells with ids {df[df.plate == plate].well_id.unique()}"

    logger.info(
        f"Constructed identity dataframe. Shape: {df.shape}. Columns: {df.columns}."
    )
    logger.info(f"Unique plate values: {df.plate.unique()}")
    logger.info(f"Values of num_mutations: {df.num_mutations.unique()}")
    logger.debug(f"{df.head()}")

    return df


def add_mutated_genes_col(
    conf_threshold: float, df: pd.DataFrame, mutation_df: pd.DataFrame
) -> pd.DataFrame:
    """Add column which tells us the number of genes which were mutated, as well as comma separated list of genes"""
    num_rows = len(df)

    signif_mutations = mutation_df[mutation_df.confidence_level <= conf_threshold]
    gene_mutation_counts = signif_mutations.groupby("mutant_ID").nunique()["gene"]
    mutated_genes = signif_mutations.groupby("mutant_ID").apply(
        lambda x: ",".join(set(x.gene))
    )
    mutated_genes = mutated_genes.reset_index().rename(columns={0: "mutated_genes"})
    df = pd.merge(df, mutated_genes, on="mutant_ID", how="left")
    df["num_mutations"] = df["mutant_ID"].apply(
        lambda x: gene_mutation_counts.get(x, 0)
    )

    assert (
        len(df) == num_rows
    ), f"Length of dataframe changed from {num_rows} to {len(df)}"
    check_plate_and_wells_are_unique(df)
    assert (
        df["mutant_ID"].notnull().all()
    ), f'Found a total of {df["mutant_ID"].isnull().sum()} null values in mutant_ID'

    return df


def create_wt_rows() -> list[dict]:
    """In this function, we create rows for the wild type plate 99"""
    rows = []
    for well in well_position_iterator():
        row_data = {
            "plate": '99',
            "well_id": well,
            "mutant_ID": "WT",
            "num_mutations": 0,
            "mutated_genes": "",
        }
        rows.append(row_data)
    return rows


def well_position_iterator():
    for i in range(1, 17):
        for j in range(1, 25):
            yield f"{chr(ord('A') + i - 1)}{j:02d}"
