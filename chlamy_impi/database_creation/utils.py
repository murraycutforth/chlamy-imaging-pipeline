import datetime
import re
from pathlib import Path
from typing import Optional
import logging

import numpy as np
import pandas as pd
from chlamy_impi.database_creation.constants import get_possible_frame_numbers
from chlamy_impi.paths import get_database_output_dir, get_csv_filename

logger = logging.getLogger(__name__)


def location_to_index(loc: str) -> tuple[int, int]:
    """Convert a location string, e.g. "A1" or "P12", to a zero-indexed tuple, e.g. (0, 0)"""
    assert 2 <= len(loc) <= 3
    letter = loc[0]
    number = int(loc[1:])

    assert letter in "ABCDEFGHIJKLMNOP"

    i = number - 1
    j = ord(letter) - ord("A")

    return i, j


def index_to_location(i: int, j: int) -> str:
    """Convert a zero-indexed tuple, e.g. (0, 0), to a location string, e.g. "A1" """
    assert 0 <= i <= 16
    assert 0 <= j <= 24

    letter = chr(ord("A") + j)
    number = i + 1

    return f"{letter}{number}"


def index_to_location_rowwise(x):
    """Convert a zero-indexed tuple, e.g. (0, 0), to a location string, e.g. "A01"

    Note: Must be A01, B12, C04, etc; not A1, B12, C4.
    """

    letter = chr(ord("A") + x.i)
    number = x.j + 1

    return f"{letter}{number:02d}"


def spreadsheet_plate_name_formatting(plate: str, filenames_npy: Optional[list[Path]] = None) -> str:
    """Convert a plate string, e.g. "Plate 01", to a equal value to the numpy files, e.g. 1"""
    assert plate.startswith("Plate "), f"Unexpected plate string: {plate}"
    number_str = plate[6:]

    assert len(number_str) <= 4

    if len(number_str) == 2:
        assert number_str[0] in "0123456789"
        assert number_str[1] in "0123456789"
        assert 1 <= int(number_str) <= 99

        if number_str[0] == "0":
            number_str = number_str[1]

    if filenames_npy is not None:
        # Then assert that number_str is in filenames_npy
        npy_filenames = set([x.stem for x in filenames_npy])
        if not number_str in npy_filenames:
            logger.warning(f"Number {number_str} not found in filenames_npy")

    assert isinstance(number_str, str)
    return number_str


def parse_name(f, return_date: int = False):
    """Parse the name of a file, e.g. `20200303_7-M4_2h-2h.npy` or `20231119_07-M3_20h_ML.npy`
    """
    try:
        f = str(f)
        parts = f.split("_")

        assert len(parts) in {3, 4}, f"Unexpected number of parts in filename: {f}, parts: {parts}"

        middle = parts[1].split("-")
        plate_num = middle[0]

        measurement_num = middle[1]

        if len(parts) == 3:
            assert len(parts[2].split(".")) == 2, f
            time_regime = parts[2].split(".")[0]
        else:
            assert len(parts[3].split(".")) == 2, f
            time_regime = parts[2] + "_" + parts[3].split(".")[0]

        assert re.match(r"M[1-9]", measurement_num), f
        assert time_regime in {
            "30s-30s",
            "1min-1min",
            "10min-10min",
            "2h-2h",
            "20h_ML",
            "20h_HL",
            "1min-5min",
            "5min-5min",
        }, f"Unexpected time regime: {time_regime}, from filename: {f}"

        if return_date:
            date = datetime.datetime.strptime(parts[0], "%Y%m%d")
            return plate_num, measurement_num, time_regime, date
        else:
            return plate_num, measurement_num, time_regime
    except Exception as e:
        logger.error(f"Error parsing filename: {e}")
        raise e


def compute_measurement_times(meta_df: pd.DataFrame) -> list[datetime.datetime]:
    """In this function, we compute the time of each y2 or npq measuremnt."""
    meta_df["Datetime"] = meta_df[["Date", "Time"]].apply(
        lambda x: pd.to_datetime(x["Date"], format="%d.%m.%y")
                  + pd.to_timedelta(x["Time"]),
        axis=1,
    )

    assert len(meta_df) <= max(get_possible_frame_numbers()) - 2, f"Unexpected number of rows in meta_df: {len(meta_df)}"
    return meta_df["Datetime"].tolist()


def save_df_to_csv(df: pd.DataFrame):
    output_dir = get_database_output_dir()
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    filename = get_csv_filename()
    df.to_csv(filename, index=False)
    logger.info(f'CSV file saved at: {filename}')



