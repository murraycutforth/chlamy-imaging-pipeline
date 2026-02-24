# Here we want to load all tif and csv files, and then check if the number of frames is correct, without any correction
import numpy as np
import pandas as pd

from chlamy_impi.database_creation.constants import get_possible_frame_numbers
from chlamy_impi.database_creation.utils import parse_name
from chlamy_impi.paths import get_npy_and_csv_filenames

filenames_meta, filenames_npy = get_npy_and_csv_filenames(dev_mode=0)

failures = []

for filename_npy, filename_meta in zip(filenames_npy, filenames_meta):
    plate_num, measurement_num, light_regime, start_date = parse_name(filename_npy.name, return_date=True)

    img_array = np.load(filename_npy)
    meta_df = pd.read_csv(filename_meta, header=0, delimiter=";").iloc[:, :-1]

    try:
        assert img_array.shape[2] in get_possible_frame_numbers()
        assert img_array.shape[2] == len(meta_df) * 2
    except AssertionError:
        failures.append((filename_npy.stem, img_array.shape[2]))

# Write out failures to text file
with open("failed_plates_17-06-2024.txt", "w") as f:
    for failure in failures:
        f.write(f"{failure[0]}: {failure[1]}\n")

