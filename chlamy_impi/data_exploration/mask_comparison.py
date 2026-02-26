from collections import defaultdict
from functools import partial
from pathlib import Path
import logging

import numpy as np
from matplotlib import pyplot as plt
from skimage import filters
from tqdm import tqdm

from chlamy_impi.lib.fv_fm_functions import compute_all_fv_fm
from chlamy_impi.lib.mask_functions import compute_threshold_mask_global, count_empty_wells, average_mask_area, \
    count_overlapping_masks

logger = logging.getLogger(__name__)

INPUT_DIR = Path("./../../output/image_processing/v6/img_array")


def try_mask_function(mask_fn, filenames):
    """mask_fn should accept an image array, and return a mask array
    """
    all_data = defaultdict(list)

    for filename in tqdm(filenames):
        img_array = np.load(filename)
        mask_array = mask_fn(img_array)

        assert mask_array.shape == (16, 24, 20, 20)

        empty_wells = count_empty_wells(mask_array)
        avg_area, std_area = average_mask_area(mask_array)
        num_overlapping = count_overlapping_masks(mask_array)
        avg_fv_fm = compute_all_fv_fm(img_array, mask_array)

        all_data["empty wells"].append(empty_wells)
        all_data["mask area"].append(avg_area)
        all_data["mask stddev"].append(std_area)
        all_data["overlapping wells"].append(num_overlapping)
        all_data["avg fv/fm"].append(np.nanmean(avg_fv_fm))

    return all_data


def plot_all_data(data_list: list[dict[str, list[float]]], mask_labels, name):
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    axs = axs.ravel()

    metrics = list(data_list[0].keys())

    for i, metric in enumerate(metrics):
        for data, label in zip(data_list, mask_labels):
            axs[i].plot(data[metric], label=label)
        axs[i].set_title(metric)
        axs[i].legend()

    plt.tight_layout()
    fig.savefig(f"{name}.png")
    plt.close()


def main():
    assert INPUT_DIR.exists()
    filenames = list(INPUT_DIR.glob("*.npy"))

    # Test min reduction

    #mask_fns = [
    #    partial(compute_threshold_mask_global, num_std=0, use_opening=False, time_reduction_fn=np.min),
    #    partial(compute_threshold_mask_global, num_std=1, use_opening=False, time_reduction_fn=np.min),
    #    partial(compute_threshold_mask_global, num_std=2, use_opening=False, time_reduction_fn=np.min),
    #    partial(compute_threshold_mask_global, num_std=3, use_opening=False, time_reduction_fn=np.min),
    #    partial(compute_threshold_mask_global, num_std=4, use_opening=False, time_reduction_fn=np.min),
    #    partial(compute_threshold_mask_global, num_std=5, use_opening=False, time_reduction_fn=np.min),
    #]

    #mask_labels = [
    #    "sigma = 0",
    #    "sigma = 1",
    #    "sigma = 2",
    #    "sigma = 3",
    #    "sigma = 4",
    #    "sigma = 5",
    #]

    #data_list = []
    #for mask_fn in mask_fns:
    #    data = try_mask_function(mask_fn, filenames)
    #    data_list.append(data)

    #plot_all_data(data_list, mask_labels, "Threshold_MinReduction_NoOpening")

    ## Test mean reduction

    #mask_fns = [
    #    partial(compute_threshold_mask_global, num_std=0, use_opening=False, time_reduction_fn=np.mean),
    #    partial(compute_threshold_mask_global, num_std=1, use_opening=False, time_reduction_fn=np.mean),
    #    partial(compute_threshold_mask_global, num_std=2, use_opening=False, time_reduction_fn=np.mean),
    #    partial(compute_threshold_mask_global, num_std=3, use_opening=False, time_reduction_fn=np.mean),
    #    partial(compute_threshold_mask_global, num_std=4, use_opening=False, time_reduction_fn=np.mean),
    #    partial(compute_threshold_mask_global, num_std=5, use_opening=False, time_reduction_fn=np.mean),
    #]

    #mask_labels = [
    #    "sigma = 0",
    #    "sigma = 1",
    #    "sigma = 2",
    #    "sigma = 3",
    #    "sigma = 4",
    #    "sigma = 5",
    #]

    #data_list = []
    #for mask_fn in mask_fns:
    #    data = try_mask_function(mask_fn, filenames)
    #    data_list.append(data)

    #plot_all_data(data_list, mask_labels, "Threshold_MeanReduction_NoOpening")

    # Test opening

    mask_fns = [
        partial(compute_threshold_mask_global, num_std=0, use_opening=True, time_reduction_fn=np.min),
        partial(compute_threshold_mask_global, num_std=1, use_opening=True, time_reduction_fn=np.min),
        partial(compute_threshold_mask_global, num_std=2, use_opening=True, time_reduction_fn=np.min),
        partial(compute_threshold_mask_global, num_std=3, use_opening=True, time_reduction_fn=np.min),
        partial(compute_threshold_mask_global, num_std=4, use_opening=True, time_reduction_fn=np.min),
        partial(compute_threshold_mask_global, num_std=5, use_opening=True, time_reduction_fn=np.min),
    ]

    mask_labels = [
        "sigma = 0",
        "sigma = 1",
        "sigma = 2",
        "sigma = 3",
        "sigma = 4",
        "sigma = 5",
    ]

    data_list = []
    for mask_fn in mask_fns:
        data = try_mask_function(mask_fn, filenames)
        data_list.append(data)

    plot_all_data(data_list, mask_labels, "Threshold_MeanReduction_Opening")

    logger.info("Program completed normally")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
