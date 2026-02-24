import itertools
import logging
from typing import Callable

import numpy as np
from skimage.morphology import binary_opening

logger = logging.getLogger(__name__)


# The following group of functions are all possible ways to generate an array of masks, given an image array


def compute_threshold_mask(
    img_arr: np.array,
    num_std: float = 3,
    use_opening: bool = False,
    time_reduction_fn: Callable = np.min,
    return_thresholds: bool = False,
) -> np.array:
    """
    Computes a threshold-based mask array for each well in a plate.
    One well has the same mask for all timepoints.

    Input:
        img_arr: 5D numpy array of shape (num_rows, num_columns, num_timepoints, height, width)
        num_std: number of standard deviations above the mean to use as the threshold
        use_opening: whether to perform a morphological opening operation on the mask
        time_reduction_fn: function to reduce the time dimension to a single image
        return_threshold: whether to return the threshold value

    Output:
        mask_arr: 4D numpy array of shape (num_rows, num_columns, height, width)
    """
    assert len(img_arr.shape) == 5
    crop_dims = img_arr.shape[-2:]

    thresholds = compute_thresholds(img_arr, num_std)
    dark_threshold, light_threshold = thresholds

    NUM_TIMESTEPS = img_arr.shape[2]
    dark_idxs = range(0, NUM_TIMESTEPS, 2)
    light_idxs = range(1, NUM_TIMESTEPS, 2)

    dark_imgs_alltime = time_reduction_fn(img_arr[:, :, dark_idxs], axis=2)
    light_imgs_alltime = time_reduction_fn(img_arr[:, :, light_idxs], axis=2)

    assert len(dark_imgs_alltime.shape) == 4
    assert len(light_imgs_alltime.shape) == 4
    assert dark_imgs_alltime.shape[2] == crop_dims[0]
    assert dark_imgs_alltime.shape[3] == crop_dims[1]
    assert light_imgs_alltime.shape[2] == crop_dims[0]
    assert light_imgs_alltime.shape[3] == crop_dims[1]

    dark_mask = dark_imgs_alltime > dark_threshold
    light_mask = light_imgs_alltime > light_threshold
    total_mask = dark_mask & light_mask

    if use_opening:
        for i, j in itertools.product(range(img_arr.shape[0]), range(img_arr.shape[1])):
            total_mask[i, j] = binary_opening(total_mask[i, j])

    if return_thresholds:
        return total_mask, (dark_threshold, light_threshold)
    else:
        return total_mask


def compute_thresholds(img_arr, num_std: float = 3.0, lighting="both"):
    """
    Computes the background threshold in the light and dark conditions

    Input:
        img_arr: 5D numpy array of shape (num_rows, num_columns, num_timepoints, height, width)
        num_std: number of standard deviations above the mean to use as the threshold
        lighting: "both" (compute 2 thresholds) or "all" (compute 1 threshold)
    Output:
        threshold: tuple of length 2, (dark_threshold, light_threshold)
            or tuple of length 2, (threshold, None)
    """
    assert len(img_arr.shape) == 5

    NUM_TIMESTEPS = img_arr.shape[2]
    dark_idxs = range(0, NUM_TIMESTEPS, 2)
    light_idxs = range(1, NUM_TIMESTEPS, 2)

    dark_img_brightness = np.mean(img_arr[:, :, dark_idxs])
    light_img_brightness = np.mean(img_arr[:, :, light_idxs])
    # verify that the dark images are darker than the light images
    assert light_img_brightness > dark_img_brightness

    if lighting == "both":
        logger.debug("Computing dark and light thresholds")
        dark_threshold = _compute_threshold(img_arr[:, :, dark_idxs], num_std)
        logger.debug(f"Dark threshold = {dark_threshold}")

        light_threshold = _compute_threshold(img_arr[:, :, light_idxs], num_std)
        logger.debug(f"Light threshold = {light_threshold}")
        out = (dark_threshold, light_threshold)
    elif lighting == "all":
        logger.debug("Computing threshold using all images")
        threshold = _compute_threshold(img_arr, num_std)
        logger.debug(f"Threshold = {threshold}")
        out = (threshold, None)
    else:
        raise ValueError("lighting must be 'both' or 'all'")

    return out


def _compute_threshold(img_arr, num_std: float = 3.0):
    """
    Computes the background threshold
    Robust to case where there is not a blank in the top left
    """
    assert len(img_arr.shape) == 5

    # First determine if top left well is indeed blank
    global_avg = np.mean(img_arr)
    global_std = np.std(img_arr)
    top_left_avg = np.mean(img_arr[0, 0])

    if abs(global_avg - top_left_avg) / global_std > 3.0:
        # We think that the top left cell is not blank
        logger.debug(f"Top left cell not blank")
        # Fall back to median of all wells, assumes that the bright spots form a minority
        threshold = np.median(img_arr)
    else:
        threshold = top_left_avg + num_std * np.std(img_arr[0, 0])
    return threshold


# The next set of functions are used to assess various properties of the masks


def count_empty_wells(mask_array):
    """
    Estimate the error due to misplating, which results in wells with no growing cells.

    Input:
        mask_array: 4D numpy array of shape (num_rows, num_columns, height, width)
    """
    mask_array_flat_im = mask_array.reshape(mask_array.shape[:2] + (-1,))
    total_wells = mask_array.shape[0] * mask_array.shape[1]
    num_good_wells = np.sum(np.max(mask_array_flat_im, axis=-1))
    empty_wells = total_wells - num_good_wells
    return empty_wells


def average_mask_area(mask_array) -> tuple[float, float]:
    sizes = np.sum(np.sum(mask_array, axis=-1), axis=-1)
    return np.mean(sizes), np.std(sizes)


def count_overlapping_masks(mask_array) -> int:
    """Perform some checks on the well masks.

    Returns the number of well masks which overlap with the boundary of the sub-image of the well.
    """
    array_shape = mask_array.shape

    num_overlapping = 0
    for i, j in itertools.product(range(array_shape[0]), range(array_shape[1])):
        if has_true_on_boundary(mask_array[i, j]):
            num_overlapping += 1

    logger.debug(f"We have found overlapping masks for {num_overlapping} masks")

    return num_overlapping


def has_true_on_boundary(arr):
    """Check if mask reaches edge of cell - should always be false"""

    # Check the top and bottom rows
    if np.any(arr[0, :]) or np.any(arr[-1, :]):
        return True

    # Check the left and right columns
    if np.any(arr[:, 0]) or np.any(arr[:, -1]):
        return True

    return False


def get_disk_mask(img_array, radius_fraction=1):
    """
    Computes an identical disk mask based on the dimensions of the
    input image array.

    Input:
        img_array: 5D numpy array of shape (num_rows, num_columns, num_timepoints, height, width)
        radius_fraction: float between 0 and 1. Fraction of the maximum disk radius
            to use for the mask.
    Output:
        disk_mask: 2D numpy boolean array of shape (height, width)
    """
    assert 1 >= radius_fraction > 0, "radius_fraction must be between 0 and 1"

    crop_dims = img_array.shape[-2:]
    CELL_WIDTH_X = crop_dims[0]
    CELL_WIDTH_Y = crop_dims[1]
    max_disk_radius = min(crop_dims) // 2
    disk_radius = int(radius_fraction * max_disk_radius)
    mask = np.zeros((CELL_WIDTH_X, CELL_WIDTH_Y), dtype=bool)
    for i in range(CELL_WIDTH_X):
        for j in range(CELL_WIDTH_Y):
            # formula for a circle centered at the center of the rectangle
            # with the given dimensions
            if (i - CELL_WIDTH_X / 2) ** 2 + (j - CELL_WIDTH_Y / 2) ** 2 < (
                disk_radius
            ) ** 2:
                mask[i, j] = 1
    return mask
