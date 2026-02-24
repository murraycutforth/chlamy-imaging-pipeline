import numpy as np
from matplotlib import pyplot as plt


def get_background_intensity(img_array, mask_array) -> np.array:
    """Get a stack of background intensities for every time step, using the top left well
    Returns a 1D numpy array of shape (num_steps,)
    """
    num_steps = img_array.shape[2]

    if not np.any(mask_array[0, 0]):
        backgrounds = np.mean(img_array[0, 0].reshape(num_steps, -1), axis=-1)
    else:
        # Fall back to median of entire image if there is not a blank in top left
        backgrounds = np.median(
            np.swapaxes(img_array, 2, 0).reshape(num_steps, -1), axis=-1
        )

    assert len(backgrounds.shape) == 1
    assert len(backgrounds) == img_array.shape[2]

    return backgrounds


def compute_all_y2_averaged(img_array, mask_array, return_std: bool = False) -> np.array:
    """Compute average Y2 for each well in an entire plate
    Returns a 3D numpy array of shape (Ni, Nj, num_steps)
    """

    img_array = subtract_background(img_array, mask_array)

    # TODO: smooth each 2D image?

    # Compute pixelwise Y2 values, for every pixel, ignoring mask
    Fm_prime_array = img_array[:, :, 3::2, ...]  # Skip Fm
    F_array = img_array[:, :, 2::2, ...]  # Skip F0
    y2_array = (Fm_prime_array - F_array) / Fm_prime_array  # shape (Ni, Nj, num_steps, 20, 20)
    num_steps = Fm_prime_array.shape[2]
    assert num_steps == F_array.shape[2]

    y2_array_means = compute_masked_mean(mask_array, num_steps, y2_array)

    assert y2_array_means.shape == (img_array.shape[0], img_array.shape[1], num_steps)
    assert np.nanmax(y2_array_means) < 2.0, f'Y2 greater than 2: {np.nanmax(y2_array_means)}'
    assert np.nanmin(y2_array_means) > -2.0, f"Y2 smaller than -2: {np.nanmin(y2_array_means)}"

    # Check that the average Y2 of all wells at each time step is positive
    for i in range(num_steps):
        assert np.nanmean(y2_array_means[:, :, i]) > -1, f"Y2 mean is very negative at time step {i}"

    if return_std:
        y2_array_stds = compute_masked_std(mask_array, num_steps, y2_array)
        return y2_array_means, y2_array_stds

    return y2_array_means


def compute_masked_mean(
    mask_array: np.array, num_steps: int, vals_array: np.array
) -> np.array:
    """Set pixels outside mask to nan, and take mean of non-nan pixels

    :param mask_array: 4D numpy array of shape (Ni, Nj, 20, 20) containing boolean values
    :param num_steps: Number of time steps
    :param vals_array: 5D numpy array of shape (Ni, Nj, num_steps, 20, 20) containing values to be averaged inside mask
    :return: 3D numpy array of shape (Ni, Nj, num_steps)
    """
    assert len(mask_array.shape) == 4
    assert len(vals_array.shape) == 5
    assert mask_array.shape[:2] == vals_array.shape[:2]
    assert mask_array.shape[2:] == vals_array.shape[3:]
    assert num_steps == vals_array.shape[2]
    assert num_steps > 0

    Ni, Nj = mask_array.shape[:2]
    mask_array = np.broadcast_to(
        mask_array[:, :, np.newaxis, ...],
        (Ni, Nj, num_steps, mask_array.shape[-2], mask_array.shape[-1]),
    )
    vals_array[~mask_array] = np.nan

    y2_array_means = np.nanmean(vals_array.reshape(Ni, Nj, num_steps, -1), axis=-1)
    return y2_array_means


def compute_masked_std(
    mask_array: np.array, num_steps: int, vals_array: np.array
) -> np.array:
    """Set pixels outside mask to nan, and take std of non-nan pixels

    :param mask_array: 4D numpy array of shape (Ni, Nj, 20, 20) containing boolean values
    :param num_steps: Number of time steps
    :param vals_array: 5D numpy array of shape (Ni, Nj, num_steps, 20, 20) containing values to be averaged inside mask
    :return: 3D numpy array of shape (Ni, Nj, num_steps)
    """
    assert len(mask_array.shape) == 4
    assert len(vals_array.shape) == 5
    assert mask_array.shape[:2] == vals_array.shape[:2]
    assert mask_array.shape[2:] == vals_array.shape[3:]
    assert num_steps == vals_array.shape[2]
    assert num_steps > 0

    Ni, Nj = mask_array.shape[:2]
    mask_array = np.broadcast_to(
        array=mask_array[:, :, np.newaxis, ...],
        shape=(Ni, Nj, num_steps, mask_array.shape[-2], mask_array.shape[-1]),
    )
    vals_array[~mask_array] = np.nan

    y2_array_stds = np.nanstd(vals_array.reshape(Ni, Nj, num_steps, -1), axis=-1)
    return y2_array_stds


def subtract_background(img_array: np.array, mask_array: np.array) -> np.array:
    """Subtract background light intensity from each time point"""
    assert len(img_array.shape) == 5
    assert len(mask_array.shape) == 4

    backgrounds = get_background_intensity(img_array, mask_array)
    img_array = img_array - backgrounds[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
    return img_array
