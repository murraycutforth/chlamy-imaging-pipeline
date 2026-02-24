import numpy as np

from chlamy_impi.lib.y2_functions import subtract_background, compute_masked_mean


def compute_all_npq_averaged(img_array, mask_array) -> np.array:
    """Compute average NPQ for each well in an entire plate
    Returns a 3D numpy array of shape (Ni, Nj, num_steps)
    """
    img_array = subtract_background(img_array, mask_array)

    # TODO: smooth each 2D image?

    # Compute pixelwise NPQ values, for every pixel, ignoring mask
    Fm_array = img_array[:, :, 1:2, ...]
    Fm_prime_array = img_array[:, :, 3::2, ...]  # Skip Fm
    npq_array = (Fm_array - Fm_prime_array) / Fm_prime_array
    num_steps = Fm_prime_array.shape[2]
    assert num_steps == npq_array.shape[2]

    npq_array_means = compute_masked_mean(mask_array, num_steps, npq_array)

    assert npq_array_means.shape == (img_array.shape[0], img_array.shape[1], num_steps)
    assert np.nanmax(npq_array_means) < 10.0, f'NPQ greater than 10: {np.nanmax(npq_array_means)}'
    assert np.nanmin(npq_array_means) > -2, f'NPQ smaller than -2: {np.nanmin(npq_array_means)}'

    return npq_array_means


def compute_all_ynpq_averaged(img_array, mask_array) -> np.array:
    """Compute average Y(NPQ) for each well in an entire plate
    Returns a 3D numpy array of shape (Ni, Nj, num_steps)
    """
    img_array = subtract_background(img_array, mask_array)

    # Compute pixelwise Y(NPQ) values, for every pixel, ignoring mask
    # Y(NPQ) = F/Fm' - F/Fm
    F = img_array[:, :, 2::2, ...]  # Skip F0
    Fm = img_array[:, :, 1, ...]
    # broadcast Fm to the same shape as F
    Fm = Fm[:, :, np.newaxis, ...]
    Fm_prime = img_array[:, :, 3::2, ...]
    ynpq_array = (F / Fm_prime) - (F / Fm)
    num_steps = Fm_prime.shape[2]
    assert num_steps == ynpq_array.shape[2]

    ynpq_array_means = compute_masked_mean(mask_array, num_steps, ynpq_array)

    assert ynpq_array_means.shape == (img_array.shape[0], img_array.shape[1], num_steps)
    assert np.nanmax(ynpq_array_means) < 2.0, f'Y_npq greater than 2: {np.nanmax(ynpq_array_means)}'
    assert np.nanmin(ynpq_array_means) > -2.0, f"Y_npq smaller than -2: {np.nanmin(ynpq_array_means)}"

    return ynpq_array_means
