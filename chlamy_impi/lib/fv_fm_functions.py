import numpy as np
from skimage import filters

from chlamy_impi.lib.y2_functions import subtract_background


def compute_all_fv_fm_averaged(img_array, mask_array, return_std: bool = False) -> np.array:
    """Compute average Y2 for each well in an entire plate

    Args:
        img_array: 5D numpy array of shape (Ni, Nj, num_steps, res_x, res_y) containing fluorescence images
        mask_array: 4D numpy array of shape (Ni, Nj, res_x, res_y) containing boolean values
        return_std: If True, return standard deviation

    Returns a 3D numpy array of shape (Ni, Nj, num_steps) with fv/fm for each well, optionally also standard deviation
    of the pixelwise values.
    """

    img_array = subtract_background(img_array, mask_array)

    f0_array = img_array[:, :, 0, ...]
    fm_array = img_array[:, :, 1, ...]
    fv_array = fm_array - f0_array
    fv_fm_array = fv_array / fm_array

    fv_fm_array_means = compute_masked_mean(mask_array, fv_fm_array)

    assert fv_fm_array_means.shape == (img_array.shape[0], img_array.shape[1])
    assert np.nanmax(fv_fm_array_means) < 2.0, f'fv/fm greater than 2: {np.nanmax(fv_fm_array_means)}'
    assert np.nanmin(fv_fm_array_means) > -2.0, f"fv/fm smaller than -2: {np.nanmin(fv_fm_array_means)}"

    if return_std:
        fv_fm_array_stds = compute_masked_std(mask_array, fv_fm_array)
        return fv_fm_array_means, fv_fm_array_stds

    return fv_fm_array_means


def compute_masked_mean(mask_array: np.array, vals_array: np.array) -> np.array:
    """Set pixels outside mask to nan, and take mean of non-nan pixels

    :param mask_array: 4D numpy array of shape (Ni, Nj, 20, 20) containing boolean values
    :param num_steps: Number of time steps
    :param vals_array: 4D numpy array of shape (Ni, Nj, 20, 20) containing values to be averaged inside mask
    :return: 2D numpy array of shape (Ni, Nj)
    """
    assert len(mask_array.shape) == 4
    assert len(vals_array.shape) == 4
    assert mask_array.shape == vals_array.shape

    Ni, Nj = mask_array.shape[:2]
    vals_array[~mask_array] = np.nan

    fv_fm_array_means = np.nanmean(vals_array.reshape(Ni, Nj, -1), axis=-1)
    return fv_fm_array_means


def compute_masked_std(mask_array: np.array, vals_array: np.array) -> np.array:
    """Set pixels outside mask to nan, and take standard deviation of non-nan pixels

    :param mask_array: 4D numpy array of shape (Ni, Nj, 20, 20) containing boolean values
    :param num_steps: Number of time steps
    :param vals_array: 4D numpy array of shape (Ni, Nj, 20, 20) containing values to be averaged inside mask
    :return: 2D numpy array of shape (Ni, Nj)
    """
    assert len(mask_array.shape) == 4
    assert len(vals_array.shape) == 4
    assert mask_array.shape == vals_array.shape

    Ni, Nj = mask_array.shape[:2]
    vals_array[~mask_array] = np.nan

    fv_fm_array_stds = np.nanstd(vals_array.reshape(Ni, Nj, -1), axis=-1)
    return fv_fm_array_stds



#def compute_pixelwise_fv_fm(arr_0, arr_1, arr_mask, cntrl_f0, cntrl_fm) -> np.array:
#    """Compute fv/fm value for each pixel given a single well.
#    Returns a 1D array of fv/fm values for pixels inside the mask.
#    """
#    assert arr_mask.shape == arr_0.shape
#    assert arr_mask.shape == arr_1.shape
#
#    # I have removed the smoothing, since we are already averaging over
#    #arr_0 = filters.gaussian(arr_0, sigma=1, channel_axis=None)
#    #arr_1 = filters.gaussian(arr_1, sigma=1, channel_axis=None)
#
#    f0_arr = arr_0[arr_mask] - cntrl_f0
#    fm_arr = arr_1[arr_mask] - cntrl_fm
#    fv_arr = fm_arr - f0_arr
#
#    return fv_arr / fm_arr
#
#
#def compute_all_fv_fm_averaged(img_array, mask_array) -> np.array:
#    """Compute average fv/fm for each well in an entire plate
#
#    Args:
#        img_array: 5D numpy array of shape (Ni, Nj, num_steps, res_x, res_y) containing fluorescence images
#        mask_array: 2D numpy array of shape (Ni, Nj) containing boolean values
#
#    Returns a 2D numpy array of shape (Ni, Nj)
#    """
#    cntrl_f0, cntrl_fm = get_background_intensity(img_array, mask_array)
#
#    all_fv_fm = np.zeros(shape=img_array.shape[:2], dtype=np.float32)
#
#    for i in range(img_array.shape[0]):
#        for j in range(img_array.shape[1]):
#            all_fv_fm[i, j] = np.mean(
#                compute_pixelwise_fv_fm(
#                    img_array[i, j, 0],
#                    img_array[i, j, 1],
#                    mask_array[i, j],
#                    cntrl_f0,
#                    cntrl_fm,
#                )
#            )
#
#    return all_fv_fm
#
#
#def compute_all_fv_fm_pixelwise(img_array, mask_array) -> np.array:
#    """Compute pixelwise fv/fm for each well in an entire plate. Outside the mask, the value is set to NaN."""
#    cntrl_f0, cntrl_fm = get_background_intensity(img_array, mask_array)
#
#    all_fv_fm = np.zeros_like(img_array[:, :, 0, ...])
#
#    for i in range(img_array.shape[0]):
#        for j in range(img_array.shape[1]):
#            fv_fm_nonzero = compute_pixelwise_fv_fm(
#                img_array[i, j, 0],
#                img_array[i, j, 1],
#                mask_array[i, j],
#                cntrl_f0,
#                cntrl_fm,
#            )
#
#            all_fv_fm[i, j][mask_array[i, j]] = fv_fm_nonzero
#            all_fv_fm[i, j][~mask_array[i, j]] = np.nan
#
#    return all_fv_fm
#
#
#def get_background_intensity(img_array, mask_array):
#    if not np.any(mask_array[0, 0]):
#        cntrl_f0 = np.mean(img_array[0, 0, 0])  # Use mean of blank well
#        cntrl_fm = np.mean(img_array[0, 0, 1])
#    else:
#        cntrl_f0 = np.median(
#            img_array[:, :, 0, ...]
#        )  # Fall back to global median intensity
#        cntrl_fm = np.median(img_array[:, :, 1, ...])
#
#    assert (
#        abs(cntrl_f0 - cntrl_fm) < 20.0
#    ), f"f0 control: {cntrl_f0}, fm control: {cntrl_fm}"
#
#    return cntrl_f0, cntrl_fm
