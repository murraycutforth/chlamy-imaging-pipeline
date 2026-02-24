import itertools
import logging

import numpy as np
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


def visualise_channels(tif, savedir, max_channels=None):
    logger.debug(f"Writing out plots of all time points in {savedir}")

    savedir.mkdir(parents=True, exist_ok=True)

    shape = tif.shape

    if max_channels is None:
        max_channels = shape[0]

    for channel in range(min(shape[0], max_channels)):
        fig, ax = plt.subplots(1, 1)
        im = ax.imshow(tif[channel, :, :], vmin=0, vmax=500)
        fig.colorbar(im, ax=ax)
        fig.savefig(savedir / f"{channel}.png")
        fig.clf()
        plt.close(fig)

    for channel in range(0, min(shape[0], max_channels) - 1, 2):
        fig, ax = plt.subplots(1, 1)
        diff = tif[channel + 1, :, :].astype(np.float32) - tif[channel, :, :].astype(np.float32)
        im = ax.imshow(diff, vmin=-50, vmax=250)
        fig.colorbar(im, ax=ax)
        fig.savefig(savedir / f"difference_{channel + 1}_{channel}.png")
        fig.clf()
        plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    ax.imshow(np.mean(tif, axis=0))
    fig.savefig(savedir / f"avg.png")
    fig.clf()
    plt.close(fig)


def visualise_well_histograms(img_array, name, savedir):

    savedir.mkdir(parents=True, exist_ok=True)
    array_shape = img_array.shape

    all_hists = []

    for i, j in itertools.product(range(array_shape[0]), range(array_shape[1])):
        well_arr_vals = img_array[i, j].ravel()
        hist, edges = np.histogram(well_arr_vals, bins=500, range=(0, 500))
        all_hists.append(hist)

    fig, ax = plt.subplots(1, 1)

    for vals in all_hists:
        ax.plot(edges[:-1], vals, linewidth=1, c="black", alpha=0.2)

    ax.set_yscale("log")
    ax.set_ylabel("Number of pixels")
    ax.set_xlabel("Intensity")
    ax.set_title(f"{name}: intensity histogram of wells")

    fig.savefig(savedir / f"{name}_well_hists.png")
    fig.clf()
    plt.close(fig)


def visualise_grid_crop(tif, img_array, i_vals, j_vals, well_coords, savedir, max_channels=5):
    logger.debug(f"Writing out plots of grid crop in {savedir}")
    savedir.mkdir(parents=True, exist_ok=True)

    img_shape = tif.shape
    array_shape = img_array.shape

    iv, jv = np.meshgrid(i_vals, j_vals, indexing="ij")
    iv2, jv2 = np.meshgrid(i_vals, j_vals, indexing="xy")

    for channel in range(min(img_shape[0], max_channels)):
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(tif[channel, :, :])
        # Draw well centre coords
        ax.scatter(list(zip(*well_coords))[1], list(zip(*well_coords))[0], facecolors="red", edgecolors=None, marker="o", s=8)
        # Draw grid
        ax.plot(jv, iv, color="red")
        ax.plot(jv2, iv2, color="red")
        fig.savefig(savedir / f"{channel}_grid.png")
        fig.clf()
        plt.close(fig)

        fig, axs = plt.subplots(array_shape[0], array_shape[1])
        for i, j in itertools.product(range(array_shape[0]), range(array_shape[1])):
            ax = axs[i, j]
            ax.axis("off")
            ax.imshow(img_array[i, j, channel], vmin=tif[channel].min(), vmax=tif[channel].max())
        fig.savefig(savedir / f"{channel}_subimage_array.png")
        fig.clf()
        plt.close(fig)


def visualise_mask_array(mask_array, savedir):
    logger.debug(f"Writing out plot of masks to {savedir}")
    savedir.mkdir(parents=True, exist_ok=True)

    array_shape = mask_array.shape

    fig, axs = plt.subplots(array_shape[0], array_shape[1])
    for i, j in itertools.product(range(array_shape[0]), range(array_shape[1])):
        ax = axs[i, j]
        ax.axis("off")
        ax.imshow(mask_array[i, j])
    fig.savefig(savedir / "mask_array.png")
    fig.clf()
    plt.close(fig)
