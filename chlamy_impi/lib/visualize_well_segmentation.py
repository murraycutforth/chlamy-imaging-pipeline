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


def visualise_well_mosaic(img_array, name, savepath):
    """Render all wells in a plate as a 16×24 contact-sheet mosaic.

    Each cell shows the mean of the Fm (light) frames for that well, using a
    shared intensity scale so empty wells appear dark and filled wells bright.

    Args:
        img_array: 5D array of shape (Ni, Nj, n_frames, H, W), float32
        name:      plate name string, used as figure title
        savepath:  Path where the PNG will be written
    """
    savepath.parent.mkdir(parents=True, exist_ok=True)

    Ni, Nj, n_frames, H, W = img_array.shape
    light_idxs = list(range(1, n_frames, 2))

    # Mean Fm image per well, shape (Ni, Nj, H, W)
    mean_fm = img_array[:, :, light_idxs, :, :].mean(axis=2)

    vmin = float(np.percentile(mean_fm, 1))
    vmax = float(np.percentile(mean_fm, 99))

    fig, axs = plt.subplots(Ni, Nj, figsize=(Nj * 0.6, Ni * 0.6))
    fig.suptitle(f"{name} — mean Fm per well", fontsize=8, y=1.01)

    for i in range(Ni):
        for j in range(Nj):
            ax = axs[i, j]
            ax.imshow(mean_fm[i, j], cmap="turbo", vmin=vmin, vmax=vmax,
                      interpolation="nearest")
            ax.axis("off")

    # Row labels (A–P) on left edge, column labels (1–24) on top edge
    row_labels = [chr(ord("A") + i) for i in range(Ni)]
    col_labels = [str(j + 1) for j in range(Nj)]
    for i, label in enumerate(row_labels):
        axs[i, 0].set_ylabel(label, fontsize=5, rotation=0, labelpad=4,
                             va="center")
        axs[i, 0].yaxis.set_visible(True)
        axs[i, 0].set_yticks([])
    for j, label in enumerate(col_labels):
        axs[0, j].set_title(label, fontsize=4, pad=2)

    fig.subplots_adjust(wspace=0.04, hspace=0.04)
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    fig.clf()
    plt.close(fig)
    logger.info(f"Saved well mosaic to {savepath}")


def visualise_mask_mosaic(mask_array, name, savepath):
    """Render all well masks in a plate as a 16×24 contact-sheet mosaic.

    White pixels are inside the mask (cells); black pixels are background.

    Args:
        mask_array: 4D bool array of shape (Ni, Nj, H, W)
        name:       plate name string, used as figure title
        savepath:   Path where the PNG will be written
    """
    savepath.parent.mkdir(parents=True, exist_ok=True)

    Ni, Nj = mask_array.shape[:2]

    fig, axs = plt.subplots(Ni, Nj, figsize=(Nj * 0.6, Ni * 0.6))
    fig.suptitle(f"{name} — masks", fontsize=8, y=1.01)

    for i in range(Ni):
        for j in range(Nj):
            ax = axs[i, j]
            ax.imshow(mask_array[i, j], cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.axis("off")

    row_labels = [chr(ord("A") + i) for i in range(Ni)]
    col_labels = [str(j + 1) for j in range(Nj)]
    for i, label in enumerate(row_labels):
        axs[i, 0].set_ylabel(label, fontsize=5, rotation=0, labelpad=4,
                             va="center")
        axs[i, 0].yaxis.set_visible(True)
        axs[i, 0].set_yticks([])
    for j, label in enumerate(col_labels):
        axs[0, j].set_title(label, fontsize=4, pad=2)

    fig.subplots_adjust(wspace=0.04, hspace=0.04)
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    fig.clf()
    plt.close(fig)
    logger.info(f"Saved mask mosaic to {savepath}")


def visualise_mask_heatmap(mask_area_2d, name, savepath):
    """Render a 16×24 heatmap of mask area (pixel count) for a plate.

    Args:
        mask_area_2d: 2D int array of shape (Ni, Nj) — masked pixel count per well
        name:         plate name string, used as figure title
        savepath:     Path where the PNG will be written
    """
    savepath.parent.mkdir(parents=True, exist_ok=True)

    Ni, Nj = mask_area_2d.shape
    row_labels = [chr(ord("A") + i) for i in range(Ni)]
    col_labels = [str(j + 1) for j in range(Nj)]

    fig, ax = plt.subplots(figsize=(Nj * 0.5, Ni * 0.5))
    im = ax.imshow(mask_area_2d, cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax, label="Masked pixels", shrink=0.6)
    ax.set_title(f"{name} — mask area", fontsize=9)
    ax.set_xticks(range(Nj))
    ax.set_xticklabels(col_labels, fontsize=5)
    ax.set_yticks(range(Ni))
    ax.set_yticklabels(row_labels, fontsize=6)

    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    fig.clf()
    plt.close(fig)
    logger.info(f"Saved mask heatmap to {savepath}")


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
