import unittest
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from segment_multiwell_plate.segment_multiwell_plate import _average_d_min, find_well_centres, \
    _generate_grid_crop_coordinates, correct_image_rotation

from chlamy_impi.database_creation.utils import parse_name
from chlamy_impi.paths import find_all_cleaned_tif_images, CLEANED_RAW_DATA_DIR
from chlamy_impi.well_segmentation_preprocessing.main import load_image
from chlamy_impi.well_segmentation_preprocessing.well_segmentation_assertions import assert_expected_shape

# Output directory for rotation correction visualisations
_OUTPUT_DIR = Path(__file__).parent.parent / "output" / "rotation_correction"

_BLOB_LOG_KWARGS = {"threshold": 0.12, "min_sigma": 1, "max_sigma": 2, "exclude_border": 10}
_PEAK_FINDER_KWARGS = {"peak_prominence": 1 / 25, "filter_threshold": 0.2, "width": 2}

# Number of plates to sample in the broad 2-D test
_N_PLATES = 5


class TestRotationCorrection(unittest.TestCase):

    def test_rotation_correction_real_data_2d(self):
        """Run rotation correction on the first N cleaned plates and save diagnostic PNGs.

        Saves one PNG per plate to output/rotation_correction/.
        """
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        filenames = find_all_cleaned_tif_images()[:_N_PLATES]

        if not filenames:
            self.skipTest("No cleaned TIF files found — run Stage 0 first")

        for im_path in filenames:
            with self.subTest(plate=im_path.stem):
                tif = load_image(im_path)
                im = tif[0]

                assert im.shape == (480, 640)
                assert np.std(im) > 1e-6

                well_coords = find_well_centres(im, **_BLOB_LOG_KWARGS)
                rotated_im, rotation_angle = correct_image_rotation(im, well_coords, _BLOB_LOG_KWARGS)
                rotated_well_coords = find_well_centres(rotated_im, **_BLOB_LOG_KWARGS)

                i_vals, j_vals = _generate_grid_crop_coordinates(im, well_coords, **_PEAK_FINDER_KWARGS)
                i_vals_rotated, j_vals_rotated = _generate_grid_crop_coordinates(
                    rotated_im, rotated_well_coords, **_PEAK_FINDER_KWARGS
                )

                fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=200)
                fig.suptitle(f'{im_path.stem}. Rotation: {rotation_angle:.2g} rad')
                axs[0].imshow(im)
                axs[0].scatter(well_coords[:, 1], well_coords[:, 0], s=1, c='red')
                iv, jv = np.meshgrid(i_vals, j_vals, indexing="ij")
                iv2, jv2 = np.meshgrid(i_vals, j_vals, indexing="xy")
                axs[0].plot(jv, iv, color="red")
                axs[0].plot(jv2, iv2, color="red")
                axs[0].set_title(f"{len(i_vals)} x {len(j_vals)}")
                axs[1].imshow(rotated_im)
                axs[1].scatter(rotated_well_coords[:, 1], rotated_well_coords[:, 0], s=1, c='red')
                iv, jv = np.meshgrid(i_vals_rotated, j_vals_rotated, indexing="ij")
                iv2, jv2 = np.meshgrid(i_vals_rotated, j_vals_rotated, indexing="xy")
                axs[1].plot(jv, iv, color="red")
                axs[1].plot(jv2, iv2, color="red")
                axs[1].set_title(f"{len(i_vals_rotated)} x {len(j_vals_rotated)}")
                fig.tight_layout()
                fig.savefig(_OUTPUT_DIR / f"{im_path.stem}_rotated.png")
                plt.close()

                plate_num, _, _ = parse_name(im_path.name)
                assert_expected_shape(i_vals_rotated, j_vals_rotated, plate_num)

    def test_rotation_correction_problem_case_3d(self):
        """Run 3-D rotation correction on the known-difficult plate 1 M5."""
        tif_path = CLEANED_RAW_DATA_DIR / "20240506_1-M5_10min-10min.tif"

        if not tif_path.exists():
            self.skipTest(f"Test plate not found: {tif_path}")

        tif = load_image(tif_path)

        assert len(tif.shape) == 3
        assert np.std(tif) > 1e-6

        well_coords = find_well_centres(tif, **_BLOB_LOG_KWARGS)
        rotated_im, rotation_angle = correct_image_rotation(tif, well_coords, _BLOB_LOG_KWARGS)
        rotated_well_coords = find_well_centres(rotated_im, **_BLOB_LOG_KWARGS)
        i_vals_rotated, j_vals_rotated = _generate_grid_crop_coordinates(
            rotated_im, rotated_well_coords, **_PEAK_FINDER_KWARGS
        )

        plate_num, _, _ = parse_name(tif_path.name)
        assert_expected_shape(i_vals_rotated, j_vals_rotated, plate_num)

    @unittest.skip("Slow test")
    def test_rotation_correction_real_data_3d(self):
        filenames = find_all_cleaned_tif_images()

        for im_path in filenames:
            tif = load_image(im_path)

            assert len(tif.shape) == 3
            assert np.std(tif) > 1e-6

            well_coords = find_well_centres(tif, **_BLOB_LOG_KWARGS)
            rotated_im, rotation_angle = correct_image_rotation(tif, well_coords, _BLOB_LOG_KWARGS)
            rotated_well_coords = find_well_centres(rotated_im, **_BLOB_LOG_KWARGS)
            i_vals_rotated, j_vals_rotated = _generate_grid_crop_coordinates(
                rotated_im, rotated_well_coords, **_PEAK_FINDER_KWARGS
            )

            plate_num, _, _ = parse_name(im_path.name)
            assert_expected_shape(i_vals_rotated, j_vals_rotated, plate_num)

    def test_average_d_min(self):
        x = np.linspace(0, 10, 11)
        y = np.linspace(0, 5, 6)
        X, Y = np.meshgrid(x, y)
        points = np.stack([X.flatten(), Y.flatten()], axis=-1)

        d_min = _average_d_min(points)
        self.assertAlmostEqual(d_min, 1, places=5)

        for theta in np.random.uniform(-0.05, 0.05, 100):
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            points_rotated = points @ rotation_matrix.T
            d_min_rotated = _average_d_min(points_rotated)
            self.assertGreater(d_min_rotated, d_min)
