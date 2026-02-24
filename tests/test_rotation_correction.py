import unittest

import numpy as np
import matplotlib.pyplot as plt
from segment_multiwell_plate.segment_multiwell_plate import _average_d_min, find_well_centres, \
    _generate_grid_crop_coordinates, correct_image_rotation

from chlamy_impi.database_creation.manual_error_correction import remove_failed_photos, remove_repeated_initial_frame_tif
from chlamy_impi.database_creation.utils import parse_name
from chlamy_impi.paths import find_all_tif_images
from chlamy_impi.well_segmentation_preprocessing.main import load_image
from chlamy_impi.well_segmentation_preprocessing.well_segmentation_assertions import assert_expected_shape


class TestRotationCorrection(unittest.TestCase):
    def test_rotation_correction_real_data_2d(self):
        # Use this test case to tune parameters on all downloaded data
        # We extract the well grid, and assert that it has the shape which we expect!

        filenames = find_all_tif_images()
        filenames.sort()
        blob_log_kwargs = {"threshold": 0.12, "min_sigma": 1, "max_sigma": 2, "exclude_border": 10}
        peak_finder_kwargs = {"peak_prominence": 1 / 25, "filter_threshold": 0.2, "width":2}

        print(f"Found {len(filenames)} tif files")

        for im_path in filenames:
            print(im_path.stem)
            tif = load_image(im_path)
            tif = remove_failed_photos(tif)
            tif = remove_repeated_initial_frame_tif(tif)
            im = tif[0]

            assert im.shape == (480, 640)
            assert np.std(im) > 1e-6

            well_coords = find_well_centres(im, **blob_log_kwargs)

            rotated_im, rotation_angle = correct_image_rotation(im, well_coords, blob_log_kwargs)

            rotated_well_coords = find_well_centres(rotated_im, **blob_log_kwargs)

            i_vals, j_vals = _generate_grid_crop_coordinates(im, well_coords, **peak_finder_kwargs)
            i_vals_rotated, j_vals_rotated = _generate_grid_crop_coordinates(rotated_im, rotated_well_coords, **peak_finder_kwargs)

            def plot_well_segmentation_result():
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
                fig.savefig(f"{im_path.stem}_rotated.png")
                plt.close()

            plot_well_segmentation_result()

            plate_num, _, _ = parse_name(im_path.name)
            assert_expected_shape(i_vals_rotated, j_vals_rotated, plate_num)

    def test_rotation_correction_problem_case_3d(self):
        filenames = find_all_tif_images()
        filenames.sort()
        blob_log_kwargs = {"threshold": 0.12, "min_sigma": 1, "max_sigma": 2, "exclude_border": 10}
        peak_finder_kwargs = {"peak_prominence": 1 / 25, "filter_threshold": 0.2, "width":2}

        for im_path in filenames:
            plate_num, measurement_num, _ = parse_name(im_path.name)

            if not (plate_num == 1 and measurement_num == "M5"):
                continue

            tif = load_image(im_path)
            tif = remove_failed_photos(tif)
            tif = remove_repeated_initial_frame_tif(tif)

            assert len(tif.shape) == 3
            assert np.std(tif) > 1e-6

            well_coords = find_well_centres(tif, **blob_log_kwargs)

            rotated_im, rotation_angle = correct_image_rotation(tif, well_coords, blob_log_kwargs)

            rotated_well_coords = find_well_centres(rotated_im, **blob_log_kwargs)

            i_vals_rotated, j_vals_rotated = _generate_grid_crop_coordinates(rotated_im, rotated_well_coords, **peak_finder_kwargs)

            assert_expected_shape(i_vals_rotated, j_vals_rotated, plate_num)

    @unittest.skip("Slow test")
    def test_rotation_correction_real_data_3d(self):
        # Use this test case to tune parameters on all downloaded data
        # We extract the well grid, and assert that it has the shape which we expect!

        filenames = find_all_tif_images()
        filenames.sort()
        blob_log_kwargs = {"threshold": 0.12, "min_sigma": 1, "max_sigma": 2, "exclude_border": 10}
        peak_finder_kwargs = {"peak_prominence": 1 / 25, "filter_threshold": 0.2, "width":2}

        print(f"Found {len(filenames)} tif files")

        for im_path in filenames:
            print(im_path.stem)
            tif = load_image(im_path)
            tif = remove_failed_photos(tif)
            tif = remove_repeated_initial_frame_tif(tif)

            assert len(tif.shape) == 3
            assert np.std(tif) > 1e-6

            well_coords = find_well_centres(tif, **blob_log_kwargs)

            rotated_im, rotation_angle = correct_image_rotation(tif, well_coords, blob_log_kwargs)

            rotated_well_coords = find_well_centres(rotated_im, **blob_log_kwargs)

            #i_vals, j_vals = _generate_grid_crop_coordinates(tif, well_coords, **peak_finder_kwargs)
            i_vals_rotated, j_vals_rotated = _generate_grid_crop_coordinates(rotated_im, rotated_well_coords, **peak_finder_kwargs)

            plate_num, _, _ = parse_name(im_path.name)
            assert_expected_shape(i_vals_rotated, j_vals_rotated, plate_num)

    def test_average_d_min(self):
        # Test that the average_d_min function works as expected
        # This is just a copy of the test from segment_multiwell_plate, to verify that the import works correctly
        x = np.linspace(0, 10, 11)
        y = np.linspace(0, 5, 6)
        X, Y = np.meshgrid(x, y)
        points = np.stack([X.flatten(), Y.flatten()], axis=-1)

        # Test that the average distance between points is correct
        d_min = _average_d_min(points)
        self.assertAlmostEqual(d_min, 1, places=5)

        # Now rotate this grid a tiny bit
        for theta in np.random.uniform(-0.05, 0.05, 100):
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            points_rotated = points @ rotation_matrix.T

            # Test that the average distance between points is larger than before
            d_min_rotated = _average_d_min(points_rotated)
            self.assertGreater(d_min_rotated, d_min)

