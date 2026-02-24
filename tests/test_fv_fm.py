import unittest
import numpy as np
from chlamy_impi.lib.fv_fm_functions import compute_all_fv_fm_averaged


class TestComputeAllFvFmAveraged(unittest.TestCase):

    def test_constant_input(self):
        img_array = np.ones((16, 24, 100, 2, 2))
        mask_array = np.ones((16, 24, 2, 2), dtype=bool)

        # Set background intensity used for control by setting top left well to 0
        mask_array[0, 0] = False
        img_array[0, 0] = 0.0

        result = compute_all_fv_fm_averaged(img_array, mask_array)
        self.assertEqual(result.shape, (16, 24))
        self.assertTrue(np.all(result.reshape(-1)[1:] == 0.0))

    def test_small_f0_fm(self):
        img_array = np.ones((2, 2, 100, 2, 2))
        mask_array = np.ones((2, 2, 2, 2), dtype=bool)

        # Set background intensity used for control by setting top left well to 0
        mask_array[0, 0] = False
        img_array[0, 0] = 0.0

        # Set Fmin to 0, so fv = fm
        img_array[:, :, 0, :, :] = 0.0

        result = compute_all_fv_fm_averaged(img_array, mask_array)
        self.assertEqual(result.shape, (2, 2))
        self.assertTrue(np.all(result.reshape(-1)[1:] == 1.0))


if __name__ == '__main__':
    unittest.main()